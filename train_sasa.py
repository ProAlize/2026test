# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for DiT using PyTorch DDP.
SASA-A-step-decay version:
- frozen local DINOv3 teacher (HF-style directory)
- DiT intermediate token extraction via forward hook (single forward pass)
- projector + cosine alignment loss
- training-phase linear decay by step (only step-based weighting)
- NO diffusion-timestep weighting (uniform across all timesteps)
- max-steps control
"""

import os
import argparse
import logging
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import AutoModel

from model_sasa import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    if dist.get_rank() == 0:
        handlers = [logging.StreamHandler()]
        if logging_dir is not None:
            handlers.append(logging.FileHandler(f"{logging_dir}/log.txt"))
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class REPAProjector(nn.Module):
    """REPA-faithful projector: 3-layer MLP with SiLU, default width 2048."""
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        # Keep old arg name for compatibility. REPA-faithful default is 2048.
        if hidden_dim is None:
            hidden_dim = 2048
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_train_phase_weight(current_step, schedule_steps, schedule="linear_decay"):
    """
    Scalar weight for the training-phase axis.
    Decays from 1.0 -> 0.0 over schedule_steps steps (linear_decay),
    or stays constant / hard-cutoff.
    """
    if schedule == "constant":
        return 1.0

    progress = min(max(current_step / max(1, schedule_steps), 0.0), 1.0)

    if schedule == "linear_decay":
        return 1.0 - progress
    if schedule == "cutoff":
        return 1.0 if current_step < schedule_steps else 0.0

    raise ValueError(f"Unknown train schedule: {schedule}")


class LocalDINOv3Teacher(nn.Module):
    """HF-style local DINOv3 teacher."""
    def __init__(self, model_dir):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        requires_grad(self.model, False)

    @torch.no_grad()
    def forward(self, x):
        out = self.model(pixel_values=x)
        h = out.last_hidden_state  # [B, N, C]

        n = h.shape[1]
        if n == 261:
            h = h[:, 5:, :]   # cls + 4 register + 256 patches
        elif n == 257:
            h = h[:, 1:, :]   # cls + 256 patches
        elif n == 256:
            pass
        else:
            raise ValueError(f"Unexpected DINO token count: {n}")
        return h


def cosine_align_loss(pred, target):
    """
    Uniform cosine alignment loss (no per-sample timestep weighting).
    pred:   [B, T, C]
    target: [B, T, C]
    returns: scalar
    """
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    cos = (pred * target).sum(dim=-1)  # [B, T]
    return (1.0 - cos).mean()


def preprocess_for_dino(x):
    """x: [B, 3, H, W] in [-1,1] -> normalized for DINOv3"""
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def extract_dino_patch_tokens(dino_model, x):
    with torch.inference_mode():
        tokens = dino_model(x)
    return tokens


def save_checkpoint(
    checkpoint_dir, train_steps, model, ema, opt, args, repa_projector=None
):
    checkpoint = {
        "model": model.module.state_dict(),
        "ema":   ema.state_dict(),
        "opt":   opt.state_dict(),
        "args":  args,
    }
    if repa_projector is not None:
        checkpoint["repa_projector"] = repa_projector.module.state_dict()

    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank   = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed   = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index    = len(glob(f"{args.results_dir}/*"))
        model_string_name   = args.model.replace("/", "-")
        experiment_name     = f"{experiment_index:03d}-{model_string_name}"
        if args.repa:
            experiment_name += (
                f"-sasa-lam{args.repa_lambda}"
                f"-train{args.repa_train_schedule}"
                f"-nodiffweight"
            )
        experiment_dir  = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir  = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger         = create_logger(None)
        checkpoint_dir = None

    # Create DiT
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model     = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")

    # VAE loading
    if args.vae_model_dir is not None:
        vae_path = args.vae_model_dir
        logger.info(f"Loading VAE from local directory: {vae_path}")
    else:
        vae_path = f"stabilityai/sd-vae-ft-{args.vae}"
        logger.info(f"Loading VAE from HuggingFace Hub: {vae_path}")

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    requires_grad(vae, False)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.max_steps is not None:
        logger.info(f"Training will stop at max_steps={args.max_steps}")

    # ---------------------------------------------------------------
    # REPA modules
    # ---------------------------------------------------------------
    dino_model     = None
    repa_projector = None
    # 确定 hook 注入的 block 索引（在 DDP wrap 之前确定，避免 module 层级变化）
    hook_layer_idx = None

    if args.repa:
        logger.info(
            "Initializing SASA alignment "
            "(single forward pass via hook, step-decay only, "
            "no diffusion-timestep weighting)..."
        )
        logger.info(f"Loading local DINOv3 teacher from: {args.dino_model_dir}")

        dino_model = LocalDINOv3Teacher(args.dino_model_dir).to(device)
        dino_model.eval()
        requires_grad(dino_model, False)

        dit_hidden_dim = model.hidden_size
        dit_depth      = model.depth

        # 默认 hook 最后一个 block
        hook_layer_idx = (
            args.repa_token_layer
            if args.repa_token_layer is not None
            else dit_depth - 1
        )

        with torch.inference_mode():
            dummy       = torch.randn(1, 3, args.image_size, args.image_size, device=device)
            dummy       = preprocess_for_dino(dummy)
            dino_tokens = extract_dino_patch_tokens(dino_model, dummy)
            dino_dim        = dino_tokens.shape[-1]
            dino_num_tokens = dino_tokens.shape[1]

        logger.info(
            f"REPA setup: DiT hidden dim = {dit_hidden_dim}, "
            f"DINO dim = {dino_dim}, DINO num_tokens = {dino_num_tokens}, "
            f"hook layer = {hook_layer_idx}/{dit_depth - 1}"
        )

        repa_projector = REPAProjector(
            in_dim=dit_hidden_dim,
            out_dim=dino_dim,
            hidden_dim=args.repa_hidden_dim
        ).to(device)

    # DDP wrap
    model = DDP(model, device_ids=[device])
    if args.repa:
        repa_projector = DDP(repa_projector, device_ids=[device])

    # Optimizer
    trainable_params = list(model.parameters())
    if args.repa:
        trainable_params += list(repa_projector.parameters())

    opt = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # Dataset
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, args.image_size)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    if args.repa:
        repa_projector.train()
        dino_model.eval()

    # Logging vars
    train_steps                = 0
    log_steps                  = 0
    running_loss               = 0.0
    running_loss_diff          = 0.0
    running_loss_align         = 0.0
    running_train_align_weight = 0.0
    start_time                 = time()

    logger.info(f"Training for {args.epochs} epochs (max_steps={args.max_steps})...")
    logger.info(
        f"Alignment schedule: train_phase={args.repa_train_schedule} "
        f"over {args.repa_schedule_steps} steps | "
        f"diffusion timestep weighting: DISABLED (uniform) | "
        f"forward hook on block[{hook_layer_idx}]"
    )

    done = False

    for epoch in range(args.epochs):
        if done:
            break

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for images, y in loader:
            images = images.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)
            current_step = train_steps

            with torch.no_grad():
                x = vae.encode(images).latent_dist.sample().mul_(0.18215)

            # ── 计算训练步数衰减权重 ────────────────────────────────────
            train_align_weight = 0.0
            if args.repa:
                train_align_weight = get_train_phase_weight(
                    current_step=current_step,
                    schedule_steps=args.repa_schedule_steps,
                    schedule=args.repa_train_schedule,
                )

            # ── 注册 hook（仅在需要对齐时） ─────────────────────────────
            # hook 在 training_losses 的单次前向中顺便抓取中间 token，
            # 避免重复做第二次完整前向，节省约 50% 显存。
            _captured_tokens = {}
            hook_handle      = None

            if args.repa and train_align_weight > 0:
                def _hook_fn(module, input, output):
                    # output: [B, T, D]，保留计算图供 projector 反向传播
                    _captured_tokens['tokens'] = output

                # model.module.blocks 对应 DDP 内部的原始 DiT
                hook_handle = model.module.blocks[hook_layer_idx].register_forward_hook(
                    _hook_fn
                )

            # ── 单次前向：diffusion loss（内部调用 model forward）──────
            t = torch.randint(
                0, diffusion.num_timesteps, (x.shape[0],), device=device
            ).long()
            model_kwargs = dict(y=y)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_diff = loss_dict["loss"].mean()

            # hook 已触发，立即移除，避免影响后续步骤
            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None

            # ── SASA 对齐 loss ──────────────────────────────────────────
            loss_align         = torch.tensor(0.0, device=device)
            alignment_computed = False

            if args.repa and train_align_weight > 0 and 'tokens' in _captured_tokens:
                dit_tokens = _captured_tokens['tokens']  # [B, T, D]，带梯度

                # Teacher tokens（clean RGB → DINO，无梯度）
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    dino_tokens = extract_dino_patch_tokens(dino_model, x_dino)

                # Projector：DiT token → DINO 特征空间
                proj_tokens = repa_projector(dit_tokens)  # [B, T, dino_dim]

                # Shape 校验
                if args.image_size == 256 and args.model.endswith("/2"):
                    assert dit_tokens.shape[1] == 256, \
                        f"Expected DiT token count 256, got {dit_tokens.shape[1]}"
                    assert dino_tokens.shape[1] == 256, \
                        f"Expected DINO token count 256, got {dino_tokens.shape[1]}"

                assert dino_tokens.ndim == 3, f"dino_tokens.ndim={dino_tokens.ndim}"
                assert dit_tokens.ndim  == 3, f"dit_tokens.ndim={dit_tokens.ndim}"
                assert proj_tokens.ndim == 3, f"proj_tokens.ndim={proj_tokens.ndim}"

                if proj_tokens.shape[1] != dino_tokens.shape[1]:
                    raise ValueError(
                        f"Token count mismatch: "
                        f"proj_tokens={proj_tokens.shape}, "
                        f"dino_tokens={dino_tokens.shape}"
                    )

                # 均匀 cosine 对齐（不按扩散时间步加权）
                loss_align = cosine_align_loss(proj_tokens, dino_tokens)
                alignment_computed = True

            # ── Total loss ──────────────────────────────────────────────
            # effective_lambda = base_lambda × train_phase_weight(step)
            # train_phase_weight 从 1.0 线性衰减到 0.0（在 schedule_steps 步内）
            effective_repa_lambda = args.repa_lambda * train_align_weight
            loss = loss_diff + effective_repa_lambda * loss_align

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Logging accumulators
            running_loss               += loss.item()
            running_loss_diff          += loss_diff.item()
            running_loss_align         += loss_align.item()
            running_train_align_weight += train_align_weight
            log_steps   += 1
            train_steps += 1

            # First-step shape log
            if train_steps == 1 and rank == 0:
                logger.info(f"images shape : {images.shape}")
                logger.info(f"latent shape : {x.shape}")
                if alignment_computed:
                    logger.info(f"dino_tokens  : {dino_tokens.shape}")
                    logger.info(f"dit_tokens   : {dit_tokens.shape}")
                    logger.info(f"proj_tokens  : {proj_tokens.shape}")

            # Periodic logging
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time      = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(
                    running_loss / log_steps, device=device)
                avg_loss_diff = torch.tensor(
                    running_loss_diff / log_steps, device=device)
                avg_loss_align = torch.tensor(
                    running_loss_align / log_steps, device=device)
                avg_train_align_weight = torch.tensor(
                    running_train_align_weight / log_steps, device=device)

                dist.all_reduce(avg_loss,               op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_diff,          op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_align,         op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_train_align_weight, op=dist.ReduceOp.SUM)

                ws = dist.get_world_size()
                avg_loss               = avg_loss.item()               / ws
                avg_loss_diff          = avg_loss_diff.item()          / ws
                avg_loss_align         = avg_loss_align.item()         / ws
                avg_train_align_weight = avg_train_align_weight.item() / ws

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Loss Diff: {avg_loss_diff:.4f}, "
                    f"Loss Align: {avg_loss_align:.4f}, "
                    f"Train Align W: {avg_train_align_weight:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss               = 0.0
                running_loss_diff          = 0.0
                running_loss_align         = 0.0
                running_train_align_weight = 0.0
                log_steps  = 0
                start_time = time()

            # Periodic checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir, train_steps,
                        model, ema, opt, args,
                        repa_projector if args.repa else None
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

            # max-steps termination
            if args.max_steps is not None and train_steps >= args.max_steps:
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir, train_steps,
                        model, ema, opt, args,
                        repa_projector if args.repa else None
                    )
                    logger.info(
                        f"Reached max_steps={args.max_steps}. "
                        f"Saved final checkpoint to {ckpt_path}"
                    )
                dist.barrier()
                done = True
                break

    model.eval()
    if args.repa:
        repa_projector.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path",    type=str, required=True)
    parser.add_argument(
        "--results-dir", type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results"
    )
    parser.add_argument(
        "--model", type=str,
        choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--image-size",        type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes",       type=int, default=1000)
    parser.add_argument("--epochs",            type=int, default=1400)
    parser.add_argument("--max-steps",         type=int, default=None,
                        help="Stop training after this many steps (overrides epochs).")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed",       type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema",
        help="VAE variant for HuggingFace Hub (ignored if --vae-model-dir is set)."
    )
    parser.add_argument(
        "--vae-model-dir", type=str, default=None,
        help="Local HF-style VAE directory. If set, --vae is ignored."
    )
    parser.add_argument("--num-workers",    type=int, default=4)
    parser.add_argument("--log-every",      type=int, default=100)
    parser.add_argument("--ckpt-every",     type=int, default=10_000)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight-decay",   type=float, default=0.0)

    # SASA / REPA args
    parser.add_argument("--repa", action="store_true",
                        help="Enable SASA alignment.")
    parser.add_argument("--repa-lambda",    type=float, default=0.1,
                        help="Base weight for alignment loss.")
    parser.add_argument("--repa-token-layer", type=int, default=None,
                        help="DiT block index to hook for token extraction. "
                             "Default: last block.")
    parser.add_argument("--repa-hidden-dim",  type=int, default=None,
                        help="REPA projector width. Default=2048 (REPA-faithful).")
    parser.add_argument(
        "--repa-train-schedule", type=str,
        choices=["constant", "linear_decay", "cutoff"],
        default="linear_decay",
        help="Training-step schedule: linear_decay decays lambda 1->0 "
             "over --repa-schedule-steps steps."
    )
    parser.add_argument(
        "--repa-schedule-steps", type=int, default=40_000,
        help="Steps over which the training-phase weight decays to 0."
    )
    parser.add_argument(
        "--dino-model-dir", type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/tmp/DiT/"
                "dinov3-vitb16-pretrain-lvd1689m",
        help="Local HF-style DINOv3 model directory."
    )

    args = parser.parse_args()

    if args.vae_model_dir is not None and not os.path.isdir(args.vae_model_dir):
        raise FileNotFoundError(
            f"--vae-model-dir not found or is not a directory: {args.vae_model_dir}"
        )

    main(args)
