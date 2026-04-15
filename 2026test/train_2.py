# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for DiT using PyTorch DDP.
Fixed REPA version:
- frozen local DINOv3 teacher (HF-style directory)
- DiT intermediate token extraction
- projector + cosine alignment loss
"""

import os
import argparse
import logging
import shutil
import subprocess
import sys
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

from models_2 import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
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
    """
    Center cropping implementation from ADM.
    """
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
    """3-layer MLP projector (REPA build_mlp): Linear→SiLU→Linear→SiLU→Linear."""
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: [B, T, C]
        return self.net(x)


class LocalDINOv3Teacher(nn.Module):
    """
    HF-style local DINOv3 teacher.

    Expects model_dir to contain:
      - config.json
      - preprocessor_config.json
      - model.safetensors
    """
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
        """
        x: [B, 3, H, W], already normalized for DINO
        return: patch tokens [B, 256, C] for 256x256 patch16
        """
        out = self.model(pixel_values=x)
        h = out.last_hidden_state  # [B, N, C]

        n = h.shape[1]
        if n == 261:
            # cls + 4 register + 256 patches
            h = h[:, 5:, :]
        elif n == 257:
            # cls + 256 patches
            h = h[:, 1:, :]
        elif n == 256:
            pass
        else:
            raise ValueError(f"Unexpected DINO token count: {n}")

        return h


def cosine_align_loss(pred, target):
    """
    pred:   [B, T, C]
    target: [B, T, C]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cos = (pred * target).sum(dim=-1)  # [B, T]
    return 1.0 - cos.mean()


def preprocess_for_dino(x):
    """
    x: [B, 3, H, W], usually in [-1, 1]
    returns normalized tensor for DINOv3 teacher
    """
    if x.min() < 0:
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]

    # If in future image size may differ, uncomment resize:
    # x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def extract_dino_patch_tokens(dino_model, x):
    """
    Extract patch tokens from HF-style local DINOv3 teacher.
    Returns:
        [B, 256, C] for 256x256 image with patch16
    """
    with torch.inference_mode():
        tokens = dino_model(x)
    return tokens


def load_vae(args, device, rank, logger):
    """
    Load the SD VAE once on rank 0, then reuse the local cache on all ranks.
    This avoids concurrent remote fetches from multiple DDP workers.
    """
    vae_source = args.vae_model_dir or f"stabilityai/sd-vae-ft-{args.vae}"
    is_local_dir = os.path.isdir(vae_source)

    if rank == 0 and not is_local_dir:
        logger.info(f"Prefetching VAE weights from: {vae_source}")
        prefetch_vae = AutoencoderKL.from_pretrained(vae_source)
        del prefetch_vae

    dist.barrier(device_ids=[device])

    vae = AutoencoderKL.from_pretrained(
        vae_source,
        local_files_only=not is_local_dir,
    ).to(device)
    vae.eval()
    requires_grad(vae, False)
    return vae


def get_alignment_weight(step, args):
    """
    Compute the effective alignment weight at the current training step.
    """
    if not args.repa:
        return 0.0
    if args.repa_schedule == "constant":
        return args.repa_lambda
    if args.repa_schedule == "linear":
        if step >= args.repa_schedule_steps:
            return 0.0
        progress = step / max(1, args.repa_schedule_steps)
        return args.repa_lambda * (1.0 - progress)
    raise ValueError(f"Unsupported repa schedule: {args.repa_schedule}")


@torch.no_grad()
def evaluate_fid(
    ema,
    diffusion,
    vae,
    args,
    device,
    step,
    logger,
):
    """
    Generate class-conditional samples with EMA weights and compute FID
    against a reference image directory via pytorch-fid.
    """
    if not args.fid_ref_dir:
        logger.info("Skipping FID evaluation: --fid-ref-dir is not set.")
        return None
    if not os.path.isdir(args.fid_ref_dir):
        logger.info(f"Skipping FID evaluation: reference dir not found: {args.fid_ref_dir}")
        return None

    eval_base_dir = getattr(args, "experiment_dir", args.results_dir)
    eval_root = os.path.join(eval_base_dir, "fid_eval", f"step_{step:07d}")
    sample_dir = os.path.join(eval_root, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    ema.eval()
    if args.model.endswith("/2") and args.image_size == 256:
        logger.info(
            f"Running FID evaluation at step={step} with "
            f"{args.fid_num_samples} samples, batch_size={args.fid_batch_size}, "
            f"sampling_steps={args.fid_sampling_steps}."
        )

    latent_size = args.image_size // 8
    using_cfg = args.fid_cfg_scale > 1.0
    sample_diffusion = create_diffusion(str(args.fid_sampling_steps))

    total_saved = 0
    while total_saved < args.fid_num_samples:
        n = min(args.fid_batch_size, args.fid_num_samples - total_saved)
        z = torch.randn(n, ema.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.fid_cfg_scale)
            sample_fn = ema.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = ema.forward

        samples = sample_diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for sample in samples:
            Image.fromarray(sample).save(os.path.join(sample_dir, f"{total_saved:06d}.png"))
            total_saved += 1

    fid_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        sample_dir,
        args.fid_ref_dir,
        "--device",
        fid_device,
        "--batch-size",
        str(args.fid_batch_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.info(f"FID evaluation failed at step={step}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return None

    fid_value = None
    for line in proc.stdout.splitlines():
        if "FID:" in line:
            try:
                fid_value = float(line.split("FID:")[-1].strip())
            except ValueError:
                fid_value = None
            break

    if fid_value is None:
        logger.info(f"FID evaluation finished but could not parse output.\nstdout:\n{proc.stdout}")
    else:
        logger.info(f"(step={step:07d}) FID: {fid_value:.4f}")

    if not args.keep_fid_samples:
        shutil.rmtree(eval_root, ignore_errors=True)
    return fid_value


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}"
        if args.repa:
            experiment_name += f"-repa-lam{args.repa_lambda}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        args.experiment_dir = experiment_dir
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None

    # Create DiT:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")

    vae = load_vae(args, device, rank, logger)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optional Fixed REPA modules:
    dino_model = None
    repa_projector = None

    if args.repa:
        logger.info("Initializing Fixed REPA...")
        logger.info(f"Loading local DINOv3 teacher from: {args.dino_model_dir}")

        dino_model = LocalDINOv3Teacher(args.dino_model_dir).to(device)
        dino_model.eval()
        requires_grad(dino_model, False)

        # Infer token dims
        dit_hidden_dim = model.hidden_size
        with torch.inference_mode():
            dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
            dummy = preprocess_for_dino(dummy)
            dino_tokens = extract_dino_patch_tokens(dino_model, dummy)
            dino_dim = dino_tokens.shape[-1]
            dino_num_tokens = dino_tokens.shape[1]

        logger.info(
            f"REPA setup: DiT hidden dim = {dit_hidden_dim}, "
            f"DINO dim = {dino_dim}, DINO num_tokens = {dino_num_tokens}"
        )

        repa_projector = REPAProjector(
            in_dim=dit_hidden_dim,
            out_dim=dino_dim,
            hidden_dim=args.repa_hidden_dim
        ).to(device)

    # Wrap trainable modules with DDP:
    model = DDP(model, device_ids=[device])

    if args.repa:
        repa_projector = DDP(repa_projector, device_ids=[device])

    # Setup optimizer:
    trainable_params = list(model.parameters())
    if args.repa:
        trainable_params += list(repa_projector.parameters())

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
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

    # Prepare models:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    if args.repa:
        repa_projector.train()
        dino_model.eval()

    # Logging vars
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    running_loss_diff = 0.0
    running_loss_align = 0.0
    start_time = time()

    logger.info(
        f"Training for up to {args.epochs} epochs "
        f"(max_steps={args.max_steps}, repa_schedule={args.repa_schedule}, "
        f"repa_schedule_steps={args.repa_schedule_steps})..."
    )

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for images, y in loader:
            current_step = train_steps
            align_weight = get_alignment_weight(current_step, args)
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                # RGB -> latent
                x = vae.encode(images).latent_dist.sample().mul_(0.18215)

            # Standard diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).long()
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_diff = loss_dict["loss"].mean()

            # REPA alignment loss
            loss_align = torch.tensor(0.0, device=device)

            if args.repa and align_weight > 0:
                # Teacher tokens from clean RGB image
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    dino_tokens = extract_dino_patch_tokens(dino_model, x_dino)   # [B, 256, 768]

                # Student tokens from noisy latent
                noise = torch.randn_like(x)
                x_t = diffusion.q_sample(x, t, noise=noise)

                dit_model = model.module
                _, dit_tokens = dit_model.forward_with_tokens(
                    x_t, t, y, token_layer=args.repa_token_layer
                )   # [B, T, D]

                proj_tokens = repa_projector(dit_tokens)  # [B, T, 768]

                # sanity checks
                if args.image_size == 256 and args.model.endswith("/2"):
                    assert dit_tokens.shape[1] == 256, \
                        f"Expected DiT token count 256, got {dit_tokens.shape[1]}"
                    assert dino_tokens.shape[1] == 256, \
                        f"Expected DINO token count 256, got {dino_tokens.shape[1]}"

                assert dino_tokens.ndim == 3, f"dino_tokens ndim={dino_tokens.ndim}"
                assert dit_tokens.ndim == 3, f"dit_tokens ndim={dit_tokens.ndim}"
                assert proj_tokens.ndim == 3, f"proj_tokens ndim={proj_tokens.ndim}"

                if proj_tokens.shape[1] != dino_tokens.shape[1]:
                    raise ValueError(
                        f"Token mismatch: proj_tokens={proj_tokens.shape}, dino_tokens={dino_tokens.shape}"
                    )

                loss_align = cosine_align_loss(proj_tokens, dino_tokens)

            loss = loss_diff + align_weight * loss_align

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            running_loss_diff += loss_diff.item()
            running_loss_align += loss_align.item()
            log_steps += 1
            train_steps += 1

            if train_steps == 1 and rank == 0:
                logger.info(f"images shape: {images.shape}")
                logger.info(f"latent shape: {x.shape}")
                if args.repa:
                    logger.info(f"dino_tokens shape: {dino_tokens.shape}")
                    logger.info(f"dit_tokens shape: {dit_tokens.shape}")
                    logger.info(f"proj_tokens shape: {proj_tokens.shape}")

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_diff = torch.tensor(running_loss_diff / log_steps, device=device)
                avg_loss_align = torch.tensor(running_loss_align / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_diff, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_align, op=dist.ReduceOp.SUM)

                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_diff = avg_loss_diff.item() / dist.get_world_size()
                avg_loss_align = avg_loss_align.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Loss Diff: {avg_loss_diff:.4f}, "
                    f"Loss Align: {avg_loss_align:.4f}, "
                    f"Align Weight: {align_weight:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss = 0.0
                running_loss_diff = 0.0
                running_loss_align = 0.0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    if args.repa:
                        checkpoint["repa_projector"] = repa_projector.module.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if args.fid_every > 0 and train_steps % args.fid_every == 0 and train_steps > 0:
                dist.barrier()
                if rank == 0:
                    evaluate_fid(
                        ema=ema,
                        diffusion=diffusion,
                        vae=vae,
                        args=args,
                        device=device,
                        step=train_steps,
                        logger=logger,
                    )
                dist.barrier()

            if train_steps >= args.max_steps:
                logger.info(f"Reached max_steps={args.max_steps}. Stopping training.")
                break

        if train_steps >= args.max_steps:
            break

    model.eval()
    if args.repa:
        repa_projector.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument(
        "--vae-model-dir",
        type=str,
        default=None,
        help="Optional local directory for the SD VAE. If unset, rank 0 downloads/caches the HF repo first.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fid-every",
        type=int,
        default=0,
        help="Run inline FID every N training steps. Set <= 0 to disable inline FID.",
    )
    parser.add_argument("--fid-num-samples", type=int, default=5_000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-sampling-steps", type=int, default=250)
    parser.add_argument("--fid-cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--fid-ref-dir",
        type=str,
        default="/data/liuchunfa/2026qjx/ILSVRC/Data/CLS-LOC/val",
        help="Reference image directory used by pytorch-fid.",
    )
    parser.add_argument(
        "--keep-fid-samples",
        action="store_true",
        help="Keep intermediate FID sample folders instead of deleting them after evaluation.",
    )

    # Fixed REPA args
    parser.add_argument("--repa", action="store_true", help="Enable fixed REPA alignment.")
    parser.add_argument("--repa-lambda", type=float, default=0.1, help="Weight for alignment loss.")
    parser.add_argument(
        "--repa-schedule",
        type=str,
        choices=["constant", "linear"],
        default="linear",
        help="Alignment schedule over training steps.",
    )
    parser.add_argument(
        "--repa-schedule-steps",
        type=int,
        default=40_000,
        help="Number of steps over which alignment decays to zero when --repa-schedule=linear.",
    )
    parser.add_argument("--repa-token-layer", type=int, default=None, help="DiT block index to extract tokens from.")
    parser.add_argument("--repa-hidden-dim", type=int, default=None, help="Hidden dim of REPA projector MLP.")

    # Local DINOv3 teacher
    parser.add_argument(
        "--dino-model-dir",
        type=str,
        default="/home/liuchunfa/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Local HF-style DINOv3 model directory containing config.json and model.safetensors."
    )

    args = parser.parse_args()
    main(args)
