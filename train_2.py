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
import math
import random
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


def optimizer_to(optimizer, device):
    """
    Move optimizer state tensors onto the target device after resume.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


def infer_train_steps_from_checkpoint_path(checkpoint_path):
    """
    Infer global train step from a checkpoint filename like 0040000.pt.
    """
    filename = os.path.basename(checkpoint_path)
    stem, _ = os.path.splitext(filename)
    if not stem.isdigit():
        raise ValueError(f"Cannot infer train_steps from checkpoint name: {checkpoint_path}")
    return int(stem)


def normalize_resume_arg_value(key, value):
    """
    Normalize values before comparing checkpoint args to current CLI args.
    """
    if key.endswith("_path") or key.endswith("_dir"):
        if value is None:
            return None
        return os.path.abspath(value)
    return value


def validate_resume_args(current_args, checkpoint_args):
    """
    Reject resume attempts when the current experiment definition does not
    match the checkpoint's original training configuration.
    """
    required_keys = [
        "data_path",
        "model",
        "image_size",
        "num_classes",
        "global_batch_size",
        "global_seed",
        "vae",
        "vae_model_dir",
        "lr",
        "weight_decay",
        "repa",
        "repa_lambda",
        "repa_diff_schedule",
        "repa_diff_threshold",
        "repa_token_layer",
        "repa_hidden_dim",
        "dino_model_dir",
    ]

    missing_keys = []
    mismatches = []

    for key in required_keys:
        if not hasattr(checkpoint_args, key):
            missing_keys.append(key)
            continue
        current_value = normalize_resume_arg_value(key, getattr(current_args, key))
        checkpoint_value = normalize_resume_arg_value(key, getattr(checkpoint_args, key))
        if current_value != checkpoint_value:
            mismatches.append((key, checkpoint_value, current_value))

    if missing_keys or mismatches:
        lines = [f"Resume checkpoint is incompatible: {current_args.resume}"]
        if missing_keys:
            lines.append(f"Missing keys in checkpoint args: {missing_keys}")
        for key, checkpoint_value, current_value in mismatches:
            lines.append(
                f"Arg mismatch for '{key}': checkpoint={checkpoint_value!r}, current={current_value!r}"
            )
        raise ValueError("\n".join(lines))


def validate_resume_runtime(current_world_size, current_local_batch_size, checkpoint):
    """
    Reject resume attempts when the distributed runtime shape does not match.
    """
    checkpoint_world_size = checkpoint.get("world_size")
    checkpoint_local_batch_size = checkpoint.get("local_batch_size")

    if checkpoint_world_size is not None and checkpoint_world_size != current_world_size:
        raise ValueError(
            f"Resume checkpoint world_size mismatch: checkpoint={checkpoint_world_size}, "
            f"current={current_world_size}"
        )
    if checkpoint_local_batch_size is not None and checkpoint_local_batch_size != current_local_batch_size:
        raise ValueError(
            f"Resume checkpoint local_batch_size mismatch: checkpoint={checkpoint_local_batch_size}, "
            f"current={current_local_batch_size}"
        )


def infer_resume_position(train_steps, steps_per_epoch):
    """
    Infer epoch and next batch offset from the global train step counter.
    """
    if steps_per_epoch <= 0:
        return 0, 0
    return divmod(train_steps, steps_per_epoch)


def capture_rng_state(device):
    """
    Capture the local RNG states needed for exact per-rank continuation.
    """
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state, device):
    """
    Restore the local RNG states captured by capture_rng_state().
    """
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device=device)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


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


class ResumableDistributedSampler(DistributedSampler):
    """
    Distributed sampler that can resume from a batch offset within one epoch.
    """
    def __init__(
        self,
        dataset,
        *,
        local_batch_size,
        resume_epoch=0,
        resume_step_in_epoch=0,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self.local_batch_size = local_batch_size
        self.resume_epoch = resume_epoch
        self.resume_step_in_epoch = resume_step_in_epoch

    def _local_start_index(self):
        if self.epoch == self.resume_epoch and self.resume_step_in_epoch > 0:
            return self.resume_step_in_epoch * self.local_batch_size
        return 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        start_index = self._local_start_index()
        if start_index > 0:
            indices = indices[start_index:]
        return iter(indices)

    def __len__(self):
        return max(0, self.num_samples - self._local_start_index())


class REPAProjector(nn.Module):
    """
    Project DiT tokens to DINO token feature space.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
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


def cosine_align_loss_per_sample(pred, target):
    """
    pred:   [B, T, C]
    target: [B, T, C]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cos = (pred * target).sum(dim=-1)  # [B, T]
    return 1.0 - cos.mean(dim=-1)


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


def get_diffusion_alignment_weights(t, num_timesteps, schedule, threshold):
    """
    Compute per-sample alignment weights from diffusion timesteps.
    """
    if num_timesteps <= 1:
        t_norm = torch.zeros_like(t, dtype=torch.float32)
    else:
        t_norm = t.float() / float(num_timesteps - 1)

    if schedule == "constant":
        return torch.ones_like(t_norm)
    if schedule == "linear_high_noise":
        return t_norm
    if schedule == "threshold_high_noise":
        return (t_norm >= threshold).to(dtype=t_norm.dtype)
    if schedule == "cosine_high_noise":
        return 0.5 * (1.0 - torch.cos(torch.pi * t_norm))
    raise ValueError(f"Unsupported repa diffusion schedule: {schedule}")


def sample_force_drop_ids(dit_model, labels):
    """
    Sample a single classifier-free label dropout mask and reuse it across
    multiple DiT forwards within the same training step.
    """
    dropout_prob = dit_model.y_embedder.dropout_prob
    if dropout_prob <= 0:
        return None
    return (torch.rand(labels.shape[0], device=labels.device) < dropout_prob).to(dtype=torch.long)


def apply_random_horizontal_flip(images, prob=0.5):
    """
    Apply horizontal flip in the main training process so augmentation randomness
    is controlled by the checkpointed per-rank RNG state rather than DataLoader workers.
    """
    if prob <= 0:
        return images
    flip_mask = torch.rand(images.shape[0], device=images.device) < prob
    if not torch.any(flip_mask):
        return images
    images = images.clone()
    images[flip_mask] = torch.flip(images[flip_mask], dims=[-1])
    return images


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
    local_batch_size = args.global_batch_size // dist.get_world_size()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder:
    if args.resume:
        args.resume = os.path.abspath(args.resume)
        checkpoint_dir = os.path.dirname(args.resume)
        experiment_dir = os.path.dirname(checkpoint_dir)
        args.experiment_dir = experiment_dir
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Resuming experiment directory: {experiment_dir}")
            logger.info(f"Resume checkpoint: {args.resume}")
        else:
            logger = create_logger(None)
    else:
        if rank == 0:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_name = f"{experiment_index:03d}-{model_string_name}"
            if args.repa:
                schedule_name = args.repa_diff_schedule.replace("_", "-")
                experiment_name += f"-repa-taware-{schedule_name}-lam{args.repa_lambda}"
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    steps_per_epoch = math.ceil(len(dataset) / dist.get_world_size()) // local_batch_size

    # Resume state if requested:
    train_steps = 0
    resume_epoch = 0
    resume_step_in_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint state from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_args(args, checkpoint["args"])
        validate_resume_runtime(dist.get_world_size(), local_batch_size, checkpoint)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        optimizer_to(opt, device)

        if args.repa:
            if "repa_projector" not in checkpoint:
                raise KeyError("Checkpoint is missing repa_projector state while --repa is enabled.")
            repa_projector.module.load_state_dict(checkpoint["repa_projector"])

        train_steps = checkpoint.get("train_steps", infer_train_steps_from_checkpoint_path(args.resume))
        resume_epoch = checkpoint.get("epoch", None)
        resume_step_in_epoch = checkpoint.get("step_in_epoch", None)
        if resume_epoch is None or resume_step_in_epoch is None:
            resume_epoch, resume_step_in_epoch = infer_resume_position(train_steps, steps_per_epoch)
            logger.warning(
                "Checkpoint is missing epoch/step_in_epoch; inferred resume position as "
                f"epoch={resume_epoch}, step_in_epoch={resume_step_in_epoch}."
            )
        logger.info(f"Resumed training step: {train_steps}")
    else:
        update_ema(ema, model.module, decay=0)

    sampler = ResumableDistributedSampler(
        dataset,
        local_batch_size=local_batch_size,
        resume_epoch=resume_epoch,
        resume_step_in_epoch=resume_step_in_epoch,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.global_seed),
    )

    if args.resume:
        rng_state_by_rank = checkpoint.get("rng_state_by_rank")
        if rng_state_by_rank is None:
            logger.warning("Checkpoint is missing rng_state_by_rank; exact RNG continuation is unavailable.")
        else:
            if rank >= len(rng_state_by_rank):
                raise ValueError(
                    f"Checkpoint rng_state_by_rank length={len(rng_state_by_rank)} is incompatible with rank={rank}."
                )
            restore_rng_state(rng_state_by_rank[rank], device)

    # Prepare models:
    model.train()
    ema.eval()

    if args.repa:
        repa_projector.train()
        dino_model.eval()

    # Logging vars
    log_steps = 0
    running_loss = 0.0
    running_loss_diff = 0.0
    running_loss_align = 0.0
    running_diff_gate = 0.0
    start_time = time()

    logger.info(
        f"Training for up to {args.epochs} epochs "
        f"(max_steps={args.max_steps}, repa_diff_schedule={args.repa_diff_schedule}, "
        f"repa_diff_threshold={args.repa_diff_threshold})..."
    )
    if args.fid_every > 0:
        logger.warning(
            "Inline FID consumes RNG after checkpoint save and breaks strict resume equivalence. "
            "Keep --fid-every <= 0 for strict continuation."
        )

    if train_steps >= args.max_steps:
        logger.info(
            f"Resume step {train_steps} already reaches max_steps={args.max_steps}. Nothing to do."
        )
        cleanup()
        return

    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        epoch_step_start = resume_step_in_epoch if epoch == resume_epoch else 0

        for step_in_epoch, (images, y) in enumerate(loader, start=epoch_step_start):
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            images = apply_random_horizontal_flip(images)

            with torch.no_grad():
                # RGB -> latent
                x = vae.encode(images).latent_dist.sample().mul_(0.18215)

            # Standard diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            dit_model = model.module
            force_drop_ids = sample_force_drop_ids(dit_model, y)
            need_align_tokens = args.repa and args.repa_lambda > 0
            model_kwargs = dict(
                y=y,
                force_drop_ids=force_drop_ids,
                return_tokens=need_align_tokens,
                token_layer=args.repa_token_layer,
            )
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise=noise)
            loss_diff_vec = loss_dict["loss"]
            loss_diff = loss_diff_vec.mean()

            # REPA alignment loss
            loss_align = torch.tensor(0.0, device=device)
            loss = loss_diff
            diff_gate_mean = torch.tensor(0.0, device=device)
            dino_tokens = None
            dit_tokens = None
            proj_tokens = None

            if args.repa and args.repa_lambda > 0:
                diff_weights = get_diffusion_alignment_weights(
                    t=t,
                    num_timesteps=diffusion.num_timesteps,
                    schedule=args.repa_diff_schedule,
                    threshold=args.repa_diff_threshold,
                )
                diff_gate_mean = diff_weights.mean()

                # Teacher tokens from clean RGB image
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    dino_tokens = extract_dino_patch_tokens(dino_model, x_dino)   # [B, 256, 768]

                if "model_aux" not in loss_dict:
                    raise KeyError("Expected model_aux tokens from diffusion.training_losses().")
                dit_tokens = loss_dict["model_aux"]   # [B, T, D]

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

                loss_align_vec = cosine_align_loss_per_sample(proj_tokens, dino_tokens)
                loss_align = loss_align_vec.mean()
                loss = (loss_diff_vec + args.repa_lambda * diff_weights * loss_align_vec).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            running_loss_diff += loss_diff.item()
            running_loss_align += loss_align.item()
            running_diff_gate += diff_gate_mean.item()
            log_steps += 1
            train_steps += 1

            if train_steps == 1 and rank == 0:
                logger.info(f"images shape: {images.shape}")
                logger.info(f"latent shape: {x.shape}")
                if args.repa and dino_tokens is not None:
                    logger.info(f"dino_tokens shape: {dino_tokens.shape}")
                    logger.info(f"dit_tokens shape: {dit_tokens.shape}")
                    logger.info(f"proj_tokens shape: {proj_tokens.shape}")
                if args.repa:
                    logger.info(f"mean diffusion gate: {diff_gate_mean.item():.4f}")

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss_diff = torch.tensor(running_loss_diff / log_steps, device=device)
                avg_loss_align = torch.tensor(running_loss_align / log_steps, device=device)
                avg_diff_gate = torch.tensor(running_diff_gate / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_diff, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_align, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_diff_gate, op=dist.ReduceOp.SUM)

                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_loss_diff = avg_loss_diff.item() / dist.get_world_size()
                avg_loss_align = avg_loss_align.item() / dist.get_world_size()
                avg_diff_gate = avg_diff_gate.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Loss Diff: {avg_loss_diff:.4f}, "
                    f"Loss Align: {avg_loss_align:.4f}, "
                    f"Diff Gate: {avg_diff_gate:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss = 0.0
                running_loss_diff = 0.0
                running_loss_align = 0.0
                running_diff_gate = 0.0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                local_rng_state = capture_rng_state(device)
                gathered_rng_states = [None] * dist.get_world_size()
                dist.all_gather_object(gathered_rng_states, local_rng_state)
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch + 1,
                        "world_size": dist.get_world_size(),
                        "local_batch_size": local_batch_size,
                        "rng_state_by_rank": gathered_rng_states,
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
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
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

    # T-aware REPA args
    parser.add_argument("--repa", action="store_true", help="Enable REPA alignment.")
    parser.add_argument("--repa-lambda", type=float, default=0.1, help="Weight for alignment loss.")
    parser.add_argument(
        "--repa-diff-schedule",
        type=str,
        choices=["constant", "linear_high_noise", "threshold_high_noise", "cosine_high_noise"],
        default="linear_high_noise",
        help="Diffusion timestep schedule for per-sample alignment gating.",
    )
    parser.add_argument(
        "--repa-diff-threshold",
        type=float,
        default=0.5,
        help="Normalized timestep threshold used by --repa-diff-schedule=threshold_high_noise.",
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
    if not 0.0 <= args.repa_diff_threshold <= 1.0:
        parser.error("--repa-diff-threshold must be within [0, 1].")
    main(args)
