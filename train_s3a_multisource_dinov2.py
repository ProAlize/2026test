# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
S3A training script for DiT on ImageNet-style data.

S3A (Stage-Conditioned Source-and-Structure Alignment) keeps the DiT backbone
unchanged and injects teacher/self signals through an auxiliary branch:
- Multi-layer token taps
- Multi-source fusion (DINO external source + optional EMA self source)
- Dynamic source reliability routing conditioned on (step, timestep, layer)
- Holistic alignment loss (feature + affinity + spatial)
- 3D curriculum weights w(step, timestep, layer) and selective source gate
"""

import os
import sys
import math
import json
import hashlib
import random
import argparse
import logging
import subprocess
from time import time, strftime
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

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

from model_sasa import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

CHECKPOINT_FORMAT_VERSION = 5
METRICS_SCHEMA_VERSION = 5


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@torch.no_grad()
def update_ema_adapters(ema_adapters: nn.ModuleDict, student_adapters: nn.ModuleDict, decay: float = 0.999):
    """EMA-update ema_adapters from student_adapters so the self-source
    projection tracks the student adapter without sharing the exact same weights."""
    for key in ema_adapters:
        ema_params = OrderedDict(ema_adapters[key].named_parameters())
        src_params = OrderedDict(student_adapters[key].named_parameters())
        for name, param in src_params.items():
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
            handlers=handlers,
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_revision(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def atomic_write_json(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def capture_local_rng_state(loader_generator: Optional[torch.Generator] = None) -> dict:
    state = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_local_rng_state(state: dict, loader_generator: Optional[torch.Generator] = None) -> None:
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])


def gather_rng_states(loader_generator: Optional[torch.Generator] = None) -> List[dict]:
    local_state = capture_local_rng_state(loader_generator=loader_generator)
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_state)
    return gathered


#################################################################################
#                           Schedule / Weight Helpers                           #
#################################################################################


def get_train_phase_weight(
    current_step: int,
    schedule_steps: int,
    schedule: str = "cosine_decay",
    warmup_steps: int = 0,
) -> float:
    """Phase-axis weight used in 3D curriculum.

    For ``piecewise_cosine`` (default):
        [0, warmup_steps)       → 1.0     (constant full strength)
        [warmup_steps, schedule_steps) → cosine 1.0 → 0.0  (smooth decay)
        [schedule_steps, ...)   → 0.0

    For ``piecewise_linear``:
        Same structure but with linear decay in the middle phase.
    """
    if schedule == "constant":
        return 1.0

    if schedule in ("piecewise_cosine", "piecewise_linear"):
        if current_step < warmup_steps:
            return 1.0
        if current_step >= schedule_steps:
            return 0.0
        decay_len = max(1, schedule_steps - warmup_steps)
        decay_progress = (current_step - warmup_steps) / decay_len
        if schedule == "piecewise_cosine":
            return 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return 1.0 - decay_progress  # piecewise_linear

    progress = min(max(current_step / max(1, schedule_steps), 0.0), 1.0)

    if schedule == "linear_decay":
        return 1.0 - progress
    if schedule == "cosine_decay":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "cutoff":
        return 1.0 if current_step < schedule_steps else 0.0

    raise ValueError(
        f"Unknown train schedule: {schedule!r}. "
        "Choose from: constant, linear_decay, cosine_decay, cutoff, piecewise_linear, piecewise_cosine."
    )


def get_diff_timestep_weight(
    t: torch.Tensor,
    T: int,
    schedule: str = "cosine",
) -> torch.Tensor:
    """Diffusion-timestep axis weight. Returns [B], mean-normalized."""
    t_norm = t.float() / max(T - 1, 1)

    if schedule == "uniform":
        weights = torch.ones_like(t_norm)
    elif schedule == "cosine":
        weights = torch.cos(math.pi / 2.0 * t_norm)
    elif schedule == "linear_high":
        weights = 1.0 - t_norm
    elif schedule == "linear_low":
        weights = t_norm
    else:
        raise ValueError(
            f"Unknown diff_schedule: {schedule!r}. "
            "Choose from: cosine, linear_high, linear_low, uniform."
        )

    mean_w = weights.mean().clamp(min=1e-8)
    return weights / mean_w


def parse_layer_indices(spec: str, depth: int) -> List[int]:
    """
    Parse layer spec into valid block indices.

    Supported examples:
    - auto
    - quarter,mid,three_quarter,last
    - 6,13,20,27
    """
    if spec is None:
        spec = "auto"

    spec = spec.strip().lower()
    if spec == "auto":
        candidate = [
            max(0, round(depth * 0.25) - 1),
            max(0, round(depth * 0.50) - 1),
            max(0, round(depth * 0.75) - 1),
            depth - 1,
        ]
        return sorted(set(candidate))

    name_to_idx = {
        "quarter": max(0, round(depth * 0.25) - 1),
        "mid": max(0, round(depth * 0.50) - 1),
        "three_quarter": max(0, round(depth * 0.75) - 1),
        "last": depth - 1,
    }

    out = []
    for tok in [x.strip() for x in spec.split(",") if x.strip()]:
        if tok in name_to_idx:
            out.append(name_to_idx[tok])
            continue
        idx = int(tok)
        if idx < 0:
            idx = depth + idx
        if idx < 0 or idx >= depth:
            raise ValueError(
                f"Layer index out of range in --s3a-layer-indices: {tok}. "
                f"Valid range is [0, {depth - 1}] (or negative equivalent)."
            )
        out.append(idx)

    if not out:
        raise ValueError("No valid layer index is parsed from --s3a-layer-indices")
    return sorted(set(out))


def build_layer_weights(
    mode: str,
    custom_csv: Optional[str],
    layer_indices: List[int],
    depth: int,
) -> List[float]:
    """
    Build per-layer scalar weights (g_layer). Returned weights are mean-normalized.
    """
    n = len(layer_indices)
    if n == 0:
        return []

    if mode == "custom":
        if custom_csv is None:
            raise ValueError("--s3a-layer-weights is required when mode=custom")
        vals = [float(x.strip()) for x in custom_csv.split(",") if x.strip()]
        if len(vals) != n:
            raise ValueError(
                f"custom layer weights length mismatch: expected {n}, got {len(vals)}"
            )
        weights = vals
    else:
        positions = [idx / max(1, depth - 1) for idx in layer_indices]
        if mode == "uniform":
            weights = [1.0 for _ in positions]
        elif mode == "deep_focus":
            # Favor deeper blocks.
            weights = [0.5 + p for p in positions]
        elif mode == "mid_focus":
            # Favor middle blocks.
            weights = [1.0 - abs(p - 0.5) * 1.2 for p in positions]
        else:
            raise ValueError(
                f"Unknown layer weight mode: {mode}. "
                "Choose from: uniform, deep_focus, mid_focus, custom"
            )

    weights_t = torch.tensor(weights, dtype=torch.float32)
    weights_t = weights_t / weights_t.mean().clamp(min=1e-8)
    return weights_t.tolist()


def _sqrt_hw(num_tokens: int) -> Optional[int]:
    side = int(math.sqrt(num_tokens))
    return side if side * side == num_tokens else None


def resize_tokens_to_match(tokens: torch.Tensor, target_token_count: int) -> torch.Tensor:
    """
    Resize token grid count while preserving channel dimension.
    tokens: [B, N, C] -> [B, target_token_count, C]
    """
    if tokens.shape[1] == target_token_count:
        return tokens

    src_hw = _sqrt_hw(tokens.shape[1])
    dst_hw = _sqrt_hw(target_token_count)

    if src_hw is not None and dst_hw is not None:
        x = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], src_hw, src_hw)
        x = F.interpolate(x, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
        return x.reshape(tokens.shape[0], tokens.shape[2], target_token_count).transpose(1, 2)

    x = tokens.transpose(1, 2)
    x = F.interpolate(x, size=target_token_count, mode="linear", align_corners=False)
    return x.transpose(1, 2)


#################################################################################
#                            Holistic Alignment Losses                          #
#################################################################################


def cosine_distance_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cos_per_token = (pred_n * target_n).sum(dim=-1)
    cos_per_sample = cos_per_token.mean(dim=1)
    return 1.0 - cos_per_sample


def affinity_loss_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    if max_tokens is not None and max_tokens > 0 and pred.shape[1] > max_tokens:
        indices = torch.linspace(
            0,
            pred.shape[1] - 1,
            steps=max_tokens,
            device=pred.device,
        ).long()
        pred = pred.index_select(1, indices)
        target = target.index_select(1, indices)

    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    pred_aff = torch.bmm(pred_n, pred_n.transpose(1, 2))
    target_aff = torch.bmm(target_n, target_n.transpose(1, 2))
    return (pred_aff - target_aff).pow(2).mean(dim=(1, 2))


def spatial_loss_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structure-consistency loss on per-token energy map and local gradients.

    Energy maps are mean-normalized before comparison so the loss captures
    spatial *structure* rather than absolute scale differences between the
    student adapter output and the teacher features.
    """
    pred_energy = pred.norm(dim=-1).clamp(min=1e-6)
    target_energy = target.norm(dim=-1).clamp(min=1e-6)

    # Mean-normalize energy maps to remove scale mismatch.
    pred_energy = pred_energy / pred_energy.mean(dim=-1, keepdim=True).clamp(min=1e-8)
    target_energy = target_energy / target_energy.mean(dim=-1, keepdim=True).clamp(min=1e-8)

    src_hw = _sqrt_hw(pred.shape[1])
    dst_hw = _sqrt_hw(target.shape[1])

    if src_hw is not None and dst_hw is not None:
        p = pred_energy.reshape(pred.shape[0], 1, src_hw, src_hw)
        z = target_energy.reshape(target.shape[0], 1, dst_hw, dst_hw)
        if src_hw != dst_hw:
            z = F.interpolate(z, size=(src_hw, src_hw), mode="bicubic", align_corners=False)

        base = (p - z).abs().mean(dim=(1, 2, 3))

        grad_px = p[:, :, 1:, :] - p[:, :, :-1, :]
        grad_pz = p[:, :, :, 1:] - p[:, :, :, :-1]
        grad_zx = z[:, :, 1:, :] - z[:, :, :-1, :]
        grad_zz = z[:, :, :, 1:] - z[:, :, :, :-1]

        grad_loss = (grad_px - grad_zx).abs().mean(dim=(1, 2, 3))
        grad_loss = grad_loss + (grad_pz - grad_zz).abs().mean(dim=(1, 2, 3))
        return base + 0.5 * grad_loss

    # Fallback for non-square token counts.
    p = pred_energy
    z = target_energy
    if p.shape[1] != z.shape[1]:
        z = F.interpolate(z.unsqueeze(1), size=p.shape[1], mode="linear", align_corners=False).squeeze(1)
    return (p - z).abs().mean(dim=1)


def source0_min_alpha_at_step(step: int, args) -> float:
    floor = 0.0
    if (
        args.s3a_dino_alpha_floor > 0
        and args.s3a_dino_alpha_floor_steps > 0
        and step < args.s3a_dino_alpha_floor_steps
    ):
        floor_ratio = 1.0 - (step / max(1, args.s3a_dino_alpha_floor_steps))
        floor = max(floor, args.s3a_dino_alpha_floor * max(0.0, floor_ratio))
    if args.s3a_protect_source0_min_alpha > 0:
        floor = max(floor, args.s3a_protect_source0_min_alpha)
    return float(floor)


def router_policy_kl_and_gap_per_sample(
    raw_alpha: torch.Tensor,
    policy_alpha: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_safe = raw_alpha.clamp(min=1e-8)
    policy_safe = policy_alpha.clamp(min=1e-8)
    # Keep router policy close to deployed alpha policy to avoid raw/effective drift.
    kl = (raw_safe * (raw_safe.log() - policy_safe.log())).sum(dim=-1)
    gap = (raw_alpha - policy_alpha).abs().sum(dim=-1)
    return kl, gap


def make_empty_align_stats(layer_indices: List[int]) -> Dict[str, Any]:
    return {
        "used_layers": 0.0,
        "feat": 0.0,
        "attn": 0.0,
        "spatial": 0.0,
        "alpha_dino": 0.0,
        "alpha_self": 0.0,
        "gate_self": 0.0,
        "gate_self_state": 0.0,
        "diff_w": 0.0,
        "raw_alpha_dino": 0.0,
        "raw_alpha_self": 0.0,
        "router_entropy_norm": 0.0,
        "router_policy_kl": 0.0,
        "router_policy_gap": 0.0,
        "loss_fused": 0.0,
        "loss_fused_probe": 0.0,
        "loss_dino_only": 0.0,
        "loss_self_only": 0.0,
        "utility_dino": 0.0,
        "utility_self": 0.0,
        "utility_self_ema": 0.0,
        "utility_self_active_ema": 0.0,
        "utility_self_inactive_ema": 0.0,
        "utility_self_active_ema_count": 0.0,
        "utility_self_inactive_ema_count": 0.0,
        "probe_count": 0.0,
        "self_probe_count": 0.0,
        "collapse_alarm": 0.0,
        "alpha_dino_min_layer": 0.0,
        "alpha_dino_max_layer": 0.0,
        "alpha_dino_layers": [0.0 for _ in layer_indices],
    }


#################################################################################
#                    DINOv2 Teacher（local .pth + torch.hub）                  #
#################################################################################


# Supported DINOv2 model variants and their properties.
DINOV2_VARIANTS = {
    "vitb14": {"hub_name": "dinov2_vitb14", "embed_dim": 768,  "patch_size": 14},
    "vitl14": {"hub_name": "dinov2_vitl14", "embed_dim": 1024, "patch_size": 14},
    "vitg14": {"hub_name": "dinov2_vitg14", "embed_dim": 1536, "patch_size": 14},
}


class LocalDINOv2Teacher(nn.Module):
    EXPECTED_PATCH_TOKENS = 256

    def __init__(
        self,
        dinov2_repo_dir: str,
        weight_path: str,
        model_variant: str = "vitb14",
    ):
        super().__init__()
        if model_variant not in DINOV2_VARIANTS:
            raise ValueError(
                f"Unknown --dinov2-model-variant: {model_variant!r}. "
                f"Choose from: {list(DINOV2_VARIANTS.keys())}"
            )
        variant_info = DINOV2_VARIANTS[model_variant]
        self.embed_dim = variant_info["embed_dim"]
        self.model_variant = model_variant

        if dinov2_repo_dir not in sys.path:
            sys.path.insert(0, dinov2_repo_dir)

        self.model = torch.hub.load(
            dinov2_repo_dir,
            variant_info["hub_name"],
            source="local",
            pretrained=False,
        )

        raw_state = torch.load(weight_path, map_location="cpu")
        if not isinstance(raw_state, dict):
            raise ValueError(
                f"Unexpected DINO checkpoint format in {weight_path}: expected dict, "
                f"got {type(raw_state)}"
            )

        # Guard against accidentally passing the diffusion training checkpoint.
        if {"model_state", "ema_state", "opt_state"}.issubset(raw_state.keys()) or {
            "model",
            "ema",
            "opt",
        }.issubset(raw_state.keys()):
            raise ValueError(
                "The provided --dinov2-weight-path looks like a diffusion training "
                "checkpoint, not a DINOv2 teacher checkpoint."
            )

        if "teacher" in raw_state and isinstance(raw_state["teacher"], dict):
            state_dict = raw_state["teacher"]
        elif "model" in raw_state and isinstance(raw_state["model"], dict):
            state_dict = raw_state["model"]
        else:
            state_dict = raw_state

        # DDP-exported checkpoints may prefix all keys with 'module.'.
        if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to strictly load DINOv2 teacher checkpoint. "
                f"Please verify --dinov2-weight-path: {weight_path}"
            ) from exc

        self.model.eval()
        requires_grad(self.model, False)

        self._resize = transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        feats = self.model.get_intermediate_layers(x, n=1, return_class_token=False)
        patch_tokens = feats[0]
        assert patch_tokens.shape[1] == self.EXPECTED_PATCH_TOKENS, (
            "DINOv2 patch token count mismatch: "
            f"expected {self.EXPECTED_PATCH_TOKENS}, got {patch_tokens.shape[1]}"
        )
        return patch_tokens


def preprocess_for_dino(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


#################################################################################
#                              S3A Core Modules                                #
#################################################################################


class SpatiallyFaithfulAdapter(nn.Module):
    """Token MLP path + depthwise spatial path."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.in_norm = nn.LayerNorm(in_dim)

        self.token_path = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.spatial_in = nn.Linear(in_dim, hidden_dim)
        self.spatial_dw = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.spatial_out = nn.Linear(hidden_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        x_norm = self.in_norm(x)
        token_out = self.token_path(x_norm)

        side = _sqrt_hw(x.shape[1])
        if side is None:
            return self.out_norm(token_out)

        spatial = self.spatial_in(x_norm)
        spatial = spatial.transpose(1, 2).reshape(x.shape[0], -1, side, side)
        spatial = self.spatial_dw(spatial)
        spatial = spatial.reshape(x.shape[0], -1, x.shape[1]).transpose(1, 2)
        spatial = self.spatial_out(spatial)

        return self.out_norm(token_out + spatial)


class SourceReliabilityRouter(nn.Module):
    """alpha(s, t, l) for source fusion."""

    def __init__(
        self,
        in_dim: int,
        num_layer_slots: int,
        num_sources: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.layer_embed = nn.Embedding(num_layer_slots, hidden_dim)
        self.token_proj = nn.Linear(in_dim, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_sources),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        layer_slot: torch.Tensor,
        timestep_norm: torch.Tensor,
        phase_norm: torch.Tensor,
    ) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        x = self.token_proj(pooled)

        cond = torch.stack([timestep_norm, phase_norm], dim=-1)
        cond = self.cond_proj(cond)

        layer = self.layer_embed(layer_slot)
        logits = self.head(x + cond + layer)
        return F.softmax(logits, dim=-1)


class S3AAlignmentHead(nn.Module):
    """Container for layer adapters, router, and selective source gate state."""

    def __init__(
        self,
        layer_indices: List[int],
        student_dim: int,
        target_dim: int,
        adapter_hidden_dim: Optional[int],
        router_hidden_dim: int,
        use_ema_source: bool,
        use_trainable_ema_adapters: bool,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.layer_keys = [str(i) for i in self.layer_indices]
        self.use_ema_source = use_ema_source
        self.use_trainable_ema_adapters = bool(use_trainable_ema_adapters and use_ema_source)
        self.num_sources = 2 if use_ema_source else 1  # [dino, self]

        self.student_adapters = nn.ModuleDict(
            {
                k: SpatiallyFaithfulAdapter(
                    in_dim=student_dim,
                    out_dim=target_dim,
                    hidden_dim=adapter_hidden_dim,
                )
                for k in self.layer_keys
            }
        )

        if self.use_ema_source:
            self.ema_adapters = nn.ModuleDict(
                {
                    k: SpatiallyFaithfulAdapter(
                        in_dim=student_dim,
                        out_dim=target_dim,
                        hidden_dim=adapter_hidden_dim,
                    )
                    for k in self.layer_keys
                }
            )
        else:
            self.ema_adapters = None

        self.router = SourceReliabilityRouter(
            in_dim=student_dim,
            num_layer_slots=len(self.layer_indices),
            num_sources=self.num_sources,
            hidden_dim=router_hidden_dim,
        )

        self.register_buffer(
            "source_gate_mask",
            torch.ones(len(self.layer_indices), self.num_sources, dtype=torch.float32),
        )
        self.register_buffer(
            "source_inactive_steps",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.long),
        )
        self.register_buffer(
            "source_recover_steps",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.long),
        )
        self.register_buffer(
            "source_utility_active_ema",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.float32),
        )
        self.register_buffer(
            "source_utility_inactive_ema",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.float32),
        )
        self.register_buffer(
            "source_utility_active_initialized",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.bool),
        )
        self.register_buffer(
            "source_utility_inactive_initialized",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.bool),
        )
        self.register_buffer(
            "self_mitigation_windows_remaining",
            torch.zeros(1, dtype=torch.long),
        )

    @torch.no_grad()
    def set_self_mitigation_windows(self, windows: int) -> None:
        if self.num_sources <= 1:
            return
        self.self_mitigation_windows_remaining.fill_(max(0, int(windows)))
        if int(self.self_mitigation_windows_remaining.item()) > 0:
            self.source_gate_mask[:, 1] = 0.0
            self.source_inactive_steps[:, 1] = 0
            self.source_recover_steps[:, 1] = 0
            self.source_utility_active_ema[:, 1] = 0.0
            self.source_utility_inactive_ema[:, 1] = 0.0
            self.source_utility_active_initialized[:, 1] = False
            self.source_utility_inactive_initialized[:, 1] = False

    @torch.no_grad()
    def tick_self_mitigation_window(self) -> int:
        if self.num_sources <= 1:
            return 0
        remaining = int(self.self_mitigation_windows_remaining.item())
        if remaining <= 0:
            return 0
        remaining -= 1
        self.self_mitigation_windows_remaining.fill_(remaining)
        return remaining

    @torch.no_grad()
    def update_gate_state(
        self,
        layer_slot: int,
        utility_active_mean: torch.Tensor,
        utility_inactive_mean: torch.Tensor,
        utility_active_valid: torch.Tensor,
        utility_inactive_valid: torch.Tensor,
        source_ready: torch.Tensor,
        utility_off_threshold: float,
        utility_on_threshold: float,
        patience: int,
        reopen_patience: int,
        utility_ema_momentum: float,
        protect_source0: bool = True,
    ):
        # source0 is DINO, kept always available by default. Source1 (self) is
        # gated by utility rather than router confidence.
        for src_idx in range(self.num_sources):
            prev_gate_on = bool(self.source_gate_mask[layer_slot, src_idx].item() > 0.5)

            def _reset_gate_counters() -> None:
                self.source_inactive_steps[layer_slot, src_idx] = 0
                self.source_recover_steps[layer_slot, src_idx] = 0

            def _invalidate_active_ema() -> None:
                self.source_utility_active_ema[layer_slot, src_idx] = 0.0
                self.source_utility_active_initialized[layer_slot, src_idx] = False

            def _invalidate_inactive_ema() -> None:
                self.source_utility_inactive_ema[layer_slot, src_idx] = 0.0
                self.source_utility_inactive_initialized[layer_slot, src_idx] = False

            def _hard_reset_all() -> None:
                _reset_gate_counters()
                _invalidate_active_ema()
                _invalidate_inactive_ema()

            if protect_source0 and src_idx == 0:
                self.source_gate_mask[layer_slot, src_idx] = 1.0
                _reset_gate_counters()
                continue

            if (
                src_idx == 1
                and int(self.self_mitigation_windows_remaining.item()) > 0
            ):
                self.source_gate_mask[layer_slot, src_idx] = 0.0
                _hard_reset_all()
                continue

            if source_ready[src_idx] <= 0:
                self.source_gate_mask[layer_slot, src_idx] = 0.0
                _hard_reset_all()
                continue

            gate_on = prev_gate_on
            if gate_on:
                if utility_active_valid[src_idx] > 0:
                    if not bool(self.source_utility_active_initialized[layer_slot, src_idx].item()):
                        active_ema = utility_active_mean[src_idx]
                        self.source_utility_active_initialized[layer_slot, src_idx] = True
                    else:
                        prev_active_ema = self.source_utility_active_ema[layer_slot, src_idx]
                        active_ema = (
                            utility_ema_momentum * prev_active_ema
                            + (1.0 - utility_ema_momentum) * utility_active_mean[src_idx]
                        )
                    self.source_utility_active_ema[layer_slot, src_idx] = active_ema

                    if active_ema < utility_off_threshold:
                        self.source_inactive_steps[layer_slot, src_idx] += 1
                    else:
                        self.source_inactive_steps[layer_slot, src_idx] = 0
                else:
                    self.source_inactive_steps[layer_slot, src_idx] = 0

                self.source_recover_steps[layer_slot, src_idx] = 0

                if (
                    utility_active_valid[src_idx] > 0
                    and self.source_inactive_steps[layer_slot, src_idx] >= patience
                ):
                    gate_on = False
            else:
                if utility_inactive_valid[src_idx] > 0:
                    if not bool(self.source_utility_inactive_initialized[layer_slot, src_idx].item()):
                        inactive_ema = utility_inactive_mean[src_idx]
                        self.source_utility_inactive_initialized[layer_slot, src_idx] = True
                    else:
                        prev_inactive_ema = self.source_utility_inactive_ema[layer_slot, src_idx]
                        inactive_ema = (
                            utility_ema_momentum * prev_inactive_ema
                            + (1.0 - utility_ema_momentum) * utility_inactive_mean[src_idx]
                        )
                    self.source_utility_inactive_ema[layer_slot, src_idx] = inactive_ema

                    if inactive_ema > utility_on_threshold:
                        self.source_recover_steps[layer_slot, src_idx] += 1
                    else:
                        self.source_recover_steps[layer_slot, src_idx] = 0
                else:
                    self.source_recover_steps[layer_slot, src_idx] = 0

                self.source_inactive_steps[layer_slot, src_idx] = 0

                if (
                    utility_inactive_valid[src_idx] > 0
                    and self.source_recover_steps[layer_slot, src_idx] >= reopen_patience
                ):
                    gate_on = True

            if gate_on != prev_gate_on:
                _reset_gate_counters()
                # Entered-regime invalidate + first-sample seeding avoids
                # mixed-estimand carryover and zero-start bias.
                if gate_on:
                    _invalidate_active_ema()
                else:
                    _invalidate_inactive_ema()

            self.source_gate_mask[layer_slot, src_idx] = 1.0 if gate_on else 0.0

    def get_source_mask(
        self,
        layer_slot: int,
        source_ready: torch.Tensor,
        current_step: int,
        self_warmup_steps: int,
        enable_selective_gate: bool,
    ) -> torch.Tensor:
        mask = source_ready.clone()

        warmup_active = (
            self.use_ema_source
            and self.num_sources > 1
            and current_step < self_warmup_steps
        )
        if warmup_active:
            mask[1] = 0.0
            if enable_selective_gate:
                # Warmup should not be a delayed auto-open path for source1:
                # keep controller gate closed and clear runtime traces.
                with torch.no_grad():
                    self.source_gate_mask[layer_slot, 1] = 0.0
                    self.source_inactive_steps[layer_slot, 1] = 0
                    self.source_recover_steps[layer_slot, 1] = 0
                    self.source_utility_active_ema[layer_slot, 1] = 0.0
                    self.source_utility_inactive_ema[layer_slot, 1] = 0.0
                    self.source_utility_active_initialized[layer_slot, 1] = False
                    self.source_utility_inactive_initialized[layer_slot, 1] = False
        if (
            self.use_ema_source
            and self.num_sources > 1
            and int(self.self_mitigation_windows_remaining.item()) > 0
        ):
            mask[1] = 0.0

        if enable_selective_gate:
            mask = mask * self.source_gate_mask[layer_slot]
        if self.num_sources > 0 and source_ready[0] > 0:
            # Keep source0 available even if legacy gate state was stale on resume.
            mask[0] = source_ready[0]

        if mask.sum() <= 0:
            mask[0] = 1.0

        return mask

    def forward(
        self,
        student_tokens: Dict[str, torch.Tensor],
        ema_tokens: Dict[str, torch.Tensor],
        dino_tokens: torch.Tensor,
        t: torch.Tensor,
        T: int,
        current_step: int,
        phase_weight: float,
        layer_weights: List[float],
        args,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return compute_s3a_alignment_loss(
            s3a_head=self,
            student_tokens=student_tokens,
            ema_tokens=ema_tokens,
            dino_tokens=dino_tokens,
            t=t,
            T=T,
            current_step=current_step,
            phase_weight=phase_weight,
            layer_weights=layer_weights,
            args=args,
        )


#################################################################################
#                              Hooking Utilities                                #
#################################################################################


def _make_block_hook(storage: Dict[str, torch.Tensor], layer_idx: int):
    key = str(layer_idx)

    def _hook_fn(module, input, output):
        storage[key] = output

    return _hook_fn


def register_block_hooks(model_module, layer_indices: List[int], storage: Dict[str, torch.Tensor]):
    handles = []
    for idx in layer_indices:
        handle = model_module.blocks[idx].register_forward_hook(_make_block_hook(storage, idx))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


#################################################################################
#                           S3A Loss Aggregation                                #
#################################################################################


def compute_s3a_alignment_loss(
    s3a_head: S3AAlignmentHead,
    student_tokens: Dict[str, torch.Tensor],
    ema_tokens: Dict[str, torch.Tensor],
    dino_tokens: torch.Tensor,
    t: torch.Tensor,
    T: int,
    current_step: int,
    phase_weight: float,
    layer_weights: List[float],
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = dino_tokens.device
    diff_weights = get_diff_timestep_weight(t, T, args.s3a_diff_schedule)
    phase_norm = min(current_step / max(1, args.s3a_schedule_steps), 1.0)

    total_loss = torch.tensor(0.0, device=device)
    used_layers = 0

    feat_acc = 0.0
    attn_acc = 0.0
    spatial_acc = 0.0
    alpha_dino_acc = 0.0
    alpha_self_acc = 0.0
    gate_self_acc = 0.0
    gate_self_state_acc = 0.0
    raw_alpha_dino_acc = 0.0
    raw_alpha_self_acc = 0.0
    router_entropy_acc = 0.0
    router_policy_kl_acc = 0.0
    router_policy_gap_acc = 0.0
    loss_fused_acc = 0.0
    loss_fused_probe_acc = 0.0
    loss_dino_only_acc = 0.0
    loss_self_only_acc = 0.0
    utility_dino_acc = 0.0
    utility_self_acc = 0.0
    utility_self_active_ema_acc = 0.0
    utility_self_inactive_ema_acc = 0.0
    utility_self_active_ema_count = 0
    utility_self_inactive_ema_count = 0
    utility_probe_count = 0
    dino_only_count = 0
    self_only_count = 0
    collapse_alarm_count = 0
    alpha_dino_layer_values = [0.0 for _ in s3a_head.layer_indices]
    gate_patience_windows = max(1, math.ceil(args.s3a_gate_patience / max(1, args.s3a_probe_every)))
    gate_reopen_windows = max(
        1, math.ceil(args.s3a_gate_reopen_patience / max(1, args.s3a_probe_every))
    )

    for slot, layer_idx in enumerate(s3a_head.layer_indices):
        key = str(layer_idx)
        if key not in student_tokens:
            continue

        s_tokens = student_tokens[key]
        pred = s3a_head.student_adapters[key](s_tokens)

        # DINO features are extracted under inference_mode; clone to a normal
        # tensor so autograd ops (e.g., fused weighted sum) can save intermediates.
        dino_layer = resize_tokens_to_match(dino_tokens, pred.shape[1]).detach().clone()
        sources = [dino_layer]

        source_ready = torch.ones(s3a_head.num_sources, device=device)

        if s3a_head.use_ema_source:
            if key in ema_tokens:
                ema_tokens_detached = ema_tokens[key].detach().clone()
                if s3a_head.use_trainable_ema_adapters and s3a_head.ema_adapters is not None:
                    # Trainable self-side projector with gradient.
                    ema_proj = s3a_head.ema_adapters[key](ema_tokens_detached)
                    ema_proj = resize_tokens_to_match(ema_proj, pred.shape[1])
                else:
                    # Use the INDEPENDENT ema_adapters (frozen copy) to project
                    # EMA tokens.  This avoids the self-alignment shortcut where
                    # pred ≈ ema_proj trivially because same adapter + similar
                    # inputs produces near-identical outputs.
                    if s3a_head.ema_adapters is not None:
                        with torch.no_grad():
                            ema_proj = s3a_head.ema_adapters[key](ema_tokens_detached)
                        ema_proj = resize_tokens_to_match(ema_proj, pred.shape[1]).detach()
                    else:
                        # Fallback: no ema_adapters module exists.
                        with torch.no_grad():
                            ema_proj = s3a_head.student_adapters[key](ema_tokens_detached)
                        ema_proj = resize_tokens_to_match(ema_proj, pred.shape[1]).detach()
                sources.append(ema_proj)
                source_ready[1] = 1.0
            else:
                sources.append(torch.zeros_like(dino_layer))
                source_ready[1] = 0.0

        layer_slot_tensor = torch.full(
            (pred.shape[0],), slot, device=device, dtype=torch.long
        )
        t_norm = t.float() / max(T - 1, 1)
        phase_norm_tensor = torch.full_like(t_norm, fill_value=phase_norm)
        router_tokens = s_tokens.detach() if args.s3a_router_detach_input else s_tokens

        raw_alpha = s3a_head.router(
            tokens=router_tokens,
            layer_slot=layer_slot_tensor,
            timestep_norm=t_norm,
            phase_norm=phase_norm_tensor,
        )
        # During self_warmup (source[1] masked), detach router output so it
        # does NOT learn a DINO-only prior that causes softmax saturation.
        # After warmup the router starts from a near-uniform state.
        if (
            s3a_head.use_ema_source
            and s3a_head.num_sources > 1
            and current_step < args.s3a_self_warmup_steps
        ):
            raw_alpha = raw_alpha.detach()

        source_mask = s3a_head.get_source_mask(
            layer_slot=slot,
            source_ready=source_ready,
            current_step=current_step,
            self_warmup_steps=args.s3a_self_warmup_steps,
            enable_selective_gate=args.s3a_enable_selective_gate,
        )
        do_probe = (
            args.s3a_probe_every <= 1
            or (current_step % args.s3a_probe_every) == 0
        )

        def _apply_joint_min_alpha(
            alpha_local: torch.Tensor,
            min_alpha_by_source: Dict[int, float],
        ) -> torch.Tensor:
            if alpha_local.shape[-1] <= 1 or not min_alpha_by_source:
                return alpha_local

            floor = torch.zeros(alpha_local.shape[-1], device=alpha_local.device, dtype=alpha_local.dtype)
            for src_idx, min_alpha in min_alpha_by_source.items():
                if min_alpha <= 0:
                    continue
                if src_idx < 0 or src_idx >= alpha_local.shape[-1]:
                    continue
                floor[src_idx] = max(floor[src_idx], float(min_alpha))

            floor_sum = float(floor.sum().item())
            if floor_sum <= 0.0:
                return alpha_local
            if floor_sum >= 1.0:
                out = floor / floor.sum().clamp(min=1e-8)
                return out.unsqueeze(0).expand(alpha_local.shape[0], -1)

            floor_batch = floor.unsqueeze(0).expand(alpha_local.shape[0], -1)
            remain = 1.0 - floor_sum
            residual = alpha_local - floor_batch
            # Straight-through estimator: forward uses clamped residual,
            # backward passes gradient through as if unclamped.  This
            # prevents zero-gradient when raw_alpha < floor, allowing the
            # router to recover from below-floor states.
            residual_clamped = residual.clamp(min=0.0)
            residual = residual + (residual_clamped - residual).detach()
            # Use the forward-clamped sum for the branch condition to avoid
            # STE-induced negative sums triggering the fallback incorrectly.
            residual_sum_fwd = residual_clamped.sum(dim=-1, keepdim=True)
            fallback = alpha_local / alpha_local.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            residual_norm = torch.where(
                residual_sum_fwd > 1e-9,
                residual / residual_sum_fwd.clamp(min=1e-8),
                fallback,
            )
            out = floor_batch + residual_norm * remain
            out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return out

        def _compute_dino_floor(mask_vec: torch.Tensor) -> float:
            if not (s3a_head.num_sources > 1 and mask_vec[0] > 0):
                return 0.0
            return source0_min_alpha_at_step(current_step, args)

        def _build_alpha(
            mask_vec: torch.Tensor,
            extra_min_alpha_by_source: Optional[Dict[int, float]] = None,
        ) -> torch.Tensor:
            alpha_local = raw_alpha * mask_vec.unsqueeze(0)
            alpha_local = alpha_local / alpha_local.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            min_alpha_by_source: Dict[int, float] = {}
            dino_floor = _compute_dino_floor(mask_vec)
            if dino_floor > 0:
                min_alpha_by_source[0] = dino_floor
            if extra_min_alpha_by_source:
                for src_idx, min_alpha in extra_min_alpha_by_source.items():
                    if min_alpha <= 0:
                        continue
                    prev = min_alpha_by_source.get(src_idx, 0.0)
                    min_alpha_by_source[src_idx] = max(prev, float(min_alpha))
            if min_alpha_by_source:
                alpha_local = _apply_joint_min_alpha(alpha_local, min_alpha_by_source)
            return alpha_local

        alpha = _build_alpha(source_mask)
        router_policy_kl_ps = torch.zeros(pred.shape[0], device=device)
        router_policy_gap_ps = torch.zeros(pred.shape[0], device=device)
        if (
            s3a_head.num_sources > 1
            and source_ready.sum().item() > 1
            and args.s3a_router_policy_kl_lambda > 0
        ):
            router_policy_kl_ps, router_policy_gap_ps = router_policy_kl_and_gap_per_sample(
                raw_alpha=raw_alpha,
                policy_alpha=alpha.detach(),
            )

        fused = torch.zeros_like(pred)
        for src_idx, src_tokens in enumerate(sources):
            fused = fused + alpha[:, src_idx].view(-1, 1, 1) * src_tokens

        def _combined_loss_per_sample(
            pred_tokens: torch.Tensor,
            target_tokens: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            feat_ps_local = cosine_distance_per_sample(pred_tokens, target_tokens)
            attn_ps_local = affinity_loss_per_sample(
                pred_tokens,
                target_tokens,
                max_tokens=args.s3a_attn_max_tokens,
            )
            spatial_ps_local = spatial_loss_per_sample(pred_tokens, target_tokens)
            combined_ps_local = (
                args.s3a_feat_weight * feat_ps_local
                + args.s3a_attn_weight * attn_ps_local
                + args.s3a_spatial_weight * spatial_ps_local
            )
            return feat_ps_local, attn_ps_local, spatial_ps_local, combined_ps_local

        feat_loss_ps, attn_loss_ps, spatial_loss_ps, combined_ps = _combined_loss_per_sample(
            pred, fused
        )

        dino_layer_loss_mean = 0.0
        self_layer_loss_mean = 0.0
        fused_probe_loss_mean = combined_ps.mean().item()
        utility_dino_layer = 0.0
        utility_self_layer = 0.0
        utility_active_layer = torch.zeros(s3a_head.num_sources, device=device)
        utility_inactive_layer = torch.zeros(s3a_head.num_sources, device=device)
        utility_active_valid = torch.zeros(s3a_head.num_sources, device=device)
        utility_inactive_valid = torch.zeros(s3a_head.num_sources, device=device)

        if do_probe:
            with torch.no_grad():
                pred_probe = pred.detach()
                _, _, _, dino_combined_ps = _combined_loss_per_sample(pred_probe, dino_layer)
                dino_layer_loss_mean = dino_combined_ps.mean().item()
                dino_only_count += 1

                if s3a_head.num_sources > 1 and source_ready[1] > 0:
                    self_layer = sources[1]
                    _, _, _, self_combined_ps = _combined_loss_per_sample(pred_probe, self_layer)
                    self_layer_loss_mean = self_combined_ps.mean().item()
                    self_only_count += 1

                if args.s3a_utility_probe_mode == "policy_loo":
                    # Policy-consistent utility: leave-one-out / add-one under the
                    # same alpha builder used by training (same router + floor rule).
                    utility_active_per_source = torch.zeros(s3a_head.num_sources, device=device)
                    utility_inactive_per_source = torch.zeros(s3a_head.num_sources, device=device)
                    utility_active_valid_per_source = torch.zeros(
                        s3a_head.num_sources, device=device
                    )
                    utility_inactive_valid_per_source = torch.zeros(
                        s3a_head.num_sources, device=device
                    )
                    alpha_train_probe = _build_alpha(source_mask)
                    fused_train_probe = torch.zeros_like(pred_probe)
                    for src_idx, src_tokens in enumerate(sources):
                        fused_train_probe = (
                            fused_train_probe
                            + alpha_train_probe[:, src_idx].view(-1, 1, 1) * src_tokens
                        )
                    _, _, _, base_combined_ps = _combined_loss_per_sample(pred_probe, fused_train_probe)
                    fused_probe_loss_mean = base_combined_ps.mean().item()

                    for src_idx in range(s3a_head.num_sources):
                        if source_ready[src_idx] <= 0:
                            continue
                        if source_mask[src_idx] > 0:
                            if source_mask.sum().item() <= 1:
                                continue
                            cf_mask = source_mask.clone()
                            cf_mask[src_idx] = 0.0
                            cf_alpha = _build_alpha(cf_mask)
                            cf_fused = torch.zeros_like(pred_probe)
                            for k, src_tokens in enumerate(sources):
                                cf_fused = cf_fused + cf_alpha[:, k].view(-1, 1, 1) * src_tokens
                            _, _, _, cf_combined_ps = _combined_loss_per_sample(pred_probe, cf_fused)
                            utility_active_per_source[src_idx] = (
                                cf_combined_ps.mean().item() - fused_probe_loss_mean
                            )
                            utility_active_valid_per_source[src_idx] = 1.0
                        else:
                            # Gate-off recovery signal: test adding this source under
                            # current policy, without changing optimizer path.
                            cf_mask = source_mask.clone()
                            cf_mask[src_idx] = source_ready[src_idx]
                            if cf_mask.sum().item() <= 1:
                                continue
                            cf_alpha = _build_alpha(
                                cf_mask,
                                extra_min_alpha_by_source={
                                    src_idx: args.s3a_gate_reopen_probe_alpha_floor
                                },
                            )
                            cf_fused = torch.zeros_like(pred_probe)
                            for k, src_tokens in enumerate(sources):
                                cf_fused = cf_fused + cf_alpha[:, k].view(-1, 1, 1) * src_tokens
                            _, _, _, cf_combined_ps = _combined_loss_per_sample(pred_probe, cf_fused)
                            utility_inactive_per_source[src_idx] = (
                                fused_probe_loss_mean - cf_combined_ps.mean().item()
                            )
                            utility_inactive_valid_per_source[src_idx] = 1.0

                    utility_active_layer = utility_active_per_source
                    utility_inactive_layer = utility_inactive_per_source
                    utility_active_valid = utility_active_valid_per_source
                    utility_inactive_valid = utility_inactive_valid_per_source
                else:
                    # Legacy utility estimator kept for ablation/backward comparison.
                    source_mask_probe = s3a_head.get_source_mask(
                        layer_slot=slot,
                        source_ready=source_ready,
                        current_step=current_step,
                        self_warmup_steps=args.s3a_self_warmup_steps,
                        enable_selective_gate=False,
                    )
                    if args.s3a_utility_probe_mode == "raw_alpha":
                        alpha_probe = _build_alpha(source_mask_probe)
                    else:
                        alpha_probe_local = source_mask_probe / source_mask_probe.sum().clamp(min=1e-8)
                        alpha_probe = alpha_probe_local.unsqueeze(0).expand(raw_alpha.shape[0], -1)
                    fused_probe = torch.zeros_like(pred_probe)
                    for src_idx, src_tokens in enumerate(sources):
                        fused_probe = fused_probe + alpha_probe[:, src_idx].view(-1, 1, 1) * src_tokens
                    _, _, _, fused_probe_combined_ps = _combined_loss_per_sample(pred_probe, fused_probe)
                    fused_probe_loss_mean = fused_probe_combined_ps.mean().item()
                    utility_self_layer = dino_layer_loss_mean - fused_probe_loss_mean
                    if s3a_head.num_sources > 1 and source_mask_probe[1] > 0:
                        utility_dino_layer = self_layer_loss_mean - fused_probe_loss_mean

                    # Legacy mode does not separate on/off estimands; attach
                    # the scalar utility to whichever side is currently observed.
                    utility_layer_legacy = torch.zeros(s3a_head.num_sources, device=device)
                    utility_layer_legacy[0] = utility_dino_layer
                    if s3a_head.num_sources > 1:
                        utility_layer_legacy[1] = utility_self_layer
                    for src_idx in range(s3a_head.num_sources):
                        if source_ready[src_idx] <= 0:
                            continue
                        if source_mask[src_idx] > 0:
                            utility_active_layer[src_idx] = utility_layer_legacy[src_idx]
                            utility_active_valid[src_idx] = 1.0
                        else:
                            utility_inactive_layer[src_idx] = utility_layer_legacy[src_idx]
                            utility_inactive_valid[src_idx] = 1.0

                if source_mask[0] > 0 and utility_active_valid[0] > 0:
                    utility_dino_layer = utility_active_layer[0].item()
                elif source_mask[0] <= 0 and utility_inactive_valid[0] > 0:
                    utility_dino_layer = utility_inactive_layer[0].item()

                if s3a_head.num_sources > 1:
                    if source_mask[1] > 0 and utility_active_valid[1] > 0:
                        utility_self_layer = utility_active_layer[1].item()
                    elif source_mask[1] <= 0 and utility_inactive_valid[1] > 0:
                        utility_self_layer = utility_inactive_layer[1].item()
                utility_probe_count += 1

        sample_weights = diff_weights * (phase_weight * layer_weights[slot])
        router_policy_kl_loss = (sample_weights * router_policy_kl_ps).mean()
        layer_loss = (
            (sample_weights * combined_ps).mean()
            + args.s3a_router_policy_kl_lambda * router_policy_kl_loss
        )

        total_loss = total_loss + layer_loss
        used_layers += 1

        feat_acc += feat_loss_ps.mean().item()
        attn_acc += attn_loss_ps.mean().item()
        spatial_acc += spatial_loss_ps.mean().item()
        alpha_dino_acc += alpha[:, 0].mean().item()
        alpha_dino_layer_values[slot] = alpha[:, 0].mean().item()
        raw_alpha_dino_acc += raw_alpha[:, 0].mean().item()
        if s3a_head.num_sources > 1:
            raw_alpha_self_acc += raw_alpha[:, 1].mean().item()
            entropy = -(
                raw_alpha.clamp(min=1e-8) * raw_alpha.clamp(min=1e-8).log()
            ).sum(dim=-1)
            entropy = entropy / math.log(s3a_head.num_sources)
            router_entropy_acc += entropy.mean().item()
            router_policy_kl_acc += router_policy_kl_ps.mean().item()
            router_policy_gap_acc += router_policy_gap_ps.mean().item()
        else:
            router_entropy_acc += 0.0
            router_policy_kl_acc += 0.0
            router_policy_gap_acc += 0.0
        loss_fused_acc += combined_ps.mean().item()
        loss_fused_probe_acc += fused_probe_loss_mean
        loss_dino_only_acc += dino_layer_loss_mean
        utility_dino_acc += utility_dino_layer
        utility_self_acc += utility_self_layer
        if s3a_head.num_sources > 1:
            alpha_self_acc += alpha[:, 1].mean().item()
            gate_self_acc += source_mask[1].item()
            if args.s3a_enable_selective_gate:
                gate_self_state_acc += s3a_head.source_gate_mask[slot, 1].item()
            else:
                gate_self_state_acc += 1.0 if source_ready[1].item() > 0 else 0.0
            loss_self_only_acc += self_layer_loss_mean
            if do_probe:
                if bool(s3a_head.source_utility_active_initialized[slot, 1].item()):
                    utility_self_active_ema_acc += s3a_head.source_utility_active_ema[slot, 1].item()
                    utility_self_active_ema_count += 1
                if bool(s3a_head.source_utility_inactive_initialized[slot, 1].item()):
                    utility_self_inactive_ema_acc += (
                        s3a_head.source_utility_inactive_ema[slot, 1].item()
                    )
                    utility_self_inactive_ema_count += 1

            if (
                do_probe
                and
                current_step >= args.s3a_self_warmup_steps
                and source_ready[1] > 0
                and alpha[:, 0].mean().item() <= args.s3a_collapse_alpha_threshold
                and alpha[:, 1].mean().item() > args.s3a_collapse_self_threshold
                and utility_dino_layer > args.s3a_collapse_utility_threshold
                and fused_probe_loss_mean + args.s3a_collapse_margin < self_layer_loss_mean
            ):
                collapse_alarm_count += 1

        if (
            args.s3a_enable_selective_gate
            and current_step >= args.s3a_self_warmup_steps
            and do_probe
        ):
            with torch.no_grad():
                utility_active_mean = utility_active_layer.clone()
                utility_inactive_mean = utility_inactive_layer.clone()
                utility_active_valid_mean = utility_active_valid.clone()
                utility_inactive_valid_mean = utility_inactive_valid.clone()
                ready = source_ready.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(utility_active_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(utility_inactive_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(utility_active_valid_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(utility_inactive_valid_mean, op=dist.ReduceOp.SUM)

                    dist.all_reduce(ready, op=dist.ReduceOp.SUM)
                    ready = (ready > 0).float()

                utility_active_mean = torch.where(
                    utility_active_valid_mean > 0,
                    utility_active_mean / utility_active_valid_mean.clamp(min=1.0),
                    torch.zeros_like(utility_active_mean),
                )
                utility_inactive_mean = torch.where(
                    utility_inactive_valid_mean > 0,
                    utility_inactive_mean / utility_inactive_valid_mean.clamp(min=1.0),
                    torch.zeros_like(utility_inactive_mean),
                )

                s3a_head.update_gate_state(
                    layer_slot=slot,
                    utility_active_mean=utility_active_mean,
                    utility_inactive_mean=utility_inactive_mean,
                    utility_active_valid=utility_active_valid_mean,
                    utility_inactive_valid=utility_inactive_valid_mean,
                    source_ready=ready,
                    utility_off_threshold=args.s3a_gate_utility_off_threshold,
                    utility_on_threshold=args.s3a_gate_utility_on_threshold,
                    patience=gate_patience_windows,
                    reopen_patience=gate_reopen_windows,
                    utility_ema_momentum=args.s3a_gate_utility_ema_momentum,
                    protect_source0=True,
                )

    if used_layers == 0:
        return total_loss, make_empty_align_stats(s3a_head.layer_indices)

    total_loss = total_loss / used_layers

    stats = {
        "used_layers": float(used_layers),
        "feat": feat_acc / used_layers,
        "attn": attn_acc / used_layers,
        "spatial": spatial_acc / used_layers,
        "alpha_dino": alpha_dino_acc / used_layers,
        "alpha_self": alpha_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0,
        "gate_self": gate_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0,
        "gate_self_state": (
            gate_self_state_acc / used_layers if s3a_head.num_sources > 1 else 0.0
        ),
        "diff_w": diff_weights.mean().item(),
        "raw_alpha_dino": raw_alpha_dino_acc / used_layers,
        "raw_alpha_self": (
            raw_alpha_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0
        ),
        "router_entropy_norm": router_entropy_acc / used_layers,
        "router_policy_kl": router_policy_kl_acc / used_layers,
        "router_policy_gap": router_policy_gap_acc / used_layers,
        "loss_fused": loss_fused_acc / used_layers,
        "loss_fused_probe": loss_fused_probe_acc / max(1, utility_probe_count),
        "loss_dino_only": loss_dino_only_acc / max(1, dino_only_count),
        "loss_self_only": (
            loss_self_only_acc / max(1, self_only_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
        "utility_dino": utility_dino_acc / max(1, utility_probe_count),
        "utility_self": utility_self_acc / max(1, utility_probe_count),
        "utility_self_ema": (
            utility_self_active_ema_acc / max(1, utility_self_active_ema_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
        "utility_self_active_ema": (
            utility_self_active_ema_acc / max(1, utility_self_active_ema_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
        "utility_self_inactive_ema": (
            utility_self_inactive_ema_acc / max(1, utility_self_inactive_ema_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
        "utility_self_active_ema_count": float(utility_self_active_ema_count),
        "utility_self_inactive_ema_count": float(utility_self_inactive_ema_count),
        "probe_count": float(utility_probe_count),
        "self_probe_count": float(self_only_count),
        "collapse_alarm": (
            collapse_alarm_count / max(1, utility_probe_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
        "alpha_dino_min_layer": min(alpha_dino_layer_values) if used_layers > 0 else 0.0,
        "alpha_dino_max_layer": max(alpha_dino_layer_values) if used_layers > 0 else 0.0,
        "alpha_dino_layers": alpha_dino_layer_values,
    }
    return total_loss, stats


#################################################################################
#                                 Checkpoint                                    #
#################################################################################


def _checkpoint_manifest_path(checkpoint_path: str) -> str:
    return f"{checkpoint_path}.sha256.json"


def _atomic_torch_save(payload: dict, output_path: str) -> None:
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def _write_checkpoint_manifest(checkpoint_path: str, train_steps: int) -> str:
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    payload = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": os.path.getsize(checkpoint_path),
        "train_steps": int(train_steps),
        "created_at": strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def _validate_checkpoint_manifest(checkpoint_path: str, allow_missing_manifest: bool = False) -> None:
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    if not os.path.isfile(manifest_path):
        if allow_missing_manifest:
            return
        raise ValueError(
            f"Missing checkpoint manifest: {manifest_path}. "
            "Use --allow-missing-manifest to bypass legacy checkpoints."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest.get("sha256")
    if not expected:
        raise ValueError(f"Invalid checkpoint manifest (missing sha256): {manifest_path}")

    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise ValueError(
            "Checkpoint sha256 mismatch. "
            f"expected={expected}, actual={actual}, path={checkpoint_path}"
        )


def _namespace_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, argparse.Namespace):
        return vars(value).copy()
    if isinstance(value, dict):
        return dict(value)
    return {}


def _is_equal_resume_value(current: Any, saved: Any, tol: float = 1e-12) -> bool:
    if isinstance(current, (int, float, bool)) and isinstance(saved, (int, float, bool)):
        return abs(float(current) - float(saved)) <= tol
    return current == saved


def _validate_resume_contract(
    current_args,
    saved_args_dict: Dict[str, Any],
    allow_legacy_resume_args: bool = False,
) -> None:
    if not saved_args_dict:
        return
    keys = [
        "model",
        "data_path",
        "global_batch_size",
        "global_seed",
        "num_workers",
        "vae_model_dir",
        "dinov2_weight_path",
        "dinov2_model_variant",
        "s3a",
        "s3a_use_ema_source",
        "s3a_trainable_ema_adapters",
        "s3a_enable_selective_gate",
        "s3a_self_warmup_steps",
        "s3a_allow_unsafe_zero_source0_floor",
        "s3a_dino_alpha_floor",
        "s3a_dino_alpha_floor_steps",
        "s3a_protect_source0_min_alpha",
        "s3a_train_schedule",
        "s3a_schedule_steps",
        "s3a_schedule_warmup_steps",
        "s3a_diff_schedule",
        "s3a_layer_weight_mode",
        "s3a_layer_weights",
        "s3a_probe_every",
        "s3a_utility_probe_mode",
        "s3a_gate_reopen_probe_alpha_floor",
        "s3a_gate_patience",
        "s3a_gate_reopen_patience",
        "s3a_gate_utility_off_threshold",
        "s3a_gate_utility_on_threshold",
        "s3a_gate_utility_ema_momentum",
        "s3a_collapse_alpha_threshold",
        "s3a_collapse_self_threshold",
        "s3a_collapse_margin",
        "s3a_collapse_utility_threshold",
        "s3a_collapse_windows",
        "s3a_collapse_auto_mitigate",
        "s3a_collapse_mitigate_windows",
        "s3a_collapse_mitigate_cooldown_windows",
        "s3a_feat_weight",
        "s3a_attn_weight",
        "s3a_spatial_weight",
        "s3a_layer_indices",
        "s3a_lambda",
        "s3a_adapter_hidden_dim",
        "s3a_router_hidden_dim",
        "s3a_router_detach_input",
        "s3a_router_policy_kl_lambda",
        "s3a_attn_max_tokens",
        "s3a_max_grad_norm",
    ]
    backward_compatible_missing_keys = {
        # Added after legacy checkpoints; safe to backfill from current args.
        "s3a_allow_unsafe_zero_source0_floor",
        "s3a_router_policy_kl_lambda",
        "dinov2_model_variant",
        "s3a_adapter_hidden_dim",
        "s3a_schedule_warmup_steps",
    }
    backward_compatible_missing_defaults = {
        # For boolean contract flags, require legacy-equivalent default.
        "s3a_allow_unsafe_zero_source0_floor": False,
        # Objective-changing router regularizer: old checkpoints were equivalent to 0.0.
        "s3a_router_policy_kl_lambda": 0.0,
        # Legacy checkpoints always used vitb14.
        "dinov2_model_variant": "vitb14",
        # Legacy checkpoints stored None (fallback to in_dim); new default is 2048.
        "s3a_adapter_hidden_dim": None,
        # Legacy checkpoints had no warmup_steps concept (equivalent to 0).
        "s3a_schedule_warmup_steps": 0,
    }
    mismatches = []
    missing_keys = []
    for key in keys:
        if not hasattr(current_args, key):
            continue
        if key not in saved_args_dict:
            if key in backward_compatible_missing_keys:
                if key in backward_compatible_missing_defaults:
                    legacy_default = backward_compatible_missing_defaults[key]
                    if _is_equal_resume_value(getattr(current_args, key), legacy_default):
                        continue
                    missing_keys.append(key)
                    continue
                continue
            missing_keys.append(key)
            continue
        current_val = getattr(current_args, key)
        saved_val = saved_args_dict[key]
        if not _is_equal_resume_value(current_val, saved_val):
            mismatches.append((key, current_val, saved_val))
    if missing_keys and not allow_legacy_resume_args:
        raise ValueError(
            "Resume contract missing critical keys in checkpoint args: "
            f"{missing_keys[:8]}{' ...' if len(missing_keys) > 8 else ''}. "
            "Some late-added keys are auto-backfilled from current args, while "
            "safety and objective-changing keys still require legacy-equivalent defaults. "
            "Use --allow-legacy-resume-args to bypass (unsafe)."
        )
    if mismatches:
        utility_mode_mismatch = any(k == "s3a_utility_probe_mode" for (k, _, _) in mismatches)
        detail = ", ".join(
            [f"{k}(current={c}, checkpoint={s})" for (k, c, s) in mismatches[:8]]
        )
        guidance = ""
        if utility_mode_mismatch:
            guidance = (
                " Legacy checkpoints may require explicitly passing "
                "--s3a-utility-probe-mode to match checkpoint args."
            )
        raise ValueError(
            "Resume contract mismatch for critical S3A args: "
            f"{detail}. Refuse to resume with incompatible config.{guidance}"
        )


def _reset_selective_gate_runtime_state(
    target_state: Dict[str, torch.Tensor],
    expected_state: Dict[str, torch.Tensor],
) -> List[str]:
    reset_keys = (
        "source_gate_mask",
        "source_inactive_steps",
        "source_recover_steps",
        "source_utility_active_ema",
        "source_utility_inactive_ema",
        "source_utility_active_initialized",
        "source_utility_inactive_initialized",
        "self_mitigation_windows_remaining",
    )
    applied: List[str] = []
    for key in reset_keys:
        if key in expected_state:
            target_state[key] = expected_state[key].clone()
            applied.append(key)
    return applied


def _sterilize_source0_gate_lane(
    target_state: Dict[str, torch.Tensor],
    expected_state: Dict[str, torch.Tensor],
) -> List[str]:
    lane0_keys = (
        "source_gate_mask",
        "source_inactive_steps",
        "source_recover_steps",
        "source_utility_active_ema",
        "source_utility_inactive_ema",
        "source_utility_active_initialized",
        "source_utility_inactive_initialized",
    )
    applied: List[str] = []
    for key in lane0_keys:
        if key not in target_state or key not in expected_state:
            continue
        cur = target_state[key]
        exp = expected_state[key]
        if (
            not isinstance(cur, torch.Tensor)
            or not isinstance(exp, torch.Tensor)
            or cur.ndim < 2
            or exp.ndim < 2
            or cur.shape[0] != exp.shape[0]
            or cur.shape[1] == 0
            or exp.shape[1] == 0
        ):
            continue
        if key == "source_gate_mask":
            target_state[key][:, 0] = 1.0
        else:
            target_state[key][:, 0] = exp[:, 0]
        applied.append(key)
    return applied


def _migrate_legacy_s3a_state(
    loaded_state: Dict[str, torch.Tensor],
    expected_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    migrated = dict(loaded_state)
    migration_notes: List[str] = []
    legacy_utility_ema = migrated.pop("source_utility_ema", None)
    if legacy_utility_ema is not None:
        migration_notes.append("discarded_legacy_source_utility_ema")
        # Legacy single-track utility semantics are incompatible with the
        # current selective-gate runtime state; sterilize the full controller.
        reset_applied = _reset_selective_gate_runtime_state(migrated, expected_state)
        if reset_applied:
            migration_notes.append("reset_selective_gate_state_for_legacy_utility_semantics")
    source0_sterilized = _sterilize_source0_gate_lane(migrated, expected_state)
    if source0_sterilized:
        migration_notes.append("sterilized_source0_gate_lane")

    missing = [k for k in expected_state.keys() if k not in migrated]
    unexpected = [k for k in migrated.keys() if k not in expected_state]
    if unexpected:
        raise ValueError(
            "S3A checkpoint contains unexpected keys not present in current model: "
            f"{unexpected[:8]}"
        )

    legacy_fill_keys = (
        "source_gate_mask",
        "source_inactive_steps",
        "source_recover_steps",
        "source_utility_active_ema",
        "source_utility_inactive_ema",
        "source_utility_active_initialized",
        "source_utility_inactive_initialized",
        "self_mitigation_windows_remaining",
        "ema_adapters.",
    )
    disallowed_missing = [k for k in missing if not k.startswith(legacy_fill_keys)]
    if disallowed_missing:
        raise ValueError(
            "S3A checkpoint missing non-migratable keys: "
            f"{disallowed_missing[:8]}"
        )

    for key in missing:
        migrated[key] = expected_state[key]
    return migrated, missing, migration_notes


def save_checkpoint(
    checkpoint_dir: str,
    train_steps: int,
    batches_seen: int,
    model,
    ema,
    opt,
    args,
    s3a_head=None,
    rng_states: Optional[List[dict]] = None,
    s3a_runtime_state: Optional[Dict[str, Any]] = None,
) -> str:
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "train_steps": train_steps,
        "batches_seen": batches_seen,
        "model_state": model.module.state_dict(),
        "ema_state": ema.state_dict(),
        "opt_state": opt.state_dict(),
        "args": args,
    }
    if s3a_head is not None:
        checkpoint["s3a_head_state"] = s3a_head.module.state_dict()
    if rng_states is not None:
        checkpoint["rng_states"] = rng_states
    if s3a_runtime_state is not None:
        checkpoint["s3a_runtime_state"] = dict(s3a_runtime_state)

    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    _atomic_torch_save(checkpoint, checkpoint_path)
    _write_checkpoint_manifest(checkpoint_path, train_steps)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    ema,
    opt,
    s3a_head=None,
    current_args=None,
    allow_legacy_resume_args: bool = False,
    allow_missing_manifest: bool = False,
) -> Tuple[int, int, Optional[List[dict]], Dict[str, Any]]:
    _validate_checkpoint_manifest(
        checkpoint_path,
        allow_missing_manifest=allow_missing_manifest,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unexpected checkpoint format at {checkpoint_path}: {type(checkpoint)}"
        )

    format_version = int(checkpoint.get("format_version", 1))
    model_state = checkpoint.get("model_state", checkpoint.get("model"))
    ema_state = checkpoint.get("ema_state", checkpoint.get("ema"))
    opt_state = checkpoint.get("opt_state", checkpoint.get("opt"))
    s3a_state = checkpoint.get("s3a_head_state", checkpoint.get("s3a_head"))
    saved_args_dict = _namespace_to_dict(checkpoint.get("args"))
    saved_runtime_state = checkpoint.get("s3a_runtime_state")
    if not isinstance(saved_runtime_state, dict):
        saved_runtime_state = {}
    if current_args is not None:
        _validate_resume_contract(
            current_args,
            saved_args_dict,
            allow_legacy_resume_args=allow_legacy_resume_args,
        )

    if model_state is None or ema_state is None or opt_state is None:
        raise ValueError(
            "Checkpoint is missing required states. "
            "Expected model/ema/opt or model_state/ema_state/opt_state."
        )

    model.module.load_state_dict(model_state, strict=True)
    ema.load_state_dict(ema_state, strict=True)
    opt.load_state_dict(opt_state)

    if s3a_head is not None:
        if s3a_state is None:
            raise ValueError(
                "S3A checkpoint is missing s3a_head_state/s3a_head while --s3a is enabled."
            )
        expected_state = s3a_head.module.state_dict()
        migration_notes: List[str] = []
        if format_version < CHECKPOINT_FORMAT_VERSION:
            s3a_state, _, migration_notes = _migrate_legacy_s3a_state(
                loaded_state=s3a_state,
                expected_state=expected_state,
            )
        source0_sterilized = _sterilize_source0_gate_lane(s3a_state, expected_state)
        if source0_sterilized and "sterilized_source0_gate_lane" not in migration_notes:
            migration_notes.append("sterilized_source0_gate_lane")
        s3a_head.module.load_state_dict(s3a_state, strict=True)
    else:
        migration_notes = []

    resume_meta = {
        "format_version": format_version,
        "saved_args": saved_args_dict,
        "s3a_runtime_state": saved_runtime_state,
        "s3a_migration_notes": migration_notes,
    }

    if "train_steps" in checkpoint:
        train_steps = int(checkpoint["train_steps"])
        batches_seen = int(checkpoint.get("batches_seen", train_steps))
        return train_steps, batches_seen, checkpoint.get("rng_states"), resume_meta

    # Backward fallback: infer from filename like 0001000.pt
    stem = os.path.basename(checkpoint_path).split(".")[0]
    if stem.isdigit():
        step = int(stem)
        return step, step, None, resume_meta
    raise ValueError(
        "Checkpoint does not contain train_steps and filename is not numeric. "
        f"Unable to infer resume step from: {checkpoint_path}"
    )


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, (
        "global_batch_size must be divisible by world_size."
    )

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    device = local_rank
    seed = args.global_seed * dist.get_world_size() + rank
    git_revision = get_git_revision(os.getcwd())

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        model_string_name = args.model.replace("/", "-")
        run_stamp = strftime("%Y%m%d-%H%M%S")
        run_uid = os.urandom(3).hex()
        experiment_name = f"{model_string_name}-seed{args.global_seed}-{run_stamp}-{run_uid}"
        if args.s3a:
            experiment_name += (
                f"-s3a-dinov2"
                f"-lam{args.s3a_lambda}"
                f"-train{args.s3a_train_schedule}"
                f"-diff{args.s3a_diff_schedule}"
            )
        experiment_dir = os.path.join(args.results_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=False)
        metrics_jsonl_path = os.path.join(experiment_dir, "metrics.jsonl")
        resolved_args_path = os.path.join(experiment_dir, "resolved_args.json")
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        metrics_jsonl_path = None
        resolved_args_path = None

    assert args.image_size % 8 == 0, "image_size must be divisible by 8."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)

    diffusion = create_diffusion(timestep_respacing="")
    T = diffusion.num_timesteps

    if args.vae_model_dir is not None:
        vae_path = args.vae_model_dir
        logger.info(f"Loading VAE from local directory: {vae_path}")
    else:
        vae_path = f"stabilityai/sd-vae-ft-{args.vae}"
        logger.info(f"Loading VAE from HuggingFace Hub: {vae_path}")

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    requires_grad(vae, False)

    logger.info(
        f"DiT model      : {args.model}\n"
        f"DiT parameters : {sum(p.numel() for p in model.parameters()):,}\n"
        f"Diffusion T    : {T}"
    )
    if args.max_steps is not None:
        logger.info(f"Training will stop at max_steps = {args.max_steps}")

    dino_model = None
    s3a_head = None
    s3a_layer_indices: List[int] = []
    s3a_layer_weights: List[float] = []

    if args.s3a:
        logger.info(
            f"Initializing S3A alignment with DINOv2 {args.dinov2_model_variant} teacher\n"
            f"  train_schedule      : {args.s3a_train_schedule} "
            f"(decay over {args.s3a_schedule_steps} steps)\n"
            f"  diff_schedule       : {args.s3a_diff_schedule}\n"
            f"  s3a_lambda          : {args.s3a_lambda}\n"
            f"  use_ema_source      : {args.s3a_use_ema_source}\n"
            f"  selective_gate      : {args.s3a_enable_selective_gate}\n"
            f"  DINOv2 variant      : {args.dinov2_model_variant}\n"
            f"  DINOv2 repo         : {args.dinov2_repo_dir}\n"
            f"  DINOv2 weight       : {args.dinov2_weight_path}"
        )

        dino_model = LocalDINOv2Teacher(
            dinov2_repo_dir=args.dinov2_repo_dir,
            weight_path=args.dinov2_weight_path,
            model_variant=args.dinov2_model_variant,
        ).to(device)
        dino_model.eval()
        requires_grad(dino_model, False)

        dit_hidden_dim = model.hidden_size
        dit_depth = model.depth

        s3a_layer_indices = parse_layer_indices(args.s3a_layer_indices, dit_depth)
        s3a_layer_weights = build_layer_weights(
            mode=args.s3a_layer_weight_mode,
            custom_csv=args.s3a_layer_weights,
            layer_indices=s3a_layer_indices,
            depth=dit_depth,
        )

        with torch.inference_mode():
            dummy_rgb = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
            dummy_dino = preprocess_for_dino(dummy_rgb)
            dino_probe = dino_model(dummy_dino)
            dino_dim = dino_probe.shape[-1]
            dino_num_tokens = dino_probe.shape[1]

        s3a_head = S3AAlignmentHead(
            layer_indices=s3a_layer_indices,
            student_dim=dit_hidden_dim,
            target_dim=dino_dim,
            adapter_hidden_dim=args.s3a_adapter_hidden_dim,
            router_hidden_dim=args.s3a_router_hidden_dim,
            use_ema_source=args.s3a_use_ema_source,
            use_trainable_ema_adapters=args.s3a_trainable_ema_adapters,
        ).to(device)

        logger.info(
            f"S3A setup:\n"
            f"  DiT hidden_dim      : {dit_hidden_dim}\n"
            f"  DINO variant        : {args.dinov2_model_variant}\n"
            f"  DINO dim            : {dino_dim}\n"
            f"  DINO num_tokens     : {dino_num_tokens}\n"
            f"  adapter hidden_dim  : {args.s3a_adapter_hidden_dim}\n"
            f"  layer indices       : {s3a_layer_indices}\n"
            f"  layer weights       : {[round(w, 4) for w in s3a_layer_weights]}\n"
            f"  trainable ema-adpt  : {args.s3a_trainable_ema_adapters}\n"
            f"  S3A params          : {sum(p.numel() for p in s3a_head.parameters()):,}"
        )
        if not args.s3a_trainable_ema_adapters:
            # Initialize ema_adapters from student_adapters (identical starting
            # point) and freeze them.  They will be updated via
            # update_ema_adapters() after each optimizer step, NOT by backprop.
            if s3a_head.ema_adapters is not None:
                for key in s3a_head.student_adapters:
                    sd = s3a_head.student_adapters[key].state_dict()
                    s3a_head.ema_adapters[key].load_state_dict(sd)
            for name, p in s3a_head.named_parameters():
                if name.startswith("ema_adapters."):
                    p.requires_grad_(False)

    model = DDP(model, device_ids=[device])
    if args.s3a:
        s3a_head = DDP(s3a_head, device_ids=[device], broadcast_buffers=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    s3a_trainable_param_count = 0
    s3a_frozen_param_count = 0
    if args.s3a:
        for name, p in s3a_head.named_parameters():
            if p.requires_grad:
                trainable_params.append(p)
                s3a_trainable_param_count += p.numel()
            else:
                s3a_frozen_param_count += p.numel()

    opt = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    resumed_train_steps = 0
    resumed_batches_seen = 0
    resumed_rng_states: Optional[List[dict]] = None
    resumed_s3a_runtime_state: Dict[str, Any] = {}

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.global_seed + rank)

    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    logger.info(f"Dataset: {len(dataset):,} images  |  path: {args.data_path}")

    if args.resume is not None:
        (
            resumed_train_steps,
            resumed_batches_seen,
            resumed_rng_states,
            resume_meta,
        ) = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            ema=ema,
            opt=opt,
            s3a_head=s3a_head if args.s3a else None,
            current_args=args,
            allow_legacy_resume_args=args.allow_legacy_resume_args,
            allow_missing_manifest=args.allow_missing_manifest,
        )
        resumed_s3a_runtime_state = dict(resume_meta.get("s3a_runtime_state", {}))
        resumed_migration_notes = list(resume_meta.get("s3a_migration_notes", []))
        if (
            resumed_rng_states is not None
            and rank < len(resumed_rng_states)
            and resumed_rng_states[rank] is not None
        ):
            restore_local_rng_state(
                resumed_rng_states[rank],
                loader_generator=loader_generator,
            )
        logger.info(
            f"Resumed from checkpoint: {args.resume} "
            f"(train_steps={resumed_train_steps}, batches_seen={resumed_batches_seen}, "
            f"rng_restored={resumed_rng_states is not None}, "
            f"collapse_alarm_windows={int(resumed_s3a_runtime_state.get('collapse_alarm_windows', 0))}, "
            f"dino_starved_windows={int(resumed_s3a_runtime_state.get('dino_starved_windows', 0))}, "
            f"collapse_mitigation_cooldown_windows={int(resumed_s3a_runtime_state.get('collapse_mitigation_cooldown_windows', 0))}, "
            f"collapse_mitigation_trigger_count={int(resumed_s3a_runtime_state.get('collapse_mitigation_trigger_count', 0))})"
        )
        if resumed_migration_notes and rank == 0:
            logger.warning(
                "Applied legacy S3A state migration during resume: "
                + ", ".join(resumed_migration_notes)
            )

    if args.resume is None:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    if args.s3a:
        s3a_head.train()
        dino_model.eval()

    # Track whether the router output layer has been reset after self warmup.
    _router_warmup_reset_done = (resumed_train_steps >= args.s3a_self_warmup_steps)

    train_steps = resumed_train_steps
    batches_seen = resumed_batches_seen
    log_steps = 0
    running_loss = 0.0
    running_loss_diff = 0.0
    running_loss_align = 0.0
    running_phase_weight = 0.0
    running_diff_weight = 0.0
    running_feat = 0.0
    running_attn = 0.0
    running_spatial = 0.0
    running_alpha_dino = 0.0
    running_alpha_self = 0.0
    running_gate_self = 0.0
    running_gate_self_state = 0.0
    running_raw_alpha_dino = 0.0
    running_raw_alpha_self = 0.0
    running_router_entropy = 0.0
    running_router_policy_kl = 0.0
    running_router_policy_gap = 0.0
    running_loss_fused = 0.0
    running_loss_fused_probe = 0.0
    running_loss_dino_only = 0.0
    running_loss_self_only = 0.0
    running_utility_dino = 0.0
    running_utility_self = 0.0
    running_utility_self_active_ema = 0.0
    running_utility_self_inactive_ema = 0.0
    running_utility_self_active_ema_count = 0.0
    running_utility_self_inactive_ema_count = 0.0
    running_probe_count = 0.0
    running_self_probe_count = 0.0
    running_collapse_alarm = 0.0
    running_alpha_dino_min_layer = 0.0
    running_alpha_dino_max_layer = 0.0
    running_alpha_dino_layers = [0.0 for _ in s3a_layer_indices]
    collapse_alarm_windows = int(resumed_s3a_runtime_state.get("collapse_alarm_windows", 0))
    dino_starved_windows = int(resumed_s3a_runtime_state.get("dino_starved_windows", 0))
    collapse_mitigation_cooldown_windows = int(
        resumed_s3a_runtime_state.get("collapse_mitigation_cooldown_windows", 0)
    )
    collapse_mitigation_trigger_count = int(
        resumed_s3a_runtime_state.get("collapse_mitigation_trigger_count", 0)
    )
    gate_off_probe_windows = max(1, math.ceil(args.s3a_gate_patience / max(1, args.s3a_probe_every)))
    gate_reopen_probe_windows = max(
        1, math.ceil(args.s3a_gate_reopen_patience / max(1, args.s3a_probe_every))
    )
    start_time = time()

    logger.info(
        f"Training config:\n"
        f"  epochs             : {args.epochs}\n"
        f"  max_steps          : {args.max_steps}\n"
        f"  global_batch       : {args.global_batch_size}\n"
        f"  global_seed        : {args.global_seed}\n"
        f"  allow_missing_manifest : {args.allow_missing_manifest}\n"
        f"  lr                 : {args.lr}\n"
        f"  weight_decay       : {args.weight_decay}\n"
            f"  log_every          : {args.log_every}\n"
            f"  ckpt_every         : {args.ckpt_every}"
    )
    if args.s3a:
        logger.info(
            f"S3A trainable params: {s3a_trainable_param_count:,}  |  "
            f"S3A frozen params: {s3a_frozen_param_count:,}"
        )
    if rank == 0 and metrics_jsonl_path is not None:
        logger.info(f"Metrics JSONL      : {metrics_jsonl_path}")
    if args.s3a:
        logger.info(
            f"S3A config:\n"
            f"  checkpoint format   : {CHECKPOINT_FORMAT_VERSION}\n"
            f"  metrics schema      : {METRICS_SCHEMA_VERSION}\n"
            f"  phase schedule     : {args.s3a_train_schedule} over {args.s3a_schedule_steps} (warmup={args.s3a_schedule_warmup_steps})\n"
            f"  diff schedule      : {args.s3a_diff_schedule}\n"
            f"  layer indices      : {s3a_layer_indices}\n"
            f"  layer mode         : {args.s3a_layer_weight_mode}\n"
            f"  self warmup steps  : {args.s3a_self_warmup_steps}\n"
            f"  dino alpha floor   : {args.s3a_dino_alpha_floor}\n"
            f"  floor steps        : {args.s3a_dino_alpha_floor_steps}\n"
            f"  gate utility thr   : off={args.s3a_gate_utility_off_threshold}, "
            f"on={args.s3a_gate_utility_on_threshold}\n"
            f"  gate patience(step): off={args.s3a_gate_patience}, "
            f"reopen={args.s3a_gate_reopen_patience}\n"
            f"  gate patience(win) : off={gate_off_probe_windows}, "
            f"reopen={gate_reopen_probe_windows}, "
            f"ema={args.s3a_gate_utility_ema_momentum}\n"
            f"  collapse thresholds: "
            f"alpha<{args.s3a_collapse_alpha_threshold}, "
            f"self>{args.s3a_collapse_self_threshold}, "
            f"u_dino>{args.s3a_collapse_utility_threshold}, "
            f"margin={args.s3a_collapse_margin}, "
            f"windows={args.s3a_collapse_windows}\n"
            f"  collapse mitigation: "
            f"enabled={args.s3a_collapse_auto_mitigate}, "
            f"hold_windows={args.s3a_collapse_mitigate_windows}, "
            f"cooldown_windows={args.s3a_collapse_mitigate_cooldown_windows}\n"
            f"  probe every        : {args.s3a_probe_every}\n"
            f"  utility estimator  : {args.s3a_utility_probe_mode}\n"
            f"  off-probe alpha fl : {args.s3a_gate_reopen_probe_alpha_floor}\n"
            f"  source0 min alpha  : {args.s3a_protect_source0_min_alpha}\n"
            f"  router KL lambda   : {args.s3a_router_policy_kl_lambda}\n"
            f"  unsafe source0 min : {args.s3a_allow_unsafe_zero_source0_floor}\n"
            f"  loss weights       : feat={args.s3a_feat_weight}, "
            f"attn={args.s3a_attn_weight}, spatial={args.s3a_spatial_weight}"
        )
        if rank == 0:
            logger.warning(
                "--s3a-source-gate-threshold is deprecated and ignored; "
                "utility-driven gate uses --s3a-gate-utility-* thresholds."
            )

    if rank == 0 and resolved_args_path is not None:
        resolved_payload = {
            "git_revision": git_revision,
            "created_at": strftime("%Y-%m-%d %H:%M:%S"),
            "resume_from": args.resume,
            "argv": sys.argv,
            "allow_missing_manifest": bool(args.allow_missing_manifest),
            "allow_legacy_resume_args": bool(args.allow_legacy_resume_args),
            "args": _namespace_to_dict(args),
        }
        atomic_write_json(resolved_args_path, resolved_payload)
        logger.info(f"Resolved args saved at {resolved_args_path}")
    if rank == 0 and metrics_jsonl_path is not None:
        contract_row = {
            "record_type": "contract",
            "event_name": "s3a_contract",
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "created_at": strftime("%Y-%m-%d %H:%M:%S"),
            "step": int(train_steps),
            "batches_seen": int(batches_seen),
            "git_revision": git_revision,
            "resume_from": args.resume,
            "log_every": int(args.log_every),
            "s3a": bool(args.s3a),
            "s3a_use_ema_source": bool(args.s3a_use_ema_source),
            "s3a_enable_selective_gate": bool(args.s3a_enable_selective_gate),
            "s3a_self_warmup_steps": int(args.s3a_self_warmup_steps),
            "s3a_dino_alpha_floor": float(args.s3a_dino_alpha_floor),
            "s3a_dino_alpha_floor_steps": int(args.s3a_dino_alpha_floor_steps),
            "s3a_protect_source0_min_alpha": float(args.s3a_protect_source0_min_alpha),
            "s3a_source0_floor_step0": float(source0_min_alpha_at_step(0, args)),
            "s3a_source0_floor_at_resume_step": float(source0_min_alpha_at_step(train_steps, args)),
            "s3a_utility_probe_mode": args.s3a_utility_probe_mode,
            "s3a_probe_every": int(args.s3a_probe_every),
            "s3a_gate_reopen_probe_alpha_floor": float(args.s3a_gate_reopen_probe_alpha_floor),
            "s3a_router_policy_kl_lambda": float(args.s3a_router_policy_kl_lambda),
            "s3a_gate_utility_off_threshold": float(args.s3a_gate_utility_off_threshold),
            "s3a_gate_utility_on_threshold": float(args.s3a_gate_utility_on_threshold),
            "s3a_allow_unsafe_zero_warmup": bool(args.s3a_allow_unsafe_zero_warmup),
            "s3a_allow_unsafe_zero_source0_floor": bool(
                args.s3a_allow_unsafe_zero_source0_floor
            ),
            "s3a_collapse_alpha_threshold": float(args.s3a_collapse_alpha_threshold),
            "s3a_collapse_self_threshold": float(args.s3a_collapse_self_threshold),
            "s3a_collapse_margin": float(args.s3a_collapse_margin),
            "s3a_collapse_auto_mitigate": bool(args.s3a_collapse_auto_mitigate),
            "s3a_collapse_windows": int(args.s3a_collapse_windows),
            "s3a_collapse_mitigate_windows": int(args.s3a_collapse_mitigate_windows),
        }
        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(contract_row, ensure_ascii=True) + "\n")

    done = False
    final_ckpt_saved = False

    steps_per_epoch = len(loader)
    if (
        args.resume is not None
        and args.num_workers > 0
        and (resumed_batches_seen % max(1, steps_per_epoch)) != 0
    ):
        raise ValueError(
            "Exact mid-epoch resume with num_workers>0 is not guaranteed due worker-prefetch RNG state. "
            "Use --num-workers 0, or resume from an epoch-boundary checkpoint."
        )
    start_epoch = batches_seen // max(1, steps_per_epoch)
    skip_batches = batches_seen % max(1, steps_per_epoch)

    for epoch in range(start_epoch, args.epochs):
        if done:
            break

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch} ...")

        for batch_idx, (images, y) in enumerate(loader):
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            current_step = train_steps
            batches_seen += 1

            with torch.no_grad():
                x = vae.encode(images).latent_dist.sample().mul_(0.18215)

            phase_weight = 0.0
            if args.s3a:
                phase_weight = get_train_phase_weight(
                    current_step=current_step,
                    schedule_steps=args.s3a_schedule_steps,
                    schedule=args.s3a_train_schedule,
                    warmup_steps=args.s3a_schedule_warmup_steps,
                )

                # One-time router reset when self_warmup ends: zero the output
                # layer so softmax returns to uniform [0.5, 0.5] instead of
                # staying saturated at [~1, ~0] from the DINO-only warmup period.
                if (
                    args.s3a_use_ema_source
                    and not _router_warmup_reset_done
                    and current_step >= args.s3a_self_warmup_steps
                ):
                    _router_warmup_reset_done = True
                    with torch.no_grad():
                        # head[-1] is the final Linear(hidden, num_sources)
                        router_mod = s3a_head.module.router
                        router_mod.head[-1].weight.zero_()
                        router_mod.head[-1].bias.zero_()
                    if rank == 0:
                        logger.info(
                            f"  [step={current_step}] Router output layer reset "
                            "to zero (post-warmup uniform initialization)."
                        )

            student_tokens: Dict[str, torch.Tensor] = {}
            ema_tokens: Dict[str, torch.Tensor] = {}
            student_handles = []
            ema_handles = []

            if args.s3a and phase_weight > 0:
                student_handles = register_block_hooks(
                    model.module, s3a_layer_indices, student_tokens
                )

            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(
                model,
                x,
                t,
                model_kwargs,
                noise=noise,
            )
            loss_diff = loss_dict["loss"].mean()

            if student_handles:
                remove_hooks(student_handles)

            # Optional self source from EMA model (frozen).
            if (
                args.s3a
                and phase_weight > 0
                and args.s3a_use_ema_source
            ):
                ema_handles = register_block_hooks(ema, s3a_layer_indices, ema_tokens)
                with torch.inference_mode():
                    _ = ema(x_t, t, y)
                remove_hooks(ema_handles)

            loss_align = torch.tensor(0.0, device=device)
            align_stats = make_empty_align_stats(s3a_layer_indices)

            if args.s3a and phase_weight > 0 and len(student_tokens) > 0:
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    dino_tokens = dino_model(x_dino)

                loss_align, align_stats = s3a_head(
                    student_tokens=student_tokens,
                    ema_tokens=ema_tokens,
                    dino_tokens=dino_tokens,
                    t=t,
                    T=T,
                    current_step=current_step,
                    phase_weight=phase_weight,
                    layer_weights=s3a_layer_weights,
                    args=args,
                )

            loss = loss_diff + args.s3a_lambda * loss_align

            loss_finite_flag = torch.tensor(
                1 if torch.isfinite(loss) else 0,
                device=device,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_finite_flag, op=dist.ReduceOp.MIN)

            if loss_finite_flag.item() == 0:
                if rank == 0:
                    logger.warning(
                        f"Non-finite loss at step={train_steps}. Skip optimizer step."
                    )
                    if metrics_jsonl_path is not None:
                        event_row = {
                            "event": "non_finite_loss",
                            "event_name": "non_finite_loss",
                            "record_type": "event",
                            "metrics_schema_version": METRICS_SCHEMA_VERSION,
                            "step": int(train_steps),
                            "batches_seen": int(batches_seen),
                            "git_revision": git_revision,
                            "resume_from": args.resume,
                        }
                        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(event_row, ensure_ascii=True) + "\n")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.s3a_max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    args.s3a_max_grad_norm,
                )
                grad_finite_flag = torch.tensor(
                    1 if torch.isfinite(torch.as_tensor(grad_norm)) else 0,
                    device=device,
                    dtype=torch.int32,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(grad_finite_flag, op=dist.ReduceOp.MIN)

                if grad_finite_flag.item() == 0:
                    if rank == 0:
                        logger.warning(
                            f"Non-finite grad norm at step={train_steps}. Skip optimizer step."
                        )
                        if metrics_jsonl_path is not None:
                            event_row = {
                                "event": "non_finite_grad",
                                "event_name": "non_finite_grad",
                                "record_type": "event",
                                "metrics_schema_version": METRICS_SCHEMA_VERSION,
                                "step": int(train_steps),
                                "batches_seen": int(batches_seen),
                                "git_revision": git_revision,
                                "resume_from": args.resume,
                            }
                            with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(event_row, ensure_ascii=True) + "\n")
                    opt.zero_grad(set_to_none=True)
                    continue
            opt.step()
            update_ema(ema, model.module)
            if (
                args.s3a
                and args.s3a_use_ema_source
                and not args.s3a_trainable_ema_adapters
                and s3a_head.module.ema_adapters is not None
            ):
                update_ema_adapters(
                    s3a_head.module.ema_adapters,
                    s3a_head.module.student_adapters,
                )

            running_loss += loss.item()
            running_loss_diff += loss_diff.item()
            running_loss_align += loss_align.item()
            running_phase_weight += phase_weight
            running_diff_weight += align_stats["diff_w"]
            running_feat += align_stats["feat"]
            running_attn += align_stats["attn"]
            running_spatial += align_stats["spatial"]
            running_alpha_dino += align_stats["alpha_dino"]
            running_alpha_self += align_stats["alpha_self"]
            running_gate_self += align_stats["gate_self"]
            running_gate_self_state += align_stats["gate_self_state"]
            running_raw_alpha_dino += align_stats["raw_alpha_dino"]
            running_raw_alpha_self += align_stats["raw_alpha_self"]
            running_router_entropy += align_stats["router_entropy_norm"]
            running_router_policy_kl += align_stats["router_policy_kl"]
            running_router_policy_gap += align_stats["router_policy_gap"]
            running_loss_fused += align_stats["loss_fused"]
            running_loss_fused_probe += align_stats["loss_fused_probe"] * align_stats["probe_count"]
            running_loss_dino_only += align_stats["loss_dino_only"] * align_stats["probe_count"]
            running_loss_self_only += align_stats["loss_self_only"] * align_stats["self_probe_count"]
            running_utility_dino += align_stats["utility_dino"] * align_stats["probe_count"]
            running_utility_self += align_stats["utility_self"] * align_stats["probe_count"]
            running_utility_self_active_ema += (
                align_stats["utility_self_active_ema"] * align_stats["utility_self_active_ema_count"]
            )
            running_utility_self_inactive_ema += (
                align_stats["utility_self_inactive_ema"] * align_stats["utility_self_inactive_ema_count"]
            )
            running_utility_self_active_ema_count += align_stats["utility_self_active_ema_count"]
            running_utility_self_inactive_ema_count += align_stats["utility_self_inactive_ema_count"]
            running_probe_count += align_stats["probe_count"]
            running_self_probe_count += align_stats["self_probe_count"]
            running_collapse_alarm += align_stats["collapse_alarm"] * align_stats["probe_count"]
            running_alpha_dino_min_layer += align_stats["alpha_dino_min_layer"]
            running_alpha_dino_max_layer += align_stats["alpha_dino_max_layer"]
            if args.s3a and len(running_alpha_dino_layers) == len(align_stats["alpha_dino_layers"]):
                for i in range(len(running_alpha_dino_layers)):
                    running_alpha_dino_layers[i] += align_stats["alpha_dino_layers"][i]

            log_steps += 1
            train_steps += 1

            if train_steps == 1 and rank == 0:
                logger.info(f"  [step=1] images shape : {images.shape}")
                logger.info(f"  [step=1] latent shape : {x.shape}")
                logger.info(f"  [step=1] t (first 8)  : {t[:8].tolist()}")
                logger.info(f"  [step=1] phase_weight : {phase_weight:.4f}")
                if args.s3a:
                    logger.info(f"  [step=1] student tap layers: {sorted(student_tokens.keys())}")
                    logger.info(
                        "  [step=1] align stats: "
                        f"feat={align_stats['feat']:.4f}, "
                        f"attn={align_stats['attn']:.4f}, "
                        f"spatial={align_stats['spatial']:.4f}, "
                        f"alpha_dino={align_stats['alpha_dino']:.4f}, "
                        f"alpha_self={align_stats['alpha_self']:.4f}"
                    )

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                def _reduce_mean(val: float) -> float:
                    tensor = torch.tensor(val / log_steps, device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    return tensor.item() / dist.get_world_size()

                def _reduce_probe_mean(val_sum: float, probe_count_sum: float) -> Tuple[float, float]:
                    pair = torch.tensor([val_sum, probe_count_sum], device=device, dtype=torch.float32)
                    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
                    denom = max(pair[1].item(), 0.0)
                    if denom <= 0.0:
                        return 0.0, 0.0
                    return pair[0].item() / denom, denom

                avg_loss = _reduce_mean(running_loss)
                avg_loss_diff = _reduce_mean(running_loss_diff)
                avg_loss_align = _reduce_mean(running_loss_align)
                avg_phase_w = _reduce_mean(running_phase_weight)
                avg_diff_w = _reduce_mean(running_diff_weight)
                avg_feat = _reduce_mean(running_feat)
                avg_attn = _reduce_mean(running_attn)
                avg_spatial = _reduce_mean(running_spatial)
                avg_alpha_dino = _reduce_mean(running_alpha_dino)
                avg_alpha_self = _reduce_mean(running_alpha_self)
                avg_gate_self = _reduce_mean(running_gate_self)
                avg_gate_self_state = _reduce_mean(running_gate_self_state)
                avg_raw_alpha_dino = _reduce_mean(running_raw_alpha_dino)
                avg_raw_alpha_self = _reduce_mean(running_raw_alpha_self)
                avg_router_entropy = _reduce_mean(running_router_entropy)
                avg_router_policy_kl = _reduce_mean(running_router_policy_kl)
                avg_router_policy_gap = _reduce_mean(running_router_policy_gap)
                avg_loss_fused = _reduce_mean(running_loss_fused)
                avg_loss_fused_probe, _ = _reduce_probe_mean(
                    running_loss_fused_probe, running_probe_count
                )
                avg_loss_dino_only, global_probe_count = _reduce_probe_mean(
                    running_loss_dino_only, running_probe_count
                )
                avg_loss_self_only, global_self_probe_count = _reduce_probe_mean(
                    running_loss_self_only, running_self_probe_count
                )
                avg_utility_dino, _ = _reduce_probe_mean(
                    running_utility_dino, running_probe_count
                )
                avg_utility_self, _ = _reduce_probe_mean(
                    running_utility_self, running_probe_count
                )
                avg_utility_self_active_ema, global_active_ema_count = _reduce_probe_mean(
                    running_utility_self_active_ema, running_utility_self_active_ema_count
                )
                avg_utility_self_inactive_ema, global_inactive_ema_count = _reduce_probe_mean(
                    running_utility_self_inactive_ema, running_utility_self_inactive_ema_count
                )
                avg_utility_self_ema = avg_utility_self_active_ema
                avg_collapse_alarm, _ = _reduce_probe_mean(
                    running_collapse_alarm, running_probe_count
                )
                avg_alpha_dino_min_layer = _reduce_mean(running_alpha_dino_min_layer)
                avg_alpha_dino_max_layer = _reduce_mean(running_alpha_dino_max_layer)
                avg_alpha_dino_layers: List[float] = []
                for i in range(len(running_alpha_dino_layers)):
                    layer_avg_tensor = torch.tensor(
                        running_alpha_dino_layers[i] / log_steps, device=device
                    )
                    dist.all_reduce(layer_avg_tensor, op=dist.ReduceOp.SUM)
                    avg_alpha_dino_layers.append(layer_avg_tensor.item() / dist.get_world_size())
                source0_floor_active = (
                    source0_min_alpha_at_step(train_steps, args)
                    if args.s3a_use_ema_source
                    else 0.0
                )
                alpha_dino_above_floor = max(0.0, avg_alpha_dino - source0_floor_active)
                source0_excess_epsilon = 1e-3
                # Per-layer DINO starvation: trigger if ANY layer is starved,
                # not just the cross-layer average.
                any_layer_dino_starved = any(
                    max(0.0, lv - source0_floor_active) <= source0_excess_epsilon
                    for lv in avg_alpha_dino_layers
                ) if avg_alpha_dino_layers else False
                dino_starved = (
                    1.0
                    if (
                        args.s3a_use_ema_source
                        and train_steps >= args.s3a_self_warmup_steps
                        and avg_alpha_self > args.s3a_collapse_self_threshold
                        and (alpha_dino_above_floor <= source0_excess_epsilon
                             or any_layer_dino_starved)
                    )
                    else 0.0
                )
                dual_source_alive = (
                    1.0
                    if (
                        args.s3a_use_ema_source
                        and avg_alpha_self > 1e-3
                        and alpha_dino_above_floor > 0.01
                    )
                    else 0.0
                )
                dual_synergy_margin = 0.0
                dual_synergy_supported = 0.0
                if global_probe_count > 0 and global_self_probe_count > 0:
                    dual_synergy_margin = (
                        min(avg_loss_dino_only, avg_loss_self_only) - avg_loss_fused_probe
                    )
                    dual_synergy_supported = 1.0 if dual_synergy_margin > 0 else 0.0

                collapse_window_triggered = False
                if global_probe_count > 0:
                    collapse_window_triggered = (
                        args.s3a_use_ema_source
                        and global_self_probe_count > 0
                        and train_steps >= args.s3a_self_warmup_steps
                        and (alpha_dino_above_floor <= source0_excess_epsilon
                             or any_layer_dino_starved)
                        and avg_alpha_self > args.s3a_collapse_self_threshold
                        and avg_utility_dino > args.s3a_collapse_utility_threshold
                        and avg_loss_fused_probe + args.s3a_collapse_margin < avg_loss_self_only
                    )
                    collapse_alarm_windows = (
                        collapse_alarm_windows + 1 if collapse_window_triggered else 0
                    )
                    dino_starved_windows = (
                        dino_starved_windows + 1 if dino_starved > 0 else 0
                    )
                else:
                    collapse_alarm_windows = 0
                    dino_starved_windows = 0
                collapse_alarm = (
                    1.0 if collapse_alarm_windows >= args.s3a_collapse_windows else 0.0
                )
                dino_starved_alarm = (
                    1.0 if dino_starved_windows >= args.s3a_collapse_windows else 0.0
                )
                mitigation_active_windows = 0
                if args.s3a and args.s3a_use_ema_source:
                    mitigation_active_windows = int(
                        s3a_head.module.self_mitigation_windows_remaining.item()
                    )
                mitigation_triggered = 0.0
                if (
                    args.s3a
                    and args.s3a_use_ema_source
                    and args.s3a_enable_selective_gate
                    and args.s3a_collapse_auto_mitigate
                    and (collapse_alarm > 0 or dino_starved_alarm > 0)
                    and mitigation_active_windows <= 0
                    and collapse_mitigation_cooldown_windows <= 0
                ):
                    s3a_head.module.set_self_mitigation_windows(
                        args.s3a_collapse_mitigate_windows
                    )
                    mitigation_active_windows = int(
                        s3a_head.module.self_mitigation_windows_remaining.item()
                    )
                    collapse_mitigation_cooldown_windows = (
                        args.s3a_collapse_mitigate_cooldown_windows
                    )
                    collapse_mitigation_trigger_count += 1
                    mitigation_triggered = 1.0
                if collapse_alarm > 0 and rank == 0:
                    logger.warning(
                        "S3A collapse alarm triggered: "
                        f"alpha_dino={avg_alpha_dino:.4f}, "
                        f"alpha_self={avg_alpha_self:.4f}, "
                        f"loss_fused={avg_loss_fused:.6f}, "
                        f"loss_fused_probe={avg_loss_fused_probe:.6f}, "
                        f"loss_self_only={avg_loss_self_only:.6f}, "
                        f"loss_dino_only={avg_loss_dino_only:.6f}, "
                        f"utility_dino={avg_utility_dino:.6f}, "
                        f"utility_self={avg_utility_self:.6f}, "
                        f"utility_self_active_ema={avg_utility_self_active_ema:.6f}, "
                        f"utility_self_inactive_ema={avg_utility_self_inactive_ema:.6f}, "
                        f"utility_self_active_ema_n={int(global_active_ema_count)}, "
                        f"utility_self_inactive_ema_n={int(global_inactive_ema_count)}, "
                        f"probes={int(global_probe_count)}, "
                        f"self_probes={int(global_self_probe_count)}, "
                        f"windows={collapse_alarm_windows}, "
                        f"dino_starved_windows={dino_starved_windows}, "
                        f"mitigation_active={mitigation_active_windows}, "
                        f"mitigation_cooldown={collapse_mitigation_cooldown_windows}, "
                        f"mitigation_triggered={int(mitigation_triggered)}"
                    )

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Loss={avg_loss:.4f}  "
                    f"Loss_diff={avg_loss_diff:.4f}  "
                    f"Loss_align={avg_loss_align:.4f}  "
                    f"PhaseW={avg_phase_w:.4f}  "
                    f"DiffW={avg_diff_w:.4f}  "
                    f"Feat={avg_feat:.4f}  "
                    f"Attn={avg_attn:.4f}  "
                    f"Spatial={avg_spatial:.4f}  "
                    f"a_dino={avg_alpha_dino:.3f}  "
                    f"a_dino_above_floor={alpha_dino_above_floor:.3f}  "
                    f"a_self={avg_alpha_self:.3f}  "
                    f"raw_dino={avg_raw_alpha_dino:.3f}  "
                    f"raw_self={avg_raw_alpha_self:.3f}  "
                    f"H={avg_router_entropy:.3f}  "
                    f"rKL={avg_router_policy_kl:.4f}  "
                    f"rGap={avg_router_policy_gap:.3f}  "
                    f"mask_self={avg_gate_self:.3f}  "
                    f"gate_self_state={avg_gate_self_state:.3f}  "
                    f"Lfused={avg_loss_fused:.4f}  "
                    f"LfusedProbe={avg_loss_fused_probe:.4f}  "
                    f"Ldino={avg_loss_dino_only:.4f}  "
                    f"Lself={avg_loss_self_only:.4f}  "
                    f"U_dino={avg_utility_dino:.4f}  "
                    f"U_self={avg_utility_self:.4f}  "
                    f"UselfEMA={avg_utility_self_ema:.4f}  "
                    f"UselfEMA_on={avg_utility_self_active_ema:.4f}  "
                    f"UselfEMA_off={avg_utility_self_inactive_ema:.4f}  "
                    f"ProbeN={int(global_probe_count)}  "
                    f"SelfProbeN={int(global_self_probe_count)}  "
                    f"dino_starved={int(dino_starved)}  "
                    f"dino_starved_alarm={int(dino_starved_alarm)}  "
                    f"dino_starved_windows={int(dino_starved_windows)}  "
                    f"dual_alive={int(dual_source_alive)}  "
                    f"synergy_margin={dual_synergy_margin:.4f}  "
                    f"synergy={int(dual_synergy_supported)}  "
                    f"alarm={collapse_alarm:.0f}  "
                    f"mitigate={mitigation_triggered:.0f}  "
                    f"mitigateW={mitigation_active_windows}  "
                    f"eff_λ={args.s3a_lambda * avg_phase_w:.6f}  "
                    f"Steps/s={steps_per_sec:.2f}"
                )
                if rank == 0 and metrics_jsonl_path is not None:
                    metric_row = {
                        "record_type": "metric",
                        "metrics_schema_version": METRICS_SCHEMA_VERSION,
                        "utility_ema_semantics": "utility_self_ema_alias_of_active_ema",
                        "utility_probe_mode": args.s3a_utility_probe_mode,
                        "step": int(train_steps),
                        "batches_seen": int(batches_seen),
                        "git_revision": git_revision,
                        "resume_from": args.resume,
                        "global_seed": int(args.global_seed),
                        "loss": float(avg_loss),
                        "loss_diff": float(avg_loss_diff),
                        "loss_align": float(avg_loss_align),
                        "phase_weight": float(avg_phase_w),
                        "diff_weight": float(avg_diff_w),
                        "feat": float(avg_feat),
                        "attn": float(avg_attn),
                        "spatial": float(avg_spatial),
                        "alpha_dino": float(avg_alpha_dino),
                        "source0_floor_active": float(source0_floor_active),
                        "alpha_dino_above_floor": float(alpha_dino_above_floor),
                        "dino_starved": float(dino_starved),
                        "dino_starved_alarm": float(dino_starved_alarm),
                        "dino_starved_windows": int(dino_starved_windows),
                        "dual_source_alive": float(dual_source_alive),
                        "dual_synergy_margin": float(dual_synergy_margin),
                        "dual_synergy_supported": float(dual_synergy_supported),
                        "alpha_self": float(avg_alpha_self),
                        "raw_alpha_dino": float(avg_raw_alpha_dino),
                        "raw_alpha_self": float(avg_raw_alpha_self),
                        "router_entropy_norm": float(avg_router_entropy),
                        "router_policy_kl": float(avg_router_policy_kl),
                        "router_policy_gap": float(avg_router_policy_gap),
                        "gate_self": float(avg_gate_self),
                        "self_source_mask_mean": float(avg_gate_self),
                        "self_gate_state_mean": float(avg_gate_self_state),
                        "loss_fused": float(avg_loss_fused),
                        "loss_fused_probe": float(avg_loss_fused_probe),
                        "loss_dino_only": float(avg_loss_dino_only),
                        "loss_self_only": float(avg_loss_self_only),
                        "utility_dino": float(avg_utility_dino),
                        "utility_self": float(avg_utility_self),
                        "utility_self_ema": float(avg_utility_self_ema),
                        "utility_self_active_ema": float(avg_utility_self_active_ema),
                        "utility_self_inactive_ema": float(avg_utility_self_inactive_ema),
                        "utility_self_active_ema_count": int(global_active_ema_count),
                        "utility_self_inactive_ema_count": int(global_inactive_ema_count),
                        "probe_count": int(global_probe_count),
                        "self_probe_count": int(global_self_probe_count),
                        "collapse_window_score": float(avg_collapse_alarm),
                        "collapse_alarm": float(collapse_alarm),
                        "collapse_alarm_windows": int(collapse_alarm_windows),
                        "collapse_mitigation_active_windows": int(mitigation_active_windows),
                        "collapse_mitigation_cooldown_windows": int(
                            collapse_mitigation_cooldown_windows
                        ),
                        "collapse_mitigation_triggered": float(mitigation_triggered),
                        "collapse_mitigation_trigger_count": int(
                            collapse_mitigation_trigger_count
                        ),
                        "alpha_dino_min_layer": float(avg_alpha_dino_min_layer),
                        "alpha_dino_max_layer": float(avg_alpha_dino_max_layer),
                        "effective_lambda": float(args.s3a_lambda * avg_phase_w),
                        "steps_per_sec": float(steps_per_sec),
                    }
                    for layer_pos, layer_idx in enumerate(s3a_layer_indices):
                        key = f"alpha_dino_l{int(layer_idx)}"
                        metric_row[key] = float(avg_alpha_dino_layers[layer_pos])
                    with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metric_row, ensure_ascii=True) + "\n")

                if args.s3a and args.s3a_use_ema_source:
                    if mitigation_triggered <= 0:
                        if collapse_mitigation_cooldown_windows > 0:
                            collapse_mitigation_cooldown_windows -= 1
                        s3a_head.module.tick_self_mitigation_window()

                running_loss = 0.0
                running_loss_diff = 0.0
                running_loss_align = 0.0
                running_phase_weight = 0.0
                running_diff_weight = 0.0
                running_feat = 0.0
                running_attn = 0.0
                running_spatial = 0.0
                running_alpha_dino = 0.0
                running_alpha_self = 0.0
                running_gate_self = 0.0
                running_gate_self_state = 0.0
                running_raw_alpha_dino = 0.0
                running_raw_alpha_self = 0.0
                running_router_entropy = 0.0
                running_router_policy_kl = 0.0
                running_router_policy_gap = 0.0
                running_loss_fused = 0.0
                running_loss_fused_probe = 0.0
                running_loss_dino_only = 0.0
                running_loss_self_only = 0.0
                running_utility_dino = 0.0
                running_utility_self = 0.0
                running_utility_self_active_ema = 0.0
                running_utility_self_inactive_ema = 0.0
                running_utility_self_active_ema_count = 0.0
                running_utility_self_inactive_ema_count = 0.0
                running_probe_count = 0.0
                running_self_probe_count = 0.0
                running_collapse_alarm = 0.0
                running_alpha_dino_min_layer = 0.0
                running_alpha_dino_max_layer = 0.0
                running_alpha_dino_layers = [0.0 for _ in s3a_layer_indices]

                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                rng_states = gather_rng_states(loader_generator=loader_generator)
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir,
                        train_steps,
                        batches_seen,
                        model,
                        ema,
                        opt,
                        args,
                        s3a_head if args.s3a else None,
                        rng_states=rng_states,
                        s3a_runtime_state={
                            "collapse_alarm_windows": int(collapse_alarm_windows),
                            "dino_starved_windows": int(dino_starved_windows),
                            "collapse_mitigation_cooldown_windows": int(
                                collapse_mitigation_cooldown_windows
                            ),
                            "collapse_mitigation_trigger_count": int(
                                collapse_mitigation_trigger_count
                            ),
                        },
                    )
                    logger.info(
                        f"Saved checkpoint to {ckpt_path} "
                        f"(manifest: {_checkpoint_manifest_path(ckpt_path)})"
                    )
                dist.barrier()

            if args.max_steps is not None and train_steps >= args.max_steps:
                rng_states = gather_rng_states(loader_generator=loader_generator)
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir,
                        train_steps,
                        batches_seen,
                        model,
                        ema,
                        opt,
                        args,
                        s3a_head if args.s3a else None,
                        rng_states=rng_states,
                        s3a_runtime_state={
                            "collapse_alarm_windows": int(collapse_alarm_windows),
                            "dino_starved_windows": int(dino_starved_windows),
                            "collapse_mitigation_cooldown_windows": int(
                                collapse_mitigation_cooldown_windows
                            ),
                            "collapse_mitigation_trigger_count": int(
                                collapse_mitigation_trigger_count
                            ),
                        },
                    )
                    logger.info(
                        f"Reached max_steps={args.max_steps}. "
                        f"Saved final checkpoint to {ckpt_path} "
                        f"(manifest: {_checkpoint_manifest_path(ckpt_path)})"
                    )
                    final_ckpt_saved = True
                dist.barrier()
                done = True
                break

    rng_states = gather_rng_states(loader_generator=loader_generator)
    if rank == 0 and train_steps > 0 and not final_ckpt_saved:
        ckpt_path = save_checkpoint(
            checkpoint_dir,
            train_steps,
            batches_seen,
            model,
            ema,
            opt,
            args,
            s3a_head if args.s3a else None,
            rng_states=rng_states,
            s3a_runtime_state={
                "collapse_alarm_windows": int(collapse_alarm_windows),
                "dino_starved_windows": int(dino_starved_windows),
                "collapse_mitigation_cooldown_windows": int(
                    collapse_mitigation_cooldown_windows
                ),
                "collapse_mitigation_trigger_count": int(
                    collapse_mitigation_trigger_count
                ),
            },
        )
        logger.info(
            f"Saved final checkpoint to {ckpt_path} "
            f"(manifest: {_checkpoint_manifest_path(ckpt_path)})"
        )
    dist.barrier()

    model.eval()
    if args.s3a:
        s3a_head.eval()

    logger.info("Done!")
    cleanup()


#################################################################################
#                                  Entry Point                                  #
#################################################################################


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DiT + S3A training with DINOv2 teacher.\n"
            "S3A: multi-layer multi-source alignment branch with dynamic routing.\n"
            "Supports DINOv2 ViT-B/14, ViT-L/14, and ViT-G/14 via --dinov2-model-variant."
        )
    )

    parser.add_argument("--data-path", type=str, required=True, help="ImageFolder training data path.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/"
            "yuantianshuo/2026/results"
        ),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2",
        help="DiT model variant.",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-model-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow resuming from checkpoints without .sha256.json sidecar.",
    )
    parser.add_argument(
        "--allow-legacy-resume-args",
        action="store_true",
        help=(
            "Allow resume when checkpoint args miss critical contract keys. "
            "Unsafe; default is fail-closed."
        ),
    )

    parser.add_argument(
        "--dinov2-repo-dir",
        type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/"
            "yuantianshuo/tmp/dinov2"
        ),
        help="DINOv2 source directory containing hubconf.py.",
    )
    parser.add_argument(
        "--dinov2-weight-path",
        type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/"
            "tmp/dinov2/dinov2_weights/dinov2_vitb14_pretrain.pth"
        ),
        help="DINOv2 checkpoint path (.pth).",
    )
    parser.add_argument(
        "--dinov2-model-variant",
        type=str,
        choices=list(DINOV2_VARIANTS.keys()),
        default="vitb14",
        help=(
            "DINOv2 model variant. vitb14 (768-d, default), "
            "vitl14 (1024-d), vitg14 (1536-d)."
        ),
    )

    parser.add_argument("--s3a", action="store_true", help="Enable S3A alignment branch.")
    parser.add_argument("--s3a-lambda", type=float, default=0.5)

    parser.add_argument(
        "--s3a-layer-indices",
        type=str,
        default="auto",
        help="Layer spec: auto | quarter,mid,three_quarter,last | comma-separated indices.",
    )
    parser.add_argument(
        "--s3a-layer-weight-mode",
        type=str,
        choices=["uniform", "deep_focus", "mid_focus", "custom"],
        default="mid_focus",
    )
    parser.add_argument(
        "--s3a-layer-weights",
        type=str,
        default=None,
        help="Custom per-layer weights CSV, used when --s3a-layer-weight-mode=custom.",
    )

    parser.add_argument(
        "--s3a-adapter-hidden-dim",
        type=int,
        default=2048,
        help=(
            "Hidden dimension for SpatiallyFaithfulAdapter. "
            "Default 2048 matches REPA projector width. "
            "Pass 0 to use in_dim (legacy behaviour, needed for resuming old checkpoints)."
        ),
    )
    parser.add_argument("--s3a-router-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--s3a-router-detach-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to detach student tokens before routing.",
    )

    parser.add_argument(
        "--s3a-train-schedule",
        type=str,
        choices=["constant", "linear_decay", "cosine_decay", "cutoff", "piecewise_linear", "piecewise_cosine"],
        default="piecewise_cosine",
    )
    parser.add_argument("--s3a-schedule-steps", type=int, default=300_000)
    parser.add_argument(
        "--s3a-schedule-warmup-steps",
        type=int,
        default=100_000,
        help=(
            "Constant-strength phase before decay begins. "
            "Only used when --s3a-train-schedule=piecewise_linear."
        ),
    )
    parser.add_argument(
        "--s3a-diff-schedule",
        type=str,
        choices=["cosine", "linear_high", "linear_low", "uniform"],
        default="cosine",
    )

    parser.add_argument("--s3a-feat-weight", type=float, default=1.0)
    parser.add_argument("--s3a-attn-weight", type=float, default=0.5)
    parser.add_argument("--s3a-spatial-weight", type=float, default=0.5)
    parser.add_argument(
        "--s3a-attn-max-tokens",
        type=int,
        default=128,
        help="Max tokens used in affinity loss (<=0 disables subsampling).",
    )
    parser.add_argument(
        "--s3a-max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. <=0 disables clipping.",
    )

    parser.add_argument(
        "--s3a-use-ema-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMA self source in addition to DINO source.",
    )
    parser.add_argument(
        "--s3a-trainable-ema-adapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether EMA source adapters are trainable. "
            "Default off to prevent self-target shortcut collapse."
        ),
    )

    parser.add_argument(
        "--s3a-enable-selective-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable source-layer selective stop gate.",
    )
    # Legacy threshold retained for backward CLI compatibility; gate decisions
    # now use utility thresholds below.
    parser.add_argument("--s3a-source-gate-threshold", type=float, default=0.05)
    parser.add_argument(
        "--s3a-gate-patience",
        type=int,
        default=500,
        help=(
            "Step-based patience before gating off a source. "
            "Converted internally to probe windows via ceil(patience/probe_every)."
        ),
    )
    parser.add_argument(
        "--s3a-gate-reopen-patience",
        type=int,
        default=200,
        help=(
            "Step-based patience before reopening a gated source. "
            "Converted internally to probe windows via ceil(patience/probe_every)."
        ),
    )
    parser.add_argument(
        "--s3a-gate-utility-off-threshold",
        type=float,
        default=0.002,
        help="Utility EMA below this threshold counts as bad for gate-off accounting.",
    )
    parser.add_argument(
        "--s3a-gate-utility-on-threshold",
        type=float,
        default=0.005,
        help="Inactive-side utility EMA above this threshold counts as recovery for gate reopen.",
    )
    parser.add_argument(
        "--s3a-gate-utility-ema-momentum",
        type=float,
        default=0.9,
        help="EMA momentum for utility-driven gate updates.",
    )
    parser.add_argument("--s3a-self-warmup-steps", type=int, default=5000)
    parser.add_argument(
        "--s3a-allow-unsafe-zero-warmup",
        action="store_true",
        help=(
            "Allow dual-source training with self warmup <= 0. "
            "Unsafe by default and blocked unless explicitly enabled."
        ),
    )
    parser.add_argument(
        "--s3a-allow-unsafe-zero-source0-floor",
        action="store_true",
        help=(
            "Allow dual-source training with --s3a-protect-source0-min-alpha <= 0. "
            "Unsafe for collaboration objective and blocked by default."
        ),
    )
    parser.add_argument(
        "--s3a-dino-alpha-floor",
        type=float,
        default=0.1,
        help="Early-stage lower bound for DINO alpha in dual-source mode (0 disables).",
    )
    parser.add_argument(
        "--s3a-dino-alpha-floor-steps",
        type=int,
        default=8000,
        help="Number of steps to apply DINO alpha floor with linear decay to 0.",
    )
    parser.add_argument(
        "--s3a-protect-source0-min-alpha",
        type=float,
        default=0.05,
        help=(
            "Persistent minimum alpha for source0 (DINO). "
            "Applied in both train and probe alpha builders; 0 disables."
        ),
    )
    parser.add_argument(
        "--s3a-router-policy-kl-lambda",
        type=float,
        default=0.1,
        help=(
            "Weight for KL(raw_router_alpha || deployed_alpha_policy). "
            "Keeps router policy aligned with floor/gate-corrected policy to avoid raw collapse."
        ),
    )
    parser.add_argument(
        "--s3a-collapse-alpha-threshold",
        type=float,
        default=0.05,
        help=(
            "Diagnostic threshold for absolute alpha_dino collapse score. "
            "Operational mitigation uses floor-relative alpha_dino_above_floor."
        ),
    )
    parser.add_argument(
        "--s3a-collapse-self-threshold",
        type=float,
        default=0.90,
        help="Collapse alarm threshold for alpha_self.",
    )
    parser.add_argument(
        "--s3a-collapse-margin",
        type=float,
        default=0.01,
        help="Collapse alarm margin used with loss_fused_probe + margin < loss_self_only.",
    )
    parser.add_argument(
        "--s3a-collapse-utility-threshold",
        type=float,
        default=0.0,
        help="Collapse alarm requires utility_dino above this threshold.",
    )
    parser.add_argument(
        "--s3a-collapse-windows",
        type=int,
        default=3,
        help="Consecutive log windows required before emitting collapse alarm.",
    )
    parser.add_argument(
        "--s3a-collapse-auto-mitigate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic temporary self-source shutdown when collapse alarm persists.",
    )
    parser.add_argument(
        "--s3a-collapse-mitigate-windows",
        type=int,
        default=3,
        help="Number of log windows to force self source off after a collapse alarm trigger.",
    )
    parser.add_argument(
        "--s3a-collapse-mitigate-cooldown-windows",
        type=int,
        default=6,
        help="Cooldown log windows before allowing another auto-mitigation trigger.",
    )
    parser.add_argument(
        "--s3a-probe-every",
        type=int,
        default=10,
        help=(
            "Compute source-only diagnostic probes every N steps. "
            "1 means probe every step."
        ),
    )
    parser.add_argument(
        "--s3a-gate-reopen-probe-alpha-floor",
        type=float,
        default=0.05,
        help=(
            "Minimum alpha assigned to a currently gated-off source during "
            "add-one counterfactual probe for reopen utility estimation. "
            "Only used when --s3a-utility-probe-mode=policy_loo; 0 disables."
        ),
    )
    parser.add_argument(
        "--s3a-utility-probe-mode",
        type=str,
        choices=["policy_loo", "uniform", "raw_alpha"],
        default="policy_loo",
        help=(
            "Utility estimator mode. "
            "policy_loo uses policy-consistent leave-one-out/add-one utility "
            "under the same alpha policy as training."
        ),
    )

    return parser


def validate_args(args):
    # Normalize: 0 → None for adapter hidden dim (0 means "use in_dim").
    if args.s3a_adapter_hidden_dim is not None and args.s3a_adapter_hidden_dim <= 0:
        args.s3a_adapter_hidden_dim = None

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.ckpt_every <= 0:
        raise ValueError("--ckpt-every must be > 0")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0 when provided")
    if args.resume is not None and not os.path.isfile(args.resume):
        raise FileNotFoundError(f"--resume not found: {args.resume}")

    if args.vae_model_dir is not None and not os.path.isdir(args.vae_model_dir):
        raise FileNotFoundError(f"--vae-model-dir not found: {args.vae_model_dir}")

    if args.s3a:
        if not os.path.isdir(args.dinov2_repo_dir):
            raise FileNotFoundError(f"--dinov2-repo-dir not found: {args.dinov2_repo_dir}")
        if not os.path.isfile(args.dinov2_weight_path):
            raise FileNotFoundError(f"--dinov2-weight-path not found: {args.dinov2_weight_path}")
        if args.s3a_gate_patience <= 0:
            raise ValueError("--s3a-gate-patience must be > 0")
        if args.s3a_gate_reopen_patience <= 0:
            raise ValueError("--s3a-gate-reopen-patience must be > 0")
        if args.s3a_gate_utility_on_threshold < args.s3a_gate_utility_off_threshold:
            raise ValueError(
                "--s3a-gate-utility-on-threshold must be >= --s3a-gate-utility-off-threshold"
            )
        if not (0.0 <= args.s3a_gate_utility_ema_momentum < 1.0):
            raise ValueError("--s3a-gate-utility-ema-momentum must be in [0, 1)")
        if args.s3a_self_warmup_steps < 0:
            raise ValueError("--s3a-self-warmup-steps must be >= 0")
        if (
            args.s3a_use_ema_source
            and args.s3a_self_warmup_steps <= 0
            and not args.s3a_allow_unsafe_zero_warmup
        ):
            raise ValueError(
                "Unsafe S3A config rejected: dual-source requires "
                "--s3a-self-warmup-steps > 0. "
                "Override explicitly with --s3a-allow-unsafe-zero-warmup."
            )
        if args.s3a_lambda < 0:
            raise ValueError("--s3a-lambda must be >= 0")
        if args.s3a_schedule_warmup_steps < 0:
            raise ValueError("--s3a-schedule-warmup-steps must be >= 0")
        if (
            args.s3a_train_schedule in ("piecewise_linear", "piecewise_cosine")
            and args.s3a_schedule_warmup_steps >= args.s3a_schedule_steps
        ):
            raise ValueError(
                "--s3a-schedule-warmup-steps must be < --s3a-schedule-steps "
                "when using piecewise_linear or piecewise_cosine schedule."
            )
        if args.s3a_feat_weight < 0 or args.s3a_attn_weight < 0 or args.s3a_spatial_weight < 0:
            raise ValueError("S3A loss weights must be >= 0")
        if not (0.0 <= args.s3a_dino_alpha_floor < 1.0):
            raise ValueError("--s3a-dino-alpha-floor must be in [0, 1)")
        if args.s3a_dino_alpha_floor_steps < 0:
            raise ValueError("--s3a-dino-alpha-floor-steps must be >= 0")
        if not (0.0 <= args.s3a_protect_source0_min_alpha < 1.0):
            raise ValueError("--s3a-protect-source0-min-alpha must be in [0, 1)")
        if args.s3a_router_policy_kl_lambda < 0:
            raise ValueError("--s3a-router-policy-kl-lambda must be >= 0")
        if (
            args.s3a_use_ema_source
            and args.s3a_protect_source0_min_alpha <= 0
            and not args.s3a_allow_unsafe_zero_source0_floor
        ):
            raise ValueError(
                "Unsafe S3A config rejected: dual-source collaboration contract requires "
                "--s3a-protect-source0-min-alpha > 0. "
                "Override explicitly with --s3a-allow-unsafe-zero-source0-floor."
            )
        if args.s3a_collapse_windows <= 0:
            raise ValueError("--s3a-collapse-windows must be > 0")
        if args.s3a_collapse_mitigate_windows <= 0:
            raise ValueError("--s3a-collapse-mitigate-windows must be > 0")
        if args.s3a_collapse_mitigate_cooldown_windows < 0:
            raise ValueError("--s3a-collapse-mitigate-cooldown-windows must be >= 0")
        if args.s3a_collapse_utility_threshold < 0:
            raise ValueError("--s3a-collapse-utility-threshold must be >= 0")
        if args.s3a_probe_every <= 0:
            raise ValueError("--s3a-probe-every must be > 0")
        if args.s3a_probe_every > args.log_every:
            raise ValueError(
                "--s3a-probe-every must be <= --log-every to keep collapse "
                "monitoring windows semantically consistent."
            )
        if not (0.0 <= args.s3a_gate_reopen_probe_alpha_floor < 1.0):
            raise ValueError("--s3a-gate-reopen-probe-alpha-floor must be in [0, 1)")
        if args.s3a_enable_selective_gate and args.s3a_utility_probe_mode != "policy_loo":
            raise ValueError(
                "--s3a-enable-selective-gate requires --s3a-utility-probe-mode=policy_loo"
            )
        if (
            args.s3a_gate_reopen_probe_alpha_floor > 0
            and args.s3a_utility_probe_mode != "policy_loo"
        ):
            raise ValueError(
                "--s3a-gate-reopen-probe-alpha-floor requires "
                "--s3a-utility-probe-mode=policy_loo"
            )
        if args.s3a_gate_reopen_probe_alpha_floor > 0 and not args.s3a_use_ema_source:
            raise ValueError(
                "--s3a-gate-reopen-probe-alpha-floor requires --s3a-use-ema-source"
            )
        max_source0_floor = args.s3a_protect_source0_min_alpha
        if args.s3a_dino_alpha_floor > 0 and args.s3a_dino_alpha_floor_steps > 0:
            max_source0_floor = max(max_source0_floor, args.s3a_dino_alpha_floor)
        if (
            args.s3a_use_ema_source
            and args.s3a_gate_reopen_probe_alpha_floor > 0
            and max_source0_floor + args.s3a_gate_reopen_probe_alpha_floor > 1.0
        ):
            raise ValueError(
                "Inconsistent S3A alpha floors: "
                "max(--s3a-dino-alpha-floor, --s3a-protect-source0-min-alpha) + "
                "--s3a-gate-reopen-probe-alpha-floor must be <= 1.0"
            )
def cli_main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if not args.s3a:
        raise ValueError("This script is dedicated to S3A. Please pass --s3a.")

    main(args)


if __name__ == "__main__":
    cli_main()
