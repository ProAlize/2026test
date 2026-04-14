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
) -> float:
    """Phase-axis weight used in 3D curriculum."""
    if schedule == "constant":
        return 1.0

    progress = min(max(current_step / max(1, schedule_steps), 0.0), 1.0)

    if schedule == "linear_decay":
        return 1.0 - progress
    if schedule == "cosine_decay":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "cutoff":
        return 1.0 if current_step < schedule_steps else 0.0

    raise ValueError(
        f"Unknown train schedule: {schedule!r}. "
        "Choose from: constant, linear_decay, cosine_decay, cutoff."
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
    """
    pred_energy = pred.norm(dim=-1)
    target_energy = target.norm(dim=-1)

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


#################################################################################
#                    DINOv2 Teacher（local .pth + torch.hub）                  #
#################################################################################


class LocalDINOv2Teacher(nn.Module):
    EXPECTED_PATCH_TOKENS = 256

    def __init__(self, dinov2_repo_dir: str, weight_path: str):
        super().__init__()

        if dinov2_repo_dir not in sys.path:
            sys.path.insert(0, dinov2_repo_dir)

        self.model = torch.hub.load(
            dinov2_repo_dir,
            "dinov2_vitb14",
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
            "source_utility_ema",
            torch.zeros(len(self.layer_indices), self.num_sources, dtype=torch.float32),
        )

    @torch.no_grad()
    def update_gate_state(
        self,
        layer_slot: int,
        utility_mean: torch.Tensor,
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
            if protect_source0 and src_idx == 0:
                self.source_gate_mask[layer_slot, src_idx] = 1.0
                self.source_inactive_steps[layer_slot, src_idx] = 0
                self.source_recover_steps[layer_slot, src_idx] = 0
                continue

            if source_ready[src_idx] <= 0:
                self.source_gate_mask[layer_slot, src_idx] = 0.0
                self.source_inactive_steps[layer_slot, src_idx] = 0
                self.source_recover_steps[layer_slot, src_idx] = 0
                self.source_utility_ema[layer_slot, src_idx] = 0.0
                continue

            prev_ema = self.source_utility_ema[layer_slot, src_idx]
            ema = (
                utility_ema_momentum * prev_ema
                + (1.0 - utility_ema_momentum) * utility_mean[src_idx]
            )
            self.source_utility_ema[layer_slot, src_idx] = ema

            if ema < utility_off_threshold:
                self.source_inactive_steps[layer_slot, src_idx] += 1
            else:
                self.source_inactive_steps[layer_slot, src_idx] = 0

            if ema > utility_on_threshold:
                self.source_recover_steps[layer_slot, src_idx] += 1
            else:
                self.source_recover_steps[layer_slot, src_idx] = 0

            if self.source_inactive_steps[layer_slot, src_idx] >= patience:
                self.source_gate_mask[layer_slot, src_idx] = 0.0

            # Re-open source if utility recovers with hysteresis.
            if self.source_recover_steps[layer_slot, src_idx] >= reopen_patience:
                self.source_gate_mask[layer_slot, src_idx] = 1.0
                self.source_inactive_steps[layer_slot, src_idx] = 0

    def get_source_mask(
        self,
        layer_slot: int,
        source_ready: torch.Tensor,
        current_step: int,
        self_warmup_steps: int,
        enable_selective_gate: bool,
    ) -> torch.Tensor:
        mask = source_ready.clone()

        if self.use_ema_source and current_step < self_warmup_steps:
            mask[1] = 0.0

        if enable_selective_gate:
            mask = mask * self.source_gate_mask[layer_slot]

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
    raw_alpha_dino_acc = 0.0
    raw_alpha_self_acc = 0.0
    router_entropy_acc = 0.0
    loss_fused_acc = 0.0
    loss_fused_probe_acc = 0.0
    loss_dino_only_acc = 0.0
    loss_self_only_acc = 0.0
    utility_dino_acc = 0.0
    utility_self_acc = 0.0
    utility_self_ema_acc = 0.0
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
                    # Optional trainable self-side projector (disabled by default).
                    ema_proj = s3a_head.ema_adapters[key](ema_tokens_detached)
                else:
                    # Use student adapter path as a frozen projector to avoid a
                    # moving self-target shortcut in default contract.
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

        def _build_alpha(mask_vec: torch.Tensor) -> torch.Tensor:
            alpha_local = raw_alpha * mask_vec.unsqueeze(0)
            alpha_local = alpha_local / alpha_local.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            if (
                s3a_head.num_sources > 1
                and mask_vec[0] > 0
                and args.s3a_dino_alpha_floor > 0
                and args.s3a_dino_alpha_floor_steps > 0
                and current_step < args.s3a_dino_alpha_floor_steps
            ):
                floor_ratio = 1.0 - (current_step / max(1, args.s3a_dino_alpha_floor_steps))
                alpha_floor = args.s3a_dino_alpha_floor * max(0.0, floor_ratio)
                alpha_dino = alpha_local[:, 0].clamp(min=alpha_floor)
                other = alpha_local[:, 1:]
                other = other / other.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                alpha_local = torch.cat(
                    [alpha_dino.unsqueeze(-1), (1.0 - alpha_dino).unsqueeze(-1) * other],
                    dim=-1,
                )
            return alpha_local

        alpha = _build_alpha(source_mask)

        def _build_utility_probe_alpha(mask_vec: torch.Tensor) -> torch.Tensor:
            if args.s3a_utility_probe_mode == "raw_alpha":
                return _build_alpha(mask_vec)
            alpha_probe_local = mask_vec / mask_vec.sum().clamp(min=1e-8)
            return alpha_probe_local.unsqueeze(0).expand(raw_alpha.shape[0], -1)

        fused = torch.zeros_like(pred)
        for src_idx, src_tokens in enumerate(sources):
            fused = fused + alpha[:, src_idx].view(-1, 1, 1) * src_tokens

        feat_loss_ps = cosine_distance_per_sample(pred, fused)
        attn_loss_ps = affinity_loss_per_sample(
            pred,
            fused,
            max_tokens=args.s3a_attn_max_tokens,
        )
        spatial_loss_ps = spatial_loss_per_sample(pred, fused)

        combined_ps = (
            args.s3a_feat_weight * feat_loss_ps
            + args.s3a_attn_weight * attn_loss_ps
            + args.s3a_spatial_weight * spatial_loss_ps
        )

        dino_layer_loss_mean = 0.0
        self_layer_loss_mean = 0.0
        fused_probe_loss_mean = 0.0
        utility_dino_layer = 0.0
        utility_self_layer = 0.0

        if do_probe:
            # Probe mask disables selective gate to estimate source utility even
            # when a source is currently gated off.
            source_mask_probe = s3a_head.get_source_mask(
                layer_slot=slot,
                source_ready=source_ready,
                current_step=current_step,
                self_warmup_steps=args.s3a_self_warmup_steps,
                enable_selective_gate=False,
            )
            alpha_probe = _build_utility_probe_alpha(source_mask_probe)
            with torch.no_grad():
                pred_probe = pred.detach()
                fused_probe = torch.zeros_like(pred_probe)
                for src_idx, src_tokens in enumerate(sources):
                    fused_probe = fused_probe + alpha_probe[:, src_idx].view(-1, 1, 1) * src_tokens

                fused_probe_feat_ps = cosine_distance_per_sample(pred_probe, fused_probe)
                fused_probe_attn_ps = affinity_loss_per_sample(
                    pred_probe,
                    fused_probe,
                    max_tokens=args.s3a_attn_max_tokens,
                )
                fused_probe_spatial_ps = spatial_loss_per_sample(pred_probe, fused_probe)
                fused_probe_combined_ps = (
                    args.s3a_feat_weight * fused_probe_feat_ps
                    + args.s3a_attn_weight * fused_probe_attn_ps
                    + args.s3a_spatial_weight * fused_probe_spatial_ps
                )
                fused_probe_loss_mean = fused_probe_combined_ps.mean().item()

                dino_feat_ps = cosine_distance_per_sample(pred_probe, dino_layer)
                dino_attn_ps = affinity_loss_per_sample(
                    pred_probe,
                    dino_layer,
                    max_tokens=args.s3a_attn_max_tokens,
                )
                dino_spatial_ps = spatial_loss_per_sample(pred_probe, dino_layer)
                dino_combined_ps = (
                    args.s3a_feat_weight * dino_feat_ps
                    + args.s3a_attn_weight * dino_attn_ps
                    + args.s3a_spatial_weight * dino_spatial_ps
                )
                dino_layer_loss_mean = dino_combined_ps.mean().item()
                dino_only_count += 1

                if s3a_head.num_sources > 1 and source_mask_probe[1] > 0:
                    self_layer = sources[1]
                    self_feat_ps = cosine_distance_per_sample(pred_probe, self_layer)
                    self_attn_ps = affinity_loss_per_sample(
                        pred_probe,
                        self_layer,
                        max_tokens=args.s3a_attn_max_tokens,
                    )
                    self_spatial_ps = spatial_loss_per_sample(pred_probe, self_layer)
                    self_combined_ps = (
                        args.s3a_feat_weight * self_feat_ps
                        + args.s3a_attn_weight * self_attn_ps
                        + args.s3a_spatial_weight * self_spatial_ps
                    )
                    self_layer_loss_mean = self_combined_ps.mean().item()
                    self_only_count += 1

                # Utility is defined as "loss without source - full fused loss".
                # Positive means this source contributes useful signal.
                utility_self_layer = dino_layer_loss_mean - fused_probe_loss_mean
                if s3a_head.num_sources > 1 and source_mask_probe[1] > 0:
                    utility_dino_layer = self_layer_loss_mean - fused_probe_loss_mean
                utility_probe_count += 1

        sample_weights = diff_weights * (phase_weight * layer_weights[slot])
        layer_loss = (sample_weights * combined_ps).mean()

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
        else:
            router_entropy_acc += 0.0
        loss_fused_acc += combined_ps.mean().item()
        loss_fused_probe_acc += fused_probe_loss_mean
        loss_dino_only_acc += dino_layer_loss_mean
        utility_dino_acc += utility_dino_layer
        utility_self_acc += utility_self_layer
        if s3a_head.num_sources > 1:
            alpha_self_acc += alpha[:, 1].mean().item()
            gate_self_acc += source_mask[1].item()
            loss_self_only_acc += self_layer_loss_mean
            if do_probe:
                utility_self_ema_acc += s3a_head.source_utility_ema[slot, 1].item()

            if (
                do_probe
                and
                current_step >= args.s3a_self_warmup_steps
                and source_ready[1] > 0
                and alpha[:, 0].mean().item() < args.s3a_collapse_alpha_threshold
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
                utility_mean = torch.zeros(s3a_head.num_sources, device=device)
                utility_mean[0] = utility_dino_layer
                if s3a_head.num_sources > 1:
                    utility_mean[1] = utility_self_layer
                ready = source_ready.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(utility_mean, op=dist.ReduceOp.SUM)
                    utility_mean = utility_mean / dist.get_world_size()

                    dist.all_reduce(ready, op=dist.ReduceOp.SUM)
                    ready = (ready > 0).float()

                s3a_head.update_gate_state(
                    layer_slot=slot,
                    utility_mean=utility_mean,
                    source_ready=ready,
                    utility_off_threshold=args.s3a_gate_utility_off_threshold,
                    utility_on_threshold=args.s3a_gate_utility_on_threshold,
                    patience=gate_patience_windows,
                    reopen_patience=gate_reopen_windows,
                    utility_ema_momentum=args.s3a_gate_utility_ema_momentum,
                    protect_source0=True,
                )

    if used_layers == 0:
        stats = {
            "used_layers": 0,
            "feat": 0.0,
            "attn": 0.0,
            "spatial": 0.0,
            "alpha_dino": 0.0,
            "alpha_self": 0.0,
            "gate_self": 0.0,
            "diff_w": 0.0,
            "raw_alpha_dino": 0.0,
            "raw_alpha_self": 0.0,
            "router_entropy_norm": 0.0,
            "loss_fused": 0.0,
            "loss_fused_probe": 0.0,
            "loss_dino_only": 0.0,
            "loss_self_only": 0.0,
            "utility_dino": 0.0,
            "utility_self": 0.0,
            "utility_self_ema": 0.0,
            "probe_count": 0.0,
            "self_probe_count": 0.0,
            "collapse_alarm": 0.0,
            "alpha_dino_min_layer": 0.0,
            "alpha_dino_max_layer": 0.0,
            "alpha_dino_layers": [0.0 for _ in s3a_head.layer_indices],
        }
        return total_loss, stats

    total_loss = total_loss / used_layers

    stats = {
        "used_layers": float(used_layers),
        "feat": feat_acc / used_layers,
        "attn": attn_acc / used_layers,
        "spatial": spatial_acc / used_layers,
        "alpha_dino": alpha_dino_acc / used_layers,
        "alpha_self": alpha_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0,
        "gate_self": gate_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0,
        "diff_w": diff_weights.mean().item(),
        "raw_alpha_dino": raw_alpha_dino_acc / used_layers,
        "raw_alpha_self": (
            raw_alpha_self_acc / used_layers if s3a_head.num_sources > 1 else 0.0
        ),
        "router_entropy_norm": router_entropy_acc / used_layers,
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
            utility_self_ema_acc / max(1, utility_probe_count)
            if s3a_head.num_sources > 1
            else 0.0
        ),
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


def _validate_resume_contract(current_args, saved_args_dict: Dict[str, Any]) -> None:
    if not saved_args_dict:
        return
    keys = [
        "model",
        "s3a",
        "s3a_use_ema_source",
        "s3a_trainable_ema_adapters",
        "s3a_enable_selective_gate",
        "s3a_self_warmup_steps",
        "s3a_dino_alpha_floor",
        "s3a_dino_alpha_floor_steps",
        "s3a_probe_every",
        "s3a_utility_probe_mode",
        "s3a_gate_patience",
        "s3a_gate_reopen_patience",
        "s3a_gate_utility_off_threshold",
        "s3a_gate_utility_on_threshold",
        "s3a_gate_utility_ema_momentum",
        "s3a_feat_weight",
        "s3a_attn_weight",
        "s3a_spatial_weight",
        "s3a_layer_indices",
        "s3a_lambda",
    ]
    mismatches = []
    for key in keys:
        if not hasattr(current_args, key):
            continue
        if key not in saved_args_dict:
            continue
        current_val = getattr(current_args, key)
        saved_val = saved_args_dict[key]
        if not _is_equal_resume_value(current_val, saved_val):
            mismatches.append((key, current_val, saved_val))
    if mismatches:
        detail = ", ".join(
            [f"{k}(current={c}, checkpoint={s})" for (k, c, s) in mismatches[:8]]
        )
        raise ValueError(
            "Resume contract mismatch for critical S3A args: "
            f"{detail}. Refuse to resume with incompatible config."
        )


def _migrate_legacy_s3a_state(
    loaded_state: Dict[str, torch.Tensor],
    expected_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    migrated = dict(loaded_state)
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
        "source_utility_ema",
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
    return migrated, missing


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
        "format_version": 3,
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
        _validate_resume_contract(current_args, saved_args_dict)

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
        if format_version < 3:
            s3a_state, _ = _migrate_legacy_s3a_state(
                loaded_state=s3a_state,
                expected_state=expected_state,
            )
        s3a_head.module.load_state_dict(s3a_state, strict=True)

    resume_meta = {
        "format_version": format_version,
        "saved_args": saved_args_dict,
        "s3a_runtime_state": saved_runtime_state,
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
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        metrics_jsonl_path = None

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
            "Initializing S3A alignment with DINOv2 ViT-B/14 teacher\n"
            f"  train_schedule      : {args.s3a_train_schedule} "
            f"(decay over {args.s3a_schedule_steps} steps)\n"
            f"  diff_schedule       : {args.s3a_diff_schedule}\n"
            f"  s3a_lambda          : {args.s3a_lambda}\n"
            f"  use_ema_source      : {args.s3a_use_ema_source}\n"
            f"  selective_gate      : {args.s3a_enable_selective_gate}\n"
            f"  DINOv2 repo         : {args.dinov2_repo_dir}\n"
            f"  DINOv2 weight       : {args.dinov2_weight_path}"
        )

        dino_model = LocalDINOv2Teacher(
            dinov2_repo_dir=args.dinov2_repo_dir,
            weight_path=args.dinov2_weight_path,
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
            f"  DINO dim            : {dino_dim}\n"
            f"  DINO num_tokens     : {dino_num_tokens}\n"
            f"  layer indices       : {s3a_layer_indices}\n"
            f"  layer weights       : {[round(w, 4) for w in s3a_layer_weights]}\n"
            f"  trainable ema-adpt  : {args.s3a_trainable_ema_adapters}\n"
            f"  S3A params          : {sum(p.numel() for p in s3a_head.parameters()):,}"
        )

    model = DDP(model, device_ids=[device])
    if args.s3a:
        s3a_head = DDP(s3a_head, device_ids=[device], broadcast_buffers=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    s3a_trainable_param_count = 0
    s3a_frozen_param_count = 0
    if args.s3a:
        for name, p in s3a_head.named_parameters():
            if (
                not args.s3a_trainable_ema_adapters
                and ".ema_adapters." in name
            ):
                p.requires_grad_(False)
                s3a_frozen_param_count += p.numel()
                continue
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
            allow_missing_manifest=args.allow_missing_manifest,
        )
        resumed_s3a_runtime_state = dict(resume_meta.get("s3a_runtime_state", {}))
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
            f"collapse_alarm_windows={int(resumed_s3a_runtime_state.get('collapse_alarm_windows', 0))})"
        )

    if args.resume is None:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    if args.s3a:
        s3a_head.train()
        dino_model.eval()

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
    running_raw_alpha_dino = 0.0
    running_raw_alpha_self = 0.0
    running_router_entropy = 0.0
    running_loss_fused = 0.0
    running_loss_fused_probe = 0.0
    running_loss_dino_only = 0.0
    running_loss_self_only = 0.0
    running_utility_dino = 0.0
    running_utility_self = 0.0
    running_utility_self_ema = 0.0
    running_probe_count = 0.0
    running_self_probe_count = 0.0
    running_collapse_alarm = 0.0
    running_alpha_dino_min_layer = 0.0
    running_alpha_dino_max_layer = 0.0
    running_alpha_dino_layers = [0.0 for _ in s3a_layer_indices]
    collapse_alarm_windows = int(resumed_s3a_runtime_state.get("collapse_alarm_windows", 0))
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
            f"  phase schedule     : {args.s3a_train_schedule} over {args.s3a_schedule_steps}\n"
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
            f"  probe every        : {args.s3a_probe_every}\n"
            f"  probe mode         : {args.s3a_utility_probe_mode}\n"
            f"  loss weights       : feat={args.s3a_feat_weight}, "
            f"attn={args.s3a_attn_weight}, spatial={args.s3a_spatial_weight}"
        )

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
            align_stats = {
                "used_layers": 0.0,
                "feat": 0.0,
                "attn": 0.0,
                "spatial": 0.0,
                "alpha_dino": 0.0,
                "alpha_self": 0.0,
                "gate_self": 0.0,
                "diff_w": 0.0,
                "raw_alpha_dino": 0.0,
                "raw_alpha_self": 0.0,
                "router_entropy_norm": 0.0,
                "loss_fused": 0.0,
                "loss_fused_probe": 0.0,
                "loss_dino_only": 0.0,
                "loss_self_only": 0.0,
                "utility_dino": 0.0,
                "utility_self": 0.0,
                "utility_self_ema": 0.0,
                "probe_count": 0.0,
                "self_probe_count": 0.0,
                "collapse_alarm": 0.0,
                "alpha_dino_min_layer": 0.0,
                "alpha_dino_max_layer": 0.0,
                "alpha_dino_layers": [0.0 for _ in s3a_layer_indices],
            }

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
                    opt.zero_grad(set_to_none=True)
                    continue
            opt.step()
            update_ema(ema, model.module)

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
            running_raw_alpha_dino += align_stats["raw_alpha_dino"]
            running_raw_alpha_self += align_stats["raw_alpha_self"]
            running_router_entropy += align_stats["router_entropy_norm"]
            running_loss_fused += align_stats["loss_fused"]
            running_loss_fused_probe += align_stats["loss_fused_probe"] * align_stats["probe_count"]
            running_loss_dino_only += align_stats["loss_dino_only"] * align_stats["probe_count"]
            running_loss_self_only += align_stats["loss_self_only"] * align_stats["self_probe_count"]
            running_utility_dino += align_stats["utility_dino"] * align_stats["probe_count"]
            running_utility_self += align_stats["utility_self"] * align_stats["probe_count"]
            running_utility_self_ema += (
                align_stats["utility_self_ema"] * align_stats["self_probe_count"]
            )
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
                avg_raw_alpha_dino = _reduce_mean(running_raw_alpha_dino)
                avg_raw_alpha_self = _reduce_mean(running_raw_alpha_self)
                avg_router_entropy = _reduce_mean(running_router_entropy)
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
                avg_utility_self_ema, _ = _reduce_probe_mean(
                    running_utility_self_ema, running_self_probe_count
                )
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

                collapse_window_triggered = False
                if global_probe_count > 0:
                    collapse_window_triggered = (
                        args.s3a_use_ema_source
                        and global_self_probe_count > 0
                        and train_steps >= args.s3a_self_warmup_steps
                        and avg_alpha_dino < args.s3a_collapse_alpha_threshold
                        and avg_alpha_self > args.s3a_collapse_self_threshold
                        and avg_utility_dino > args.s3a_collapse_utility_threshold
                        and avg_loss_fused_probe + args.s3a_collapse_margin < avg_loss_self_only
                    )
                    collapse_alarm_windows = (
                        collapse_alarm_windows + 1 if collapse_window_triggered else 0
                    )
                collapse_alarm = (
                    1.0 if collapse_alarm_windows >= args.s3a_collapse_windows else 0.0
                )
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
                        f"utility_self_ema={avg_utility_self_ema:.6f}, "
                        f"probes={int(global_probe_count)}, "
                        f"self_probes={int(global_self_probe_count)}, "
                        f"windows={collapse_alarm_windows}"
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
                    f"a_self={avg_alpha_self:.3f}  "
                    f"raw_dino={avg_raw_alpha_dino:.3f}  "
                    f"raw_self={avg_raw_alpha_self:.3f}  "
                    f"H={avg_router_entropy:.3f}  "
                    f"gate_self={avg_gate_self:.3f}  "
                    f"Lfused={avg_loss_fused:.4f}  "
                    f"LfusedProbe={avg_loss_fused_probe:.4f}  "
                    f"Ldino={avg_loss_dino_only:.4f}  "
                    f"Lself={avg_loss_self_only:.4f}  "
                    f"U_dino={avg_utility_dino:.4f}  "
                    f"U_self={avg_utility_self:.4f}  "
                    f"UselfEMA={avg_utility_self_ema:.4f}  "
                    f"ProbeN={int(global_probe_count)}  "
                    f"SelfProbeN={int(global_self_probe_count)}  "
                    f"alarm={collapse_alarm:.0f}  "
                    f"eff_λ={args.s3a_lambda * avg_phase_w:.6f}  "
                    f"Steps/s={steps_per_sec:.2f}"
                )
                if rank == 0 and metrics_jsonl_path is not None:
                    metric_row = {
                        "step": int(train_steps),
                        "batches_seen": int(batches_seen),
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
                        "alpha_self": float(avg_alpha_self),
                        "raw_alpha_dino": float(avg_raw_alpha_dino),
                        "raw_alpha_self": float(avg_raw_alpha_self),
                        "router_entropy_norm": float(avg_router_entropy),
                        "gate_self": float(avg_gate_self),
                        "loss_fused": float(avg_loss_fused),
                        "loss_fused_probe": float(avg_loss_fused_probe),
                        "loss_dino_only": float(avg_loss_dino_only),
                        "loss_self_only": float(avg_loss_self_only),
                        "utility_dino": float(avg_utility_dino),
                        "utility_self": float(avg_utility_self),
                        "utility_self_ema": float(avg_utility_self_ema),
                        "probe_count": int(global_probe_count),
                        "self_probe_count": int(global_self_probe_count),
                        "collapse_window_score": float(avg_collapse_alarm),
                        "collapse_alarm": float(collapse_alarm),
                        "collapse_alarm_windows": int(collapse_alarm_windows),
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
                running_raw_alpha_dino = 0.0
                running_raw_alpha_self = 0.0
                running_router_entropy = 0.0
                running_loss_fused = 0.0
                running_loss_fused_probe = 0.0
                running_loss_dino_only = 0.0
                running_loss_self_only = 0.0
                running_utility_dino = 0.0
                running_utility_self = 0.0
                running_utility_self_ema = 0.0
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
            "DiT + S3A training with DINOv2 ViT-B/14 teacher.\n"
            "S3A: multi-layer multi-source alignment branch with dynamic routing."
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
        help="DINOv2 ViT-B/14 checkpoint path.",
    )

    parser.add_argument("--s3a", action="store_true", help="Enable S3A alignment branch.")
    parser.add_argument("--s3a-lambda", type=float, default=0.1)

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

    parser.add_argument("--s3a-adapter-hidden-dim", type=int, default=None)
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
        choices=["constant", "linear_decay", "cosine_decay", "cutoff"],
        default="cosine_decay",
    )
    parser.add_argument("--s3a-schedule-steps", type=int, default=40_000)
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
        default=0.0,
        help="Utility EMA below this threshold counts as bad for gate-off accounting.",
    )
    parser.add_argument(
        "--s3a-gate-utility-on-threshold",
        type=float,
        default=0.005,
        help="Utility EMA above this threshold counts as recovery for gate reopen.",
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
        "--s3a-collapse-alpha-threshold",
        type=float,
        default=0.05,
        help="Collapse alarm threshold for alpha_dino.",
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
        help="Collapse alarm margin used with loss_fused + margin < loss_self_only.",
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
        "--s3a-probe-every",
        type=int,
        default=10,
        help=(
            "Compute source-only diagnostic probes every N steps. "
            "1 means probe every step."
        ),
    )
    parser.add_argument(
        "--s3a-utility-probe-mode",
        type=str,
        choices=["uniform", "raw_alpha"],
        default="uniform",
        help=(
            "Probe mixing for utility estimation. "
            "uniform decouples utility from router confidence."
        ),
    )

    return parser


def validate_args(args):
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
        if args.s3a_feat_weight < 0 or args.s3a_attn_weight < 0 or args.s3a_spatial_weight < 0:
            raise ValueError("S3A loss weights must be >= 0")
        if not (0.0 <= args.s3a_dino_alpha_floor < 1.0):
            raise ValueError("--s3a-dino-alpha-floor must be in [0, 1)")
        if args.s3a_dino_alpha_floor_steps < 0:
            raise ValueError("--s3a-dino-alpha-floor-steps must be >= 0")
        if args.s3a_collapse_windows <= 0:
            raise ValueError("--s3a-collapse-windows must be > 0")
        if args.s3a_collapse_utility_threshold < 0:
            raise ValueError("--s3a-collapse-utility-threshold must be >= 0")
        if args.s3a_probe_every <= 0:
            raise ValueError("--s3a-probe-every must be > 0")
        if args.s3a_probe_every > args.log_every:
            raise ValueError(
                "--s3a-probe-every must be <= --log-every to keep collapse "
                "monitoring windows semantically consistent."
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
