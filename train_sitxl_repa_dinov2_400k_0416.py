# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
SiT-XL/2 + REPA training script with DINOv2 ViT-B/14 teacher.
直接基于 REPA 源码 (sihyun-yu/REPA) 的 SILoss 逻辑实现。

REPA baseline 之上的两个创新点：
  1. 二维对齐调度：训练阶段衰减 (轴1) × 扩散时间步加权 (轴2)
  2. iREPA 投影器选项：Conv2d(k=3) + spatial normalization

所有其他部分（模型、interpolant、loss 计算）与 REPA 原版严格一致。
"""

import os
import sys
import math
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
from torch.amp import autocast, GradScaler
from torchvision.datasets import ImageFolder
from torchvision import transforms

# ── REPA 版 SiT 模型导入 ────────────────────────────────────────────────────────
# 使用 REPA 修改版 SiT (sihyun-yu/REPA/models/sit.py)
# 该版本包含内置 projectors、encoder_depth、use_cfg 等
# forward 返回 (output, zs_tilde) tuple
import sys
sys.path.insert(0, "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/project/REPA")

from models.sit import SiT_models
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params   = OrderedDict(ema_model.named_parameters())
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
            handlers.append(
                logging.FileHandler(f"{logging_dir}/log.txt")
            )
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
    arr    = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    )


#################################################################################
#               训练步数衰减权重（Training Phase Weight）                        #
#################################################################################

def get_train_phase_weight(
    current_step: int,
    schedule_steps: int,
    schedule: str = "cosine_decay",
    warmup_steps: int = 0,
) -> float:
    """
    训练步数驱动的对齐权重（训练阶段衰减）。

    支持五种模式
    ------------
    constant     : 始终返回 1.0
    linear_decay : w = 1 - t/T，线性衰减
    cosine_decay : w = 0.5 * (1 + cos(π * t / T))，余弦衰减（推荐）
    cutoff       : 前 schedule_steps 步返回 1.0，之后返回 0.0
    three_phase  : 三阶段调度（需配合 warmup_steps）
                   [0, warmup_steps)           : w = 1.0（全力对齐）
                   [warmup_steps, schedule_steps) : 线性衰减 1.0 → 0.0
                   [schedule_steps, ...)        : w = 0.0（纯生成训练）
    """
    if schedule == "constant":
        return 1.0

    if schedule == "three_phase":
        if current_step < warmup_steps:
            return 1.0
        elif current_step < schedule_steps:
            decay_progress = (current_step - warmup_steps) / max(1, schedule_steps - warmup_steps)
            return 1.0 - decay_progress
        else:
            return 0.0

    progress = min(max(current_step / max(1, schedule_steps), 0.0), 1.0)

    if schedule == "linear_decay":
        return 1.0 - progress

    if schedule == "cosine_decay":
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if schedule == "cutoff":
        return 1.0 if current_step < schedule_steps else 0.0

    raise ValueError(
        f"Unknown train schedule: {schedule!r}. "
        "Choose from: constant, linear_decay, cosine_decay, cutoff, three_phase."
    )


#################################################################################
#               REPA-style 扩散时间步加权（SiT 连续时间 t ∈ [0,1]）            #
#################################################################################

def get_diff_timestep_weight(
    t: torch.Tensor,
    schedule: str = "cosine",
) -> torch.Tensor:
    """
    SiT 使用连续时间 t ∈ [0, 1]（0=干净，1=纯噪声），
    与 REPA 原版的离散 T=1000 对应关系：t_continuous = t_discrete / (T-1)

    四种模式
    --------
    cosine      : w = cos(π/2 · t)   t=0→1.0，t=1→0.0  [REPA 原版默认]
    linear_high : w = 1 - t          线性，低噪声权重高
    linear_low  : w = t              高噪声权重高（对照实验）
    uniform     : w = 1.0            均匀加权

    返回
    ----
    weights : [B]，batch 内均值归一化（mean≈1），保持 loss 量级稳定。
    """
    if schedule == "uniform":
        weights = torch.ones_like(t)
    elif schedule == "cosine":
        weights = torch.cos(math.pi / 2.0 * t)
    elif schedule == "linear_high":
        weights = 1.0 - t
    elif schedule == "linear_low":
        weights = t
    else:
        raise ValueError(
            f"Unknown diff_schedule: {schedule!r}. "
            "Choose from: cosine, linear_high, linear_low, uniform."
        )

    mean_w  = weights.mean().clamp(min=1e-8)
    weights = weights / mean_w    # batch 内归一化，均值≈1
    return weights


def cosine_align_loss_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    diff_schedule: str = "cosine",
) -> torch.Tensor:
    """
    带扩散时间步加权的余弦对齐 loss（SiT 连续时间版本）。

    参数
    ----
    pred          : [B, N, C]  DiT/SiT 中间 token 经 projector 投影后的特征
    target        : [B, N, C]  DINOv2 teacher patch token 特征
    t             : [B]        当前 batch 连续时间步 t ∈ [0, 1]
    diff_schedule : str        时间步加权模式
    """
    pred_n   = F.normalize(pred,   dim=-1)    # [B, N, C]
    target_n = F.normalize(target, dim=-1)    # [B, N, C]

    cos_per_token   = (pred_n * target_n).sum(dim=-1)    # [B, N]
    cos_per_sample  = cos_per_token.mean(dim=1)           # [B]
    loss_per_sample = -cos_per_sample                     # [B]  与 REPA SILoss 一致

    weights = get_diff_timestep_weight(t, diff_schedule)   # [B]
    loss    = (weights * loss_per_sample).mean()            # scalar
    return loss


#################################################################################
#                    DINOv2 Teacher（本地 .pth，torch.hub 方式）                #
#################################################################################

class LocalDINOv2Teacher(nn.Module):
    """
    从本地 .pth 权重加载 DINOv2 ViT-B/14，作为冻结 teacher。
    与 train_sasa_dinov2.py 中完全一致。
    """

    EXPECTED_PATCH_TOKENS = 256    # (224 / 14)² = 256

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

        state_dict = torch.load(weight_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]

        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            print(f"[DINOv2Teacher] missing keys ({len(missing)}): "
                  f"{missing[:5]} ...")
        if unexpected:
            print(f"[DINOv2Teacher] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]} ...")

        self.model.eval()
        requires_grad(self.model, False)

        self._resize = transforms.Resize(
            224,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, 3, H, W]，ImageNet 归一化
        返回 : [B, 256, 768]  patch tokens
        """
        x = self._resize(x)    # [B, 3, 224, 224]
        feats = self.model.get_intermediate_layers(
            x, n=1, return_class_token=False
        )
        patch_tokens = feats[0]    # [B, 256, 768]

        assert patch_tokens.shape[1] == self.EXPECTED_PATCH_TOKENS, (
            f"DINOv2 patch token count mismatch: "
            f"expected {self.EXPECTED_PATCH_TOKENS}, "
            f"got {patch_tokens.shape[1]}"
        )
        return patch_tokens


def preprocess_for_dino(x: torch.Tensor) -> torch.Tensor:
    """
    将 [-1, 1] 范围的图像转换为 DINOv2 期望的 ImageNet 归一化。
    x : [B, 3, H, W]
    """
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=x.device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=x.device
    ).view(1, 3, 1, 1)
    return (x - mean) / std


#################################################################################
#                          REPA / iREPA Projector                               #
#################################################################################

class REPAProjector(nn.Module):
    """
    三层 MLP，将 SiT 中间 token 投影到 DINOv2 特征空间。
    结构：Linear → SiLU → Linear → SiLU → Linear（与 REPA 原版 build_mlp 一致）
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class IREPAProjector(nn.Module):
    """
    iREPA (ICLR 2026) 的卷积投影器。
    用 Conv2d(kernel=3) 替代 MLP，天然保留空间局部结构。
    """
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 16):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, padding=1,
        )

    def forward(self, x):
        B, N, D = x.shape
        h = w = self.grid_size
        x = x.transpose(1, 2).reshape(B, D, h, w)
        x = self.conv(x)
        x = x.reshape(B, -1, N).transpose(1, 2)
        return x


def spatial_normalize(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """iREPA spatial normalization：去除 teacher feature 全局分量，增强空间对比度。"""
    mean = x.mean(dim=1, keepdim=True)
    x = x - gamma * mean
    std = x.std(dim=1, keepdim=True) + 1e-6
    return x / std


def build_projector(
    projector_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 2048,
    grid_size: int = 16,
) -> nn.Module:
    if projector_type == "repa":
        return REPAProjector(in_dim, out_dim, hidden_dim)
    elif projector_type == "irepa":
        return IREPAProjector(in_dim, out_dim, grid_size)
    else:
        raise ValueError(f"Unknown projector_type: {projector_type!r}")


#################################################################################
#                             Checkpoint                                        #
#################################################################################

def save_checkpoint(
    checkpoint_dir: str,
    train_steps: int,
    model,
    ema,
    opt,
    args,
    repa_projector=None,
) -> str:
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
#                    REPA SILoss 内联（linear path, v-prediction）              #
#################################################################################

def mean_flat(x):
    """对 batch 之外的所有维度取均值（与 REPA loss.py 一致）。"""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), \
        "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, \
        "global_batch_size must be divisible by world_size."

    rank   = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed   = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, "
        f"world_size={dist.get_world_size()}."
    )

    # ── 实验目录 ─────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index  = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name   = f"{experiment_index:03d}-{model_string_name}"
        if args.repa:
            experiment_name += (
                f"-repa-dinov2"
                f"-lam{args.repa_lambda}"
                f"-train{args.repa_train_schedule}"
                f"-diff{args.repa_diff_schedule}"
            )
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger         = create_logger(None)
        checkpoint_dir = None

    # ── 创建 REPA 版 SiT 模型 ─────────────────────────────────────────────
    assert args.image_size % 8 == 0, "image_size must be divisible by 8."
    latent_size = args.image_size // 8

    # REPA 版 SiT 需要额外参数：
    #   use_cfg      : 启用 CFG dropout
    #   z_dims       : encoder embed_dim，用于内置 projector
    #   encoder_depth: 在第 N 层提取 token 用于对齐
    #   projector_dim: 内置 3-layer MLP 的隐层维度
    #   block_kwargs : fused_attn, qk_norm
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        z_dims=[args.z_dim],
        encoder_depth=args.repa_encoder_depth,
        projector_dim=args.repa_hidden_dim,
        **block_kwargs,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)

    logger.info(
        f"SiT model      : {args.model}\n"
        f"SiT parameters : {sum(p.numel() for p in model.parameters()):,}\n"
        f"encoder_depth  : {args.repa_encoder_depth}\n"
        f"z_dims         : [{args.z_dim}]\n"
        f"projector_dim  : {args.repa_hidden_dim}"
    )
    if args.max_steps is not None:
        logger.info(f"Training will stop at max_steps = {args.max_steps}")

    # ── VAE ──────────────────────────────────────────────────────────────
    if args.vae_model_dir is not None:
        vae_path = args.vae_model_dir
        logger.info(f"Loading VAE from local directory: {vae_path}")
    else:
        vae_path = f"stabilityai/sd-vae-ft-{args.vae}"
        logger.info(f"Loading VAE from HuggingFace Hub: {vae_path}")

    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.eval()
    requires_grad(vae, False)

    # ── REPA 组件 ─────────────────────────────────────────────────────────
    dino_model     = None
    repa_projector = None
    hook_layer_idx = None

    if args.repa:
        logger.info(
            "Initializing REPA alignment with DINOv2 ViT-B/14 teacher\n"
            f"  train_schedule : {args.repa_train_schedule} "
            f"(decay over {args.repa_schedule_steps} steps)\n"
            f"  diff_schedule  : {args.repa_diff_schedule}\n"
            f"  repa_lambda    : {args.repa_lambda}\n"
            f"  projector_type : {args.projector_type}\n"
            f"  DINOv2 repo    : {args.dinov2_repo_dir}\n"
            f"  DINOv2 weight  : {args.dinov2_weight_path}"
        )

        dino_model = LocalDINOv2Teacher(
            dinov2_repo_dir=args.dinov2_repo_dir,
            weight_path=args.dinov2_weight_path,
        ).to(device)
        dino_model.eval()
        requires_grad(dino_model, False)

        # sit_hidden_dim = model.hidden_size
        # sit_depth      = model.depth
        # 从 x_embedder 反推 hidden_size，从 blocks 长度反推 depth
        sit_hidden_dim = model.x_embedder.proj.out_channels  # = 1152
        sit_depth      = len(model.blocks)                    # = 28

        # 探测 DINO 输出维度
        with torch.inference_mode():
            dummy_rgb  = torch.zeros(
                1, 3, args.image_size, args.image_size, device=device
            )
            dummy_dino = preprocess_for_dino(dummy_rgb)
            dino_probe = dino_model(dummy_dino)
            dino_dim        = dino_probe.shape[-1]    # 768
            dino_num_tokens = dino_probe.shape[1]     # 256

        if args.projector_type == "repa":
            # REPA baseline: 使用模型内置 projector (self.projectors)
            # 不需要外部 projector 和 hook
            logger.info(
                f"REPA setup (internal projector):\n"
                f"  SiT encoder_depth : {args.repa_encoder_depth}\n"
                f"  Internal projector: {args.repa_hidden_dim}d, 3-layer SiLU MLP\n"
                f"  Projector params  : "
                f"{sum(p.numel() for p in model.projectors.parameters()):,}"
            )

        elif args.projector_type == "irepa":
            # iREPA: 需要外部 Conv projector + hook
            hook_layer_idx = args.repa_encoder_depth - 1

            grid_size = int(dino_num_tokens ** 0.5)
            repa_projector = build_projector(
                projector_type="irepa",
                in_dim=sit_hidden_dim,
                out_dim=dino_dim,
                grid_size=grid_size,
            ).to(device)

            logger.info(
                f"iREPA setup (external Conv projector):\n"
                f"  hook block       : [{hook_layer_idx} / {sit_depth - 1}]\n"
                f"  Conv projector   : in={sit_hidden_dim}, out={dino_dim}, k=3\n"
                f"  Projector params : "
                f"{sum(p.numel() for p in repa_projector.parameters()):,}"
            )

    # ── DDP wrap ──────────────────────────────────────────────────────────
    model = DDP(model, device_ids=[device])
    if args.repa and repa_projector is not None:
        repa_projector = DDP(repa_projector, device_ids=[device])

    # ── 优化器 ────────────────────────────────────────────────────────────
    # model.parameters() 已包含 REPA SiT 内置 projector 的参数
    trainable_params = list(model.parameters())
    if args.repa and repa_projector is not None:
        # iREPA 外部 Conv projector 的额外参数
        trainable_params += list(repa_projector.parameters())

    opt = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Mixed Precision（与 REPA 原版 fp16 一致）──────────────────────────
    scaler = GradScaler("cuda", enabled=args.mixed_precision == "fp16")
    mp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no":   torch.float32,
    }[args.mixed_precision]

    # ── 数据集 ────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Lambda(
            lambda pil_image: center_crop_arr(pil_image, args.image_size)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            inplace=True,
        ),
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(
        f"Dataset: {len(dataset):,} images  |  path: {args.data_path}"
    )

    # ── 初始化 EMA ────────────────────────────────────────────────────────
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    if args.repa:
        if repa_projector is not None:
            repa_projector.train()
        dino_model.eval()

    # ── 日志累计变量 ──────────────────────────────────────────────────────
    train_steps                = 0
    log_steps                  = 0
    running_loss               = 0.0
    running_loss_denoising     = 0.0
    running_loss_align         = 0.0
    running_train_align_weight = 0.0
    running_avg_diff_weight    = 0.0
    start_time                 = time()

    logger.info(
        f"Training config:\n"
        f"  epochs         : {args.epochs}\n"
        f"  max_steps      : {args.max_steps}\n"
        f"  global_batch   : {args.global_batch_size}\n"
        f"  lr             : {args.lr}\n"
        f"  weight_decay   : {args.weight_decay}\n"
        f"  log_every      : {args.log_every}\n"
        f"  ckpt_every     : {args.ckpt_every}"
    )
    if args.repa:
        logger.info(
            f"REPA schedule:\n"
            f"  train_phase    : {args.repa_train_schedule} "
            f"over {args.repa_schedule_steps} steps\n"
            f"  diff_timestep  : {args.repa_diff_schedule}\n"
            f"  hook block     : [{hook_layer_idx}]"
        )

    done = False

    # ════════════════════════════════════════════════════════════════════
    #                           主训练循环
    # ════════════════════════════════════════════════════════════════════
    for epoch in range(args.epochs):
        if done:
            break

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch} ...")

        for images, y in loader:
            images = images.to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True)
            current_step = train_steps

            # ── VAE 编码 ─────────────────────────────────────────────────
            with torch.no_grad():
                # SiT 约定：x1 = 干净的 latent（数据端）
                x1 = vae.encode(images).latent_dist.sample().mul_(0.18215)

            # ── 训练步数衰减权重 ──────────────────────────────────────────
            train_align_weight = 0.0
            if args.repa:
                train_align_weight = get_train_phase_weight(
                    current_step=current_step,
                    schedule_steps=args.repa_schedule_steps,
                    schedule=args.repa_train_schedule,
                    warmup_steps=args.repa_warmup_steps,
                )

            # ── 注册 forward hook（仅 iREPA 模式需要，捕获中间 token）──────
            _captured_tokens = {}
            hook_handle      = None

            if (args.repa and train_align_weight > 0
                    and args.projector_type == "irepa"):
                def _hook_fn(module, input, output):
                    _captured_tokens['tokens'] = output

                hook_handle = (
                    model.module.blocks[hook_layer_idx]
                    .register_forward_hook(_hook_fn)
                )

            # ── REPA SILoss 内联：linear interpolant + v-prediction ────────
            # 与 REPA 源码 loss.py 的 SILoss.__call__ 严格一致
            # t ∈ [0,1]：t=0 干净数据，t=1 纯噪声
            time_input = torch.rand(
                (x1.shape[0], 1, 1, 1), device=device, dtype=x1.dtype
            )
            alpha_t  = 1.0 - time_input          # t=0 → alpha=1（数据）
            sigma_t  = time_input                 # t=0 → sigma=0（无噪声）
            noises   = torch.randn_like(x1)
            x_t      = alpha_t * x1 + sigma_t * noises
            # v-prediction target: d_alpha/dt * x + d_sigma/dt * noise
            model_target = -x1 + noises           # d_alpha=-1, d_sigma=1

            model_kwargs = dict(y=y)
            with autocast("cuda", dtype=mp_dtype, enabled=(args.mixed_precision != "no")):
                model_output, zs_tilde = model(x_t, time_input.flatten(), **model_kwargs)
                loss_denoising = mean_flat(
                    (model_output - model_target) ** 2
                ).mean()

            # t_continuous: 用于二维调度的轴 2（扩散时间步加权）
            # t=0 干净 → 权重大，与 get_diff_timestep_weight cosine 一致
            t_continuous = time_input.squeeze()    # [B]

            # hook 触发后立即移除
            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None

            # ── REPA 对齐 loss ────────────────────────────────────────────
            loss_align         = torch.tensor(0.0, device=device)
            avg_diff_weight    = 0.0
            alignment_computed = False

            if (args.repa and train_align_weight > 0):

                # Teacher tokens（clean RGB → DINOv2）
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    dino_tokens = dino_model(x_dino)        # [B, 256, 768]

                # iREPA: 对 teacher features 做 spatial normalization
                if args.projector_type == "irepa":
                    dino_tokens = spatial_normalize(dino_tokens)

                # ── 获取 projected tokens ──────────────────────────────
                if args.projector_type == "repa":
                    # REPA baseline: 使用模型内置 projector 的输出 zs_tilde
                    # 与 REPA loss.py SILoss 数学等价的 batch 级实现
                    # zs_tilde = [tensor [B, N, z_dim]]（单 encoder 只有 1 个元素）
                    z_tilde = zs_tilde[0]                          # [B, N, z_dim]
                    z_tilde_n = F.normalize(z_tilde, dim=-1)       # [B, N, z_dim]
                    z_n       = F.normalize(dino_tokens, dim=-1)   # [B, N, D]
                    # 逐 token cos，在 N 维取均值 → [B]
                    proj_loss = -(z_n * z_tilde_n).sum(dim=-1).mean(dim=-1)  # [B]

                    # 轴 2：扩散时间步加权（REPA baseline 用 uniform 时等价于无加权）
                    weights = get_diff_timestep_weight(t_continuous, args.repa_diff_schedule)
                    loss_align = (weights * proj_loss).mean()
                    with torch.no_grad():
                        avg_diff_weight = weights.mean().item()

                elif args.projector_type == "irepa":
                    # iREPA: 使用外部 Conv projector + hook 捕获的 token
                    if 'tokens' in _captured_tokens:
                        sit_tokens  = _captured_tokens['tokens']
                        proj_tokens = repa_projector(sit_tokens)

                        if proj_tokens.shape[1] != dino_tokens.shape[1]:
                            raise ValueError(
                                f"Token count mismatch: "
                                f"proj={proj_tokens.shape}, "
                                f"dino={dino_tokens.shape}"
                            )

                        loss_align = cosine_align_loss_weighted(
                            pred=proj_tokens,
                            target=dino_tokens,
                            t=t_continuous,
                            diff_schedule=args.repa_diff_schedule,
                        )
                        with torch.no_grad():
                            w = get_diff_timestep_weight(
                                t_continuous, args.repa_diff_schedule
                            )
                            avg_diff_weight = w.mean().item()

                alignment_computed = True

            # ── Total loss ────────────────────────────────────────────────
            # L = L_denoising + λ_eff · L_align
            # λ_eff = repa_lambda × train_phase_weight(step)
            effective_repa_lambda = args.repa_lambda * train_align_weight
            loss = loss_denoising + effective_repa_lambda * loss_align

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.max_grad_norm is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model.module)

            # ── 日志累计 ──────────────────────────────────────────────────
            running_loss               += loss.item()
            running_loss_denoising     += loss_denoising.item()
            running_loss_align         += loss_align.item()
            running_train_align_weight += train_align_weight
            running_avg_diff_weight    += avg_diff_weight
            log_steps   += 1
            train_steps += 1

            # 第一步调试日志
            if train_steps == 1 and rank == 0:
                logger.info(f"  [step=1] images shape     : {images.shape}")
                logger.info(f"  [step=1] latent (x1) shape: {x1.shape}")
                if t_continuous is not None:
                    logger.info(
                        f"  [step=1] t_cont (first 8) : "
                        f"{t_continuous[:8].tolist()}"
                    )
                if alignment_computed:
                    logger.info(
                        f"  [step=1] dino_tokens : {dino_tokens.shape}"
                    )
                    if args.projector_type == "irepa" and 'tokens' in _captured_tokens:
                        logger.info(
                            f"  [step=1] sit_tokens  : {_captured_tokens['tokens'].shape}"
                        )
                        logger.info(
                            f"  [step=1] proj_tokens : {proj_tokens.shape}"
                        )
                    elif args.projector_type == "repa":
                        logger.info(
                            f"  [step=1] zs_tilde[0] : {zs_tilde[0].shape}"
                        )
                logger.info(
                    f"  [step=1] train_align_weight : {train_align_weight:.4f}"
                )
                logger.info(
                    f"  [step=1] effective_lambda   : "
                    f"{effective_repa_lambda:.6f}"
                )

            # ── 定期日志 ──────────────────────────────────────────────────
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time      = time()
                steps_per_sec = log_steps / (end_time - start_time)

                def _reduce_mean(val: float) -> float:
                    tensor = torch.tensor(val / log_steps, device=device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    return tensor.item() / dist.get_world_size()

                avg_loss               = _reduce_mean(running_loss)
                avg_loss_denoising     = _reduce_mean(running_loss_denoising)
                avg_loss_align         = _reduce_mean(running_loss_align)
                avg_train_align_weight = _reduce_mean(
                    running_train_align_weight
                )
                avg_diff_w_log         = _reduce_mean(running_avg_diff_weight)

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Loss={avg_loss:.4f}  "
                    f"Loss_denoising={avg_loss_denoising:.4f}  "
                    f"Loss_align={avg_loss_align:.4f}  "
                    f"TrainW={avg_train_align_weight:.4f}  "
                    f"DiffW={avg_diff_w_log:.4f}  "
                    f"eff_λ="
                    f"{args.repa_lambda * avg_train_align_weight:.6f}  "
                    f"Steps/s={steps_per_sec:.2f}"
                )

                running_loss               = 0.0
                running_loss_denoising     = 0.0
                running_loss_align         = 0.0
                running_train_align_weight = 0.0
                running_avg_diff_weight    = 0.0
                log_steps  = 0
                start_time = time()

            # ── 定期 checkpoint ───────────────────────────────────────────
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir, train_steps,
                        model, ema, opt, args,
                        repa_projector if args.repa else None,
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

            # ── max_steps 终止 ────────────────────────────────────────────
            if (args.max_steps is not None
                    and train_steps >= args.max_steps):
                if rank == 0:
                    ckpt_path = save_checkpoint(
                        checkpoint_dir, train_steps,
                        model, ema, opt, args,
                        repa_projector if args.repa else None,
                    )
                    logger.info(
                        f"Reached max_steps={args.max_steps}. "
                        f"Saved final checkpoint to {ckpt_path}"
                    )
                dist.barrier()
                done = True
                break

    # ── 收尾 ─────────────────────────────────────────────────────────────
    model.eval()
    if args.repa and repa_projector is not None:
        repa_projector.eval()

    logger.info("Done!")
    cleanup()


#################################################################################
#                                  Entry Point                                  #
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "SiT-XL/2 + REPA training with DINOv2 ViT-B/14 teacher.\n"
            "训练步数权重: linear_decay / cosine_decay / constant / cutoff\n"
            "扩散时间步权重: cosine / linear_high / linear_low / uniform"
        )
    )

    # ── 基础训练参数 ───────────────────────────────────────────────────
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="ImageFolder 格式的训练数据目录。",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/"
            "yuantianshuo/2026/results"
        ),
    )
    parser.add_argument(
        "--model", type=str,
        choices=list(SiT_models.keys()),
        default="SiT-XL/2",
    )
    parser.add_argument(
        "--image-size", type=int, choices=[256, 512], default=256,
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument(
        "--fused-attn", action=argparse.BooleanOptionalAction, default=True,
        help="启用 fused attention（REPA 默认 True）。",
    )
    parser.add_argument(
        "--qk-norm", action=argparse.BooleanOptionalAction, default=False,
        help="启用 QK normalization（REPA 默认 False）。",
    )
    parser.add_argument(
        "--cfg-prob", type=float, default=0.1,
        help="CFG label dropout 概率（REPA 默认 0.1）。",
    )
    parser.add_argument(
        "--z-dim", type=int, default=768,
        help="DINOv2 encoder embed_dim，用于 SiT 内置 projector 的 z_dims。",
    )
    parser.add_argument("--epochs",      type=int, default=1400)
    parser.add_argument("--max-steps",   type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed",       type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="mse",
    )
    parser.add_argument("--vae-model-dir", type=str, default=None)
    parser.add_argument("--num-workers",   type=int, default=4)
    parser.add_argument("--log-every",     type=int, default=100)
    parser.add_argument("--ckpt-every",    type=int, default=10_000)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight-decay",  type=float, default=0.0)
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="梯度裁剪最大范数（REPA 原版 max_grad_norm=1.0）。",
    )
    parser.add_argument(
        "--mixed-precision", type=str,
        choices=["no", "fp16", "bf16"], default="fp16",
        help="Mixed precision 训练（REPA 原版用 fp16）。",
    )

    # ── SiT Interpolant 已内联（linear path + v-prediction，与 REPA 一致）
    # 不再需要 transport 参数

    # ── DINOv2 路径参数 ────────────────────────────────────────────────
    parser.add_argument(
        "--dinov2-repo-dir", type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/"
            "yuantianshuo/tmp/dinov2"
        ),
    )
    parser.add_argument(
        "--dinov2-weight-path", type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/"
            "tmp/dinov2/dinov2_weights/dinov2_vitb14_pretrain.pth"
        ),
    )

    # ── REPA 参数 ──────────────────────────────────────────────────────
    # --repa 默认启用，本脚本必须使用 REPA 对齐
    parser.add_argument(
        "--projector-type", type=str,
        choices=["repa", "irepa"], default="repa",
        help="repa: 3-layer MLP [REPA]; irepa: Conv2d(k=3) + spatial norm [iREPA]",
    )
    parser.add_argument("--repa-lambda",       type=float, default=0.5,
                        help="REPA 原版默认 0.5，对应 --proj-coeff=0.5。")
    parser.add_argument(
        "--repa-encoder-depth", type=int, default=8,
        help=(
            "REPA 原版 --encoder-depth=8，即从第 8 个 block 提取 token。\n"
            "SiT-XL/2 共 28 层，默认 hook 第 7 层（0-indexed）。"
        ),
    )
    parser.add_argument("--repa-hidden-dim",   type=int,   default=2048,
                        help="Projector MLP 隐层维度（REPA 原版 projector_dim=2048）。")
    parser.add_argument(
        "--repa-train-schedule", type=str,
        choices=["constant", "linear_decay", "cosine_decay", "cutoff", "three_phase"],
        default="three_phase",
        help=(
            "轴1 训练步数权重调度。默认 three_phase:\n"
            "  [0, warmup_steps) = 1.0\n"
            "  [warmup_steps, warmup_steps+schedule_steps) = 线性衰减\n"
            "  [warmup_steps+schedule_steps, ∞) = 0.0"
        ),
    )
    parser.add_argument(
        "--repa-warmup-steps", type=int, default=100_000,
        help=(
            "three_phase: 前 N 步保持满权重（默认 100k）。\n"
            "0~100k 全力对齐，100k 开始衰减。"
        ),
    )
    parser.add_argument(
        "--repa-schedule-steps", type=int, default=300_000,
        help=(
            "three_phase: 衰减截止步数（绝对值）。默认 300k。\n"
            "衰减区间 = [warmup_steps, schedule_steps) = [100k, 300k)，\n"
            "300k 后权重归零，纯生成训练。"
        ),
    )
    parser.add_argument(
        "--repa-diff-schedule", type=str,
        choices=["cosine", "linear_high", "linear_low", "uniform"],
        default="cosine",
    )

    args = parser.parse_args()

    # 本脚本强制启用 REPA 对齐
    args.repa = True

    # ── 路径校验 ───────────────────────────────────────────────────────
    if args.vae_model_dir is not None:
        if not os.path.isdir(args.vae_model_dir):
            raise FileNotFoundError(
                f"--vae-model-dir not found: {args.vae_model_dir}"
            )
    if not os.path.isdir(args.dinov2_repo_dir):
        raise FileNotFoundError(
            f"--dinov2-repo-dir not found: {args.dinov2_repo_dir}"
        )
    if not os.path.isfile(args.dinov2_weight_path):
        raise FileNotFoundError(
            f"--dinov2-weight-path not found: {args.dinov2_weight_path}"
        )

    main(args)