# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
SASA-A training script with diffusion-timestep weighting (aligned with REPA).
Teacher: DINOv2 ViT-B/14 (local .pth weights via torch.hub)
训练步数权重调度：linear_decay / cosine_decay / constant / cutoff
扩散时间步权重：cosine / linear_high / linear_low / uniform
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
) -> float:
    """
    训练步数驱动的对齐权重（训练阶段衰减）。
    与扩散时间步加权相互独立，最终有效 lambda = base_lambda × 本函数返回值。

    支持四种模式
    ------------
    constant     : 始终返回 1.0，全程保持固定对齐压力
    linear_decay : w = 1 - t/T
                   从 1.0 线性衰减到 0.0，历时 schedule_steps 步
    cosine_decay : w = 0.5 * (1 + cos(π * t / T))
                   从 1.0 余弦衰减到 0.0，历时 schedule_steps 步
                   特点：前期衰减慢（充分对齐建立表征），
                         后期衰减快（快速退出专注生成质量）
    cutoff       : 前 schedule_steps 步返回 1.0，之后返回 0.0

    数值对比（T=40000步，部分节点）
    --------------------------------
    step  | linear | cosine
    ------+--------+-------
    0     | 1.000  | 1.000
    8k    | 0.800  | 0.905   ← cosine前期权重更高
    20k   | 0.500  | 0.500   ← 中点相同
    32k   | 0.200  | 0.095   ← cosine后期权重更低
    40k   | 0.000  | 0.000
    """
    if schedule == "constant":
        return 1.0

    # 归一化进度 progress ∈ [0, 1]
    progress = min(max(current_step / max(1, schedule_steps), 0.0), 1.0)

    if schedule == "linear_decay":
        # 线性：均匀衰减
        return 1.0 - progress

    if schedule == "cosine_decay":
        # 半周期余弦：平滑衰减，前慢后快
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if schedule == "cutoff":
        # 硬截断：前段全开，后段全关
        return 1.0 if current_step < schedule_steps else 0.0

    raise ValueError(
        f"Unknown train schedule: {schedule!r}. "
        "Choose from: constant, linear_decay, cosine_decay, cutoff."
    )


#################################################################################
#               REPA-style 扩散时间步加权（Diffusion Timestep Weight）           #
#################################################################################

def get_diff_timestep_weight(
    t: torch.Tensor,
    T: int,
    schedule: str = "cosine",
) -> torch.Tensor:
    """
    与原版 REPA 的 loss.py 保持一致的扩散时间步权重。

    参数
    ----
    t        : [B]  当前 batch 的扩散时间步（整数，范围 [0, T-1]）
    T        : int  扩散总步数（通常 1000）
    schedule : str  加权模式

    四种模式
    --------
    cosine      : w = cos(π/2 · t/(T-1))
                  t=0(干净图像) → 1.0，t=T-1(纯噪声) → 0.0  [REPA 默认]
    linear_high : w = 1 - t/(T-1)   线性，低噪声区间权重高
    linear_low  : w = t/(T-1)       线性，高噪声区间权重高（对照实验）
    uniform     : w = 1.0           均匀，无加权

    返回
    ----
    weights : [B]，已做 batch 内均值归一化（mean≈1），
              保持 loss 绝对量级不随 schedule 改变。
    """
    t_norm = t.float() / max(T - 1, 1)    # [B]，值域 [0, 1]

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

    # batch 内均值归一化：loss 量级与 schedule 无关
    mean_w  = weights.mean().clamp(min=1e-8)
    weights = weights / mean_w             # [B]，均值≈1
    return weights


def cosine_align_loss_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    T: int,
    diff_schedule: str = "cosine",
) -> torch.Tensor:
    """
    带扩散时间步加权的余弦对齐 loss（与 REPA 原版对齐）。

    参数
    ----
    pred          : [B, N, C]  DiT 中间 token 经 projector 投影后的特征
    target        : [B, N, C]  DINO teacher patch token 特征
    t             : [B]        当前 batch 扩散时间步（整数）
    T             : int        扩散总步数
    diff_schedule : str        时间步加权模式

    计算流程
    --------
    1. pred / target 在特征维度 C 做 L2 归一化
    2. 逐 token 余弦相似度，在 token 维度 N 取均值 → [B]
    3. 转为 loss：1 - cos_sim → [B]
    4. 按扩散时间步权重 w(t) 加权，取 batch 均值 → scalar

    数学表达
    --------
        w(b)   = get_diff_timestep_weight(t[b], T, schedule)
        cos(b) = mean_n [ <pred_norm[b,n], target_norm[b,n]> ]
        loss   = mean_b [ w(b) · (1 - cos(b)) ]
    """
    pred_n   = F.normalize(pred,   dim=-1)    # [B, N, C]
    target_n = F.normalize(target, dim=-1)    # [B, N, C]

    cos_per_token  = (pred_n * target_n).sum(dim=-1)    # [B, N]
    cos_per_sample = cos_per_token.mean(dim=1)           # [B]
    loss_per_sample = 1.0 - cos_per_sample               # [B]

    weights = get_diff_timestep_weight(t, T, diff_schedule)   # [B]
    loss    = (weights * loss_per_sample).mean()               # scalar
    return loss


#################################################################################
#                    DINOv2 Teacher（本地 .pth，torch.hub 方式）                #
#################################################################################

class LocalDINOv2Teacher(nn.Module):
    """
    从本地 .pth 权重加载 DINOv2 ViT-B/14，作为冻结 teacher。

    加载策略
    --------
    1. 将 dinov2_repo_dir 加入 sys.path，使 torch.hub 能找到本地代码。
    2. 用 torch.hub.load(source='local') 实例化模型结构（不联网）。
    3. 手动 load_state_dict 加载 .pth 权重。

    输入输出
    --------
    输入 : [B, 3, H, W]，已经过 preprocess_for_dino（ImageNet 归一化）
    内部 : 自动 resize 到 224×224（DINOv2 ViT-B/14 patch_size=14 的要求）
    输出 : [B, 256, 768]  patch tokens（已去掉 CLS token）
           256 = (224/14)²，768 = ViT-B hidden dim
    """

    EXPECTED_PATCH_TOKENS = 256    # (224 / 14)² = 256

    def __init__(self, dinov2_repo_dir: str, weight_path: str):
        """
        参数
        ----
        dinov2_repo_dir : DINOv2 源码根目录（含 hubconf.py）
        weight_path     : dinov2_vitb14_pretrain.pth 的完整路径
        """
        super().__init__()

        # ── 1. 将 dinov2 repo 加入 Python 路径 ──────────────────────────
        if dinov2_repo_dir not in sys.path:
            sys.path.insert(0, dinov2_repo_dir)

        # ── 2. 本地实例化模型（不联网） ───────────────────────────────────
        self.model = torch.hub.load(
            dinov2_repo_dir,
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )

        # ── 3. 加载本地 .pth 权重 ────────────────────────────────────────
        state_dict = torch.load(weight_path, map_location="cpu")
        # 兼容不同权重包装格式
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "teacher" in state_dict:
            state_dict = state_dict["teacher"]

        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            print(
                f"[DINOv2Teacher] missing keys ({len(missing)}): "
                f"{missing[:5]} ..."
            )
        if unexpected:
            print(
                f"[DINOv2Teacher] unexpected keys ({len(unexpected)}): "
                f"{unexpected[:5]} ..."
            )

        self.model.eval()
        requires_grad(self.model, False)

        # ── 4. 输入 resize（256→224） ─────────────────────────────────────
        # DINOv2 ViT-B/14 要求输入尺寸可被 patch_size=14 整除
        # 256×256 不满足（256/14≈18.28），需 resize 到 224×224
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

        # get_intermediate_layers：return_class_token=False → 只返回 patch tokens
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
    std  = torch.tensor(
        [0.229, 0.224, 0.225], device=x.device
    ).view(1, 3, 1, 1)
    return (x - mean) / std


#################################################################################
#                          REPA Projector                                       #
#################################################################################

class REPAProjector(nn.Module):
    """
    两层 MLP，将 DiT 中间 token 投影到 DINOv2 特征空间。
    结构：Linear → GELU → Linear（与 REPA 原版一致）
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


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
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger         = create_logger(None)
        checkpoint_dir = None

    # ── 创建 DiT ─────────────────────────────────────────────────────────
    assert args.image_size % 8 == 0, "image_size must be divisible by 8."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model     = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")
    T         = diffusion.num_timesteps    # 通常 1000

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

    logger.info(
        f"DiT model      : {args.model}\n"
        f"DiT parameters : {sum(p.numel() for p in model.parameters()):,}\n"
        f"Diffusion T    : {T}"
    )
    if args.max_steps is not None:
        logger.info(f"Training will stop at max_steps = {args.max_steps}")

    # ── REPA 组件 ─────────────────────────────────────────────────────────
    dino_model     = None
    repa_projector = None
    hook_layer_idx = None

    if args.repa:
        logger.info(
            "Initializing REPA alignment with DINOv2 ViT-B/14 teacher\n"
            f"  train_schedule : {args.repa_train_schedule} "
            f"(decay over {args.repa_schedule_steps} steps)\n"
            f"  diff_schedule  : {args.repa_diff_schedule} "
            f"(REPA-style timestep weighting)\n"
            f"  repa_lambda    : {args.repa_lambda}\n"
            f"  DINOv2 repo    : {args.dinov2_repo_dir}\n"
            f"  DINOv2 weight  : {args.dinov2_weight_path}"
        )

        # 实例化 DINOv2 teacher
        dino_model = LocalDINOv2Teacher(
            dinov2_repo_dir=args.dinov2_repo_dir,
            weight_path=args.dinov2_weight_path,
        ).to(device)
        dino_model.eval()
        requires_grad(dino_model, False)

        dit_hidden_dim = model.hidden_size
        dit_depth      = model.depth

        # hook 注入的 DiT block 索引（默认最后一层）
        hook_layer_idx = (
            args.repa_token_layer
            if args.repa_token_layer is not None
            else dit_depth - 1
        )

        # dummy forward 探测 DINO 输出维度
        with torch.inference_mode():
            dummy_rgb  = torch.zeros(
                1, 3, args.image_size, args.image_size, device=device
            )
            dummy_dino = preprocess_for_dino(dummy_rgb)
            dino_probe = dino_model(dummy_dino)
            dino_dim        = dino_probe.shape[-1]    # 768
            dino_num_tokens = dino_probe.shape[1]     # 256

        logger.info(
            f"REPA setup:\n"
            f"  DiT hidden_dim  : {dit_hidden_dim}\n"
            f"  DINO dim        : {dino_dim}\n"
            f"  DINO num_tokens : {dino_num_tokens}\n"
            f"  hook block      : [{hook_layer_idx} / {dit_depth - 1}]"
        )

        repa_projector = REPAProjector(
            in_dim=dit_hidden_dim,
            out_dim=dino_dim,
            hidden_dim=args.repa_hidden_dim,
        ).to(device)

        logger.info(
            f"  Projector params: "
            f"{sum(p.numel() for p in repa_projector.parameters()):,}"
        )

    # ── DDP wrap ──────────────────────────────────────────────────────────
    model = DDP(model, device_ids=[device])
    if args.repa:
        repa_projector = DDP(repa_projector, device_ids=[device])

    # ── 优化器 ────────────────────────────────────────────────────────────
    trainable_params = list(model.parameters())
    if args.repa:
        trainable_params += list(repa_projector.parameters())

    opt = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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
        repa_projector.train()
        dino_model.eval()

    # ── 日志累计变量 ──────────────────────────────────────────────────────
    train_steps                = 0
    log_steps                  = 0
    running_loss               = 0.0
    running_loss_diff          = 0.0
    running_loss_align         = 0.0
    running_train_align_weight = 0.0
    running_avg_diff_weight    = 0.0
    start_time                 = time()

    # 打印完整训练配置
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
            f"  diff_timestep  : {args.repa_diff_schedule} "
            f"(T={T})\n"
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
                x = vae.encode(images).latent_dist.sample().mul_(0.18215)

            # ── 训练步数衰减权重（训练阶段轴） ────────────────────────────
            # 支持 linear_decay / cosine_decay / constant / cutoff
            train_align_weight = 0.0
            if args.repa:
                train_align_weight = get_train_phase_weight(
                    current_step=current_step,
                    schedule_steps=args.repa_schedule_steps,
                    schedule=args.repa_train_schedule,
                )

            # ── 注册 forward hook（仅对齐期间） ──────────────────────────
            # 在 diffusion.training_losses 单次前向中捕获中间 token，
            # 避免重复推理，节省约 50% 显存。
            _captured_tokens = {}
            hook_handle      = None

            if args.repa and train_align_weight > 0:
                def _hook_fn(module, input, output):
                    # output: [B, N_tok, D]，保留梯度供 projector 反向传播
                    _captured_tokens['tokens'] = output

                hook_handle = (
                    model.module.blocks[hook_layer_idx]
                    .register_forward_hook(_hook_fn)
                )

            # ── 采样扩散时间步 + 单次前向 diffusion loss ─────────────────
            t = torch.randint(0, T, (x.shape[0],), device=device).long()
            model_kwargs = dict(y=y)

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss_diff = loss_dict["loss"].mean()

            # hook 触发后立即移除，避免影响后续步骤
            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None

            # ── REPA 对齐 loss（带扩散时间步加权） ───────────────────────
            loss_align         = torch.tensor(0.0, device=device)
            avg_diff_weight    = 0.0
            alignment_computed = False

            if (args.repa
                    and train_align_weight > 0
                    and 'tokens' in _captured_tokens):

                dit_tokens = _captured_tokens['tokens']    # [B, N_tok, D]

                # Teacher tokens（clean RGB → DINOv2，无梯度）
                x_dino = preprocess_for_dino(images)
                with torch.inference_mode():
                    # LocalDINOv2Teacher 内部自动 resize 到 224×224
                    dino_tokens = dino_model(x_dino)        # [B, 256, 768]

                # Projector：DiT token → DINOv2 特征空间
                proj_tokens = repa_projector(dit_tokens)    # [B, N_tok, 768]

                # Token 数量一致性校验
                if proj_tokens.shape[1] != dino_tokens.shape[1]:
                    raise ValueError(
                        f"Token count mismatch: "
                        f"proj={proj_tokens.shape}, "
                        f"dino={dino_tokens.shape}"
                    )

                # 带扩散时间步加权的余弦对齐 loss（与 REPA 原版一致）
                # w(t) = cos(π/2·t/T) [cosine模式]
                # loss = mean_b[ w(t[b]) · (1 - cos_sim(proj[b], dino[b])) ]
                loss_align = cosine_align_loss_weighted(
                    pred=proj_tokens,
                    target=dino_tokens,
                    t=t,
                    T=T,
                    diff_schedule=args.repa_diff_schedule,
                )

                # 记录本 batch 平均扩散时间步权重（仅用于日志监控）
                with torch.no_grad():
                    w = get_diff_timestep_weight(
                        t, T, args.repa_diff_schedule
                    )
                    avg_diff_weight = w.mean().item()

                alignment_computed = True

            # ── Total loss ────────────────────────────────────────────────
            # L = L_diff + λ_eff · L_align
            # λ_eff = repa_lambda × train_phase_weight(step)
            #   train_phase_weight：训练步数轴衰减（linear/cosine/cutoff）
            #   L_align 内部：扩散时间步轴加权（与 REPA 原版一致）
            effective_repa_lambda = args.repa_lambda * train_align_weight
            loss = loss_diff + effective_repa_lambda * loss_align

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # ── 日志累计 ──────────────────────────────────────────────────
            running_loss               += loss.item()
            running_loss_diff          += loss_diff.item()
            running_loss_align         += loss_align.item()
            running_train_align_weight += train_align_weight
            running_avg_diff_weight    += avg_diff_weight
            log_steps   += 1
            train_steps += 1

            # 第一步调试日志：打印各张量形状和时间步权重样例
            if train_steps == 1 and rank == 0:
                logger.info(f"  [step=1] images shape : {images.shape}")
                logger.info(f"  [step=1] latent shape : {x.shape}")
                logger.info(f"  [step=1] t (first 8)  : {t[:8].tolist()}")
                if alignment_computed:
                    logger.info(
                        f"  [step=1] dino_tokens  : {dino_tokens.shape}"
                    )
                    logger.info(
                        f"  [step=1] dit_tokens   : {dit_tokens.shape}"
                    )
                    logger.info(
                        f"  [step=1] proj_tokens  : {proj_tokens.shape}"
                    )
                    w8 = get_diff_timestep_weight(
                        t[:8], T, args.repa_diff_schedule
                    )
                    logger.info(
                        f"  [step=1] diff_weights (first 8): {w8.tolist()}"
                    )
                logger.info(
                    f"  [step=1] train_align_weight : {train_align_weight:.4f}"
                )
                logger.info(
                    f"  [step=1] effective_lambda   : {effective_repa_lambda:.6f}"
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
                avg_loss_diff          = _reduce_mean(running_loss_diff)
                avg_loss_align         = _reduce_mean(running_loss_align)
                avg_train_align_weight = _reduce_mean(
                    running_train_align_weight
                )
                avg_diff_w_log         = _reduce_mean(running_avg_diff_weight)

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Loss={avg_loss:.4f}  "
                    f"Loss_diff={avg_loss_diff:.4f}  "
                    f"Loss_align={avg_loss_align:.4f}  "
                    f"TrainW={avg_train_align_weight:.4f}  "
                    f"DiffW={avg_diff_w_log:.4f}  "
                    f"eff_λ="
                    f"{args.repa_lambda * avg_train_align_weight:.6f}  "
                    f"Steps/s={steps_per_sec:.2f}"
                )

                running_loss               = 0.0
                running_loss_diff          = 0.0
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
    if args.repa:
        repa_projector.eval()

    logger.info("Done!")
    cleanup()


#################################################################################
#                                  Entry Point                                  #
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "DiT + REPA training with DINOv2 ViT-B/14 teacher.\n"
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
        help="实验结果根目录。",
    )
    parser.add_argument(
        "--model", type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2",
        help="DiT 模型变体。",
    )
    parser.add_argument(
        "--image-size", type=int, choices=[256, 512], default=256,
        help="训练图像分辨率。",
    )
    parser.add_argument(
        "--num-classes", type=int, default=1000,
        help="类别数（ImageNet=1000）。",
    )
    parser.add_argument(
        "--epochs", type=int, default=1400,
        help="最大训练 epoch 数（被 max-steps 覆盖）。",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="达到该步数后停止训练（覆盖 epochs）。",
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=256,
        help="全局 batch size（所有 GPU 合计）。",
    )
    parser.add_argument(
        "--global-seed", type=int, default=0,
        help="全局随机种子。",
    )
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema",
        help="VAE 变体（HuggingFace Hub）。若设置 --vae-model-dir 则忽略。",
    )
    parser.add_argument(
        "--vae-model-dir", type=str, default=None,
        help="本地 HF 格式 VAE 目录。若设置则忽略 --vae。",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker 数量。",
    )
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="每隔多少步打印一次日志。",
    )
    parser.add_argument(
        "--ckpt-every", type=int, default=10_000,
        help="每隔多少步保存一次 checkpoint。",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="AdamW 学习率。",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW weight decay。",
    )

    # ── DINOv2 路径参数 ────────────────────────────────────────────────
    parser.add_argument(
        "--dinov2-repo-dir", type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/"
            "yuantianshuo/tmp/dinov2"
        ),
        help="DINOv2 源码根目录（含 hubconf.py）。",
    )
    parser.add_argument(
        "--dinov2-weight-path", type=str,
        default=(
            "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/"
            "tmp/dinov2/dinov2_weights/dinov2_vitb14_pretrain.pth"
        ),
        help="DINOv2 ViT-B/14 预训练权重 .pth 文件路径。",
    )

    # ── REPA 参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--repa", action="store_true",
        help="启用 REPA 表征对齐。",
    )
    parser.add_argument(
        "--repa-lambda", type=float, default=0.1,
        help="对齐 loss 的基础权重 λ。有效 λ = repa_lambda × train_phase_weight。",
    )
    parser.add_argument(
        "--repa-token-layer", type=int, default=None,
        help="用于提取中间 token 的 DiT block 索引。默认：最后一层。",
    )
    parser.add_argument(
        "--repa-hidden-dim", type=int, default=None,
        help="Projector MLP 隐层维度。默认：DiT hidden_size。",
    )
    parser.add_argument(
        "--repa-train-schedule", type=str,
        choices=["constant", "linear_decay", "cosine_decay", "cutoff"],
        default="cosine_decay",
        help=(
            "训练步数权重调度（训练阶段轴）:\n"
            "  constant     : w=1.0，全程固定\n"
            "  linear_decay : w 从 1→0 线性衰减\n"
            "  cosine_decay : w 从 1→0 余弦衰减（推荐）\n"
            "                 前期衰减慢（充分对齐），后期快速退出\n"
            "  cutoff       : 前N步 w=1，之后 w=0"
        ),
    )
    parser.add_argument(
        "--repa-schedule-steps", type=int, default=40_000,
        help=(
            "训练步数衰减周期 T（train_phase 轴）。\n"
            "linear_decay / cosine_decay 在此步数内从 1.0 衰减到 0.0。"
        ),
    )
    parser.add_argument(
        "--repa-diff-schedule", type=str,
        choices=["cosine", "linear_high", "linear_low", "uniform"],
        default="cosine",
        help=(
            "扩散时间步加权模式（与 REPA 原版 loss.py 对应）:\n"
            "  cosine      : w=cos(π/2·t/T)，t小权重大 [REPA原版默认]\n"
            "  linear_high : w=1-t/T，线性，低噪声权重高\n"
            "  linear_low  : w=t/T，高噪声权重高（对照实验）\n"
            "  uniform     : w=1.0，无加权（退化为均匀版本）"
        ),
    )

    args = parser.parse_args()

    # ── 路径合法性校验 ─────────────────────────────────────────────────
    if args.vae_model_dir is not None:
        if not os.path.isdir(args.vae_model_dir):
            raise FileNotFoundError(
                f"--vae-model-dir not found: {args.vae_model_dir}"
            )
    if args.repa:
        if not os.path.isdir(args.dinov2_repo_dir):
            raise FileNotFoundError(
                f"--dinov2-repo-dir not found: {args.dinov2_repo_dir}"
            )
        if not os.path.isfile(args.dinov2_weight_path):
            raise FileNotFoundError(
                f"--dinov2-weight-path not found: {args.dinov2_weight_path}"
            )

    main(args)
