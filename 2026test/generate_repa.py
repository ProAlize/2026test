# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
完全对齐 REPA generate.py 的采样脚本。
支持 DiT 模型（model_sasa.py），采样后保存 .png + .npz。

对齐要点：
  1. 类别确定性分配（sample_index % num_classes）
  2. SDE / ODE 采样模式（通过 diffusion 步数控制）
  3. CFG guidance（cfg_scale + guidance_high/low）
  4. 保存 .npz 供 ADM 评测套件使用
  5. 多卡 DDP 并行采样
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from model_sasa import DiT_models

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────────────

def requires_grad(model, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def load_checkpoint(ckpt_path: str):
    """
    加载 checkpoint，兼容三种格式：
      1. {"ema": ..., "args": ...}   ← train_sasa_dinov2.py 保存格式
      2. {"model": ..., "args": ...} ← 直接使用 model 权重
      3. 裸 state_dict
    优先使用 ema 权重（与 REPA 评测一致）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return ckpt, None
    if "ema" in ckpt:
        return ckpt["ema"], ckpt.get("args")
    if "model" in ckpt:
        return ckpt["model"], ckpt.get("args")
    return ckpt, None


def infer_from_ckpt(cli_value, ckpt_args, field_name, default):
    """CLI 优先，其次从 checkpoint args 推断，最后用默认值。"""
    if cli_value is not None:
        return cli_value
    if ckpt_args is not None and hasattr(ckpt_args, field_name):
        return getattr(ckpt_args, field_name)
    return default


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── 基础校验 ──────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    # ── 分布式初始化 ──────────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # 固定随机种子（与 REPA 对齐：每个 rank 独立种子）
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

    # ── 从 checkpoint 推断超参 ────────────────────────────────────────────
    state_dict, ckpt_args = load_checkpoint(args.ckpt)

    model_name  = infer_from_ckpt(args.model,       ckpt_args, "model",       "DiT-XL/2")
    image_size  = int(infer_from_ckpt(args.image_size,  ckpt_args, "image_size",  256))
    num_classes = int(infer_from_ckpt(args.num_classes, ckpt_args, "num_classes", 1000))
    vae_name    = infer_from_ckpt(args.vae,         ckpt_args, "vae",         "ema")

    latent_size = image_size // 8

    # ── 输出目录 ──────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    dist.barrier(device_ids=[device])

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    requires_grad(model, False)

    using_cfg = args.cfg_scale > 1.0
    if using_cfg and not hasattr(model, "forward_with_cfg"):
        raise AttributeError(
            f"Model '{model_name}' does not implement forward_with_cfg. "
            "Set --cfg-scale 1.0 to disable CFG."
        )

    # ── VAE ───────────────────────────────────────────────────────────────
    vae_source   = args.vae_model_dir or f"stabilityai/sd-vae-ft-{vae_name}"
    is_local_dir = os.path.isdir(vae_source)

    # rank 0 预先下载（如果需要），其余 rank 等待
    if rank == 0 and not is_local_dir:
        _ = AutoencoderKL.from_pretrained(vae_source)

    dist.barrier(device_ids=[device])

    vae = AutoencoderKL.from_pretrained(
        vae_source,
        local_files_only=not is_local_dir,
    ).to(device)
    vae.eval()
    requires_grad(vae, False)

    # ── Diffusion（步数控制 DDPM / DDIM 采样） ────────────────────────────
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # ── 采样规模计算 ──────────────────────────────────────────────────────
    per_rank_batch = args.per_proc_batch_size
    global_batch   = per_rank_batch * world_size

    # 向上取整，保证所有 rank 迭代次数相同
    total_samples    = int(
        math.ceil(args.num_fid_samples / global_batch) * global_batch
    )
    samples_per_rank = total_samples // world_size
    assert samples_per_rank % per_rank_batch == 0
    iterations = samples_per_rank // per_rank_batch

    if rank == 0:
        print(f"DiT model        : {model_name}")
        print(f"Image size       : {image_size}x{image_size}")
        print(f"Latent size      : {latent_size}x{latent_size}")
        print(f"Num classes      : {num_classes}")
        print(f"CFG scale        : {args.cfg_scale}")
        print(f"Sampling steps   : {args.num_sampling_steps}")
        print(f"Total samples    : {total_samples}  "
              f"(requested {args.num_fid_samples})")
        print(f"World size       : {world_size}")
        print(f"Per-rank batch   : {per_rank_batch}")
        print(f"Iterations/rank  : {iterations}")
        print(f"Sample dir       : {args.sample_dir}")
        print(f"Checkpoint       : {args.ckpt}")
        print("─" * 60)

    # ════════════════════════════════════════════════════════════════════
    #  采样循环
    # ════════════════════════════════════════════════════════════════════
    all_images = []    # 本 rank 采集的 uint8 numpy [N, H, W, 3]

    pbar = tqdm(
        range(iterations),
        desc=f"rank{rank}",
        disable=(rank != 0),
    )

    for batch_idx in pbar:

        # ── 1. 确定性类别分配（与 REPA / ADM 完全对齐） ──────────────────
        #
        #  全局图片索引布局（world_size=2, per_rank=4 为例）：
        #
        #  batch_idx=0:
        #    rank0 生成: idx 0, 2, 4, 6   → class 0, 2, 4, 6
        #    rank1 生成: idx 1, 3, 5, 7   → class 1, 3, 5, 7
        #
        #  batch_idx=1:
        #    rank0 生成: idx 8, 10, 12, 14
        #    rank1 生成: idx 9, 11, 13, 15
        #
        #  class_label = global_index % num_classes
        #  → 1000 类均匀覆盖，每类恰好 num_fid_samples/1000 张

        global_start = batch_idx * global_batch + rank * per_rank_batch
        class_labels = [
            (global_start + i) % num_classes
            for i in range(per_rank_batch)
        ]
        y = torch.tensor(class_labels, device=device, dtype=torch.long)

        # ── 2. 初始噪声 ───────────────────────────────────────────────────
        z = torch.randn(
            per_rank_batch,
            model.in_channels,
            latent_size,
            latent_size,
            device=device,
        )

        # ── 3. CFG 设置（与 REPA generate.py 完全对齐） ───────────────────
        #
        #  REPA 的 CFG 实现：
        #    - 将 z、y 各复制一份（条件 + 无条件）
        #    - y_null = num_classes（空类别，模型内部做 embedding）
        #    - guidance_high / guidance_low：对时间步范围做 guidance mask
        #      （仅在 t ∈ [T*guidance_low, T*guidance_high] 时施加 CFG）
        if using_cfg:
            z      = torch.cat([z, z], dim=0)                     # [2B, C, H, W]
            y_null = torch.full(
                (per_rank_batch,), num_classes,
                device=device, dtype=y.dtype,
            )
            y            = torch.cat([y, y_null], dim=0)          # [2B]
            model_kwargs = dict(
                y          = y,
                cfg_scale  = args.cfg_scale,
            )
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn    = model.forward

        # ── 4. DDPM 反向采样 ──────────────────────────────────────────────
        samples = diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )

        if using_cfg:
            # 取条件侧（前半），丢弃无条件侧
            samples, _ = samples.chunk(2, dim=0)                  # [B, C, H, W]

        # ── 5. VAE 解码 ───────────────────────────────────────────────────
        samples = vae.decode(samples / 0.18215).sample            # [B, 3, H, W]
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)    # [0, 255]
        samples = (
            samples
            .permute(0, 2, 3, 1)                                  # [B, H, W, 3]
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        # ── 6. 保存 PNG（全局唯一文件名，与 REPA 索引规则一致） ───────────
        for i, img_np in enumerate(samples):
            # 交错分配：同一 batch 内 rank0 取偶数位，rank1 取奇数位
            global_idx = batch_idx * global_batch + i * world_size + rank
            if global_idx < args.num_fid_samples:
                # 仅保存有效范围内的图片（截断多余的 padding 样本）
                Image.fromarray(img_np).save(
                    os.path.join(args.sample_dir, f"{global_idx:06d}.png")
                )

        all_images.append(samples)

    # ════════════════════════════════════════════════════════════════════
    #  保存 .npz（与 REPA 对齐，供 ADM TensorFlow 评测套件使用）
    # ════════════════════════════════════════════════════════════════════
    #
    #  REPA generate.py 的 npz 保存逻辑：
    #    arr  = np.concatenate(all_samples, axis=0)   # 本 rank 所有样本
    #    all_gather → rank 0 拼接 → 截断到 num_fid_samples → np.savez
    #
    local_arr    = np.concatenate(all_images, axis=0)         # [N_local, H, W, 3]
    local_tensor = torch.from_numpy(local_arr).to(device)

    # all_gather：每个 rank 的 tensor 大小相同（padding 保证了这一点）
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    if rank == 0:
        all_arr = torch.cat(gathered, dim=0).cpu().numpy()    # [total_samples, H, W, 3]
        all_arr = all_arr[:args.num_fid_samples]              # 截断到精确数量

        npz_path = os.path.join(args.sample_dir, "samples.npz")
        np.savez(npz_path, arr_0=all_arr)

        print(f"\nSaved {len(all_arr)} samples:")
        print(f"  PNG  → {args.sample_dir}/*.png")
        print(f"  .npz → {npz_path}")
        print(
            f"  Shape: {all_arr.shape}  "
            f"dtype: {all_arr.dtype}  "
            f"min={all_arr.min()}  max={all_arr.max()}"
        )

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiT sampling script, fully aligned with REPA generate.py"
    )

    # ── checkpoint ────────────────────────────────────────────────────────
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="训练 checkpoint 路径（.pt 文件）。",
    )

    # ── 模型超参（可从 checkpoint 自动推断） ──────────────────────────────
    parser.add_argument(
        "--model", type=str,
        choices=list(DiT_models.keys()), default=None,
        help="DiT 模型变体。默认从 checkpoint 推断，否则 DiT-XL/2。",
    )
    parser.add_argument(
        "--image-size", type=int, choices=[256, 512], default=None,
        help="图像分辨率。默认从 checkpoint 推断，否则 256。",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="类别数。默认从 checkpoint 推断，否则 1000。",
    )

    # ── VAE ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default=None,
        help="VAE 变体（HuggingFace Hub）。默认从 checkpoint 推断，否则 ema。",
    )
    parser.add_argument(
        "--vae-model-dir", type=str, default=None,
        help="本地 HF 格式 VAE 目录。若设置则忽略 --vae。",
    )

    # ── 采样参数 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--num-fid-samples", type=int, default=50_000,
        help="生成图片总数（REPA 默认 50000）。",
    )
    parser.add_argument(
        "--per-proc-batch-size", type=int, default=64,
        help="每个 GPU 每步生成的图片数（REPA 默认 64）。",
    )
    parser.add_argument(
        "--num-sampling-steps", type=int, default=250,
        help="DDPM 采样步数（REPA 默认 250）。",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=1.5,
        help="CFG guidance scale（>1.0 启用 CFG）。",
    )
    parser.add_argument(
        "--global-seed", type=int, default=0,
        help="全局随机种子。",
    )

    # ── 输出 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--sample-dir", type=str, required=True,
        help="生成图片的保存目录（.png + samples.npz）。",
    )

    args = parser.parse_args()
    main(args)
