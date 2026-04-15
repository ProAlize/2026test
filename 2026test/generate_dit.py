# generate_dit.py
"""
DiT-XL/2 模型的图像生成脚本（用于FID评估）
- 使用 EMA 权重
- 使用 DDPM diffusion 采样
- 支持 classifier-free guidance
- 生成 50000 张图像并保存为 .npz 格式
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 从你的项目导入
sys.path.insert(0, "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/2026test")
from model_sasa import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def create_npz_from_sample_folder(sample_dir: str, num: int = 50000) -> str:
    """把 sample_dir 下的 PNG 打包成 .npz"""
    sample_dir = Path(sample_dir)
    png_files  = sorted(sample_dir.glob("*.png"))[:num]

    print(f"[INFO] 共找到 {len(png_files)} 张 PNG，正在打包 .npz ...")
    samples = []
    for f in tqdm(png_files, desc="读取图像"):
        samples.append(np.array(Image.open(f)))

    samples  = np.stack(samples)              # (N, H, W, 3)
    npz_path = str(sample_dir / "samples.npz")
    np.savez(npz_path, arr_0=samples)
    print(f"[INFO] 已保存 {len(samples)} 张图像 → {npz_path}")
    return npz_path


# ─────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────

def main(args):
    assert torch.cuda.is_available(), "需要 GPU"

    # ── 分布式初始化 ────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + rank)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"[INFO] 样本输出目录: {args.sample_dir}")

    dist.barrier()

    # ── 加载 checkpoint ─────────────────────────────────────────
    if rank == 0:
        print(f"[INFO] 加载 checkpoint: {args.ckpt}")

    ckpt = torch.load(
        args.ckpt,
        map_location=f"cuda:{device}",
        weights_only=False,
    )

    # 优先使用 EMA 权重
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
        if rank == 0:
            print("[INFO] 使用 EMA 权重")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        if rank == 0:
            print("[INFO] 使用 model 权重（未找到 EMA）")
    else:
        state_dict = ckpt
        if rank == 0:
            print("[INFO] 直接使用 checkpoint（无 model/ema key）")

    # ── 构建模型 ────────────────────────────────────────────────
    latent_size = args.image_size // 8          # 256 // 8 = 32
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    # 严格加载（若 key 不匹配会报错，便于排查）
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if rank == 0:
        print(f"[INFO] 模型加载完成 | missing={len(missing)} | unexpected={len(unexpected)}")

    model.eval()

    # ── VAE ─────────────────────────────────────────────────────
    if args.vae_model_dir:
        vae = AutoencoderKL.from_pretrained(args.vae_model_dir).to(device)
        if rank == 0:
            print(f"[INFO] 加载本地 VAE: {args.vae_model_dir}")
    else:
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
        ).to(device)
        if rank == 0:
            print(f"[INFO] 加载 HuggingFace VAE: stabilityai/sd-vae-ft-{args.vae}")
    vae.eval()

    # ── Diffusion ───────────────────────────────────────────────
    diffusion = create_diffusion(
        str(args.num_sampling_steps)   # timestep_respacing="250" 等
    )

    # ── 采样数量规划 ────────────────────────────────────────────
    n                  = args.per_proc_batch_size
    global_batch_size  = n * world_size
    # 向上取整到 global_batch_size 的整数倍
    total_samples      = int(
        np.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    )
    iterations         = total_samples // global_batch_size

    if rank == 0:
        print(f"[INFO] 目标样本数: {args.num_fid_samples}")
        print(f"[INFO] 实际生成数: {total_samples}")
        print(f"[INFO] 每 GPU batch: {n}  |  GPU 数: {world_size}  |  迭代: {iterations}")

    # ── 采样循环 ────────────────────────────────────────────────
    pbar         = range(iterations)
    pbar         = tqdm(pbar, desc="生成中") if rank == 0 else pbar
    saved_count  = 0                          # 当前 rank 已保存的样本数

    for i in pbar:
        # 随机类别标签 0~999
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # CFG: 拼接 unconditional（类别 = num_classes）
        y_null    = torch.full_like(y, args.num_classes)
        y_combined = torch.cat([y, y_null], dim=0)          # (2n,)

        # 初始噪声
        z = torch.randn(
            n, 4, latent_size, latent_size,
            device=device,
        )
        z_combined = torch.cat([z, z], dim=0)               # (2n, 4, 32, 32)

        # model_kwargs
        model_kwargs = dict(
            y=y_combined,
            cfg_scale=args.cfg_scale,
        )

        # DDPM p_sample_loop（内置 CFG via forward_with_cfg）
        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,         # 使用 CFG forward
                z_combined.shape,
                noise=z_combined,               # CFG 两路必须共享初始噪声
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
            )
            # p_sample_loop 返回 (2n, C, H, W)，取前 n 个（conditional 部分）
            samples = samples[:n]

            # VAE 解码: latent → 像素空间
            samples = vae.decode(samples / 0.18215).sample  # (n, 3, 256, 256)

        # 转换到 uint8
        samples = (samples * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()  # (n, H, W, 3)

        # 保存 PNG
        for j, img_arr in enumerate(samples):
            # 全局唯一 index = 迭代 i × global_batch + 本 rank 偏移 + j
            global_idx = i * global_batch_size + rank * n + j
            if global_idx >= args.num_fid_samples:
                break
            save_path = os.path.join(args.sample_dir, f"{global_idx:06d}.png")
            Image.fromarray(img_arr).save(save_path)
            saved_count += 1

    dist.barrier()

    # ── rank 0 打包 .npz ────────────────────────────────────────
    if rank == 0:
        npz_path = create_npz_from_sample_folder(
            args.sample_dir,
            num=args.num_fid_samples,
        )
        print(f"\n[DONE] .npz 文件已保存: {npz_path}")
        print(f"[DONE] 使用以下命令计算 FID:")
        print(f"  python evaluator.py <ref_stats.npz> {npz_path}")

    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT-XL/2 图像生成（FID评估）")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/"
                "dit_xl_sasa_stepdecay_80k/010-DiT-XL-2-sasa-lam0.1-trainlinear_decay-nodiffweight/"
                "checkpoints/0080000.pt",
        help="checkpoint 文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DiT-XL/2",
        choices=list(DiT_models.keys()),
        help="模型架构",
    )
    parser.add_argument(
        "--vae-model-dir",
        type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/models/DiT-XL-2-256/vae",
        help="本地 VAE 目录（HuggingFace 格式）",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="ema",
        choices=["ema", "mse"],
        help="HuggingFace VAE 变体（vae-model-dir 为空时使用）",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/"
                "dit_xl_sasa_stepdecay_80k/010-DiT-XL-2-sasa-lam0.1-trainlinear_decay-nodiffweight/"
                "fid_samples/0080000",
        help="生成图像的保存目录",
    )
    parser.add_argument("--image-size",         type=int,   default=256)
    parser.add_argument("--num-classes",        type=int,   default=1000)
    parser.add_argument("--num-fid-samples",    type=int,   default=50000)
    parser.add_argument("--per-proc-batch-size",type=int,   default=64)
    parser.add_argument("--num-sampling-steps", type=int,   default=250)
    parser.add_argument("--cfg-scale",          type=float, default=1.5)
    parser.add_argument("--global-seed",        type=int,   default=0)

    args = parser.parse_args()
    main(args)
