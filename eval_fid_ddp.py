import argparse
import hashlib
import json
import math
import os
import shutil
from time import strftime

import torch
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from pytorch_fid.fid_score import (
    InceptionV3,
    calculate_activation_statistics,
    calculate_frechet_distance,
)

from diffusion import create_diffusion
from models_2 import DiT_models


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "ema_state" in checkpoint:
            return checkpoint["ema_state"], checkpoint.get("args")
        if "ema" in checkpoint:
            return checkpoint["ema"], checkpoint.get("args")
        if "model_state" in checkpoint:
            return checkpoint["model_state"], checkpoint.get("args")
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"], checkpoint.get("args")
    return checkpoint, checkpoint.get("args") if isinstance(checkpoint, dict) else None


def infer_from_ckpt(cli_value, ckpt_args, field_name, default):
    if cli_value is not None:
        return cli_value
    if ckpt_args is not None and hasattr(ckpt_args, field_name):
        return getattr(ckpt_args, field_name)
    return default


def load_vae(vae_model_dir, vae_name, device, rank):
    vae_source = vae_model_dir or f"stabilityai/sd-vae-ft-{vae_name}"
    is_local_dir = os.path.isdir(vae_source)

    if rank == 0 and not is_local_dir:
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


def collect_image_files_case_insensitive(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm"}
    files = []
    for cur_root, _, filenames in os.walk(root_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(cur_root, name))
    files.sort()
    return files


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(file_list, root_dir):
    h = hashlib.sha256()
    root_abs = os.path.abspath(root_dir)
    for path in file_list:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, root_abs)
        size_bytes = os.path.getsize(abs_path)
        h.update(rel_path.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size_bytes).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def build_batch_from_indices(
    batch_indices,
    seed_base,
    in_channels,
    latent_size,
    num_classes,
    device,
):
    batch_size = len(batch_indices)
    z = torch.empty(batch_size, in_channels, latent_size, latent_size, device=device)
    y = torch.empty(batch_size, dtype=torch.long, device=device)
    for i, sample_index in enumerate(batch_indices):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed_base + sample_index))
        z[i] = torch.randn(in_channels, latent_size, latent_size, generator=g, device=device)
        y[i] = torch.randint(0, num_classes, (1,), generator=g, device=device).item()
    return z, y


def main(args):
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "Offline FID evaluation requires CUDA."
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.isdir(args.fid_ref_dir), f"Reference directory not found: {args.fid_ref_dir}"
    assert args.fid_num_samples > 0, "--fid-num-samples must be > 0"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed)

    state_dict, ckpt_args = load_checkpoint(args.ckpt)
    model_name = infer_from_ckpt(args.model, ckpt_args, "model", "DiT-XL/2")
    image_size = int(infer_from_ckpt(args.image_size, ckpt_args, "image_size", 256))
    num_classes = int(infer_from_ckpt(args.num_classes, ckpt_args, "num_classes", 1000))
    vae_name = infer_from_ckpt(args.vae, ckpt_args, "vae", "ema")

    latent_size = image_size // 8
    model = DiT_models[model_name](input_size=latent_size, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    requires_grad(model, False)

    vae = load_vae(args.vae_model_dir, vae_name, device, rank)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    using_cfg = args.cfg_scale > 1.0

    preflight_ok = torch.tensor(1, device=device, dtype=torch.int32)
    preflight_error = ""
    if rank == 0:
        try:
            os.makedirs(args.eval_root, exist_ok=True)
            if os.path.isdir(args.sample_dir):
                existing = collect_image_files_case_insensitive(args.sample_dir)
                if args.overwrite:
                    shutil.rmtree(args.sample_dir, ignore_errors=True)
                elif len(existing) > 0:
                    raise RuntimeError(
                        f"sample_dir is not empty and --overwrite is not set: {args.sample_dir}"
                    )
            os.makedirs(args.sample_dir, exist_ok=True)
        except Exception as exc:
            preflight_ok.fill_(0)
            preflight_error = str(exc)

    dist.broadcast(preflight_ok, src=0)
    if preflight_ok.item() == 0:
        dist.destroy_process_group()
        raise RuntimeError(preflight_error or "rank0 preflight failed")

    if rank == 0:
        print(f"[{strftime('%Y-%m-%d %H:%M:%S')}] Offline FID evaluation")
        print(f"Checkpoint: {args.ckpt}")
        print(f"Eval root: {args.eval_root}")
        print(f"Sample dir: {args.sample_dir}")
        print(f"Reference dir: {args.fid_ref_dir}")
        print(
            f"Sampling {args.fid_num_samples} images with "
            f"{args.num_sampling_steps} steps, cfg={args.cfg_scale}, world_size={world_size}"
        )
        print(f"Global seed: {args.global_seed}")
    dist.barrier(device_ids=[device])

    per_rank_batch = args.per_proc_batch_size
    global_batch_size = per_rank_batch * world_size
    total_samples = int(math.ceil(args.fid_num_samples / global_batch_size) * global_batch_size)
    rank_sample_indices = list(range(rank, total_samples, world_size))
    samples_needed_this_rank = len(rank_sample_indices)
    assert samples_needed_this_rank % per_rank_batch == 0, \
        "Per-rank sample count must be divisible by per-proc batch size."
    iterations = samples_needed_this_rank // per_rank_batch

    progress = tqdm(range(iterations), disable=(rank != 0))
    for iter_idx in progress:
        start = iter_idx * per_rank_batch
        end = start + per_rank_batch
        batch_indices = rank_sample_indices[start:end]
        z, y = build_batch_from_indices(
            batch_indices=batch_indices,
            seed_base=args.global_seed,
            in_channels=model.in_channels,
            latent_size=latent_size,
            num_classes=num_classes,
            device=device,
        )

        if using_cfg:
            z = torch.cat([z, z], dim=0)
            y_null = torch.full((len(batch_indices),), num_classes, device=device, dtype=y.dtype)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

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
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            sample_index = batch_indices[i]
            Image.fromarray(sample).save(os.path.join(args.sample_dir, f"{sample_index:06d}.png"))

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

    if rank != 0:
        return

    sample_files = collect_image_files_case_insensitive(args.sample_dir)
    ref_files = collect_image_files_case_insensitive(args.fid_ref_dir)

    if len(sample_files) == 0:
        raise RuntimeError(f"No generated images found in sample_dir: {args.sample_dir}")
    if len(ref_files) == 0:
        raise RuntimeError(f"No reference images found in fid_ref_dir: {args.fid_ref_dir}")
    if len(sample_files) < args.fid_num_samples:
        raise RuntimeError(
            f"Not enough generated samples. requested={args.fid_num_samples}, "
            f"generated={len(sample_files)}"
        )

    scored_sample_files = sample_files[:args.fid_num_samples]

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    fid_model = InceptionV3([block_idx]).to(device)
    fid_model.eval()

    m1, s1 = calculate_activation_statistics(
        scored_sample_files,
        fid_model,
        batch_size=args.fid_batch_size,
        dims=dims,
        device=f"cuda:{device}",
        num_workers=args.fid_num_workers,
    )
    m2, s2 = calculate_activation_statistics(
        ref_files,
        fid_model,
        batch_size=args.fid_batch_size,
        dims=dims,
        device=f"cuda:{device}",
        num_workers=args.fid_num_workers,
    )
    fid_value = float(calculate_frechet_distance(m1, s1, m2, s2))
    ckpt_sha256 = sha256_file(args.ckpt)
    sample_manifest = manifest_sha256(scored_sample_files, args.sample_dir)
    ref_manifest = manifest_sha256(ref_files, args.fid_ref_dir)

    metrics = {
        "checkpoint": os.path.abspath(args.ckpt),
        "checkpoint_sha256": ckpt_sha256,
        "model": model_name,
        "image_size": image_size,
        "num_classes": num_classes,
        "global_seed": args.global_seed,
        "world_size": world_size,
        "fid_num_samples": args.fid_num_samples,
        "requested_sample_count": args.fid_num_samples,
        "generated_sample_count": len(sample_files),
        "scored_sample_count": len(scored_sample_files),
        "per_proc_batch_size": per_rank_batch,
        "fid_batch_size": args.fid_batch_size,
        "num_sampling_steps": args.num_sampling_steps,
        "cfg_scale": args.cfg_scale,
        "sample_dir": os.path.abspath(args.sample_dir),
        "fid_ref_dir": os.path.abspath(args.fid_ref_dir),
        "sample_image_count": len(sample_files),
        "ref_image_count": len(ref_files),
        "sample_manifest_sha256": sample_manifest,
        "ref_manifest_sha256": ref_manifest,
        "fid_num_workers": args.fid_num_workers,
        "returncode": 0,
        "fid": fid_value,
    }

    metrics_path = os.path.join(args.eval_root, "metrics.json")
    stdout_path = os.path.join(args.eval_root, "fid_compute_stdout.txt")
    stderr_path = os.path.join(args.eval_root, "fid_compute_stderr.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(
            "FID computed in-process with case-insensitive file collection.\n"
            f"generated_samples={len(sample_files)}, scored_samples={len(scored_sample_files)}, "
            f"ref_images={len(ref_files)}\n"
            f"fid={fid_value}\n"
        )
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write("")

    print(f"FID: {fid_value:.4f}")

    if not args.keep_samples:
        shutil.rmtree(args.sample_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a DiT/S3A checkpoint.")
    parser.add_argument("--eval-root", type=str, required=True, help="Directory to store samples and FID outputs.")
    parser.add_argument("--fid-ref-dir", type=str, required=True, help="Reference image directory for pytorch-fid.")
    parser.add_argument("--sample-dir", type=str, default=None, help="Optional override for the generated sample directory.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default=None)
    parser.add_argument("--vae-model-dir", type=str, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=8)
    parser.add_argument("--fid-num-samples", type=int, default=50_000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--fid-num-workers", type=int, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.sample_dir is None:
        args.sample_dir = os.path.join(args.eval_root, "samples")

    main(args)
