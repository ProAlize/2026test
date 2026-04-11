import argparse
import contextlib
import io
import json
import math
import os
import pathlib
import shutil
import traceback
from time import strftime

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as TF
from PIL import Image
from diffusers.models import AutoencoderKL
from pytorch_fid.fid_score import (
    IMAGE_EXTENSIONS,
    calculate_frechet_distance,
)
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from diffusion import create_diffusion
from models_2 import DiT_models


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
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


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        return checkpoint["ema"], checkpoint.get("args")
    return checkpoint, None


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


def collect_image_files(path):
    """
    Recursively collect image files with case-insensitive extension matching.
    This avoids pytorch-fid's default behavior, which misses files like `.JPEG`.
    """
    root = pathlib.Path(path)
    files = sorted(
        str(file)
        for file in root.rglob("*")
        if file.is_file() and file.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )
    return files


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, image_size=None):
        self.files = files
        self.image_size = image_size
        self.to_tensor = TF.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        if self.image_size is not None:
            image = center_crop_arr(image, self.image_size)
        return self.to_tensor(image)


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, image_size=None):
    model.eval()

    if batch_size > len(files):
        print(
            "Warning: batch size is bigger than the data size. "
            "Setting batch size to data size"
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, image_size=image_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))
    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]

    return pred_arr


def calculate_activation_statistics(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1, image_size=None):
    act = get_activations(
        files,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
        image_size=image_size,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid_for_paths(sample_path, reference_path, batch_size, device, dims=2048, num_workers=1, image_size=None):
    """
    Compute FID directly through pytorch-fid library functions so we can
    support uppercase extensions and keep detailed stdout/stderr logs.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            inception = InceptionV3([block_idx]).to(device)
            inception.eval()
            requires_grad(inception, False)

            if sample_path.endswith(".npz"):
                with np.load(sample_path) as f:
                    mu_sample, sigma_sample = f["mu"][:], f["sigma"][:]
            else:
                sample_files = collect_image_files(sample_path)
                if not sample_files:
                    raise ValueError(f"No sample images found in: {sample_path}")
                mu_sample, sigma_sample = calculate_activation_statistics(
                    sample_files, inception, batch_size, dims, device, num_workers, image_size=image_size
                )

            if reference_path.endswith(".npz"):
                with np.load(reference_path) as f:
                    mu_ref, sigma_ref = f["mu"][:], f["sigma"][:]
            else:
                ref_files = collect_image_files(reference_path)
                if not ref_files:
                    raise ValueError(
                        f"No reference images found in: {reference_path}. "
                        "If this directory contains files like `.JPEG`, the old CLI-based "
                        "pytorch-fid path would miss them due to case-sensitive globbing."
                    )
                mu_ref, sigma_ref = calculate_activation_statistics(
                    ref_files, inception, batch_size, dims, device, num_workers, image_size=image_size
                )

            fid_value = float(calculate_frechet_distance(mu_sample, sigma_sample, mu_ref, sigma_ref))
        return 0, fid_value, stdout_buffer.getvalue(), stderr_buffer.getvalue()
    except Exception:
        return 1, None, stdout_buffer.getvalue(), stderr_buffer.getvalue() + traceback.format_exc()


def main(args):
    torch.set_grad_enabled(False)
    assert torch.cuda.is_available(), "Offline FID evaluation requires CUDA."
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    assert os.path.isdir(args.fid_ref_dir), f"Reference directory not found: {args.fid_ref_dir}"

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

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

    if rank == 0:
        os.makedirs(args.eval_root, exist_ok=True)
        if args.overwrite and os.path.isdir(args.sample_dir):
            shutil.rmtree(args.sample_dir, ignore_errors=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"[{strftime('%Y-%m-%d %H:%M:%S')}] Offline FID evaluation")
        print(f"Checkpoint: {args.ckpt}")
        print(f"Eval root: {args.eval_root}")
        print(f"Sample dir: {args.sample_dir}")
        print(f"Reference dir: {args.fid_ref_dir}")
        print(
            f"Sampling {args.fid_num_samples} images with "
            f"{args.num_sampling_steps} steps, cfg={args.cfg_scale}, world_size={world_size}"
        )
    dist.barrier(device_ids=[device])

    per_rank_batch = args.per_proc_batch_size
    global_batch_size = per_rank_batch * world_size
    total_samples = int(math.ceil(args.fid_num_samples / global_batch_size) * global_batch_size)
    samples_needed_this_rank = total_samples // world_size
    assert samples_needed_this_rank % per_rank_batch == 0, \
        "Per-rank sample count must be divisible by per-proc batch size."
    iterations = samples_needed_this_rank // per_rank_batch

    progress = tqdm(range(iterations), disable=(rank != 0))
    total_offset = 0
    for _ in progress:
        z = torch.randn(per_rank_batch, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (per_rank_batch,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.full((per_rank_batch,), num_classes, device=device, dtype=y.dtype)
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
            sample_index = i * world_size + rank + total_offset
            Image.fromarray(sample).save(os.path.join(args.sample_dir, f"{sample_index:06d}.png"))
        total_offset += global_batch_size

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

    if rank != 0:
        return

    returncode, fid_value, fid_stdout, fid_stderr = compute_fid_for_paths(
        sample_path=args.sample_dir,
        reference_path=args.fid_ref_dir,
        batch_size=args.fid_batch_size,
        device=f"cuda:{device}" if torch.cuda.is_available() else "cpu",
        image_size=image_size,
    )
    metrics = {
        "checkpoint": os.path.abspath(args.ckpt),
        "model": model_name,
        "image_size": image_size,
        "num_classes": num_classes,
        "fid_num_samples": args.fid_num_samples,
        "per_proc_batch_size": per_rank_batch,
        "fid_batch_size": args.fid_batch_size,
        "num_sampling_steps": args.num_sampling_steps,
        "cfg_scale": args.cfg_scale,
        "sample_dir": os.path.abspath(args.sample_dir),
        "fid_ref_dir": os.path.abspath(args.fid_ref_dir),
        "returncode": returncode,
        "fid": fid_value,
    }

    metrics_path = os.path.join(args.eval_root, "metrics.json")
    stdout_path = os.path.join(args.eval_root, "pytorch_fid_stdout.txt")
    stderr_path = os.path.join(args.eval_root, "pytorch_fid_stderr.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(fid_stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(fid_stderr)

    if returncode != 0:
        print(f"FID evaluation failed. See {stdout_path} and {stderr_path}.")
        raise SystemExit(returncode)

    if fid_value is None:
        print(f"FID finished but could not parse a numeric result. See {stdout_path}.")
    else:
        print(f"FID: {fid_value:.4f}")

    if not args.keep_samples:
        shutil.rmtree(args.sample_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a train_2.py checkpoint.")
    parser.add_argument("--eval-root", type=str, required=True, help="Directory to store samples and FID outputs.")
    parser.add_argument("--fid-ref-dir", type=str, required=True, help="Reference image directory for pytorch-fid.")
    parser.add_argument("--sample-dir", type=str, default=None, help="Optional override for the generated sample directory.")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default=None)
    parser.add_argument("--vae-model-dir", type=str, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=8)
    parser.add_argument("--fid-num-samples", type=int, default=5_000)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.sample_dir is None:
        args.sample_dir = os.path.join(args.eval_root, "samples")

    main(args)
