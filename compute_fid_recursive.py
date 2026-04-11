#!/usr/bin/env python3
import argparse
import json
import pathlib

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from pytorch_fid.fid_score import IMAGE_EXTENSIONS, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
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


def collect_image_files(path: str) -> list[str]:
    root = pathlib.Path(path)
    return sorted(
        str(file)
        for file in root.rglob("*")
        if file.is_file() and file.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )


class ImagePathDataset(Dataset):
    def __init__(self, files: list[str], image_size: int | None = None):
        self.files = files
        self.image_size = image_size
        self.to_tensor = TF.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.files[index]).convert("RGB")
        if self.image_size is not None:
            image = center_crop_arr(image, self.image_size)
        return self.to_tensor(image)


def get_activations(
    files: list[str],
    model: InceptionV3,
    device: str,
    dims: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int | None = None,
) -> np.ndarray:
    dataset = ImagePathDataset(files, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(files)),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(dataloader, desc="Extracting Inception features"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]
    return pred_arr


def calculate_stats(
    files: list[str],
    model: InceptionV3,
    device: str,
    dims: int,
    batch_size: int,
    num_workers: int,
    image_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    act = get_activations(
        files,
        model,
        device=device,
        dims=dims,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    return np.mean(act, axis=0), np.cov(act, rowvar=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive FID (case-insensitive extensions).")
    parser.add_argument("--sample-dir", type=str, required=True)
    parser.add_argument("--ref-dir", type=str, required=True)
    parser.add_argument("--ref-stats", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_files = collect_image_files(args.sample_dir)
    if not sample_files:
        raise RuntimeError(f"No sample images found in: {args.sample_dir}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    inception = InceptionV3([block_idx]).to(args.device)
    inception.eval()
    for p in inception.parameters():
        p.requires_grad = False

    mu_sample, sigma_sample = calculate_stats(
        sample_files,
        inception,
        device=args.device,
        dims=args.dims,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    ref_stats_path = pathlib.Path(args.ref_stats) if args.ref_stats else None
    ref_files: list[str] | None = None
    if ref_stats_path and ref_stats_path.is_file():
        cached = np.load(ref_stats_path)
        mu_ref = cached["mu"]
        sigma_ref = cached["sigma"]
        num_ref_images = int(cached["num_ref_images"]) if "num_ref_images" in cached else -1
    else:
        ref_files = collect_image_files(args.ref_dir)
        if not ref_files:
            raise RuntimeError(f"No reference images found in: {args.ref_dir}")
        mu_ref, sigma_ref = calculate_stats(
            ref_files,
            inception,
            device=args.device,
            dims=args.dims,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
        num_ref_images = len(ref_files)
        if ref_stats_path:
            ref_stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                ref_stats_path,
                mu=mu_ref,
                sigma=sigma_ref,
                num_ref_images=num_ref_images,
                ref_dir=str(pathlib.Path(args.ref_dir).resolve()),
                image_size=-1 if args.image_size is None else int(args.image_size),
                dims=int(args.dims),
            )

    fid = float(calculate_frechet_distance(mu_sample, sigma_sample, mu_ref, sigma_ref))
    result = {
        "fid": fid,
        "sample_dir": str(pathlib.Path(args.sample_dir).resolve()),
        "ref_dir": str(pathlib.Path(args.ref_dir).resolve()),
        "num_sample_images": len(sample_files),
        "num_ref_images": num_ref_images,
        "ref_stats": str(ref_stats_path.resolve()) if ref_stats_path else None,
        "batch_size": args.batch_size,
        "dims": args.dims,
        "device": args.device,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = pathlib.Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
