#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


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


def collect_image_files(root: pathlib.Path) -> list[pathlib.Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ADM-style .npz (arr_0: NHWC uint8) from image folder."
    )
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--out-npz", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--allow-fewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = pathlib.Path(args.image_dir).resolve()
    out_npz = pathlib.Path(args.out_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = collect_image_files(image_dir)
    if not files:
        raise RuntimeError(f"No images found under: {image_dir}")

    if len(files) < args.num_images and not args.allow_fewer:
        raise RuntimeError(
            f"Found only {len(files)} images under {image_dir}, fewer than requested {args.num_images}. "
            "Use --allow-fewer to proceed."
        )

    target_n = min(args.num_images, len(files))
    selected = files[:target_n]
    samples = np.empty((target_n, args.image_size, args.image_size, 3), dtype=np.uint8)

    for i, path in enumerate(tqdm(selected, desc="Building arr_0")):
        img = Image.open(path).convert("RGB")
        img = center_crop_arr(img, args.image_size)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape != (args.image_size, args.image_size, 3):
            raise RuntimeError(f"Unexpected image shape at {path}: {arr.shape}")
        samples[i] = arr

    np.savez(out_npz, arr_0=samples)
    print(f"Saved: {out_npz}")
    print(f"arr_0 shape: {samples.shape}, dtype: {samples.dtype}")


if __name__ == "__main__":
    main()
