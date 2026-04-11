import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from pytorch_fid.fid_score import IMAGE_EXTENSIONS, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


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


def collect_image_files(path):
    root = os.path.abspath(path)
    files = sorted(
        os.path.join(dp, fn)
        for dp, _, fns in os.walk(root)
        for fn in fns
        if os.path.splitext(fn)[1].lower().lstrip(".") in IMAGE_EXTENSIONS
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


def get_mu_sigma(path, model, batch_size, dims, device, num_workers, image_size):
    if path.endswith(".npz"):
        with np.load(path) as f:
            mu, sigma = f["mu"][:], f["sigma"][:]
        return mu, sigma

    files = collect_image_files(path)
    if not files:
        raise ValueError(f"No image files found in: {path}")
    return calculate_activation_statistics(
        files,
        model=model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
        image_size=image_size,
    )


def build_inception(dims, device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()
    for p in inception.parameters():
        p.requires_grad = False
    return inception


def cmd_stats(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    inception = build_inception(args.dims, device)
    files = collect_image_files(args.input)
    if not files:
        raise ValueError(f"No image files found in: {args.input}")
    mu, sigma = calculate_activation_statistics(
        files,
        inception,
        batch_size=args.batch_size,
        dims=args.dims,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    np.savez(args.output, mu=mu, sigma=sigma)
    print(f"[{datetime.now().strftime('%F %T')}] Saved stats: {args.output}")
    print(f"images={len(files)} dims={args.dims} device={device}")


def cmd_fid(args):
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    inception = build_inception(args.dims, device)
    mu_s, sigma_s = get_mu_sigma(
        args.samples,
        inception,
        batch_size=args.batch_size,
        dims=args.dims,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    mu_r, sigma_r = get_mu_sigma(
        args.ref,
        inception,
        batch_size=args.batch_size,
        dims=args.dims,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    fid = float(calculate_frechet_distance(mu_s, sigma_s, mu_r, sigma_r))
    metrics = {
        "timestamp": datetime.now().strftime("%F %T"),
        "samples": os.path.abspath(args.samples),
        "ref": os.path.abspath(args.ref),
        "dims": args.dims,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "device": device,
        "fid": fid,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    print(f"FID: {fid:.6f}")
    if args.output_json:
        print(f"Saved metrics: {args.output_json}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    parser_stats = subparsers.add_parser("stats")
    parser_stats.add_argument("--input", type=str, required=True)
    parser_stats.add_argument("--output", type=str, required=True)
    parser_stats.add_argument("--device", type=str, default="cuda:0")
    parser_stats.add_argument("--batch-size", type=int, default=32)
    parser_stats.add_argument("--num-workers", type=int, default=8)
    parser_stats.add_argument("--dims", type=int, choices=[64, 192, 768, 2048], default=2048)
    parser_stats.add_argument("--image-size", type=int, default=256)
    parser_stats.set_defaults(func=cmd_stats)

    parser_fid = subparsers.add_parser("fid")
    parser_fid.add_argument("--samples", type=str, required=True)
    parser_fid.add_argument("--ref", type=str, required=True)
    parser_fid.add_argument("--output-json", type=str, default=None)
    parser_fid.add_argument("--device", type=str, default="cuda:0")
    parser_fid.add_argument("--batch-size", type=int, default=32)
    parser_fid.add_argument("--num-workers", type=int, default=8)
    parser_fid.add_argument("--dims", type=int, choices=[64, 192, 768, 2048], default=2048)
    parser_fid.add_argument("--image-size", type=int, default=256)
    parser_fid.set_defaults(func=cmd_fid)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
