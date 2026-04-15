# img2npz.py
import numpy as np
from PIL import Image
import os
import glob

sample_dir = "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_repa_dinov2_400k/005-DiT-XL-2-repa-dinov2-lam0.1-trainlinear_decay-diffcosine/checkpoints/10k_samples/0010000"
output_npz = "/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_repa_dinov2_400k/005-DiT-XL-2-repa-dinov2-lam0.1-trainlinear_decay-diffcosine/10k_samples.npz"

# 支持 png/jpg
img_paths = sorted(
    glob.glob(os.path.join(sample_dir, "*.png")) +
    glob.glob(os.path.join(sample_dir, "*.jpg"))
)
print(f"找到 {len(img_paths)} 张图片")

images = []
for p in img_paths:
    img = Image.open(p).convert("RGB")
    images.append(np.array(img))

images = np.stack(images, axis=0)  # shape: (N, H, W, 3)
print(f"图像数组 shape: {images.shape}")

np.savez(output_npz, arr_0=images)
print(f"已保存至 {output_npz}")
