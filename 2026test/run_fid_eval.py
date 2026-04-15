# run_fid_eval.py
"""
针对DiT-XL/2模型计算FID的完整流程脚本
支持自动遍历checkpoint目录中的所有权重文件
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path


def find_checkpoints(ckpt_dir, pattern="*.pt"):
    """查找目录中所有checkpoint文件"""
    ckpt_dir = Path(ckpt_dir)
    checkpoints = []
    
    # 支持多种格式
    for pattern in ["*.pt", "*.pth", "checkpoint*.pt", "model*.pt"]:
        found = sorted(glob.glob(str(ckpt_dir / pattern)))
        checkpoints.extend(found)
    
    # 去重并排序
    checkpoints = sorted(list(set(checkpoints)))
    return checkpoints


def generate_samples(
    ckpt_path,
    output_dir,
    model="DiT-XL/2",
    num_fid_samples=50000,
    per_proc_batch_size=64,
    num_gpus=8,
    cfg_scale=1.5,
    num_steps=250,
    seed=0,
    image_size=256,
    num_classes=1000,
):
    """调用generate脚本生成样本"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成的npz文件路径
    npz_path = output_dir / "samples.npz"
    
    if npz_path.exists():
        print(f"[INFO] 已存在样本文件: {npz_path}，跳过生成步骤")
        return str(npz_path)
    
    print(f"[INFO] 开始生成样本，使用checkpoint: {ckpt_path}")
    print(f"[INFO] 输出目录: {output_dir}")
    
    # 构建生成命令 - 适配DiT的generate.py
    cmd = [
        "torchrun",
        f"--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "generate.py",  # 根据实际情况修改generate脚本路径
        "--model", model,
        "--num-fid-samples", str(num_fid_samples),
        "--ckpt", ckpt_path,
        "--per-proc-batch-size", str(per_proc_batch_size),
        "--num-sampling-steps", str(num_steps),
        "--cfg-scale", str(cfg_scale),
        "--seed", str(seed),
        "--image-size", str(image_size),
        "--num-classes", str(num_classes),
        "--sample-dir", str(output_dir),
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] 生成失败，返回码: {result.returncode}")
        return None
    
    # 查找生成的npz文件
    npz_files = list(output_dir.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] 未找到生成的npz文件")
        return None
    
    return str(npz_files[0])


def compute_fid(
    sample_npz_path,
    ref_batch_path=None,
    ref_stats_path=None,
):
    """
    使用ADM evaluator计算FID
    需要先安装: pip install git+https://github.com/openai/guided-diffusion
    """
    
    if ref_stats_path is None and ref_batch_path is None:
        print("[ERROR] 需要提供参考数据集统计信息或参考图像")
        return None
    
    if ref_stats_path:
        # 使用预计算的统计数据（推荐）
        cmd = [
            "python", "-m", "pytorch_fid",
            sample_npz_path,
            ref_stats_path,
            "--device", "cuda",
        ]
    else:
        cmd = [
            "python", "-m", "pytorch_fid",
            sample_npz_path,
            ref_batch_path,
            "--device", "cuda",
        ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.stdout


def compute_fid_adm(sample_npz_path, ref_npz_path):
    """使用OpenAI ADM evaluator计算FID（与REPA论文一致）"""
    
    # ADM evaluator脚本路径，需要从guided-diffusion下载
    evaluator_script = "evaluations/evaluator.py"
    
    if not os.path.exists(evaluator_script):
        print(f"[WARNING] 未找到ADM evaluator: {evaluator_script}")
        print("[INFO] 请从以下地址下载: https://github.com/openai/guided-diffusion/tree/main/evaluations")
        print("[INFO] 或使用pytorch-fid作为替代")
        return None
    
    cmd = [
        "python", evaluator_script,
        ref_npz_path,       # 参考数据统计
        sample_npz_path,    # 生成样本
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="计算DiT模型的FID分数")
    
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/mnt/tidal-alsh01/dataset/redaigc/yuantianshuo/2026/results/dit_xl_sasa_stepdecay_80k/010-DiT-XL-2-sasa-lam0.1-trainlinear_decay-nodiffweight",
        help="checkpoint目录路径",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="指定单个checkpoint文件路径（如果指定则忽略--ckpt-dir的自动搜索）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="生成样本的输出目录，默认在ckpt-dir下创建fid_results文件夹",
    )
    parser.add_argument(
        "--ref-stats",
        type=str,
        default=None,
        help="参考数据集统计文件路径(.npz格式，ImageNet预计算的统计数据)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DiT-XL/2",
        help="模型架构名称",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        choices=[256, 512],
        help="生成图像分辨率",
    )
    parser.add_argument(
        "--num-fid-samples",
        type=int,
        default=50000,
        help="用于FID计算的生成样本数量",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="classifier-free guidance的scale",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=250,
        help="采样步数",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="使用的GPU数量",
    )
    parser.add_argument(
        "--per-proc-batch-size",
        type=int,
        default=64,
        help="每个GPU的batch size",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="评估目录中所有checkpoint",
    )
    parser.add_argument(
        "--generate-script",
        type=str,
        default="generate.py",
        help="生成脚本路径",
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.ckpt_dir, "fid_results")
    
    # 确定要评估的checkpoint
    if args.ckpt_path:
        checkpoints = [args.ckpt_path]
    elif args.eval_all:
        checkpoints = find_checkpoints(args.ckpt_dir)
        if not checkpoints:
            print(f"[ERROR] 在 {args.ckpt_dir} 中未找到checkpoint文件")
            sys.exit(1)
        print(f"[INFO] 找到 {len(checkpoints)} 个checkpoint文件:")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    else:
        # 使用最新的checkpoint
        checkpoints = find_checkpoints(args.ckpt_dir)
        if not checkpoints:
            print(f"[ERROR] 在 {args.ckpt_dir} 中未找到checkpoint文件")
            sys.exit(1)
        checkpoints = [checkpoints[-1]]  # 取最新的
        print(f"[INFO] 使用最新checkpoint: {checkpoints[0]}")
    
    # 对每个checkpoint计算FID
    results = {}
    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).stem
        sample_output_dir = os.path.join(args.output_dir, ckpt_name)
        
        print(f"\n{'='*60}")
        print(f"[INFO] 处理checkpoint: {ckpt_name}")
        print(f"{'='*60}")
        
        # 生成样本
        npz_path = generate_samples(
            ckpt_path=ckpt_path,
            output_dir=sample_output_dir,
            model=args.model,
            num_fid_samples=args.num_fid_samples,
            per_proc_batch_size=args.per_proc_batch_size,
            num_gpus=args.num_gpus,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            image_size=args.image_size,
        )
        
        if npz_path is None:
            print(f"[ERROR] 生成样本失败，跳过 {ckpt_name}")
            continue
        
        # 计算FID
        if args.ref_stats:
            fid_output = compute_fid_adm(npz_path, args.ref_stats)
        else:
            print("[WARNING] 未提供参考统计数据，跳过FID计算")
            print(f"[INFO] 生成的样本保存在: {npz_path}")
            print("[INFO] 请使用以下命令手动计算FID:")
            print(f"  python evaluator.py <ref_stats.npz> {npz_path}")
            continue
        
        results[ckpt_name] = fid_output
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("[INFO] FID评估结果汇总:")
    print(f"{'='*60}")
    for ckpt_name, result in results.items():
        print(f"\nCheckpoint: {ckpt_name}")
        print(result)


if __name__ == "__main__":
    main()
