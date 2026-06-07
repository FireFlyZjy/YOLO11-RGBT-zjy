"""
批量验证脚本: 扫描指定目录下所有 best.pt, 逐个验证并输出到 val_results.csv

用法示例:
    # 扫描 runs/ 下所有 best.pt 并验证
    python batch_val.py --search_dir runs/FLIR/26dual-test

    # 指定数据集和设备
    python batch_val.py --search_dir runs/test --data ultralytics/cfg/datasets/flir.yaml --device 0

    # 只验证包含 vHeat 的权重
    python batch_val.py --search_dir runs/ --filter vHeat

    # 查看所有参数
    python batch_val.py --help
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# 禁用 ultralytics 在线检查, 避免网络代理导致的加载失败
os.environ["ULTRALYTICS_OFFLINE"] = "1"

from ultralytics import YOLO


# ============================================================================
# 默认配置 — 直接修改此处, 或通过命令行 --参数名 覆盖
# ============================================================================
DEFAULT_SEARCH_DIR = "runs"
DEFAULT_DATA = "ultralytics/cfg/datasets/flir.yaml"
DEFAULT_DEVICE = "0"
DEFAULT_BATCH = 4
DEFAULT_IMGSZ = 640
DEFAULT_WORKERS = 0
DEFAULT_USE_SIMOTM = "RGBT"
DEFAULT_CHANNELS = 4
DEFAULT_CSV_PATH = "val_results.csv"
DEFAULT_FILTER = ""  # 文件名过滤关键词, 留空则扫描全部


def main():
    parser = argparse.ArgumentParser(
        description="批量验证脚本 — 扫描目录下所有 best.pt, 逐个验证并输出 CSV"
    )
    parser.add_argument("--search_dir", type=str, default=DEFAULT_SEARCH_DIR,
                        help="搜索根目录 (递归扫描子目录)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="数据集 YAML 路径")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="CUDA 设备号, 'cpu' 表示 CPU")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="验证批次大小")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="输入图像尺寸")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="数据加载线程数")
    parser.add_argument("--use_simotm", type=str, default=DEFAULT_USE_SIMOTM,
                        help="模态组合: RGBT/RGBRGB6C/RGBT_IR/...")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                        help="输入通道数")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH,
                        help="输出 CSV 路径")
    parser.add_argument("--filter", type=str, default=DEFAULT_FILTER,
                        help="文件名过滤关键词 (如 vHeat), 留空则扫描全部")

    args = parser.parse_args()

    search_root = Path(args.search_dir)
    if not search_root.exists():
        print(f"[ERROR] 搜索目录不存在: {search_root}")
        sys.exit(1)

    # ---- 递归查找所有 best.pt ----
    all_pts = sorted(search_root.rglob("best.pt"))
    if args.filter:
        all_pts = [p for p in all_pts if args.filter in str(p)]

    if not all_pts:
        print(f"[WARN] 在 {search_root} 下未找到 best.pt 文件"
              + (f" (过滤: {args.filter})" if args.filter else ""))
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"  批量验证配置")
    print(f"{'='*70}")
    print(f"  搜索目录:     {search_root}")
    print(f"  找到权重数:   {len(all_pts)}")
    if args.filter:
        print(f"  过滤关键词:   {args.filter}")
    print(f"  数据集:       {args.data}")
    print(f"  模态:         {args.use_simotm}  (通道: {args.channels})")
    print(f"  Device:       {args.device}  |  Batch: {args.batch}  |  imgsz: {args.imgsz}")
    print(f"  输出 CSV:     {args.csv}")
    print(f"{'='*70}\n")

    # ---- CSV 列定义 (同 train_results.csv 格式) ----
    columns = [
        "Time",
        "Model",
        "YAML",
        "Pretrained",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "FLOPs(G)",
        "Params(M)",
        "FPS",
        "Preprocess(ms)",
        "Inference(ms)",
        "Postprocess(ms)",
        "Total(ms)",
        "Epochs",
        "Optimizer",
        "use_simotm",
        "Channels",
        "Batch",
        "Workers",
        "Device",
        "imgsz",
        "Data",
        "Project",
        "Name",
    ]

    csv_path = Path(args.csv)
    file_exists = csv_path.exists()

    success_count = 0
    fail_count = 0

    for i, pt_path in enumerate(all_pts):
        # ---- 从路径推断模型名称 ----
        # 路径形如: runs/FLIR/26dual-test/yolo26s-RGBT-midfusion-vHeat_Block/weights/best.pt
        # 取倒数第三个路径段作为实验名
        parts = pt_path.parts
        try:
            w_idx = parts.index("weights")
            exp_name = parts[w_idx - 1] if w_idx > 0 else "unknown"
        except ValueError:
            exp_name = pt_path.parent.parent.name if pt_path.parent.parent else "unknown"

        project = str(pt_path.parents[2]) if len(pt_path.parents) >= 3 else ""

        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(all_pts)}] {exp_name}")
        print(f"  路径: {pt_path}")
        print(f"{'─'*60}")

        try:
            # ---- 加载权重 (离线模式, 避免网络代理问题) ----
            model = YOLO(str(pt_path), task='detect', verbose=False)

            # ---- 获取 FLOPs / Params ----  (从 checkpoint 元数据提取) ----
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            ckpt_dict = ckpt if isinstance(ckpt, dict) else {}

            model_yaml = ""
            pretrained = ""
            epochs = ""
            optimizer = ""
            flops_g = 0.0
            params_m = 0.0

            if ckpt_dict:
                # 从 train_args 或直接字段提取
                train_args = ckpt_dict.get("train_args", {}) or ckpt_dict.get("args", {}) or {}
                if isinstance(train_args, dict):
                    model_yaml = str(train_args.get("model", "")) or ""
                    pretrained = str(train_args.get("pretrained", "")) or ""
                    epochs = str(train_args.get("epochs", "")) or ""
                    optimizer = str(train_args.get("optimizer", "")) or ""
                model_yaml = model_yaml or str(ckpt_dict.get("model_yaml", "")) or ""
                epochs = epochs or str(ckpt_dict.get("epoch", "")) or ""

            # 使用 model.info() 获取参数量 (fused 模型也可获取)
            try:
                n_l, n_p, n_g, flops = model.info(verbose=False)
                params_m = n_p / 1e6
                if flops and flops > 0:
                    flops_g = flops
            except Exception:
                pass

            # 如果 info() 没给出 FLOPs, 从 state_dict 估算参数量
            if params_m == 0:
                try:
                    sd = model.model.state_dict()
                    params_m = sum(v.numel() for v in sd.values()) / 1e6
                except Exception:
                    params_m = 0.0

            # 如果 FLOPs 仍为 0, 尝试从 YAML 重建模型计算
            if flops_g == 0 and model_yaml and Path(model_yaml).exists():
                try:
                    ref_model = YOLO(model_yaml, verbose=False)
                    _, n_p_ref, _, flops_ref = ref_model.info(verbose=False)
                    if flops_ref and flops_ref > 0:
                        flops_g = flops_ref
                    if n_p_ref and n_p_ref > 0:
                        params_m = n_p_ref / 1e6
                except Exception:
                    pass

            # ---- 运行验证 ----
            metrics = model.val(
                data=args.data,
                device=args.device,
                batch=args.batch,
                imgsz=args.imgsz,
                workers=args.workers,
                use_simotm=args.use_simotm,
                channels=args.channels,
                verbose=False,
            )

            # ---- 提取指标 ----
            rd = metrics.results_dict
            precision = rd.get("metrics/precision(B)", 0)
            recall = rd.get("metrics/recall(B)", 0)
            map50 = rd.get("metrics/mAP50(B)", 0)
            map50_95 = rd.get("metrics/mAP50-95(B)", 0)

            # ---- 提取速度 ----
            speed = metrics.speed
            preprocess_time = speed.get("preprocess", 0)
            inference_time = speed.get("inference", 0)
            postprocess_time = speed.get("postprocess", 0)
            total_time = preprocess_time + inference_time + speed.get("loss", 0) + postprocess_time
            fps = 1000.0 / total_time if total_time > 0 else 0

            # ---- 输出结果 ----
            print(f"  mAP50: {map50:.4f}  |  mAP50-95: {map50_95:.4f}  |  "
                  f"Precision: {precision:.4f}  |  Recall: {recall:.4f}")
            print(f"  Params: {params_m:.2f}M  |  FLOPs: {flops_g:.2f}G  |  FPS: {fps:.1f}")

            # ---- 写入 CSV ----
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True

                writer.writerow({
                    "Time": now_str,
                    "Model": exp_name,
                    "YAML": model_yaml,
                    "Pretrained": pretrained,
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "mAP50": f"{map50:.4f}",
                    "mAP50-95": f"{map50_95:.4f}",
                    "FLOPs(G)": f"{flops_g:.2f}",
                    "Params(M)": f"{params_m:.2f}",
                    "FPS": f"{fps:.1f}",
                    "Preprocess(ms)": f"{preprocess_time:.2f}",
                    "Inference(ms)": f"{inference_time:.2f}",
                    "Postprocess(ms)": f"{postprocess_time:.2f}",
                    "Total(ms)": f"{total_time:.2f}",
                    "Epochs": epochs,
                    "Optimizer": optimizer,
                    "use_simotm": args.use_simotm,
                    "Channels": str(args.channels),
                    "Batch": str(args.batch),
                    "Workers": str(args.workers),
                    "Device": args.device,
                    "imgsz": str(args.imgsz),
                    "Data": args.data,
                    "Project": project,
                    "Name": exp_name,
                })

            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"  [FAILED] {type(e).__name__}: {e}")

    # ---- 汇总 ----
    print(f"\n{'='*70}")
    print(f"  批量验证完成")
    print(f"{'='*70}")
    print(f"  成功: {success_count}  |  失败: {fail_count}  |  总计: {len(all_pts)}")
    print(f"  结果已保存到: {csv_path.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
