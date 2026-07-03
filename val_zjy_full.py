"""
YOLO 批量验证脚本 — 自动扫描 runs/ 下所有 best.pt，读取 args.yaml 恢复训练参数，逐个验证并输出到 CSV

用法示例:
    # 扫描 runs/ 下所有 best.pt 并验证（默认）
    python val_zjy_full.py

    # 指定扫描目录
    python val_zjy_full.py --root_dir runs/FLIR

    # 指定输出 CSV
    python val_zjy_full.py --csv_name flir_results.csv

    # 覆盖设备 / 批量大小等（所有模型统一使用）
    python val_zjy_full.py --device 0 --batch 8

    # 跳过已在 CSV 中的模型（默认开启）
    python val_zjy_full.py --no-skip-existing   # 重新验证所有模型

    # 单模型模式（保留旧功能）
    python val_zjy_full.py --weights path/to/best.pt
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

# ============================================================================
# 默认参数
# ============================================================================

DEFAULT_ROOT_DIR = "runs"                  # 扫描根目录
DEFAULT_CSV_DIR = "runs"                   # CSV 保存目录
DEFAULT_CSV_NAME = "val_results.csv"       # CSV 文件名
DEFAULT_DEVICE = "0"                       # CUDA 设备
DEFAULT_WORKERS = 0                        # 数据加载线程数
DEFAULT_BATCH = 4                          # 验证批次大小
DEFAULT_IMGSZ = 640                        # 输入图像尺寸

# --- args.yaml 缺失时的 fallback 默认值 ---
DEFAULT_FALLBACK_DATA = "ultralytics/cfg/datasets/flir.yaml"
DEFAULT_FALLBACK_USE_SIMOTM = "RGBT"
DEFAULT_FALLBACK_CHANNELS = 4

# ============================================================================


def find_model_file(weights_input):
    """在给定路径中查找模型权重文件 (.pt)"""
    p = Path(weights_input)
    if p.is_file() and p.suffix == ".pt":
        return p
    if p.is_dir():
        for candidate in ["last.pt", "best.pt"]:
            f = p / "weights" / candidate
            if f.exists():
                return f
            f = p / candidate
            if f.exists():
                return f
        pt_files = list(p.rglob("*.pt"))
        if pt_files:
            return pt_files[0]
    return None


def discover_best_pt_files(root_dir):
    """递归扫描 root_dir 下所有 weights/best.pt 文件，按模型名排序"""
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERROR] 扫描目录不存在: {root}")
        return []
    files = sorted(root.rglob("weights/best.pt"))
    if not files:
        print(f"[WARNING] 在 '{root}' 下未找到任何 best.pt 文件")
    return files


def read_args_yaml(model_dir):
    """读取模型训练目录下的 args.yaml，恢复训练参数"""
    args_path = model_dir / "args.yaml"
    if not args_path.exists():
        print(f"  [WARNING] args.yaml 未找到: {args_path}，使用默认值")
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return args if args else {}


def validate_single_model(best_pt_path, csv_path, index, total,
                          device_override=None, batch_override=None,
                          workers_override=None, imgsz_override=None,
                          fallback_data=None, fallback_use_simotm=None,
                          fallback_channels=None):
    """验证单个模型，将结果追加到 CSV"""
    model_dir = best_pt_path.parent.parent  # .../weights/best.pt → .../weights → .../
    model_name = model_dir.name

    # ---- 从 args.yaml 恢复训练参数 ----
    args = read_args_yaml(model_dir)

    data = args.get("data", fallback_data or DEFAULT_FALLBACK_DATA)
    use_simotm = args.get("use_simotm", fallback_use_simotm or DEFAULT_FALLBACK_USE_SIMOTM)
    channels = args.get("channels", fallback_channels or DEFAULT_FALLBACK_CHANNELS)
    imgsz = imgsz_override or args.get("imgsz", DEFAULT_IMGSZ)
    batch = batch_override or args.get("batch", DEFAULT_BATCH)
    workers = workers_override or args.get("workers", DEFAULT_WORKERS)
    device = device_override or args.get("device", DEFAULT_DEVICE)
    model_yaml = args.get("model", "")
    pretrained = args.get("pretrained", "")
    epochs = args.get("epochs", 0)
    optimizer = args.get("optimizer", "")
    project = args.get("project", "")
    name = args.get("name", model_name)
    pairs_rgb_ir = args.get("pairs_rgb_ir", None)

    # ---- 打印进度 ----
    print(f"\n[{index}/{total}] {model_name}")
    print(f"  Weights:     {best_pt_path}")
    print(f"  Data:        {data}")
    print(f"  Device:      {device}  |  Batch: {batch}  |  imgsz: {imgsz}")

    # ---- 加载模型 ----
    model = YOLO(str(best_pt_path))

    # ---- 获取 FLOPs 与 Params ----
    n_l, n_p, n_g, flops = model.info(verbose=True)
    params_m = n_p / 1e6
    flops_g = flops if flops else 0

    # ---- 运行验证 ----
    val_kwargs = dict(
        data=data,
        workers=workers,
        device=device,
        batch=batch,
        use_simotm=use_simotm,
        channels=channels,
        imgsz=imgsz,
        project=project,
        name=f"{model_name}_val",
    )
    if pairs_rgb_ir is not None:
        val_kwargs["pairs_rgb_ir"] = pairs_rgb_ir

    metrics = model.val(**val_kwargs)

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

    # ---- 打印结果摘要 ----
    print(f"  Results:     P={precision:.4f}  R={recall:.4f}  "
          f"mAP50={map50:.4f}  mAP50-95={map50_95:.4f}  "
          f"FPS={fps:.2f}  FLOPs={flops_g:.2f}G  Params={params_m:.2f}M")

    # ---- 写入 CSV ----
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [
        "Time",
        "Model",
        "Model_YAML",
        "Data",
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
        "use_simotm",
        "Channels",
        "Batch",
        "Workers",
        "Device",
        "imgsz",
        "Epochs",
        "Optimizer",
        "Project",
        "Name",
    ]

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Time": now_str,
            "Model": model_name,
            "Model_YAML": model_yaml,
            "Data": data,
            "Pretrained": pretrained,
            "Precision": f"{precision:.4f}",
            "Recall": f"{recall:.4f}",
            "mAP50": f"{map50:.4f}",
            "mAP50-95": f"{map50_95:.4f}",
            "FLOPs(G)": f"{flops_g:.2f}",
            "Params(M)": f"{params_m:.2f}",
            "FPS": f"{fps:.2f}",
            "Preprocess(ms)": f"{preprocess_time:.2f}",
            "Inference(ms)": f"{inference_time:.2f}",
            "Postprocess(ms)": f"{postprocess_time:.2f}",
            "Total(ms)": f"{total_time:.2f}",
            "use_simotm": use_simotm,
            "Channels": str(channels),
            "Batch": str(batch),
            "Workers": str(workers),
            "Device": str(device),
            "imgsz": str(imgsz),
            "Epochs": str(epochs),
            "Optimizer": optimizer,
            "Project": project,
            "Name": name,
        })

    print(f"  -> 结果已追加到 {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 批量验证脚本 — 自动扫描 runs/ 下所有 best.pt，逐个验证并输出到 CSV"
    )

    # 扫描模式（与 --weights 互斥）
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR,
                        help="扫描根目录，递归查找所有 best.pt (默认: runs/)")

    # 单模型模式（与 --root_dir 互斥，保留旧功能）
    parser.add_argument("--weights", type=str, default=None,
                        help="单模型模式：指定权重路径（.pt 文件或目录）")

    # CSV 输出
    parser.add_argument("--csv_dir", type=str, default=DEFAULT_CSV_DIR)
    parser.add_argument("--csv_name", type=str, default=DEFAULT_CSV_NAME)

    # 参数覆盖（不指定则使用各模型 args.yaml 中的训练参数）
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖所有模型的验证设备")
    parser.add_argument("--batch", type=int, default=None,
                        help="覆盖所有模型的 batch 大小")
    parser.add_argument("--workers", type=int, default=None,
                        help="覆盖所有模型的 workers 数")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="覆盖所有模型的输入图像尺寸")

    # args.yaml 缺失时的 fallback 参数
    parser.add_argument("--fallback-data", type=str, default=DEFAULT_FALLBACK_DATA,
                        help="args.yaml 缺失时使用的数据集路径 (默认: flir.yaml)")
    parser.add_argument("--fallback-use-simotm", type=str, default=DEFAULT_FALLBACK_USE_SIMOTM,
                        help="args.yaml 缺失时使用的模态 (默认: RGBT)")
    parser.add_argument("--fallback-channels", type=int, default=DEFAULT_FALLBACK_CHANNELS,
                        help="args.yaml 缺失时使用的通道数 (默认: 4)")

    # 跳过选项
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已在 CSV 中存在的模型 (默认开启)")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing",
                        help="重新验证所有模型，即使 CSV 中已有")

    args = parser.parse_args()

    csv_path = Path(args.csv_dir) / args.csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 收集已有模型的集合（用于 skip-existing） ----
    existing_models = set()
    if args.skip_existing and csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("Model", "").strip()
                data = row.get("Data", "").strip()
                if model_name:
                    existing_models.add((model_name, data))
        if existing_models:
            print(f"CSV 中已存在 {len(existing_models)} 条记录，将跳过这些模型")
            print(f"  (使用 --no-skip-existing 强制重新验证)")

    # ---- 模式1: 单模型模式（--weights） ----
    if args.weights:
        weights_path = find_model_file(args.weights)
        if weights_path is None:
            print(f"[ERROR] 在 '{args.weights}' 中未找到 .pt 文件")
            sys.exit(1)
        validate_single_model(
            weights_path, csv_path, index=1, total=1,
            device_override=args.device,
            batch_override=args.batch,
            workers_override=args.workers,
            imgsz_override=args.imgsz,
            fallback_data=args.fallback_data,
            fallback_use_simotm=args.fallback_use_simotm,
            fallback_channels=args.fallback_channels,
        )
        return

    # ---- 模式2: 批量扫描模式（默认） ----
    best_pt_files = discover_best_pt_files(args.root_dir)
    if not best_pt_files:
        print(f"未找到任何 best.pt，退出。")
        sys.exit(1)

    total = len(best_pt_files)
    print(f"\n{'=' * 60}")
    print(f"  批量验证: 共发现 {total} 个 best.pt")
    print(f"  扫描目录: {args.root_dir}")
    print(f"  输出 CSV: {csv_path}")
    if args.device:
        print(f"  设备覆盖: {args.device}")
    if args.batch:
        print(f"  Batch覆盖: {args.batch}")
    print(f"  Fallback data: {args.fallback_data} (用于缺失 args.yaml 的模型)")
    print(f"{'=' * 60}\n")

    # ---- 预筛选：跳过已有模型 ----
    to_validate = []
    skipped = 0
    for bp in best_pt_files:
        model_dir = bp.parent.parent
        model_name = model_dir.name
        args_dict = read_args_yaml(model_dir)
        data = args_dict.get("data", "")

        if args.skip_existing and (model_name, data) in existing_models:
            skipped += 1
            continue
        to_validate.append(bp)

    if skipped > 0:
        print(f"跳过 {skipped} 个已在 CSV 中的模型")
    if not to_validate:
        print("所有模型均已验证过，无需操作。")
        print(f"  (使用 --no-skip-existing 强制重新验证)")
        return

    print(f"待验证: {len(to_validate)} 个模型\n")

    # ---- 逐个验证 ----
    for idx, bp in enumerate(to_validate, start=1):
        validate_single_model(
            bp, csv_path, index=idx, total=len(to_validate),
            device_override=args.device,
            batch_override=args.batch,
            workers_override=args.workers,
            imgsz_override=args.imgsz,
            fallback_data=args.fallback_data,
            fallback_use_simotm=args.fallback_use_simotm,
            fallback_channels=args.fallback_channels,
        )

    print(f"\n{'=' * 60}")
    print(f"  批量验证完成！共验证 {len(to_validate)} 个模型")
    print(f"  结果保存至: {csv_path}")
    if skipped > 0:
        print(f"  (已跳过 {skipped} 个已有记录)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
