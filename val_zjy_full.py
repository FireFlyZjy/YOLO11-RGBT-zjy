import argparse
import csv
import sys
from pathlib import Path

from ultralytics import YOLO

# ============================================================================
# 可修改的超参数 / Configurable Hyperparameters
# 涵盖 val_zjy.py 中所有参数（包括被注释掉的备选值）
# ============================================================================

# --- 模型与数据集 ---
# 模型权重路径，支持目录（自动查找 last.pt/best.pt）或直接 .pt 文件路径
# 示例: "runs/FLIR/26dual-test/yolo26n-RGBT-midfusion-ASPP-V1/weights"
DEFAULT_WEIGHTS = "runs/FLIR/26dual-test/yolo26n-RGBT-midfusion-ASPP-V1/weights/best.pt"

# 数据集配置文件
DEFAULT_DATA = "ultralytics/cfg/datasets/flir.yaml"

# --- 验证参数 ---
DEFAULT_WORKERS = 0                                      # 数据加载线程数
DEFAULT_DEVICE = "0"                                     # CUDA 设备，'cpu' 表示 CPU
DEFAULT_BATCH = 4                                        # 验证批次大小
DEFAULT_IMGSZ = 640                                      # 输入图像尺寸

# --- 预处理方法 (use_simotm) ---
# 可选值: "RGBT", "RGBRGB6C", "SimOTMBBS", "RGBT_IR", "RGBRGB", "Gray"
DEFAULT_USE_SIMOTM = "RGBT"                              # 当前使用: RGBT
# DEFAULT_USE_SIMOTM = "RGBRGB6C"                        # 备选: RGBRGB6C

# --- 输入通道数 ---
DEFAULT_CHANNELS = 4                                     # 当前使用: 4
# DEFAULT_CHANNELS = 6                                   # 备选: 6

# --- 结果保存 ---
DEFAULT_PROJECT = "runs/FLIR/val"                        # 验证结果保存目录
DEFAULT_NAME = "test"                             # 实验名称

# --- CSV 输出 ---
DEFAULT_CSV_DIR = r"C:\Users\Patrick\Desktop\exp_results"  # CSV 保存目录
DEFAULT_CSV_NAME = "val_results.csv"                     # CSV 文件名

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


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 验证脚本 — 输出 Precision/Recall/mAP/FLOPs/Params/FPS 并追加到 CSV"
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help="模型权重目录路径（自动查找 last.pt/best.pt）或直接 .pt 文件路径",
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--use_simotm", type=str, default=DEFAULT_USE_SIMOTM)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    parser.add_argument("--csv_dir", type=str, default=DEFAULT_CSV_DIR)
    parser.add_argument("--csv_name", type=str, default=DEFAULT_CSV_NAME)

    args = parser.parse_args()

    # ---- 查找模型文件 ----
    weights_path = find_model_file(args.weights)
    if weights_path is None:
        print(f"[ERROR] 在 '{args.weights}' 中未找到 .pt 文件")
        sys.exit(1)

    weights_str = str(weights_path)
    print(f"Loading model: {weights_str}")

    # ---- 提取模型名称：weights 目录的上一级目录名 ----
    model_name = weights_path.parent.parent.name if weights_path.parent.name == "weights" else weights_path.parent.name
    print(f"Model name: {model_name}")

    # ---- 加载模型 ----
    model = YOLO(weights_str)

    # ---- 获取 FLOPs 与 Params ----
    # model.info() 同时打印模型摘要并返回 (n_layers, n_params, n_gradients, flops)
    n_l, n_p, n_g, flops = model.info(verbose=True)
    params_m = n_p / 1e6      # 参数量 (M)
    flops_g = flops           # 计算量 (G)

    # ---- 运行验证 ----
    print("\nRunning validation...")
    metrics = model.val(
        data=args.data,
        workers=args.workers,
        device=args.device,
        batch=args.batch,
        use_simotm=args.use_simotm,
        channels=args.channels,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
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
    # loss_time 不计入 visible FPS but available
    total_time = preprocess_time + inference_time + speed.get("loss", 0) + postprocess_time
    total_time_visible = preprocess_time + inference_time + postprocess_time
    fps = 1000.0 / total_time if total_time > 0 else 0

    # ---- 控制台输出 ----
    print("\n" + "=" * 80)
    print("                            VALIDATION RESULTS")
    print("=" * 80)
    print(f"  Model:              {model_name}")
    print(f"  Dataset:            {args.data}")
    print(f"  use_simotm:         {args.use_simotm}")
    print(f"  Channels:           {args.channels}")
    print("-" * 80)
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  mAP50:              {map50:.4f}")
    print(f"  mAP50-95:           {map50_95:.4f}")
    print(f"  FLOPs:              {flops_g:.2f} G")
    print(f"  Params:             {params_m:.2f} M")
    print("-" * 80)
    print(f"  FPS:                {fps:.2f}")
    print(f"    Preprocess Time:  {preprocess_time:.2f} ms")
    print(f"    Inference Time:   {inference_time:.2f} ms")
    print(f"    Postprocess Time: {postprocess_time:.2f} ms")
    print(f"    Total Time:       {total_time:.2f} ms")
    print("=" * 80)

    # ---- 保存到 CSV (追加模式) ----
    csv_dir = Path(args.csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / args.csv_name

    columns = [
        "Model",
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
        "Data",
        "Project",
        "Name",
    ]

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Model": model_name,
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
            "use_simotm": args.use_simotm,
            "Channels": str(args.channels),
            "Batch": str(args.batch),
            "Workers": str(args.workers),
            "Device": args.device,
            "imgsz": str(args.imgsz),
            "Data": args.data,
            "Project": args.project,
            "Name": args.name,
        })

    print(f"\nResults appended to: {csv_path}")

    # ---- 返回 metrics 供程序化调用 ----
    return metrics


if __name__ == "__main__":
    main()
