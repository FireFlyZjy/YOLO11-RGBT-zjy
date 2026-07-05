"""
YOLO 训练+验证一体化脚本 (全参数 argparse 支持)

用法示例:
    # 使用默认参数训练
    python train_AddModules.py

    # 只改模型和数据集
    python train_AddModules.py --model_yaml path/to/model.yaml --data path/to/data.yaml

    # 调整训练超参数 (顺序任意, 未指定的使用默认值)
    python train_AddModules.py --epochs 200 --batch 8 --lr0 0.001 --device 0,1

    # 断点续训
    python train_AddModules.py --resume runs/.../weights/last.pt --epochs 150

    # 查看所有可配置参数
    python train_AddModules.py --help
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

class _DummyEMA:
    """占位EMA — 跳过参数更新但保留模型引用, 兼容所有 EMA 属性和 checkpoint。"""
    def __init__(self, model):
        from copy import deepcopy
        self.ema_model = deepcopy(model).eval()
        self.updates = 0
    def update(self, model): pass
    def update_attr(self, model, include=None): pass
    @property
    def ema(self): return self.ema_model

# ============================================================================
# 默认超参数 — 直接修改此处改变默认值, 或通过命令行 --参数名 覆盖
# ============================================================================

# --- 模型配置 ---
# 模型 YAML 配置文件路径
DEFAULT_MODEL_YAML = "ultralytics/cfg/models/26-RGBT/2026-06-27/flickerformer/yolo26-RGBT-midfusion-SCAM.yaml"

# --- 训练结果保存 ---
DEFAULT_PROJECT = "runs/FLIR/26dual-test"
DEFAULT_NAME = "yolo26s-RGBT-midfusion-SCAM"

# --- 数据集 ---
DEFAULT_DATA = "ultralytics/cfg/datasets/flir.yaml"

# --- 预训练权重 ---
DEFAULT_PRETRAINED = "weights/yolo26s.pt"
# 续训 checkpoint 路径 (e.g. runs/.../weights/last.pt)，留空则从头训练
DEFAULT_RESUME = ''

# --- 模态配置 ---
# 可选值: "RGBT", "RGBRGB6C", "SimOTMBBS", "RGBT_IR", "RGBRGB", "Gray"
DEFAULT_USE_SIMOTM = "RGBT"                              # 当前使用: RGBT
# DEFAULT_USE_SIMOTM = "RGBRGB6C"                        # 备选: RGBRGB6C (已被注释)

DEFAULT_CHANNELS = 4                                     # 当前使用: 4
# DEFAULT_CHANNELS = 6                                   # 备选: 6 (已被注释)

# --- pairs_rgb_ir 配置 ---
# 可选值: ['visible','infrared'], ['rgb', 'ir'], ['images', 'images_ir'], ['images', 'image']
# DEFAULT_PAIRS_RGB_IR = ['visible','infrared']           # (已被注释)

# --- 训练参数 ---
DEFAULT_EPOCHS = 100                                     # 训练轮数
DEFAULT_BATCH = 4                                        # 批次大小
DEFAULT_IMGSZ = 640                                      # 输入图像尺寸
DEFAULT_WORKERS = 0                                      # 数据加载线程数
DEFAULT_DEVICE = "0"                                     # CUDA 设备，'cpu' 表示 CPU
DEFAULT_OPTIMIZER = "SGD"                                # 优化器，可选 SGD/Adam/AdamW 等
DEFAULT_LR0 = 0.0                                        # 初始学习率 (0 = 使用YOLO默认值)
DEFAULT_CLOSE_MOSAIC = 10                                # mosaic 增强关闭轮数

# --- 数据增强 ---
DEFAULT_CACHE = False                                    # 是否缓存数据集到内存
DEFAULT_AMP = True                                       # 自动混合精度 (默认开启)
DEFAULT_FRACTION = 1.0                                   # 数据集使用比例, <1.0 则只取部分数据

# --- 双模态配对 ---
# 可选: ['visible','infrared'] / ['rgb', 'ir'] / ['images', 'images_ir'] / ['images', 'image']
DEFAULT_PAIRS_RGB_IR = ''                                # 留空使用数据集默认配置

# --- 验证参数 (训练后自动验证) ---
# 验证时使用训练后的 best.pt

# --- CSV 输出 ---
DEFAULT_CSV_DIR = r"runs"
DEFAULT_CSV_NAME = "train_results.csv"

# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 训练+验证一体化脚本 — 训练完成后自动输出指标到控制台并追加到 CSV"
    )
    # 模型 YAML
    parser.add_argument("--model_yaml", type=str, default=DEFAULT_MODEL_YAML,
                        help="模型 YAML 配置文件路径")
    # 训练结果保存
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    # 数据集
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    # 预训练权重
    parser.add_argument("--pretrained", type=str, default=DEFAULT_PRETRAINED,
                        help="预训练权重路径")
    # 模态
    parser.add_argument("--use_simotm", type=str, default=DEFAULT_USE_SIMOTM)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    # 训练
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER)
    parser.add_argument("--lr0", type=float, default=DEFAULT_LR0,
                        help="初始学习率 (0=使用YOLO默认值)")
    parser.add_argument("--close_mosaic", type=int, default=DEFAULT_CLOSE_MOSAIC)
    parser.add_argument("--cache", type=lambda x: x.lower() == 'true' if isinstance(x, str) else x,
                        default=DEFAULT_CACHE)
    parser.add_argument("--amp", type=lambda x: x.lower() == 'true' if isinstance(x, str) else x,
                        default=DEFAULT_AMP)
    parser.add_argument("--ema", type=lambda x: x.lower() == 'true' if isinstance(x, str) else x,
                        default=True, help="是否启用EMA (自定义检测头需--ema false)")
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                        help="数据集使用比例, <1.0则只取部分(调试用)")
    parser.add_argument("--pairs_rgb_ir", type=str, default=DEFAULT_PAIRS_RGB_IR,
                        help="双模态配对规则, 如 'visible,infrared' (逗号分隔)")
    # 恢复训练
    parser.add_argument("--resume", type=str, default=DEFAULT_RESUME,
                        help="续训 checkpoint 路径 (e.g. runs/.../weights/last.pt)，留空则从头训练")
    # CSV
    parser.add_argument("--csv_dir", type=str, default=DEFAULT_CSV_DIR)
    parser.add_argument("--csv_name", type=str, default=DEFAULT_CSV_NAME)

    args = parser.parse_args()

    resume_mode = bool(args.resume)

    # ---- 提取模型名称 (从 YAML 文件名) ----
    yaml_stem = Path(args.model_yaml).stem
    if yaml_stem.startswith("yolo26-RGBT-midfusion-"):
        model_short = yaml_stem[len("yolo26-RGBT-midfusion-"):]
    elif yaml_stem.startswith("yolo26-RGBT-"):
        model_short = yaml_stem[len("yolo26-RGBT-"):]
    else:
        model_short = yaml_stem

    # 解析 pairs_rgb_ir
    pairs = None
    if args.pairs_rgb_ir:
        pairs = [x.strip() for x in args.pairs_rgb_ir.split(",")]

    # ---- 启动时打印全部配置 ----
    print("=" * 60)
    print("  训练配置")
    print("=" * 60)
    print(f"  模型YAML:     {args.model_yaml}")
    print(f"  模型简称:     {model_short}")
    if resume_mode:
        print(f"  续训路径:     {args.resume}")
    else:
        print(f"  预训练权重:   {args.pretrained}")
    print(f"  数据集:       {args.data}")
    print(f"  模态:         {args.use_simotm}  (通道数: {args.channels})")
    if pairs:
        print(f"  配对规则:     {pairs}")
    print(f"  保存路径:     {args.project}/{args.name}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch:        {args.batch}")
    print(f"  Imgsz:        {args.imgsz}")
    print(f"  Device:       {args.device}")
    print(f"  Optimizer:    {args.optimizer}" + (f"  lr0={args.lr0}" if args.lr0 > 0 else "  lr0=default"))
    print(f"  Close Mosaic: {args.close_mosaic}")
    print(f"  AMP:          {args.amp}")
    print(f"  Fraction:     {args.fraction}")
    print(f"  Workers:      {args.workers}")
    print(f"  Cache:        {args.cache}")
    print(f"  CSV:          {args.csv_dir}/{args.csv_name}")
    print("=" * 60)

    # ---- 步骤1: 加载模型 ----
    print("\n[Step 1/4] Loading model...")

    if resume_mode:
        # 续训模式: 从 checkpoint 恢复
        print(f"Resume checkpoint: {args.resume}")
        model = YOLO(args.resume, task='detect')
    else:
        # 从头训练模式: 加载 YAML + 预训练权重
        model = YOLO(args.model_yaml, task='detect').load(args.pretrained)

    # ---- 获取 FLOPs 与 Params (训练前) ----
    n_l, n_p, n_g, flops = model.info(verbose=True)
    params_m = n_p / 1e6
    flops_g = flops if flops else 0

    # ---- 步骤2: 训练 ----
    print("\n[Step 2/4] Training..." + (" (resume)" if resume_mode else ""))

    # 公共训练参数
    common_train_args = dict(
        data=args.data,
        cache=args.cache,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        device=args.device,
        use_simotm=args.use_simotm,
        channels=args.channels,
        project=args.project,
        name=args.name,
    )
    if args.lr0 > 0:
        common_train_args["lr0"] = args.lr0
    if not args.amp:
        common_train_args["amp"] = False
    if args.fraction < 1.0:
        common_train_args["fraction"] = args.fraction
    if pairs:
        common_train_args["pairs_rgb_ir"] = pairs


    if resume_mode:
        # resume=True 从 checkpoint 恢复 optimizer / 学习率调度器等状态
        common_train_args["resume"] = True
    else:
        common_train_args["optimizer"] = args.optimizer

    # 自定义检测头(DecoupledHead/ATAH)需关闭EMA, 通过callback替换为占位对象
    if not args.ema:
        model.add_callback("on_train_start", lambda t: setattr(t, 'ema', _DummyEMA(t.model)))

    # 预创建保存目录，避免 Ultralytics 内部保存 results.csv 时父目录不存在
    Path(args.project, args.name).mkdir(parents=True, exist_ok=True)

    train_result = model.train(**common_train_args)

    # ---- 步骤3: 验证训练好的模型 ----
    print("\n[Step 3/4] Validating trained model...")

    # 训练后 best.pt 的路径: project/name/weights/best.pt
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        # 尝试 last.pt
        best_pt = Path(args.project) / args.name / "weights" / "last.pt"

    if best_pt.exists():
        best_pt_str = str(best_pt)
        print(f"Validating: {best_pt_str}")

        # 加载训练好的模型
        val_model = YOLO(best_pt_str, task='detect')

        # 运行验证
        metrics = val_model.val(
            data=args.data,
            workers=args.workers,
            device=args.device,
            batch=args.batch,
            use_simotm=args.use_simotm,
            channels=args.channels,
            project=args.project,
            name=args.name + "_val",
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
        total_time = preprocess_time + inference_time + speed.get("loss", 0) + postprocess_time
        fps = 1000.0 / total_time if total_time > 0 else 0

        # ---- 步骤4: 控制台输出 ----
        print("\n" + "=" * 80)
        print("                        TRAINING + VALIDATION RESULTS")
        print("=" * 80)
        print(f"  Model:              {model_short}")
        print(f"  YAML:               {args.model_yaml}")
        print(f"  Dataset:            {args.data}")
        print(f"  Epochs:             {args.epochs}")
        print(f"  Optimizer:          {args.optimizer}")
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
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Time": now_str,
                "Model": model_short,
                "YAML": args.model_yaml,
                "Pretrained": args.resume if resume_mode else args.pretrained,
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
                "Epochs": str(args.epochs),
                "Optimizer": args.optimizer,
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

    else:
        print(f"[WARNING] 训练后的权重文件未找到: {best_pt}")
        print("训练可能未正常完成，无法运行验证。")


if __name__ == "__main__":
    main()
