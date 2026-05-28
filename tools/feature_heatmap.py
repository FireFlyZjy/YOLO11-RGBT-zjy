"""
特征图热力图可视化工具 (Feature Map Heatmap Visualization)

功能:
  注册 forward hook 以捕获模型中指定层的特征图。
  将特征图保存为 JET 彩色热力图 (使用 OpenCV 的 applyColorMap)。
  可选择保存 .npy 文件以便后续分析 (如计算特征分布、可视化激活区域等)。

用法示例:
  # 可视化所有层
  python tools/feature_heatmap.py --model_yaml ultralytics/cfg/models/yolo11n.yaml --image_path ./demo.jpg

  # 指定特定层 (按索引)
  python tools/feature_heatmap.py --model_yaml ... --image_path ./demo.jpg --layer_names 2 5 10

  # 保存 npy 文件以便进一步分析
  python tools/feature_heatmap.py --model_yaml ... --image_path ./demo.jpg --save_npy

  # RGBT 4通道模型
  python tools/feature_heatmap.py --model_yaml .../yolo26-RGBT-midfusion.yaml --image_path ./demo.jpg --channels 4

参考来源: objectdetection-tricks/tricks_3.py
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 终端颜色代码
# ============================================================================
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


class FeatureHook:
    """
    特征图捕获钩子。

    用于在模型前向传播时捕获指定层的输出特征图。

    用法:
        hook = FeatureHook()
        handle = layer.register_forward_hook(hook)
        model(input)
        feature_maps = hook.features
    """

    def __init__(self):
        """初始化钩子，存储特征图和模块信息。"""
        self.features = None      # 存储捕获的特征图张量
        self.module_name = ""     # 存储模块名称
        self.module_type = ""     # 存储模块类型

    def __call__(self, module, module_input, module_output):
        """
        钩子回调函数，在前向传播后自动调用。

        参数:
            module (nn.Module): 当前模块
            module_input (tuple): 输入张量元组
            module_output: 输出张量 (或 list/tuple 等)
        """
        self.module_type = module.__class__.__name__

        # 只捕获 4D 特征图 (batch, channels, height, width)
        if isinstance(module_output, torch.Tensor) and module_output.dim() == 4:
            # 分离张量并转移到 CPU
            self.features = module_output.detach().cpu()


def preprocess_image(image_path, input_size=640, channels=3):
    """
    加载并预处理输入图像，使其符合模型输入要求。

    处理流程:
      1. 读取图片
      2. 调整尺寸到 input_size
      3. 归一化到 [0, 1]
      4. 调整通道顺序为 (C, H, W)
      5. 添加 batch 维度

    参数:
        image_path (str): 图片文件路径
        input_size (int): 模型输入尺寸 (默认: 640)
        channels (int): 模型输入通道数 (3=RGB, 4=RGBT, 6=RGBRGB6C)

    返回:
        tuple: (torch.Tensor, ndarray, tuple)
            - input_tensor: 处理后的输入张量 [1, C, H, W]
            - original_image: 原始图像 (用于显示)
            - original_size: 原始图像尺寸 (H, W)
    """
    print(f"{BLUE}加载图片: {image_path}{RESET}")

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    original_image = image.copy()
    original_size = image.shape[:2]  # (H, W)

    print(f"  原始尺寸: {original_size}")
    print(f"  目标尺寸: {input_size}")

    # 调整尺寸到模型输入大小
    resized = cv2.resize(image, (input_size, input_size))

    # 转换为 float32 并归一化到 [0, 1]
    resized = resized.astype(np.float32) / 255.0

    # OpenCV 读取的是 BGR 格式，保持 BGR 顺序
    # 转换为 (C, H, W) 格式
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)

    # 处理通道数不匹配的情况
    actual_channels = tensor.shape[1]
    if actual_channels < channels:
        # 如果实际通道数少于需要，复制 RGB 通道来填充
        repeats = channels // actual_channels + 1
        tensor = tensor.repeat(1, repeats, 1, 1)[:, :channels, :, :]
        print(f"{YELLOW}  通道数从 {actual_channels} 扩展到 {channels}{RESET}")
    elif actual_channels > channels:
        # 如果实际通道数多于需要，截取前 channels 个通道
        tensor = tensor[:, :channels, :, :]
        print(f"{YELLOW}  通道数从 {actual_channels} 裁剪到 {channels}{RESET}")

    print(f"{GREEN}  输入张量形状: {list(tensor.shape)}{RESET}")
    return tensor, original_image, original_size


def save_heatmap(feature_map, save_dir, stage_name, layer_type, max_channels=32, save_npy=False):
    """
    将特征图保存为 JET 彩色热力图。

    处理流程:
      1. 取 batch 索引 0 的特征图
      2. 按通道维度切分，取前 max_channels 个通道
      3. 对每个通道的值归一化到 [0, 255]
      4. 应用 OpenCV 的 COLORMAP_JET 生成彩色热力图
      5. 保存为 PNG 图片

    参数:
        feature_map (torch.Tensor): 特征图张量 [1, C, H, W]
        save_dir (Path): 保存目录
        stage_name (str): 阶段名称 (用于文件名)
        layer_type (str): 模块类型 (用于文件名)
        max_channels (int): 最大保存的通道数 (默认: 32)
        save_npy (bool): 是否同时保存 .npy 文件
    """
    if feature_map is None:
        print(f"{YELLOW}  跳过空特征图{RESET}")
        return

    # 获取特征图尺寸
    batch_size, channels, height, width = feature_map.shape

    # 跳过 1x1 的特征图 (通常是最后的检测头输出，空间信息太少)
    if height <= 1 or width <= 1:
        return

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名 (使用阶段名称和模块类型)
    safe_name = f"{stage_name}_{layer_type}".replace(".", "_").replace("/", "_")
    heatmap_path = save_dir / f"{safe_name}_heatmap.png"

    # 选择要可视化的通道数 (不超过实际通道数和最大限制)
    n_visualize = min(max_channels, channels)

    # 取 batch 0 的特征图
    feat = feature_map[0].cpu().numpy()  # (C, H, W)

    # 如果通道数很多，均匀采样 n_visualize 个通道进行展示
    if channels > n_visualize:
        indices = np.linspace(0, channels - 1, n_visualize, dtype=int)
        selected = feat[indices]
    else:
        selected = feat[:n_visualize]

    # 将各通道排列成网格 (方便一次性保存为一张大图)
    # 计算网格尺寸 (尽量接近方形)
    grid_cols = int(np.ceil(np.sqrt(n_visualize)))
    grid_rows = int(np.ceil(n_visualize / grid_cols))

    # 创建网格画布
    grid_height = height * grid_rows
    grid_width = width * grid_cols
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # 逐一处理每个通道
    for i in range(n_visualize):
        channel_data = selected[i]  # (H, W)

        # 归一化到 [0, 255]
        if channel_data.max() > channel_data.min():
            normalized = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
        else:
            # 全零特征图
            normalized = np.zeros_like(channel_data)
        normalized_255 = (normalized * 255).astype(np.uint8)

        # 应用 JET 彩色映射
        colored = cv2.applyColorMap(normalized_255, cv2.COLORMAP_JET)

        # 计算在网格中的位置
        row_idx = i // grid_cols
        col_idx = i % grid_cols
        y_start = row_idx * height
        x_start = col_idx * width

        # 将彩色特征图放入网格
        grid_image[y_start:y_start + height, x_start:x_start + width] = colored

    # 保存热力图网格
    cv2.imwrite(str(heatmap_path), grid_image)
    print(f"{GREEN}  保存热力图: {heatmap_path} ({n_visualize}/{channels} 通道){RESET}")

    # 如果指定，保存 .npy 文件
    if save_npy:
        npy_path = save_dir / f"{safe_name}.npy"
        np.save(str(npy_path), feat)
        print(f"  保存 NPY: {npy_path} (形状: {feat.shape}){RESET}")


# ============================================================================
# 模型加载函数
# ============================================================================
def load_model(model_yaml, channels):
    """
    从 YAML 配置文件加载模型。

    参数:
        model_yaml (str): 模型 YAML 文件路径
        channels (int): 输入通道数

    返回:
        nn.Module: 加载好的 PyTorch 模型
    """
    from ultralytics import YOLO

    print(f"{BLUE}正在加载模型: {model_yaml}{RESET}")
    print(f"  输入通道数: {channels}")

    try:
        model = YOLO(model_yaml).model
    except Exception as e:
        print(f"{RED}加载模型失败: {e}{RESET}")
        print(f"{YELLOW}尝试直接解析 YAML 构建模型...{RESET}")

        try:
            from ultralytics.nn.tasks import parse_model
            import yaml

            with open(model_yaml, "r") as f:
                yaml_cfg = yaml.safe_load(f)

            model, _ = parse_model(yaml_cfg, ch=channels)
        except Exception as e2:
            print(f"{RED}备用加载也失败: {e2}{RESET}")
            raise

    model.eval()
    print(f"{GREEN}模型加载成功! 参数量: {sum(p.numel() for p in model.parameters()):,}{RESET}")

    return model


# ============================================================================
# 主函数
# ============================================================================
def main():
    """解析命令行参数并执行特征图热力图可视化。"""
    parser = argparse.ArgumentParser(
        description="特征图热力图可视化工具 (Feature Map Heatmap Visualization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 可视化默认模型的所有层
  python tools/feature_heatmap.py --image_path ./data/demo.jpg

  # 可视化指定模型的指定层
  python tools/feature_heatmap.py --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml --image_path ./data/demo.jpg --layer_names 2 5 10 15

  # RGBT 4通道输入，同时保存 npy
  python tools/feature_heatmap.py --model_yaml ... --image_path ./data/demo.jpg --channels 4 --save_npy

  # 只可视化前 16 个通道 (减少输出文件大小)
  python tools/feature_heatmap.py --model_yaml ... --image_path ./data/demo.jpg --max_channels 16
        """,
    )

    parser.add_argument("--model_yaml", type=str, default=None,
                        help="模型 YAML 配置文件路径 (默认: 使用 YOLO11n)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="输入图片路径 (必需)")
    parser.add_argument("--img_size", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--channels", type=int, default=3,
                        help="输入通道数 (RGB=3, RGBT=4, RGBRGB6C=6, 默认: 3)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (默认: cpu, 可选: cuda:0)")
    parser.add_argument("--layer_names", type=int, nargs="+", default=None,
                        help="要可视化的层索引列表 (默认: 所有层, 例如: --layer_names 2 5 10)")
    parser.add_argument("--max_channels", type=int, default=32,
                        help="每层最大可视化的通道数 (默认: 32)")
    parser.add_argument("--save_dir", type=str, default="runs/feature_heatmap",
                        help="热力图保存目录 (默认: ./runs/feature_heatmap)")
    parser.add_argument("--save_npy", action="store_true", default=False,
                        help="是否同时保存 .npy 文件 (默认: 不保存)")

    args = parser.parse_args()

    # ===== 检查输入图片是否存在 =====
    if not os.path.exists(args.image_path):
        print(f"{RED}错误: 图片文件未找到: {args.image_path}{RESET}")
        return

    # ===== 设备选择 =====
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"{BLUE}使用设备: {device}{RESET}")

    # ===== 加载模型 =====
    if args.model_yaml:
        model = load_model(args.model_yaml, args.channels)
    else:
        print(f"{YELLOW}未指定模型 YAML，使用默认 YOLO11n{RESET}")
        from ultralytics import YOLO
        model = YOLO("yolo11n.yaml").model
        model.eval()

    model = model.to(device)

    # ===== 预处理图片 =====
    input_tensor, original_image, original_size = preprocess_image(
        args.image_path, args.img_size, args.channels
    )
    input_tensor = input_tensor.to(device)

    # ===== 注册钩子 =====
    # 确定要监控的层
    hooks = {}
    handles = []

    # 获取模型的所有子模块
    # YOLO 的 model.model 是一个 Sequential 容器
    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        modules = model.model
    elif isinstance(model, nn.Sequential):
        modules = model
    else:
        # 对于其他结构，遍历 named_modules
        modules = {name: mod for name, mod in model.named_modules()}

    if isinstance(modules, nn.Sequential):
        # Sequential 容器: 按索引访问
        for idx, module in enumerate(modules):
            # 如果指定了层索引，只监控指定的层
            if args.layer_names is not None and idx not in args.layer_names:
                continue

            hook = FeatureHook()
            handle = module.register_forward_hook(hook)
            handles.append(handle)
            hooks[idx] = hook
    else:
        # 对于 named_modules 的情况
        for name, module in modules.items():
            if args.layer_names is not None and name not in [str(n) for n in args.layer_names]:
                continue

            hook = FeatureHook()
            handle = module.register_forward_hook(hook)
            handles.append(handle)
            hooks[name] = hook

    print(f"{BLUE}已注册 {len(hooks)} 个特征图钩子{RESET}")

    # ===== 创建保存目录 =====
    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始图像 (用于对比参考)
    original_save_path = save_dir / "original_input.jpg"
    cv2.imwrite(str(original_save_path), original_image)
    print(f"  保存原始图像: {original_save_path}")

    # ===== 执行前向传播 =====
    print(f"\n{BLUE}开始前向传播以捕获特征图...{RESET}")

    try:
        with torch.no_grad():
            _ = model(input_tensor)
    except Exception as e:
        print(f"{RED}前向传播出错: {e}{RESET}")
        import traceback
        traceback.print_exc()

        # 清除钩子
        for handle in handles:
            handle.remove()
        return

    # ===== 保存特征图 =====
    print(f"\n{BLUE}保存特征图热力图...{RESET}")

    saved_count = 0
    for layer_id, hook in hooks.items():
        if hook.features is not None:
            save_heatmap(
                hook.features,
                save_dir,
                f"layer_{layer_id}",
                hook.module_type,
                max_channels=args.max_channels,
                save_npy=args.save_npy,
            )
            saved_count += 1
        else:
            print(f"{YELLOW}  层 {layer_id} ({hook.module_type}): 无特征图输出{RESET}")

    # ===== 清除钩子 =====
    for handle in handles:
        handle.remove()

    print(f"\n{GREEN}特征图可视化完成!{RESET}")
    print(f"  共保存 {saved_count} 层特征图")
    print(f"  保存路径: {save_dir.absolute()}")


if __name__ == "__main__":
    main()
