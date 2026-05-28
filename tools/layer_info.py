"""
模型逐层特征形状调试工具 (Per-Layer Feature Shape Debugger)

功能:
  为模型所有层注册 forward hook，打印每一层的输入/输出张量形状。
  对于融合架构 (如 RGBT 双模态融合)，此工具可以快速定位维度不匹配的位置。
  支持 YOLO 和 RTDETR 系列模型。

用法示例:
  # 分析默认模型
  python tools/layer_info.py

  # 分析指定模型 YAML
  python tools/layer_info.py --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml

  # 自定义输入尺寸
  python tools/layer_info.py --model_yaml ... --img_size 640 --batch_size 1 --channels 4

参考来源: objectdetection-tricks/tricks_13.py
"""

import argparse
import warnings

import torch
import torch.nn as nn

# 忽略非关键警告 (如 YOLO 版本兼容性警告)
warnings.filterwarnings("ignore")


# ============================================================================
# 终端颜色代码
# ============================================================================
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


class LayerShapeHook:
    """
    层形状钩子管理器。

    为模型的每一层注册 forward hook，在前向传播时自动打印该层的
    输入形状和输出形状。这对于调试多模态融合模型中的张量维度问题非常有用。

    用法:
        hook_manager = LayerShapeHook(model)
        model(input_data)  # 自动打印每一层的形状信息
        hook_manager.remove()  # 清除所有钩子
    """

    def __init__(self, model, show_input=True, show_output=True):
        """
        初始化钩子管理器。

        参数:
            model (nn.Module): PyTorch 模型
            show_input (bool): 是否打印输入形状
            show_output (bool): 是否打印输出形状
        """
        self.model = model
        self.show_input = show_input
        self.show_output = show_output
        self.handles = []   # 保存所有钩子的句柄，用于后续移除
        self.layer_count = 0  # 层计数器

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """
        为模型的所有子模块注册 forward hook。

        对于 Sequential 容器，遍历其所有子模块并分别注册。
        对于普通模块，直接注册。
        """
        # 如果模型本身是 Sequential，遍历每一层
        if isinstance(self.model, nn.Sequential):
            for idx, module in enumerate(self.model):
                self._register_single_hook(module, idx)
        else:
            # 遍历模型的所有命名子模块
            for name, module in self.model.named_modules():
                # 跳过容器类模块 (Sequential, ModuleList) 和模型本身
                # 只对实际的运算模块 (Conv, BatchNorm, C3k2 等) 注册钩子
                if isinstance(module, (nn.Sequential, nn.ModuleList)):
                    continue
                # 跳过没有可训练参数的模块 (如 nn.Identity)
                self._register_named_hook(module, name)

    def _register_single_hook(self, module, idx):
        """
        为单个模块注册钩子 (基于索引)。

        参数:
            module (nn.Module): 要注册的模块
            idx (int): 层的索引编号
        """
        handle = module.register_forward_hook(
            lambda mod, inp, out, idx=idx: self._hook_fn(idx, mod, inp, out)
        )
        self.handles.append(handle)
        self.layer_count += 1

    def _register_named_hook(self, module, name):
        """
        为单个模块注册钩子 (基于名称)。

        参数:
            module (nn.Module): 要注册的模块
            name (str): 模块的名称路径
        """
        handle = module.register_forward_hook(
            lambda mod, inp, out, name=name: self._named_hook_fn(name, mod, inp, out)
        )
        self.handles.append(handle)

    @staticmethod
    def _format_tensor_shape(data):
        """
        格式化张量形状为可读字符串。

        参数:
            data: 可以是 torch.Tensor、list、tuple、dict 或 None

        返回:
            str: 格式化后的形状字符串
        """
        if data is None:
            return "None"

        if isinstance(data, torch.Tensor):
            return str(list(data.shape))

        if isinstance(data, (list, tuple)):
            shapes = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    shapes.append(str(list(item.shape)))
                elif isinstance(item, dict):
                    # 处理 YOLO Head 输出 (包含 "one2one" 和 "one2many" 的 dict)
                    dict_shapes = {}
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            dict_shapes[k] = str(list(v.shape))
                        elif isinstance(v, (list, tuple)):
                            dict_shapes[k] = [str(list(x.shape)) if isinstance(x, torch.Tensor) else str(type(x).__name__) for x in v]
                        else:
                            dict_shapes[k] = str(type(v).__name__)
                    shapes.append(f"dict({dict_shapes})")
                else:
                    shapes.append(str(type(item).__name__))
            return ", ".join(shapes)

        if isinstance(data, dict):
            parts = []
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    parts.append(f"{k}: {list(v.shape)}")
                elif isinstance(v, (list, tuple)):
                    tensor_shapes = [str(list(x.shape)) if isinstance(x, torch.Tensor) else str(type(x).__name__) for x in v]
                    parts.append(f"{k}: [{', '.join(tensor_shapes)}]")
                else:
                    parts.append(f"{k}: {str(type(v).__name__)}")
            return "{" + ", ".join(parts) + "}"

        return str(type(data).__name__)

    def _hook_fn(self, idx, module, inp, out):
        """
        基于索引的钩子回调函数。

        参数:
            idx (int): 层索引
            module (nn.Module): 当前层模块
            inp (tuple): 输入张量元组
            out: 输出 (tensor, list, tuple 或 dict)
        """
        module_type = module.__class__.__name__
        in_shape = self._format_tensor_shape(inp[0] if isinstance(inp, (list, tuple)) and len(inp) == 1 else inp)
        out_shape = self._format_tensor_shape(out)

        if self.show_input and self.show_output:
            print(f"  [{idx:>3}] {module_type:<50} in: {in_shape:<40} out: {out_shape}")
        elif self.show_output:
            print(f"  [{idx:>3}] {module_type:<50} out: {out_shape}")

    def _named_hook_fn(self, name, module, inp, out):
        """
        基于名称的钩子回调函数。

        参数:
            name (str): 模块名称路径
            module (nn.Module): 当前层模块
            inp (tuple): 输入张量元组
            out: 输出 (tensor, list, tuple 或 dict)
        """
        module_type = module.__class__.__name__
        out_shape = self._format_tensor_shape(out)

        if self.show_output:
            print(f"  {name:<60} [{module_type:<30}] out: {out_shape}")

    def remove(self):
        """移除所有已注册的钩子。"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        print(f"{GREEN}已移除 {self.layer_count} 个钩子{RESET}")


# ============================================================================
# 模型加载函数
# ============================================================================
def load_model(model_yaml, channels):
    """
    从 YAML 配置文件加载模型。

    参数:
        model_yaml (str): 模型 YAML 文件路径
        channels (int): 输入通道数 (RGB=3, RGBT=4, RGBRGB6C=6)

    返回:
        nn.Module: 加载好的 PyTorch 模型
    """
    from ultralytics import YOLO

    print(f"{BLUE}正在加载模型: {model_yaml}{RESET}")
    print(f"  输入通道数: {channels}")

    try:
        # 使用 YOLO API 加载模型
        # YOLO 类会自动解析 YAML 并构建完整的模型
        model = YOLO(model_yaml).model
    except Exception as e:
        print(f"{RED}加载模型失败: {e}{RESET}")
        print(f"{YELLOW}尝试直接解析 YAML 构建模型...{RESET}")

        # 备用方案: 直接使用 parse_model
        try:
            from ultralytics.nn.tasks import parse_model
            import yaml

            with open(model_yaml, "r") as f:
                yaml_cfg = yaml.safe_load(f)

            model, _ = parse_model(yaml_cfg, ch=channels)
        except Exception as e2:
            print(f"{RED}备用加载也失败: {e2}{RESET}")
            raise

    # 切换到评估模式
    model.eval()
    print(f"{GREEN}模型加载成功! 参数量: {sum(p.numel() for p in model.parameters()):,}{RESET}")

    return model


# ============================================================================
# 主函数
# ============================================================================
def main():
    """解析命令行参数并执行逐层形状分析。"""
    parser = argparse.ArgumentParser(
        description="模型逐层特征形状调试工具 (Layer Feature Shape Debugger)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 使用默认 YOLO11n 模型
  python tools/layer_info.py

  # 分析自定义融合模型
  python tools/layer_info.py --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml

  # 自定义输入尺寸 (RGBT 4通道)
  python tools/layer_info.py --model_yaml .../yolo26-RGBT-midfusion.yaml --img_size 640 --channels 4

  # 只看输出形状，不显示输入
  python tools/layer_info.py --model_yaml ... --no_input
        """,
    )

    parser.add_argument("--model_yaml", type=str, default=None,
                        help="模型 YAML 配置文件路径 (默认: 使用 YOLO11n)")
    parser.add_argument("--img_size", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小 (默认: 1)")
    parser.add_argument("--channels", type=int, default=3,
                        help="输入通道数 (RGB=3, RGBT=4, RGBRGB6C=6, 默认: 3)")
    parser.add_argument("--no_input", action="store_true", default=False,
                        help="不显示输入形状，只显示输出形状")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (默认: cpu, 可选: cuda:0)")

    args = parser.parse_args()

    # ===== 设备选择 =====
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"{BLUE}使用设备: {device}{RESET}")

    # ===== 加载模型 =====
    if args.model_yaml:
        model = load_model(args.model_yaml, args.channels)
    else:
        # 使用默认 YOLO11n 模型
        print(f"{YELLOW}未指定模型 YAML，使用默认 YOLO11n{RESET}")
        from ultralytics import YOLO
        model = YOLO("yolo11n.yaml").model
        model.eval()

    model = model.to(device)

    # ===== 创建随机输入 =====
    dummy_input = torch.randn(args.batch_size, args.channels, args.img_size, args.img_size).to(device)
    print(f"{BLUE}输入张量形状: {list(dummy_input.shape)}{RESET}")
    print(f"{ORANGE}{'=' * 120}{RESET}")
    print(f"{'Layer Info':^120}")
    print(f"{ORANGE}{'=' * 120}{RESET}")

    # ===== 注册钩子并运行前向传播 =====
    hook_manager = LayerShapeHook(model, show_input=not args.no_input, show_output=True)

    print(f"\n{BLUE}开始逐层分析 (共 {hook_manager.layer_count} 个注册点)...{RESET}\n")

    try:
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"\n{RED}前向传播出错: {e}{RESET}")
        print(f"{YELLOW}提示: 检查模型 YAML 中的通道数是否与 --channels 参数匹配。{RESET}")
        print(f"{YELLOW}      RGBT 模型需要 --channels 4, RGBRGB6C 需要 --channels 6{RESET}")

        # 打印详细错误堆栈
        import traceback
        traceback.print_exc()
    finally:
        # 清除钩子，避免内存泄漏
        hook_manager.remove()

    print(f"\n{GREEN}逐层形状分析完成!{RESET}")


if __name__ == "__main__":
    main()
