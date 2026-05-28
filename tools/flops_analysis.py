"""
模型逐层 FLOPs/Params 分析工具 (Per-Layer FLOPs/Params Analysis)

功能:
  使用 thop.profile 计算模型的逐层 FLOPs (浮点运算次数) 和参数量。
  生成格式化表格输出，并标记超出阈值的层 (如参数量过大、计算量过大)。
  适用于 YOLO 和 RTDETR 系列模型。

用法示例:
  # 分析默认模型
  python tools/flops_analysis.py

  # 分析指定模型
  python tools/flops_analysis.py --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml

  # 自定义输入尺寸和通道数
  python tools/flops_analysis.py --model_yaml ... --img_size 640 --channels 4 --batch_size 1

  # 设置 FLOPs 告警阈值 (超过 1G FLOPs 的层标红)
  python tools/flops_analysis.py --model_yaml ... --flops_warn 1.0

参考来源: objectdetection-tricks/tricks_10.py
"""

import argparse
import warnings

import torch
from prettytable import PrettyTable

# 忽略非关键警告
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


def clever_format(num, format_str="%.3f"):
    """
    将大数字格式化为人类可读的形式 (自动选择单位)。

    类似于 thop.clever_format，但在此独立实现以避免依赖问题。

    参数:
        num (float): 原始数字
        format_str (str): 格式化字符串

    返回:
        str: 格式化后的字符串，如 "1.234G", "567.890M"
    """
    if num >= 1e9:
        return f"{format_str % (num / 1e9)}G"
    elif num >= 1e6:
        return f"{format_str % (num / 1e6)}M"
    elif num >= 1e3:
        return f"{format_str % (num / 1e3)}K"
    else:
        return f"{format_str % num}"


def format_with_warning(value, unit, threshold, format_str="%.3f"):
    """
    格式化值并添加颜色警告 (如果超过阈值)。

    参数:
        value (float): 原始值 (FLOPs 或 Params)
        unit (str): 单位类型 ("G", "M", "K", "")
        threshold (float): 告警阈值 (与 value 同单位)
        format_str (str): 数值格式化字符串

    返回:
        str: 格式化后的字符串 (可能包含 ANSI 颜色码)
    """
    # 确定实际的单位缩放
    if unit == "G":
        scaled = value / 1e9
    elif unit == "M":
        scaled = value / 1e6
    elif unit == "K":
        scaled = value / 1e3
    else:
        scaled = value

    formatted = f"{format_str % scaled}{unit}"

    # 如果超过阈值，标记为红色
    if scaled > threshold:
        return f"{RED}{formatted} !{RESET}"
    return formatted


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

    # 融合 Conv + BN 层以提高准确度
    model.fuse()
    model.eval()

    print(f"{GREEN}模型加载成功! 总参数量: {sum(p.numel() for p in model.parameters()):,}{RESET}")

    return model


# ============================================================================
# FLOPs 分析函数
# ============================================================================
def analyze_flops(model, input_tensor, batch_size, flops_warn=1.0, params_warn=5.0):
    """
    分析模型的逐层 FLOPs 和参数量。

    参数:
        model (nn.Module): PyTorch 模型
        input_tensor (torch.Tensor): 输入张量
        batch_size (int): 批次大小 (用于归一化 FLOPs)
        flops_warn (float): FLOPs 警告阈值 (单位: G)，超过此值的层被标记
        params_warn (float): Params 警告阈值 (单位: M)，超过此值的层被标记
    """
    try:
        from thop import profile
    except ImportError:
        print(f"{RED}错误: 需要安装 thop 库。请执行: pip install thop{RESET}")
        print(f"{YELLOW}或者使用以下命令安装: pip install thop --upgrade{RESET}")
        return

    print(f"\n{BLUE}开始逐层 FLOPs/Params 分析...{RESET}")
    print(f"  FLOPs 告警阈值: {flops_warn}G/层")
    print(f"  Params 告警阈值: {params_warn}M/层")

    # ===== 执行 profile =====
    try:
        # ret_layer_info=True 会返回每个子模块的详细统计
        total_flops, total_params, layers_info = profile(
            model, [input_tensor],
            verbose=False,
            ret_layer_info=True,
        )
    except Exception as e:
        print(f"{RED}FLOPs 分析失败: {e}{RESET}")
        print(f"{YELLOW}提示: 某些自定义算子可能不被 thop 支持。{RESET}")
        print(f"{YELLOW}      可以尝试更新 thop 或忽略特定算子。{RESET}")
        import traceback
        traceback.print_exc()
        return

    # ===== 格式化总结果 =====
    # 对于 YOLO，实际 FLOPs 需要乘以 2 (前向+反向) 并除以 batch_size 进行归一化
    effective_flops = total_flops * 2 / batch_size
    total_flops_str = clever_format(effective_flops)
    total_params_str = clever_format(total_params)

    print(f"\n{GREEN}总 FLOPs: {total_flops_str}  (原始: {total_flops:.3e}, 有效: {effective_flops:.3e}){RESET}")
    print(f"{GREEN}总 Params: {total_params_str}  (原始: {total_params:,}){RESET}")

    # ===== 创建逐层表格 =====
    table = PrettyTable()
    table.title = f"逐层 FLOPs/Params 分析 (总FLOPs: {total_flops_str}, 总Params: {total_params_str})"
    table.field_names = ["层索引 (Layer ID)", "FLOPs", "Params", "模块类型 (Module Type)"]

    # 检查是否存在按索引组织的层信息
    # thop 的返回格式: layers_info['model'][2] 包含 {layer_id: (flops, params)} 的字典
    if "model" in layers_info and 2 in layers_info["model"]:
        layer_flops_dict = layers_info["model"][2]
    else:
        # 尝试从其他结构中提取
        print(f"{YELLOW}警告: 无法解析逐层信息结构。可能是 thop 版本不同导致的。{RESET}")
        print(f"{YELLOW}layers_info 的 keys: {list(layers_info.keys()) if hasattr(layers_info, 'keys') else type(layers_info)}{RESET}")
        layer_flops_dict = {}

    # 如果找到了逐层信息，逐行输出
    if layer_flops_dict:
        for layer_id in sorted(layer_flops_dict.keys(), key=int):
            data = layer_flops_dict[layer_id]
            layer_flops = data[0] * 2 / batch_size  # 乘以2并除以batch
            layer_params = data[1]

            flops_str = format_with_warning(layer_flops, "G", flops_warn)
            params_str = format_with_warning(layer_params, "M", params_warn)

            # 尝试获取模块类型 (如果可以从模型结构中读取)
            module_type = ""
            try:
                layer_idx = int(layer_id)
                if hasattr(model, "model") and isinstance(model.model, (list, torch.nn.Sequential)):
                    if layer_idx < len(model.model):
                        module_type = type(model.model[layer_idx]).__name__
            except (ValueError, IndexError, AttributeError):
                module_type = ""

            table.add_row([layer_id, flops_str, params_str, module_type])

        # 添加总计行
        total_flops_str = clever_format(effective_flops)
        total_params_str = clever_format(total_params)
        table.add_row(["总计 (Total)", total_flops_str, total_params_str, ""])

    else:
        # 如果没有逐层信息，至少显示总量
        table.add_row(["N/A", total_flops_str, total_params_str, "所有层合并"])

    # 设置对齐方式
    table.align["层索引 (Layer ID)"] = "r"
    table.align["FLOPs"] = "r"
    table.align["Params"] = "r"
    table.align["模块类型 (Module Type)"] = "l"

    print(f"\n{ORANGE}{'=' * 100}{RESET}")
    print(table)
    print(f"{ORANGE}{'=' * 100}{RESET}")

    # ===== 汇总分析 =====
    if layer_flops_dict:
        all_layer_flops = [v[0] * 2 / batch_size for v in layer_flops_dict.values()]
        all_layer_params = [v[1] for v in layer_flops_dict.values()]

        max_flops_layer = max(layer_flops_dict.keys(), key=lambda k: layer_flops_dict[k][0] * 2 / batch_size)
        max_params_layer = max(layer_flops_dict.keys(), key=lambda k: layer_flops_dict[k][1])

        print(f"\n{BLUE}分析汇总:{RESET}")
        print(f"  总层数 (有统计信息的): {len(layer_flops_dict)}")
        print(f"  平均每层 FLOPs: {clever_format(sum(all_layer_flops) / len(all_layer_flops))}")
        print(f"  平均每层 Params: {clever_format(sum(all_layer_params) / len(all_layer_params))}")
        print(f"  FLOPs 最大层: {max_flops_layer} ({clever_format(layer_flops_dict[max_flops_layer][0] * 2 / batch_size)})")
        print(f"  Params 最大层: {max_params_layer} ({clever_format(layer_flops_dict[max_params_layer][1])})")

        # 标记超过阈值的层
        flops_over = sum(1 for v in layer_flops_dict.values() if v[0] * 2 / batch_size > flops_warn * 1e9)
        params_over = sum(1 for v in layer_flops_dict.values() if v[1] > params_warn * 1e6)
        if flops_over > 0:
            print(f"{RED}  警告: {flops_over} 层的 FLOPs 超过阈值 {flops_warn}G{RESET}")
        if params_over > 0:
            print(f"{RED}  警告: {params_over} 层的参数量超过阈值 {params_warn}M{RESET}")

    return total_flops, total_params


# ============================================================================
# 主函数
# ============================================================================
def main():
    """解析命令行参数并执行逐层 FLOPs/Params 分析。"""
    parser = argparse.ArgumentParser(
        description="模型逐层 FLOPs/Params 分析工具 (Per-Layer FLOPs/Params Analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 分析默认 YOLO11n 模型
  python tools/flops_analysis.py

  # 分析自定义 RGBT 融合模型
  python tools/flops_analysis.py --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml --channels 4

  # 设置告警阈值
  python tools/flops_analysis.py --model_yaml ... --flops_warn 1.0 --params_warn 2.0
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
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备 (默认: cpu, 可选: cuda:0)")
    parser.add_argument("--flops_warn", type=float, default=1.0,
                        help="FLOPs 告警阈值 (单位: Giga, 默认: 1.0G)")
    parser.add_argument("--params_warn", type=float, default=5.0,
                        help="Params 告警阈值 (单位: Million, 默认: 5.0M)")

    args = parser.parse_args()

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
        model.fuse()
        model.eval()

    model = model.to(device)

    # ===== 创建随机输入 =====
    dummy_input = torch.randn(args.batch_size, args.channels, args.img_size, args.img_size).to(device)
    print(f"{BLUE}输入张量形状: {list(dummy_input.shape)}{RESET}")

    # ===== 执行分析 =====
    analyze_flops(model, dummy_input, args.batch_size, args.flops_warn, args.params_warn)

    print(f"\n{GREEN}FLOPs/Params 分析完成!{RESET}")


if __name__ == "__main__":
    main()
