"""
检测错误分析可视化工具 (TP/FP/FN Visualization)

功能:
  将模型预测结果与真值标签进行比对，可视化:
    - TP (True Positive): 正确检测 - 绿色框
    - FN (False Negative): 漏检 - 红色框 (真值中有但模型没检测到)
    - FP (False Positive): 误检 - 蓝色框 (模型误检了不存在的目标)
  统计每张图片和整体的 TP/FP/FN 数量，输出汇总报告。

用法示例:
  # 基本用法
  python tools/error_analysis.py --gt_dir ./datasets/val/labels --pred_dir ./runs/detect/predict/labels --image_dir ./datasets/val/images

  # 设置 IoU 阈值
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --iou_threshold 0.5

  # 指定类别名称
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --classes person car bus

  # 指定图片后缀
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --image_ext .png

参考来源: objectdetection-tricks/tricks_1.py
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================================
# 终端颜色代码
# ============================================================================
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def xywh2xyxy(boxes):
    """
    将 YOLO 格式的边界框坐标 (x_center, y_center, width, height) 转换为
    左上角+右下角格式 (x1, y1, x2, y2)。

    参数:
        boxes (ndarray): 形状为 (N, 4) 的数组，每行为 [x_center, y_center, width, height]

    返回:
        ndarray: 形状为 (N, 4) 的数组，每行为 [x1, y1, x2, y2]
    """
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - w/2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - h/2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2 = x_center + w/2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2 = y_center + h/2
    return converted


def compute_iou(boxes1, boxes2):
    """
    计算两组边界框之间的 IoU (Intersection over Union)。

    参数:
        boxes1 (ndarray): 形状为 (N, 4) 的第一组框 [x1, y1, x2, y2]
        boxes2 (ndarray): 形状为 (M, 4) 的第二组框 [x1, y1, x2, y2]

    返回:
        ndarray: 形状为 (N, M) 的 IoU 矩阵
    """
    # 计算交集区域的左上角和右下角坐标
    x_left = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)    # (N, M)
    y_top = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)     # (N, M)
    x_right = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)   # (N, M)
    y_bottom = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)  # (N, M)

    # 计算交集面积 (负值表示无重叠，取0)
    intersection = np.maximum(0, x_right - x_left + 1) * np.maximum(0, y_bottom - y_top + 1)

    # 计算各框面积
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)  # (M,)

    # 计算并集面积
    union = area1[:, None] + area2[None, :] - intersection

    # IoU = 交集 / 并集
    return intersection / np.maximum(union, 1e-6)


def draw_box(img, box, color, thickness=2):
    """
    在图像上绘制矩形边界框。

    参数:
        img (ndarray): 输入图像
        box (array-like): [x1, y1, x2, y2] 坐标
        color (tuple): (B, G, R) 颜色
        thickness (int): 线宽

    返回:
        ndarray: 绘制后的图像
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def load_yolo_labels(file_path, img_width, img_height):
    """
    加载 YOLO 格式标签文件，返回类别ID和像素坐标。

    参数:
        file_path (str): 标签文件路径
        img_width (int): 图像宽度
        img_height (int): 图像高度

    返回:
        tuple: (ndarray, ndarray)
            - class_ids: 形状为 (N,) 的类别ID数组
            - boxes_xyxy: 形状为 (N, 4) 的边界框数组 [x1, y1, x2, y2] (像素坐标)
    """
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return np.array([]), np.array([]).reshape(0, 4)

    if not lines:
        return np.array([]), np.array([]).reshape(0, 4)

    data_list = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            values = [float(x) for x in parts[:5]]
            data_list.append(values)
        except ValueError:
            continue

    if not data_list:
        return np.array([]), np.array([]).reshape(0, 4)

    data = np.array(data_list, dtype=np.float32)
    class_ids = data[:, 0].astype(int)

    # YOLO 格式: [class_id, x_center, y_center, width, height] (归一化)
    boxes_yolo = data[:, 1:5]

    # 转换为像素坐标
    boxes_xyxy = xywh2xyxy(boxes_yolo)
    boxes_xyxy[:, [0, 2]] *= img_width   # x1, x2
    boxes_xyxy[:, [1, 3]] *= img_height  # y1, y2

    return class_ids, boxes_xyxy


# ============================================================================
# 核心分析函数
# ============================================================================
def analyze_errors(
    gt_dir, pred_dir, image_dir, classes=None,
    iou_threshold=0.45, image_ext=".jpg", save_dir="error_vis",
):
    """
    分析检测结果中的 TP/FP/FN 并生成可视化。

    匹配策略:
      1. 对每张图片，计算所有真值框和预测框的 IoU
      2. 对每个真值框，找到 IoU 最高的预测框 (且类别相同)
      3. 如果最高 IoU >= 阈值，视为 TP；否则视为 FN
      4. 未被匹配的预测框视为 FP

    参数:
        gt_dir (str): 真值标签目录 (YOLO 格式)
        pred_dir (str): 预测结果目录 (YOLO 格式)
        image_dir (str): 原始图片目录
        classes (list): 类别名称列表 (可选)
        iou_threshold (float): IoU 匹配阈值 (默认: 0.45)
        image_ext (str): 图片文件后缀 (默认: .jpg)
        save_dir (str): 可视化结果保存目录
    """
    # 可视化颜色定义
    # 绿色: TP (正确检测)  红色: FN (漏检)  蓝色: FP (误检)
    COLOR_TP = (0, 255, 0)    # 绿色
    COLOR_FN = (0, 0, 255)    # 红色
    COLOR_FP = (255, 0, 0)    # 蓝色

    # 创建保存目录 (如果已存在则清空)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有标签文件列表
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]

    if not gt_files:
        print(f"{RED}错误: 在 {gt_dir} 中未找到任何 .txt 标签文件{RESET}")
        return

    print(f"{BLUE}开始分析, 共 {len(gt_files)} 张图片{RESET}")
    print(f"  IoU 阈值: {iou_threshold}")
    print(f"  标签目录: {gt_dir}")
    print(f"  预测目录: {pred_dir}")
    print(f"  图片目录: {image_dir}")

    # ===== 全局统计 =====
    total_tp, total_fn, total_fp = 0, 0, 0
    per_image_results = []  # 保存每张图片的统计信息

    # 逐图片处理
    for gt_file in tqdm(gt_files, desc="分析检测结果"):
        # 文件名 (不含后缀)
        stem = os.path.splitext(gt_file)[0]

        # 读取图片
        img_path = os.path.join(image_dir, f"{stem}{image_ext}")
        image = cv2.imread(img_path)

        if image is None:
            # 尝试其他后缀
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                alt_path = os.path.join(image_dir, f"{stem}{ext}")
                if os.path.exists(alt_path):
                    image = cv2.imread(alt_path)
                    found = True
                    break
            if not found:
                print(f"{RED}  图片未找到: {img_path}{RESET}")
                per_image_results.append((stem, 0, 0, 0))
                continue

        h, w = image.shape[:2]

        # ---- 加载真值标签 ----
        gt_path = os.path.join(gt_dir, gt_file)
        gt_class_ids, gt_boxes = load_yolo_labels(gt_path, w, h)

        # ---- 加载预测结果 ----
        pred_path = os.path.join(pred_dir, gt_file)
        if os.path.exists(pred_path):
            pred_class_ids, pred_boxes = load_yolo_labels(pred_path, w, h)
        else:
            pred_class_ids, pred_boxes = np.array([]), np.array([]).reshape(0, 4)

        # ---- IoU 匹配 ----
        tp_count, fn_count, fp_count = 0, 0, 0
        matched_pred = set()  # 记录已匹配的预测框索引

        # 对每个真值框，寻找最佳匹配的预测框
        for gt_idx in range(len(gt_class_ids)):
            if len(pred_class_ids) == 0:
                # 没有预测框，全部视为漏检
                draw_box(image, gt_boxes[gt_idx], COLOR_FN)
                fn_count += 1
                continue

            # 计算当前真值框与所有预测框的 IoU
            ious = compute_iou(gt_boxes[gt_idx:gt_idx + 1], pred_boxes)[0]
            # 按 IoU 从高到低排序
            sorted_indices = ious.argsort()[::-1]

            matched = False
            for p_idx in sorted_indices:
                if ious[p_idx] < iou_threshold:
                    # IoU 低于阈值，停止搜索
                    break
                if p_idx in matched_pred:
                    # 该预测框已被其他真值框匹配
                    continue
                if gt_class_ids[gt_idx] != pred_class_ids[p_idx]:
                    # 类别不匹配
                    continue

                # ---- 匹配成功: 视为 TP ----
                draw_box(image, pred_boxes[p_idx], COLOR_TP)
                matched_pred.add(p_idx)
                tp_count += 1
                matched = True
                break

            if not matched:
                # ---- 未匹配到合适的预测框: 视为 FN ----
                draw_box(image, gt_boxes[gt_idx], COLOR_FN)
                fn_count += 1

        # ---- 剩余的未匹配预测框: 视为 FP ----
        for p_idx in range(len(pred_class_ids)):
            if p_idx not in matched_pred:
                draw_box(image, pred_boxes[p_idx], COLOR_FP)
                fp_count += 1

        # ---- 保存可视化结果 ----
        save_path = os.path.join(save_dir, f"{stem}{image_ext}")
        cv2.imwrite(save_path, image)

        # ---- 记录本次统计 ----
        per_image_results.append((stem, tp_count, fn_count, fp_count))
        total_tp += tp_count
        total_fn += fn_count
        total_fp += fp_count

    # ===== 输出汇总报告 =====
    print(f"\n{ORANGE}{'=' * 60}{RESET}")
    print(f"{ORANGE}检测错误分析汇总{RESET}")
    print(f"{ORANGE}{'=' * 60}{RESET}")

    # 表格头
    print(f"{'图片名称':<40} {'TP':>6} {'FN':>6} {'FP':>6} {'总计':>6}")
    print("-" * 64)

    # 每张图片的结果
    for stem, tp, fn, fp in per_image_results:
        total = tp + fn + fp
        print(f"{stem:<40} {tp:>6} {fn:>6} {fp:>6} {total:>6}")

    # 分隔线
    print("-" * 64)

    # 总计
    grand_total = total_tp + total_fn + total_fp
    print(f"{'总计':<40} {total_tp:>6} {total_fn:>6} {total_fp:>6} {grand_total:>6}")

    # 指标计算
    # Precision = TP / (TP + FP): 预测为正类的样本中，实际为正类的比例
    # Recall    = TP / (TP + FN): 实际为正类的样本中，被正确预测的比例
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{BLUE}性能指标 (IoU={iou_threshold}):{RESET}")
    print(f"  Precision: {precision:.3f}  ({total_tp}/{total_tp + total_fp})")
    print(f"  Recall:    {recall:.3f}  ({total_tp}/{total_tp + total_fn})")
    print(f"  F1-score:  {f1:.3f}")

    print(f"\n{GREEN}可视化结果保存至: {os.path.abspath(save_dir)}{RESET}")
    print(f"{GREEN}分析完成!{RESET}")

    # 将结果写入文本文件
    result_file = os.path.join(save_dir, "analysis_result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"检测错误分析结果 (IoU阈值: {iou_threshold})\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'图片名称':<40} {'TP':>6} {'FN':>6} {'FP':>6} {'总计':>6}\n")
        f.write("-" * 64 + "\n")
        for stem, tp, fn, fp in per_image_results:
            total = tp + fn + fp
            f.write(f"{stem:<40} {tp:>6} {fn:>6} {fp:>6} {total:>6}\n")
        f.write("-" * 64 + "\n")
        f.write(f"{'总计':<40} {total_tp:>6} {total_fn:>6} {total_fp:>6} {grand_total:>6}\n")
        f.write(f"\nPrecision: {precision:.3f}\n")
        f.write(f"Recall:    {recall:.3f}\n")
        f.write(f"F1-score:  {f1:.3f}\n")

    print(f"{BLUE}详细结果已保存至: {result_file}{RESET}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """解析命令行参数并执行错误分析。"""
    parser = argparse.ArgumentParser(
        description="检测错误分析可视化工具 (TP/FP/FN Visualization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 基础分析
  python tools/error_analysis.py --gt_dir ./datasets/val/labels --pred_dir ./runs/detect/predict/labels --image_dir ./datasets/val/images

  # 设置 IoU 阈值
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --iou_threshold 0.5

  # 指定类别名称和图片后缀
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --classes person car bus --image_ext .png

  # 指定输出目录
  python tools/error_analysis.py --gt_dir ... --pred_dir ... --image_dir ... --save_dir ./my_error_vis
        """,
    )

    parser.add_argument("--gt_dir", type=str, required=True,
                        help="真值标签目录路径 (YOLO 格式 .txt 文件, 必需)")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="预测结果目录路径 (YOLO 格式 .txt 文件, 必需)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="原始图片目录路径 (必需)")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="类别名称列表 (可选, 仅用于输出显示)")
    parser.add_argument("--iou_threshold", type=float, default=0.45,
                        help="IoU 匹配阈值 (默认: 0.45, 范围: 0~1)")
    parser.add_argument("--image_ext", type=str, default=".jpg",
                        help="图片文件后缀 (默认: .jpg)")
    parser.add_argument("--save_dir", type=str, default="error_vis",
                        help="可视化结果保存目录 (默认: ./error_vis)")

    args = parser.parse_args()

    # ===== 参数校验 =====
    if not os.path.exists(args.gt_dir):
        print(f"{RED}错误: 真值标签目录不存在: {args.gt_dir}{RESET}")
        return
    if not os.path.exists(args.pred_dir):
        print(f"{RED}错误: 预测结果目录不存在: {args.pred_dir}{RESET}")
        return
    if not os.path.exists(args.image_dir):
        print(f"{RED}错误: 图片目录不存在: {args.image_dir}{RESET}")
        return

    if args.iou_threshold < 0 or args.iou_threshold > 1:
        print(f"{RED}错误: IoU 阈值必须在 [0, 1] 范围内{RESET}")
        return

    # ===== 执行分析 =====
    analyze_errors(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        image_dir=args.image_dir,
        classes=args.classes,
        iou_threshold=args.iou_threshold,
        image_ext=args.image_ext,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
