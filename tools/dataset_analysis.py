"""
数据集小/中/大目标分析工具 (Dataset Small/Medium/Large Object Analysis)

功能:
  统计数据集中每个类别的实例数量，并根据像素面积将目标分为三类:
    - 小目标 (Small):  area < 32*32  (1024像素)
    - 中目标 (Medium): 32*32 <= area <= 96*96  (1024~9216像素)
    - 大目标 (Large):  area > 96*96  (9216像素)
  支持 YOLO 格式标签和 COCO JSON 格式。

用法示例:
  # YOLO 格式分析
  python tools/dataset_analysis.py --labels_dir ./datasets/train/labels --images_dir ./datasets/train/images --classes person car bus

  # COCO JSON 格式分析
  python tools/dataset_analysis.py --coco_json ./datasets/annotations/instances_train.json

  # 启用可视化 (在标签上绘制检测框)
  python tools/dataset_analysis.py --labels_dir ./datasets/train/labels --images_dir ./datasets/train/images --visual_box --save_path ./vis_boxes

参考来源: objectdetection-tricks/tricks_15.py
"""

import os
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm


# ============================================================================
# 颜色工具
# ============================================================================
# 预定义的颜色列表，用于可视化时不同类别显示不同颜色
COLOR_LIST = [
    (255, 0, 0),         # 红色
    (0, 255, 0),         # 绿色
    (0, 0, 255),         # 蓝色
    (255, 165, 0),       # 橙色
    (255, 255, 0),       # 黄色
    (0, 255, 255),       # 青色
    (255, 0, 255),       # 品红
    (255, 255, 255),     # 白色
    (128, 0, 0),         # 棕色
    (0, 128, 0),         # 深绿色
    (0, 0, 128),         # 深蓝色
    (128, 128, 0),       # 橄榄色
    (0, 128, 128),       # 蓝绿色
    (128, 0, 128),       # 紫色
    (192, 192, 192),     # 银色
    (255, 99, 71),       # 番茄色
    (0, 255, 127),       # 春绿色
    (255, 105, 180),     # 深粉色
    (70, 130, 180),      # 钢蓝色
]

# 终端颜色代码，用于输出信息高亮
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

# 小/中/大目标的像素面积阈值
# object_info[0] = 32*32 = 1024 (小目标上限)
# object_info[1] = 96*96 = 9216 (大目标下限)
OBJECT_INFO = [32 * 32, 96 * 96]

# 支持的图片后缀
IMAGE_POSTFIX = ["jpg", "png", "bmp", "tif", "jpeg"]


def get_color_by_class(class_id):
    """
    根据类别索引返回固定颜色，确保同一类别在不同图片中颜色一致。

    参数:
        class_id (int): 类别ID

    返回:
        tuple: (B, G, R) 颜色值
    """
    return COLOR_LIST[class_id % len(COLOR_LIST)]


def draw_detections(box, name, color, img):
    """
    在图片上绘制检测框和类别标签。

    参数:
        box (list): [xmin, ymin, xmax, ymax] 边界框坐标 (像素)
        name (str): 类别名称
        color (tuple): (B, G, R) 框颜色
        img (ndarray): 输入图像

    返回:
        ndarray: 绘制后的图像
    """
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))

    # 根据图像大小自适应调整线宽和字体大小
    line_thickness = max(1, int(min(height, width) / 400))
    font_scale = min(height, width) / 1000
    font_thickness = max(1, int(min(height, width) / 400))
    text_offset_y = int(min(height, width) / 100)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
    cv2.putText(
        img, str(name),
        (xmin, ymin - text_offset_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (0, 255, 0), font_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


# ============================================================================
# 数据加载函数
# ============================================================================
def load_yolo_labels(label_path, image_width, image_height):
    """
    加载 YOLO 格式标签文件。
    YOLO 格式: class_id x_center y_center width height
    所有坐标值均为归一化的相对坐标 (0~1)。

    参数:
        label_path (str): 标签文件路径 (*.txt)
        image_width (int): 图像宽度 (像素)
        image_height (int): 图像高度 (像素)

    返回:
        list[dict]: 每个元素包含 class_id, xmin, ymin, xmax, ymax, area
    """
    objects = []
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"{RED}标签文件未找到: {label_path}{RESET}")
        return objects

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            cls_id = int(float(parts[0]))
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
        except (ValueError, IndexError):
            # 跳过格式错误的行
            continue

        # 计算像素坐标 (左上角和右下角)
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        area = width * height

        objects.append({
            "class_id": cls_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "area": area,
        })

    return objects


def load_coco_json(coco_path):
    """
    加载 COCO JSON 格式的标注文件。

    COCO JSON 结构:
    {
        "images": [{"id": 1, "width": ..., "height": ..., "file_name": ...}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x,y,w,h], "area": ...}, ...],
        "categories": [{"id": 1, "name": "person"}, ...]
    }

    参数:
        coco_path (str): COCO JSON 文件路径

    返回:
        tuple: (classes_list, img_to_annots) 类别列表和图像ID到标注的映射
    """
    import json

    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 解析类别信息: 建立 category_id -> category_name 的映射
    categories = {}
    for cat in coco_data.get("categories", []):
        categories[cat["id"]] = cat["name"]

    # 按类别ID排序，确保顺序一致
    sorted_cat_ids = sorted(categories.keys())
    classes_list = [categories[cid] for cid in sorted_cat_ids]

    # 建立 category_id -> 连续索引 的映射
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}

    # 解析图片信息: image_id -> (width, height)
    image_info = {}
    for img in coco_data.get("images", []):
        image_info[img["id"]] = (img["width"], img["height"], img.get("file_name", ""))

    # 解析标注信息: 按 image_id 分组
    img_to_annots = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_info:
            continue

        width, height, fname = image_info[img_id]
        bbox = ann["bbox"]  # COCO 格式: [x, y, w, h]
        x, y, w, h = bbox
        cls_id = cat_id_to_idx[ann["category_id"]]
        area = ann.get("area", w * h)

        obj = {
            "class_id": cls_id,
            "xmin": x,
            "ymin": y,
            "xmax": x + w,
            "ymax": y + h,
            "area": area,
        }

        if img_id not in img_to_annots:
            img_to_annots[img_id] = {
                "width": width,
                "height": height,
                "file_name": fname,
                "objects": [],
            }
        img_to_annots[img_id]["objects"].append(obj)

    print(f"{ORANGE}COCO JSON 加载完成: {len(image_info)} 张图片, {sum(len(v['objects']) for v in img_to_annots.values())} 个标注, {len(classes_list)} 个类别{RESET}")

    return classes_list, img_to_annots


def get_images_and_labels(images_dir, labels_dir):
    """
    匹配图片和标签文件，返回 标签路径 -> 图片路径 的字典。

    参数:
        images_dir (str): 图片目录路径
        labels_dir (str): 标签目录路径

    返回:
        dict: {label_path: image_path} 的映射字典
    """
    # 收集所有标签文件 (.txt)
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    label_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in label_files}

    # 收集所有图片文件 (多种后缀)
    image_dict = {}
    for ext in IMAGE_POSTFIX:
        for p in glob.glob(os.path.join(images_dir, f"*.{ext}")):
            image_dict[os.path.splitext(os.path.basename(p))[0]] = p

    print(f"{ORANGE}图片数量: {len(image_dict)}, 标签数量: {len(label_dict)}{RESET}")

    # 根据文件名 (不含后缀) 进行匹配
    matched = {}
    for stem in label_dict:
        if stem in image_dict:
            matched[label_dict[stem]] = image_dict[stem]

    print(f"匹配成功: {len(matched)} 对{RESET}")
    return matched


# ============================================================================
# 核心统计函数
# ============================================================================
def analyze_dataset_yolo(images_labels_dict, classes, visual_box=False, save_path="visual_boxes"):
    """
    分析 YOLO 格式数据集的 S/M/L 目标分布。

    参数:
        images_labels_dict (dict): {label_path: image_path} 映射
        classes (list): 类别名称列表
        visual_box (bool): 是否生成可视化结果
        save_path (str): 可视化结果保存目录
    """
    if visual_box and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"可视化结果将保存到: {save_path}")

    # 初始化统计字典: 每个类别统计小(s)、中(m)、大(l)目标和总数(num)
    classes_dict = {cls: {"s": 0, "m": 0, "l": 0, "num": 0} for cls in classes}

    # 逐张图片处理
    for label_path, image_path in tqdm(images_labels_dict.items(), desc="分析数据集中"):
        # 读取图片获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            print(f"{RED}图片读取失败，跳过: {image_path}{RESET}")
            continue

        h, w = image.shape[:2]

        # 加载该图片的 YOLO 标签
        objects = load_yolo_labels(label_path, w, h)
        if not objects:
            continue

        for obj in objects:
            cls_id = obj["class_id"]
            area = obj["area"]

            # 检查类别ID是否在有效范围内
            if cls_id >= len(classes):
                print(f"{YELLOW}警告: 类别ID {cls_id} 超出范围 (共 {len(classes)} 类), 跳过{RESET}")
                continue

            class_name = classes[cls_id]
            classes_dict[class_name]["num"] += 1

            # 根据面积分类: small / medium / large
            if area < OBJECT_INFO[0]:
                classes_dict[class_name]["s"] += 1
            elif area > OBJECT_INFO[1]:
                classes_dict[class_name]["l"] += 1
            else:
                classes_dict[class_name]["m"] += 1

        # 可视化: 在图片上绘制所有标注框
        if visual_box:
            for obj in objects:
                cls_id = obj["class_id"]
                if cls_id >= len(classes):
                    continue
                class_name = classes[cls_id]
                box = [obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]]
                image = draw_detections(box, class_name, get_color_by_class(cls_id), image)

            out_path = os.path.join(save_path, os.path.basename(image_path))
            cv2.imwrite(out_path, image)

    # 打印统计表格
    print_summary_table(classes_dict)


def analyze_dataset_coco(coco_path, classes, visual_box=False, save_path="visual_boxes"):
    """
    分析 COCO JSON 格式数据集的 S/M/L 目标分布。

    参数:
        coco_path (str): COCO JSON 文件路径
        classes (list): 类别名称列表
        visual_box (bool): 是否生成可视化结果
        save_path (str): 可视化结果保存目录
    """
    import json

    if visual_box and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # 加载 COCO 数据
    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 解析类别
    cat_id_to_name = {}
    for cat in coco_data.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]

    # 构建 image_id -> image_info 映射
    images_info = {}
    for img in coco_data.get("images", []):
        images_info[img["id"]] = img

    # 初始化统计字典
    classes_dict = {cls: {"s": 0, "m": 0, "l": 0, "num": 0} for cls in classes}

    # 逐标注处理
    for ann in tqdm(coco_data.get("annotations", []), desc="分析COCO标注"):
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, "unknown")

        # 如果类别不在指定列表中则跳过
        if cat_name not in classes_dict:
            continue

        cls_id = classes.index(cat_name)
        bbox = ann["bbox"]  # [x, y, w, h]
        area = bbox[2] * bbox[3]

        classes_dict[cat_name]["num"] += 1

        if area < OBJECT_INFO[0]:
            classes_dict[cat_name]["s"] += 1
        elif area > OBJECT_INFO[1]:
            classes_dict[cat_name]["l"] += 1
        else:
            classes_dict[cat_name]["m"] += 1

        # 可视化
        if visual_box:
            img_id = ann["image_id"]
            if img_id not in images_info:
                continue

            img_info = images_info[img_id]
            img_path = os.path.join(os.path.dirname(coco_path), "..", img_info.get("file_name", ""))
            if not os.path.exists(img_path):
                continue

            save_img_name = f"{img_id:012d}.jpg"
            save_img_path = os.path.join(save_path, save_img_name)
            if not os.path.exists(save_img_path):
                img_array = cv2.imread(img_path)
                if img_array is None:
                    continue
                cv2.imwrite(save_img_path, img_array)
            else:
                img_array = cv2.imread(save_img_path)

            x, y, w_box, h_box = map(int, bbox)
            cv2.rectangle(img_array, (x, y), (x + w_box, y + h_box), get_color_by_class(cls_id), 2)
            cv2.putText(img_array, cat_name, (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(save_img_path, img_array)

    # 打印统计表格
    print_summary_table(classes_dict)


def print_summary_table(classes_dict):
    """
    打印 S/M/L 统计汇总表格。

    参数:
        classes_dict (dict): 统计字典，结构为 {class_name: {"s": int, "m": int, "l": int, "num": int}}
    """
    # 计算各类别总和
    total_s = sum(v["s"] for v in classes_dict.values())
    total_m = sum(v["m"] for v in classes_dict.values())
    total_l = sum(v["l"] for v in classes_dict.values())
    total_num = sum(v["num"] for v in classes_dict.values())

    if total_num == 0:
        print(f"{RED}未检测到任何目标，请检查数据路径和格式。{RESET}")
        return

    # 创建 PrettyTable 表格
    table = PrettyTable()
    table.field_names = ["类别 (Category)", "小目标 S (<1024px)", "中目标 M (1024~9216px)", "大目标 L (>9216px)", "总数 (Total)"]

    # 添加每个类别的行
    for category, values in classes_dict.items():
        s, m, l, num = values["s"], values["m"], values["l"], values["num"]
        if num == 0:
            table.add_row([category, "0 (0.0%)", "0 (0.0%)", "0 (0.0%)", 0])
        else:
            table.add_row([
                category,
                f"{s} ({s / num:.1%})",
                f"{m} ({m / num:.1%})",
                f"{l} ({l / num:.1%})",
                num,
            ])

    # 添加总计行
    table.add_row([
        "总计 (All)",
        f"{total_s} ({total_s / total_num:.1%})",
        f"{total_m} ({total_m / total_num:.1%})",
        f"{total_l} ({total_l / total_num:.1%})",
        total_num,
    ])

    # 左对齐类别列
    table.align["类别 (Category)"] = "l"

    print(f"\n{GREEN}数据集 S/M/L 目标分布统计:{RESET}")
    print(table)

    # 额外输出汇总信息
    print(f"\n{BLUE}汇总信息:{RESET}")
    print(f"  总目标数: {total_num}")
    print(f"  小目标数: {total_s} ({total_s / total_num:.1%})")
    print(f"  中目标数: {total_m} ({total_m / total_num:.1%})")
    print(f"  大目标数: {total_l} ({total_l / total_num:.1%})")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """解析命令行参数并执行数据集分析。"""
    parser = argparse.ArgumentParser(
        description="数据集小/中/大目标分析工具 (Dataset S/M/L Analysis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # YOLO 格式分析
  python tools/dataset_analysis.py --labels_dir ./datasets/train/labels --images_dir ./datasets/train/images --classes person car bus

  # COCO JSON 格式分析 (二选一)
  python tools/dataset_analysis.py --coco_json ./datasets/annotations/instances_train.json

  # 可视化标注框
  python tools/dataset_analysis.py --labels_dir ... --images_dir ... --visual_box --save_path ./vis_boxes

  # 从文件读取类别列表 (每行一个类别名)
  python tools/dataset_analysis.py --labels_dir ... --images_dir ... --classes_file ./my_classes.txt
        """,
    )

    # ===== 输入数据参数 =====
    # YOLO 格式参数组 (与 COCO 格式二选一)
    yolo_group = parser.add_argument_group("YOLO 格式参数")
    yolo_group.add_argument("--labels_dir", type=str, default=None,
                            help="YOLO 标签目录路径 (包含 *.txt 文件)")
    yolo_group.add_argument("--images_dir", type=str, default=None,
                            help="图片目录路径 (与 labels_dir 配对使用)")

    # COCO JSON 格式参数组
    coco_group = parser.add_argument_group("COCO JSON 格式参数 (与YOLO格式二选一)")
    coco_group.add_argument("--coco_json", type=str, default=None,
                            help="COCO JSON 标注文件路径")

    # ===== 类别参数 =====
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="类别名称列表，按顺序对应标签中的 class_id (例如: --classes person car bus)")
    parser.add_argument("--classes_file", type=str, default=None,
                        help="类别名称文件 (每行一个类别名, 优先级低于 --classes)")

    # ===== 可视化参数 =====
    parser.add_argument("--visual_box", action="store_true", default=False,
                        help="是否在图片上绘制检测框并保存")
    parser.add_argument("--save_path", type=str, default="visual_boxes",
                        help="可视化结果保存目录 (默认: ./visual_boxes)")

    # ===== 阈值参数 =====
    parser.add_argument("--small_threshold", type=int, default=32,
                        help="小目标边长阈值 (默认: 32, 面积=1024)")
    parser.add_argument("--large_threshold", type=int, default=96,
                        help="大目标边长阈值 (默认: 96, 面积=9216)")

    args = parser.parse_args()

    # 更新目标面积阈值 (如果用户指定了不同值)
    global OBJECT_INFO
    OBJECT_INFO = [args.small_threshold ** 2, args.large_threshold ** 2]

    # ===== 获取类别列表 =====
    classes = args.classes
    if classes is None and args.classes_file:
        # 从文件读取类别列表
        try:
            with open(args.classes_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            print(f"{ORANGE}从文件读取了 {len(classes)} 个类别{RESET}")
        except FileNotFoundError:
            print(f"{RED}类别文件未找到: {args.classes_file}{RESET}")
            return

    # ===== 执行分析 =====
    if args.coco_json:
        # ---- COCO JSON 格式分析 ----
        if not os.path.exists(args.coco_json):
            print(f"{RED}COCO 文件未找到: {args.coco_json}{RESET}")
            return

        print(f"{BLUE}使用 COCO JSON 格式分析: {args.coco_json}{RESET}")

        # 如果未指定类别，从 COCO 文件自动读取
        if classes is None:
            import json
            with open(args.coco_json, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
            classes = [cat["name"] for cat in sorted(coco_data.get("categories", []), key=lambda x: x["id"])]
            print(f"{ORANGE}从 COCO 文件自动读取了 {len(classes)} 个类别: {classes}{RESET}")

        analyze_dataset_coco(args.coco_json, classes, args.visual_box, args.save_path)

    elif args.labels_dir and args.images_dir:
        # ---- YOLO 格式分析 ----
        if not os.path.exists(args.labels_dir):
            print(f"{RED}标签目录未找到: {args.labels_dir}{RESET}")
            return
        if not os.path.exists(args.images_dir):
            print(f"{RED}图片目录未找到: {args.images_dir}{RESET}")
            return

        print(f"{BLUE}使用 YOLO 格式分析:{RESET}")
        print(f"  标签目录: {args.labels_dir}")
        print(f"  图片目录: {args.images_dir}")

        # 匹配图片和标签
        matched = get_images_and_labels(args.images_dir, args.labels_dir)
        if not matched:
            print(f"{RED}未找到匹配的图片-标签对，请检查目录路径。{RESET}")
            return

        # 如果未指定类别，使用占位符类别名
        if classes is None:
            # 先扫描所有标签，自动检测类别数量
            max_cls_id = -1
            for label_path in matched:
                try:
                    with open(label_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                cls_id = int(line.split()[0])
                                max_cls_id = max(max_cls_id, cls_id)
                except (FileNotFoundError, ValueError, IndexError):
                    continue

            num_classes = max_cls_id + 1 if max_cls_id >= 0 else 1
            classes = [f"class_{i}" for i in range(num_classes)]
            print(f"{YELLOW}未指定类别名称，使用占位符: {classes}{RESET}")

        analyze_dataset_yolo(matched, classes, args.visual_box, args.save_path)

    else:
        print(f"{RED}错误: 请指定输入数据。使用以下方式之一:{RESET}")
        print("  1. --labels_dir + --images_dir (YOLO格式)")
        print("  2. --coco_json (COCO格式)")
        print("  使用 --help 查看完整参数说明。")
        return

    print(f"{GREEN}分析完成!{RESET}")


if __name__ == "__main__":
    main()
