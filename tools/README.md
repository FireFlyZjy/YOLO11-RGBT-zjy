# YOLO RGB-Infrared Detection Tools

本目录包含 5 个独立的 YOLO RGBT 检测辅助工具脚本。每个脚本均可独立运行，用于数据集分析、模型调试和结果可视化。

---

## 工具列表

### 1. dataset_analysis.py — 数据集 S/M/L 目标分析

统计数据集中每个类别的实例数量，并按像素面积将目标分为小(S)、中(M)、大(L)三类。

**分类标准:**
- 小目标 (Small): 面积 < 32x32 (1024 像素)
- 中目标 (Medium): 32x32 ~ 96x96 (1024~9216 像素)
- 大目标 (Large): 面积 > 96x96 (9216 像素)

**用法:**
```bash
# YOLO 格式分析
python tools/dataset_analysis.py \
    --labels_dir ./datasets/train/labels \
    --images_dir ./datasets/train/images \
    --classes person car bus

# COCO JSON 格式分析
python tools/dataset_analysis.py \
    --coco_json ./datasets/annotations/instances_train.json

# 可视化标注框
python tools/dataset_analysis.py \
    --labels_dir ./datasets/train/labels \
    --images_dir ./datasets/train/images \
    --visual_box --save_path ./vis_boxes

# 从文件读取类别列表
python tools/dataset_analysis.py \
    --labels_dir ./datasets/train/labels \
    --images_dir ./datasets/train/images \
    --classes_file ./my_classes.txt
```

**参数说明:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--labels_dir` | str | None | YOLO 标签目录 |
| `--images_dir` | str | None | 图片目录 |
| `--coco_json` | str | None | COCO JSON 文件 (与 YOLO 格式二选一) |
| `--classes` | str list | None | 类别名称列表 |
| `--classes_file` | str | None | 类别名称文件 |
| `--visual_box` | flag | False | 是否绘制检测框 |
| `--save_path` | str | visual_boxes | 可视化保存目录 |
| `--small_threshold` | int | 32 | 小目标边长阈值 |
| `--large_threshold` | int | 96 | 大目标边长阈值 |

---

### 2. layer_info.py — 模型逐层特征形状调试器

为模型所有层注册 forward hook，打印每层的输入/输出张量形状。用于调试融合架构中的维度不匹配问题。

**用法:**
```bash
# 分析默认模型 (YOLO11n)
python tools/layer_info.py

# 分析自定义融合模型
python tools/layer_info.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml

# RGBT 模型 (4通道输入)
python tools/layer_info.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml \
    --img_size 640 --channels 4

# 只看输出形状
python tools/layer_info.py \
    --model_yaml ... \
    --no_input
```

**参数说明:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_yaml` | str | None | 模型 YAML 路径 |
| `--img_size` | int | 640 | 输入尺寸 |
| `--batch_size` | int | 1 | 批次大小 |
| `--channels` | int | 3 | 输入通道数 (RGBT=4, RGBRGB6C=6) |
| `--no_input` | flag | False | 不显示输入形状 |
| `--device` | str | cpu | 运行设备 |

---

### 3. flops_analysis.py — 逐层 FLOPs/Params 分析

使用 thop.profile 计算模型逐层 FLOPs 和参数量，生成格式化表格，标记超出阈值的层。

**用法:**
```bash
# 分析默认模型
python tools/flops_analysis.py

# 分析自定义模型
python tools/flops_analysis.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml \
    --channels 4

# 设置告警阈值
python tools/flops_analysis.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml \
    --flops_warn 1.0 --params_warn 5.0
```

**参数说明:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_yaml` | str | None | 模型 YAML 路径 |
| `--img_size` | int | 640 | 输入尺寸 |
| `--batch_size` | int | 1 | 批次大小 |
| `--channels` | int | 3 | 输入通道数 |
| `--device` | str | cpu | 运行设备 |
| `--flops_warn` | float | 1.0 | FLOPs 告警阈值 (G) |
| `--params_warn` | float | 5.0 | Params 告警阈值 (M) |

---

### 4. feature_heatmap.py — 特征图热力图可视化

捕获模型指定层的特征图，保存为 JET 彩色热力图，可选择同时保存 .npy 文件。

**用法:**
```bash
# 可视化所有层
python tools/feature_heatmap.py \
    --image_path ./data/demo.jpg

# 可视化指定层
python tools/feature_heatmap.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml \
    --image_path ./data/demo.jpg \
    --layer_names 2 5 10 15

# RGBT 4通道 + 保存 npy
python tools/feature_heatmap.py \
    --model_yaml ultralytics/cfg/models/26-RGBT/yolo26-RGBT-midfusion.yaml \
    --image_path ./data/demo.jpg \
    --channels 4 --save_npy

# 限制每层通道数
python tools/feature_heatmap.py \
    --image_path ./data/demo.jpg \
    --max_channels 16
```

**参数说明:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_yaml` | str | None | 模型 YAML 路径 |
| `--image_path` | str | 必需 | 输入图片路径 |
| `--img_size` | int | 640 | 输入尺寸 |
| `--channels` | int | 3 | 输入通道数 |
| `--device` | str | cpu | 运行设备 |
| `--layer_names` | int list | None | 要可视化的层索引 |
| `--max_channels` | int | 32 | 每层最大通道数 |
| `--save_dir` | str | runs/feature_heatmap | 保存目录 |
| `--save_npy` | flag | False | 保存 .npy 文件 |

---

### 5. error_analysis.py — TP/FP/FN 错误分析可视化

将预测结果与真值标签比对，绘制彩色边界框并统计指标。

**颜色编码:**
- 绿色: TP (True Positive, 正确检测)
- 红色: FN (False Negative, 漏检)
- 蓝色: FP (False Positive, 误检)

**用法:**
```bash
# 基本用法
python tools/error_analysis.py \
    --gt_dir ./datasets/val/labels \
    --pred_dir ./runs/detect/predict/labels \
    --image_dir ./datasets/val/images

# 设置 IoU 阈值
python tools/error_analysis.py \
    --gt_dir ./datasets/val/labels \
    --pred_dir ./runs/detect/predict/labels \
    --image_dir ./datasets/val/images \
    --iou_threshold 0.5

# 指定类别和图片后缀
python tools/error_analysis.py \
    --gt_dir ./datasets/val/labels \
    --pred_dir ./runs/detect/predict/labels \
    --image_dir ./datasets/val/images \
    --classes person car bus \
    --image_ext .png
```

**参数说明:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gt_dir` | str | 必需 | 真值标签目录 |
| `--pred_dir` | str | 必需 | 预测结果目录 |
| `--image_dir` | str | 必需 | 图片目录 |
| `--classes` | str list | None | 类别名称列表 |
| `--iou_threshold` | float | 0.45 | IoU 匹配阈值 |
| `--image_ext` | str | .jpg | 图片后缀 |
| `--save_dir` | str | error_vis | 保存目录 |

---

## 参考来源

本目录下的工具脚本参考自 [objectdetection-tricks](https://github.com/yooyoo95/objectdetection-tricks) 项目:

| 工具 | 参考文件 |
|------|----------|
| dataset_analysis.py | tricks_15.py |
| layer_info.py | tricks_13.py |
| flops_analysis.py | tricks_10.py |
| feature_heatmap.py | tricks_3.py |
| error_analysis.py | tricks_1.py |

---

## 注意事项

1. **依赖安装**: 部分工具需要额外依赖，请确保已安装: `pip install thop prettytable tqdm`
2. **模型加载**: 所有工具均支持从 YAML 配置文件加载 YOLO 系列模型 (包括自定义融合模型)
3. **双模态模型**: RGBT 模型输入为 4 通道，使用时需指定 `--channels 4`
4. **路径格式**: Windows 下请使用绝对路径或相对路径，避免路径中包含空格
5. **GPU 支持**: 部分工具支持 `--device cuda:0` 参数使用 GPU 加速
