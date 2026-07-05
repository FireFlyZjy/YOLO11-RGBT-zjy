# RT-SFOD 训练策略使用指南

> 创建日期: 2026-07-05
> 论文: RT-SFOD (ECCV 2026)
> 来源: RT-SFOD/scripts/YOLO26/stage2_rtsfod_yolo26.py
> 相关文档: [指令.md](指令.md) | [创新点.md](创新点.md) | [改进位置.md](改进位置.md)

---

## 一、RT-SFOD 概述

RT-SFOD 是一个半监督目标检测框架，核心创新在于**训练策略**而非网络架构模块。

### 核心策略

| 策略 | 全称 | 作用 |
|------|------|------|
| **DHF** | Dual-Head pseudo-label Fusion | 双头伪标签融合，提升检测质量 |
| **MARD** | Multi-scale Adaptive Representation Diversification | 多尺度特征多样性正则化 |

### 与其他模块的区别

| 模块类型 | 示例 | 使用方式 |
|----------|------|----------|
| 网络架构模块 | CKConv, μPCAD, WDAM, C2Former | 直接调用 yaml 训练 |
| 训练策略 | MARD_Loss, DualHeadFusion | 需要单独脚本 |

---

## 二、文件位置

```
ultralytics/cfg/models/26-RGBT/2026-07-05/train/
├── train_MARD.py          # MARD 损失训练脚本
└── inference_DHF.py       # DHF 双头融合推理脚本

ultralytics/nn/ExtraModules/loss/
└── RT_SFOD_Loss.py        # 损失函数实现
```

---

## 三、MARD 损失训练

### 3.1 MARD 损失原理

MARD 在 P3/P4/P5 多尺度特征上应用两种正则化损失：

1. **Variance Loss (方差损失)**
   - 公式: `relu(gamma - std).mean()`
   - 作用: 鼓励特征多样性，避免特征坍缩

2. **Covariance Loss (协方差损失)**
   - 公式: `off_diag.pow(2).sum() / (c * (c - 1))`
   - 作用: 减少特征冗余，提升特征效率

3. **MARD Loss (组合损失)**
   - 公式: `alpha * variance_loss + beta * covariance_loss`

### 3.2 使用方式

#### 标准训练模式（不使用 MARD 损失）

```bash
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --data ultralytics/cfg/datasets/flir.yaml \
    --epochs 100 --batch 4 --device 0
```

#### 带 MARD 损失模式（需要自定义训练循环）

```bash
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --data ultralytics/cfg/datasets/flir.yaml \
    --epochs 100 --batch 4 --device 0 --use_mard
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | YAML 配置文件路径 |
| `--data` | flir.yaml | 数据集配置文件路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch` | 4 | 批次大小 |
| `--device` | 0 | 设备 |
| `--mard_alpha` | 1.0 | 方差损失权重 |
| `--mard_beta` | 0.1 | 协方差损失权重 |
| `--mard_gamma` | 1.0 | 方差阈值 |
| `--mard_weight` | 0.05 | MARD 损失总权重 |
| `--use_mard` | False | 是否使用 MARD 损失 |

### 3.4 代码集成示例

```python
from ultralytics.nn.ExtraModules.loss.RT_SFOD_Loss import MARD_Loss

# 创建 MARD 损失函数
mard_loss_fn = MARD_Loss(alpha=1.0, beta=0.1, gamma=1.0)

# 在训练循环中使用
feats = [p3, p4, p5]  # 从模型中间层获取
mard_result = mard_loss_fn(feats)

# 添加到总损失
total_loss = det_loss + 0.05 * mard_result['mard_loss']
```

---

## 四、DHF 双头融合推理

### 4.1 DHF 融合原理

DHF 利用 YOLO 的 one2one 和 one2many 双头输出进行融合：

1. **one2one 头**: 高置信度锚点
2. **one2many 头**: 补充检测
3. **去重融合**: 移除重叠框，保留互补检测

### 4.2 使用方式

#### 标准推理模式（不使用 DHF 融合）

```bash
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/images \
    --device 0
```

#### 带 DHF 融合模式（需要模型支持双头输出）

```bash
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/images \
    --device 0 --use_dhf
```

### 4.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | YAML 配置文件路径 |
| `--weights` | None | 权重文件路径 |
| `--source` | 必填 | 图像/视频/目录路径 |
| `--device` | 0 | 设备 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.45 | NMS IoU 阈值 |
| `--tau_o2o` | 0.5 | one2one 头置信度阈值 |
| `--tau_o2m` | 0.5 | one2many 头置信度阈值 |
| `--tau_no` | 0.2 | 新框 IoU 阈值 |
| `--tau_dup` | 0.7 | 去重 IoU 阈值 |
| `--use_dhf` | False | 是否使用 DHF 融合 |

### 4.4 代码集成示例

```python
from ultralytics.nn.ExtraModules.loss.RT_SFOD_Loss import DualHeadFusion

# 创建 DHF 融合器
dhf = DualHeadFusion(tau_o2o=0.5, tau_o2m=0.5, tau_no=0.2, tau_dup=0.7)

# 获取双头输出
boxes_one2one = model.predict_one2one(source)  # one2one 头输出
boxes_one2many = model.predict_one2many(source)  # one2many 头输出

# 融合
fused_boxes = dhf(boxes_one2one, boxes_one2many)
```

---

## 五、完整训练流程示例

### 5.1 标准训练（推荐）

```bash
# 1. 使用 CKConv 模块训练
D:\Anaconda\envs\torch310\python.exe -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/26-RGBT/2026-07-05/conv/yolo26-RGBT-midfusion-CKConv.yaml')
model.train(data='ultralytics/cfg/datasets/flir.yaml', epochs=100, batch=4, device='0', workers=0, use_simotm='RGBT', channels=4, close_mosaic=0)
"

# 2. 使用 μPCAD_2D 模块训练
D:\Anaconda\envs\torch310\python.exe -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml')
model.train(data='ultralytics/cfg/datasets/flir.yaml', epochs=100, batch=4, device='0', workers=0, use_simotm='RGBT', channels=4, close_mosaic=0)
"
```

### 5.2 使用 MARD 损失训练

```bash
# 标准训练模式
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --data ultralytics/cfg/datasets/flir.yaml \
    --epochs 100 --batch 4 --device 0

# 带 MARD 损失模式
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --data ultralytics/cfg/datasets/flir.yaml \
    --epochs 100 --batch 4 --device 0 --use_mard
```

### 5.3 使用 DHF 融合推理

```bash
# 标准推理模式
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/images \
    --device 0

# 带 DHF 融合模式
D:\Anaconda\envs\torch310\python.exe ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py \
    --model ultralytics/cfg/models/26-RGBT/2026-07-05/frequency/yolo26-RGBT-midfusion-uPCAD.yaml \
    --weights runs/detect/train/weights/best.pt \
    --source path/to/images \
    --device 0 --use_dhf
```

---

## 六、Agent 快速参考

### 6.1 判断使用哪种方式

| 场景 | 使用方式 |
|------|----------|
| 使用网络架构模块（CKConv, μPCAD 等） | 直接调用 yaml 训练 |
| 需要 MARD 损失正则化 | 使用 `train_MARD.py` 脚本 |
| 需要 DHF 双头融合推理 | 使用 `inference_DHF.py` 脚本 |

### 6.2 关键文件

| 文件 | 作用 |
|------|------|
| `ultralytics/nn/ExtraModules/loss/RT_SFOD_Loss.py` | 损失函数实现 |
| `ultralytics/cfg/models/26-RGBT/2026-07-05/train/train_MARD.py` | MARD 训练脚本 |
| `ultralytics/cfg/models/26-RGBT/2026-07-05/train/inference_DHF.py` | DHF 推理脚本 |

### 6.3 注意事项

1. **网络架构模块**：所有 yaml 文件都可以直接用标准训练命令
2. **MARD 损失**：需要修改 ultralytics 训练循环才能真正集成，当前脚本演示了使用方式
3. **DHF 融合**：需要模型支持 one2one 和 one2many 双头输出，当前脚本演示了使用方式
4. **参数调优**：MARD 权重建议从 0.05 开始，根据训练效果调整
