# COI 创新点（Conv-One-Identity · 三重残差卷积）

> 模块位于 `ultralytics/nn/ExtraModules/conv/COI.py`，来源论文 **HVPNet（RGB-T 显著目标检测）**。
> ⚠️ **归属声明**：COI 为**复现 / 移植**自 HVPNet 的模块（Conv-One-Identity），**并非本工作的原创贡献**。本文件用于论文中将其作为"大规模 RGB-T 融合基准中的高分复现基线"来描述，请务必保留 HVPNet 原始出处引用，切勿作为本文创新点声称。

## 1. 机制概述
COI 以**三条并行路径求和**构成的极轻量残差组件：
1. **Shortcut**：恒等映射 + BN；
2. **Depthwise Conv**：保持通道独立（k×k 深度可分离卷积）；
3. **1×1 Conv**：跨通道线性组合。

三条输出相加后经 **GELU** 激活：

```
x → proj → {BN(x), BN(DWConv(x)), BN(1x1Conv(x))} → sum → GELU
```

## 2. 为何在 RGB-T 检测中得分高
- **极轻量且强表达**：深度可分离 + 1×1 跨通道组合 + 残差，在极小参数下保持强特征变换能力。
- 易嵌入 backbone，作为基础构件广泛替换普通 Conv。
- 在 FLIR（yolo26s, RGBT-4ch, 150 epoch, imgsz640）上 mAP50-95 = **0.5216**，为高分基线中**算力最低**者（108.96G）。

## 3. 实测指标（FLIR, yolo26s, 150 epoch, imgsz640, RGBT-4ch）
| 指标 | 数值 |
|---|---|
| mAP50 | 0.8799 |
| mAP50-95 | **0.5216** |
| FLOPs(G) | 108.96 |
| Params(M) | 14.96 |
| FPS | 29.14 |

> 相对 DWR/DASI：精度接近（差距 <0.005 mAP50-95），但算力低约 23%–30%，属"精度-算力均衡"型最强复现基线。

## 4. 在本文中的定位
COI（源自 HVPNet）是高分基线中**效率最优**的一位，与 DWR/DASI 共同界定了"复现模块能达到的精度-算力上界"。本文 DCAF 的设计目标之一，即在不引入 Transformer/大空洞卷积的前提下，逼近该组上界并进一步降低算力。
