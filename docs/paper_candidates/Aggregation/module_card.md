# Aggregation — 多尺度密集注意力聚合

## 基本信息

| 项目 | 内容 |
|------|------|
| **模块名** | Aggregation |
| **类别** | neck（颈部多尺度聚合，from 为列表） |
| **来源论文** | HVPNet (RGB-T 显著目标检测) |
| **来源仓库** | exps/HVPNet-main |
| **源码** | `ultralytics/nn/ExtraModules/neck/Aggregation.py` |
| **YAML 用法** | `[[P5, P4, P3], 1, Aggregation, [c2]]`（3 尺度输入） |
| **参数签名** | `__init__(self, c1, c2)`（c1 可为 list `[c_p3, c_p4, c_p5]`） |

## 核心机制

Aggregation 接收 3 个尺度金字塔特征（P3/P4/P5），通过逐元素乘进行跨尺度交互，再密集拼接输出。

### 三步流程

1. **统一投影**: 3 个尺度各自 1x1 Conv 投影到统一通道 c2
2. **逐元素乘跨尺度交互**:
   - `x2_1 = conv(upsample(x1)) * x2`（深层 P5 上采样后与中层 P4 相乘）
   - `x3_1 = conv(upsample²(x1)) * conv(upsample(x2)) * x3`（深层双重上采样与中层上采样后与浅层 P3 三者相乘）
3. **密集拼接**: 交互后的特征 + 上采样的原始特征 → concat → 3x3 Conv 融合 → 1x1 Conv 输出

### 伪代码

```
x1, x2, x3 = P5, P4, P3  # 从小分辨率到大分辨率
x1, x2, x3 = proj[0](x1), proj[1](x2), proj[2](x3)

x2_1 = conv_up(upsample(x1)) * x2
x3_1 = conv_up(upsample(upsample(x1))) * conv_up(upsample(x2)) * x3

x2_2 = concat(x2_1, conv_up(upsample(x1)))
x3_2 = concat(x3_1, conv_up(upsample(x2_2)))

output = conv1x1(conv3x3(x3_2))
```

## 网络结构

```
P5 ──proj──┬── upsample ── conv ──×── x2_1 ──┐
P4 ──proj──┤                               concat → x2_2
           └── upsample ── conv ───────────┘
                                              ↓ upsample ── conv
P5 ──proj──┬── upsample² ── conv ───┐
P4 ──proj──┤── upsample ── conv ────×── x3_1 ──┐
P3 ──proj──┤──────────────────────────┘        concat → x3_2 → conv3x3 → conv1x1 → output
           └──────────────────────────────────┘
```

## 实验结果

| 配置 | Epochs | Precision | Recall | mAP50 | mAP50-95 | Params(M) | FLOPs(G) | FPS |
|------|--------|-----------|--------|-------|----------|-----------|----------|-----|
| Aggregation | 100 | 0.8659 | 0.7833 | 0.8674 | 0.5058 | 33.67 | 232.24 | 20.69 |
| **基线** | 100 | 0.8370 | 0.7774 | 0.8495 | 0.4900 | 15.24 | 34.41 | 52.67 |

### vs 基线涨点

| mAP50 涨点 | mAP50-95 涨点 |
|-----------|---------------|
| +0.0179 | +0.0158 |

## 插入策略

- **插入位置**: Neck 中 P3 层之后（层 32），输入 `[P5(23), P4(22), P3(31)]`
- **原因**: 跨尺度逐元素乘让深层语义指导浅层定位，小目标在浅层(P3)有强响应但缺语义，在深层(P5)有语义但分辨率低，逐元素乘是天然的小目标-大目标特征互补
- **层号位移**: 在 neck 中插入新层，后续层号 +2

## 论文出处

> **论文**: HVPNet (RGB-T Salient Object Detection)
> **来源**: exps/HVPNet-main 仓库
> **模块文件**: neck/Aggregation.py

## 创新点与局限

**优点**:
- 逐元素乘跨尺度交互，深度融合不同分辨率特征
- Precision=0.8659（全榜最高之一）
- 生成的 attention map 可用于后续特征加权

**局限**:
- 参数量最大（33.67M），FLOPs 最高（232.24G）
- FPS 仅 20.69，对显存和速度要求高的场景需权衡
- 卫星超大图场景需评估显存可行性
