# ProgressiveAgg — 渐进式多尺度特征聚合

## 基本信息

| 项目 | 内容 |
|------|------|
| **模块名** | ProgressiveAgg |
| **类别** | neck（颈部多尺度聚合，from 为列表） |
| **来源论文** | ProGRess (rmae-progress) + HVPNet 密集融合 |
| **来源仓库** | exps/rmae-progress + exps/HVPNet-main |
| **源码** | `ultralytics/nn/ExtraModules/neck/ProgressiveAgg.py` |
| **YAML 用法** | `[[P3, P4, P5], 1, ProgressiveAgg, [c2, use_lcar]]` |
| **参数签名** | `__init__(self, c1, c2, use_lcar=True)` |

## 核心机制

ProgressiveAgg 融合了 ProGRess 的渐进跳跃融合思想和 HVPNet 的密集融合策略，由三个子模块组成：

### 1. PLF (Progressive Leapwise Fusion)

从深到浅渐进融合（P5→P4→P3），每个阶段将深层特征上采样后与当前层特征拼接 → 1x1 Conv 融合：

```
fused[P5] = feats[P5]
fused[P4] = FusionBlock(feats[P4], upsample(fused[P5]))
fused[P3] = FusionBlock(feats[P3], upsample(fused[P4]))
```

### 2. LCAR (Lightweight Channel Attention Residual)

每层融合后经轻量 1x1 通道注意力 + 残差连接：

```
gate = x * sigmoid(conv1x1(x))  # 通道注意力
output = gate + x               # 残差
```

### 3. Bottleneck 融合

所有层上采样到最大分辨率（P3）后拼接 → 1x1 Conv 融合输出。

## 网络结构

```
P3 ──proj──┐
P4 ──proj──┤     ┌── PLF (深→浅) ──┐
P5 ──proj──┘     │                 │
                  │  fused[P5] ←────┘
                  │  fused[P4] ←── FusionBlock(P4, up(fused[P5]))
                  │  fused[P3] ←── FusionBlock(P3, up(fused[P4]))
                  │
                  ├── LCAR (通道注意力+残差) × 3
                  │
                  ├── 上采样到 P3 分辨率
                  │
                  └── concat → Bottleneck(1x1) → output
```

## 实验结果

| 配置 | Epochs | Precision | Recall | mAP50 | mAP50-95 | Params(M) | FLOPs(G) | FPS |
|------|--------|-----------|--------|-------|----------|-----------|----------|-----|
| ProgressiveAgg | 100 | 0.8536 | 0.7945 | 0.8654 | 0.4998 | 16.43 | 42.64 | 88.14 |
| **基线** | 100 | 0.8370 | 0.7774 | 0.8495 | 0.4900 | 15.24 | 34.41 | 52.67 |

### vs 基线涨点

| mAP50 涨点 | mAP50-95 涨点 |
|-----------|---------------|
| +0.0159 | +0.0098 |

## 插入策略

- **插入位置**: Neck 中 P3 层之后（层 32），输入 `[P3(31), P4(22), P5(23)]`
- **原因**: 渐进式融合比一次性 concat 更平滑地传递多尺度信息，LCAR 增强跨模态特征的通道选择性
- **层号位移**: 在 neck 中插入新层，后续层号 +2

## 论文出处

> **论文灵感**: ProGRess (PLF + LCAR) + HVPNet (密集融合)
> **来源**: exps/rmae-progress + exps/HVPNet-main
> **模块文件**: neck/ProgressiveAgg.py

## 创新点与局限

**优点**:
- **FPS=88.14（全榜最高）**，精度-速度均衡最优 neck
- 参数量仅 16.43M（接近基线 15.24M），FLOPs 42.64G（仅比基线多 8G）
- 渐进式融合保留多尺度信息，轻量通道注意力增强通道选择性
- 对不同尺度目标（近景大目标 vs 远景小目标）的自适应聚合效果好

**局限**:
- mAP50-95 涨点幅度（+0.0098）小于 COI（+0.0188）和 DASI（+0.0332）
- 精度不是最高，但综合速度-精度最优
