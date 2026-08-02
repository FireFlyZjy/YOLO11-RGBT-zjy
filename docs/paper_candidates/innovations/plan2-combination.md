# 方案2: DASI-ProgAgg — Backbone+Neck 双层组合增强

## 基本信息

| 项目 | 内容 |
|------|------|
| **方案名** | DASI-ProgAgg |
| **YAML** | `2026-08-02/neck/yolo26-RGBT-midfusion-DASI-ProgAgg.yaml` |
| **代码改动** | 无（复用现有 DASI + ProgressiveAgg） |
| **层号位移** | Neck +2（插入 ProgressiveAgg 层 32） |
| **日期** | 2026-08-02 |

## 设计

### Backbone — DASI 浅层小目标增强

| 层号 | 原模块 | 新模块 | args | 说明 |
|------|--------|--------|------|------|
| 2 | Conv [64,3,2] | DASI [64] | s=1 (无 stride) | P1 全分辨率, 分辨率提升一级 |
| 12 | Conv [64,3,2] | DASI [64] | s=1 (无 stride) | IR 分支同上 |

DASI 无 stride → 分辨率提升一级 → 浅层特征图更大 → 小目标空间细节更丰富

### Neck — ProgressiveAgg 渐进多尺度聚合

| 新层号 | 模块 | 输入 | args | 说明 |
|--------|------|------|------|------|
| 32 | ProgressiveAgg | [31, 22, 23] | [256, True] | P3+P4+P5 渐进融合 |

ProgressiveAgg 内部使用 `F.interpolate` 处理尺寸不匹配，兼容 DASI 的提升分辨率。

### 层号位移对照

| 基线层号 | 新层号 | 模块 | 说明 |
|----------|--------|------|------|
| 31 | 31 | C3k2 [256] | P3, 不变 |
| — | 32 | ProgressiveAgg | 新插入 |
| 32 | 33 | Concat [31,32] | 融合 P3 |
| 33 | 34 | C3k2 [256] | P3_enh |
| 34 | 35 | Conv [256,3,2] | P4 下采样 |
| 35 | 36 | Concat | cat head P4 |
| 36 | 37 | C3k2 [512] | P4 |
| 37 | 38 | Conv [512,3,2] | P5 下采样 |
| 38 | 39 | Concat | cat head P5 |
| — | 40 | C3k2 [1024] | P5 |
| 38 | 41 | Detect [34,37,40] | 引用更新 |

## 核心创新

### 1. 双层叠加

| 层级 | 模块 | 机制 | 作用 |
|------|------|------|------|
| Backbone 浅层 | DASI | 4级空洞卷积(dilation=2/4/6/8) | 浅层小目标特征增强 |
| Neck | ProgressiveAgg | PLF渐进融合 + LCAR通道注意力 | 多尺度特征聚合 |

形成 "浅层增强 → 多层聚合" 的完整链路。

### 2. 分辨率协同

- DASI 无 stride → backbone 输出分辨率提升一级（P3: 160×160 vs 基线 80×80）
- ProgressiveAgg 的 FusionBlock 使用 `F.interpolate` 自适应尺寸匹配
- 更大特征图 → ProgressiveAgg 有更多空间信息可聚合

### 3. 互补机制

| 维度 | DASI | ProgressiveAgg |
|------|------|----------------|
| 位置 | backbone 浅层 | neck |
| 核心机制 | 大空洞率多尺度感受野 | 渐进跳跃融合 + 通道注意力 |
| 增强目标 | 小目标稀疏响应 | 多尺度特征互补 |
| 单独表现 | mAP50=0.886 (精度第1) | FPS=88.14 (速度第1) |

## 验证结果 (model.info)

| 指标 | DASI 单独 | ProgAgg 单独 | 组合实测 |
|------|---------|-----------|---------|
| Layers | — | — | 572 |
| Params(M) | 15.57 | 16.43 | 16.76 |
| FLOPs(G) | 142.46 | 42.64 | 175.36 |
| vs 基线 | +108.05 | +8.23 | +140.95 |

FLOPs 约 175G，在 DASI(142G)+ProgAgg(43G) 之间，叠加效应显著。

## 论文价值

- **创新点**: Backbone + Neck 双层组合增强，验证叠加效果
- **实验设计**: 消融对比 — DASI单独 vs ProgAgg单独 vs 组合
- **理论问题**: backbone 浅层增强和 neck 多尺度聚合是否互补？还是冗余？

## 相关文件

- YAML: `ultralytics/cfg/models/26-RGBT/2026-08-02/neck/yolo26-RGBT-midfusion-DASI-ProgAgg.yaml`
- DASI 源码: `ultralytics/nn/ExtraModules/attention/DASI.py`
- ProgressiveAgg 源码: `ultralytics/nn/ExtraModules/neck/ProgressiveAgg.py`
- DASI module card: `docs/paper_candidates/DASI/module_card.md`
- ProgressiveAgg module card: `docs/paper_candidates/ProgressiveAgg/module_card.md`
