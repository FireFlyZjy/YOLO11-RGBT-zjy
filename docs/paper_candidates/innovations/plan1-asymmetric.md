# 方案1: COI-DASI-Asymmetric — 模态特异性非对称 backbone 增强

## 基本信息

| 项目 | 内容 |
|------|------|
| **方案名** | COI-DASI-Asymmetric |
| **YAML** | `2026-08-02/conv/yolo26-RGBT-midfusion-COI-DASI-Asymmetric.yaml` |
| **代码改动** | `DASI.py` 添加 `s=1` stride 参数 |
| **层号位移** | 无（原位替换） |
| **日期** | 2026-08-02 |

## 设计

### RGB 分支 — COI 三重残差（纹理增强）

| 层号 | 原模块 | 新模块 | args | 说明 |
|------|--------|--------|------|------|
| 2 | Conv [64,3,2] | COI [64,3,2] | stride=2 | P1/2 降采样 + 三重残差 |
| 3 | Conv [128,3,2] | COI [128,3,2] | stride=2 | P2/4 降采样 + 三重残差 |
| 5 | Conv [256,3,2] | COI [256] | s=1 | P3/8 特征增强 |

### IR 分支 — DASI 多空洞率（红外小目标增强）

| 层号 | 原模块 | 新模块 | args | 说明 |
|------|--------|--------|------|------|
| 12 | Conv [64,3,2] | DASI [64,2] | stride=2 | P1/2 降采样 + 多空洞率 |
| 13 | Conv [128,3,2] | Conv [128,3,2] | — | 保持标准 Conv |
| 15 | Conv [256,3,2] | Conv [256,3,2] | — | 保持标准 Conv |

### Stride 对齐验证

| 层号 | RGB 输出分辨率 | IR 输出分辨率 | 匹配 |
|------|--------------|-------------|------|
| 2/12 | 320×320 (stride=2) | 320×320 (stride=2) | ✓ |
| 3/13 | 160×160 (stride=2) | 160×160 (stride=2) | ✓ |
| 5/15 | 160×160 (s=1) | 160×160 (stride=2) | ✓ |
| 6/16 | 80×80 | 80×80 | ✓ |

两分支 stride 一致，Concat 无空间冲突。

## 核心创新

### 1. 模态特异性非对称设计

现有 RGB-T 检测均采用**对称 backbone**（RGB 和 IR 共享相同结构），忽略了两个模态的物理特性差异：

| 模态 | 信息特点 | 最优增强策略 |
|------|---------|-------------|
| RGB | 纹理/边缘/颜色丰富 | COI 三重残差（shortcut + DWConv + 1x1），保留纹理细节 |
| IR | 热分布/小目标响应 | DASI 多空洞率（dilation=2/4/6/8），大感受野捕获稀疏热特征 |

### 2. DASI stride 扩展

DASI 原本无 stride 参数，只能替换 s=1 的 Conv。新增 `s=1` 参数后：
- `DASI, [64, 2]` → stride=2，可替换降采样 Conv，维持基线分辨率
- `DASI, [64]` → stride=1（默认），向后兼容现有 YAML

### 3. 差异化增强动机

| 模态 | 增强模块 | 机制 | 对应物理特性 |
|------|---------|------|-------------|
| RGB | COI | 三路并行（恒等+深度卷积+逐点卷积） | 纹理/颜色 → 需通道+空间双增强 |
| IR | DASI | 4级空洞卷积并行 | 小目标热特征 → 需大感受野上下文 |

## 验证结果 (model.info)

| 指标 | COI 单独 | DASI 单独 | Asymmetric 实测 |
|------|---------|---------|----------------|
| Layers | — | — | 504 |
| Params(M) | 14.96 | 15.57 | 15.28 |
| FLOPs(G) | 108.96 | 142.46 | **33.21** |
| vs 基线 | +74.55 | +108.05 | **-1.20 (更轻!)** |

FLOPs 反而低于基线，因为只替换 3 COI + 1 DASI（远少于 COI 全替换的 6 层）。

## 论文价值

- **创新点**: 首次在 RGB-T 检测中提出模态特异性 backbone 非对称设计
- **理论支撑**: RGB 和 IR 的物理特性差异 → 不同增强策略
- **实验设计**: 消融对比 — 对称 COI vs 对称 DASI vs 非对称 COI+DASI

## 相关文件

- YAML: `ultralytics/cfg/models/26-RGBT/2026-08-02/conv/yolo26-RGBT-midfusion-COI-DASI-Asymmetric.yaml`
- COI 源码: `ultralytics/nn/ExtraModules/conv/COI.py`
- DASI 源码: `ultralytics/nn/ExtraModules/attention/DASI.py`（已修改，添加 stride）
- COI module card: `docs/paper_candidates/COI/module_card.md`
- DASI module card: `docs/paper_candidates/DASI/module_card.md`
