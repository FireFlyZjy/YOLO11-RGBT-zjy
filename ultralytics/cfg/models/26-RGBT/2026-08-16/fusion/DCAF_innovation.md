# DCAF 创新点（Discrepancy-Compensated Adaptive Fusion · 差异补偿自适应融合）

> 原创模块，位于 `ultralytics/nn/ExtraModules/fusion/DCAF.py`，模型配置 `yolo26-RGBT-midfusion-DCAF.yaml`。
> 本文档用于论文撰写，汇总本模块的创新点、动机、与现有工作的区别。

## 1. 一句话定位
DCAF 是一种**轻量、局部、互补感知**的 RGB-T 双模态融合模块，作为中期融合中 `Concat` 的**原位替代**，插入双分支骨干的 P3 / P4 / P5 对接处。

## 2. 现有方法的不足（问题动机）
| 类别 | 代表模块 | 缺陷 |
|---|---|---|
| 全局质量加权 | QualityWeightedFusion | 每个模态仅一个**整图标量**权重，对模态的**局部失效**盲目（如暗角处 RGB 已失效但权重不变） |
| 差分 + 全局池化 | DMAF | 对差分图做 GAP，**空间判别力被压没** |
| 边缘混合 | EdgeBlendFusion | Sobel 边缘 + 固定混合，**无可靠性建模、无差异补偿** |
| Transformer/交叉注意力 | ICAFusion / C2Former / HAFFormer / MARSS | 表达力强但**参数量大、训练不稳、近似 O(N²)** |

## 3. 我们的两个创新机制
### 3.1 局部可靠性重加权（Local Reliability Reweighting）
通过共享的轻量瓶颈卷积（1×1 → SiLU → 1×1）为 vis / ir 各自预测**逐位置、逐通道**的可靠性图
`r_v, r_i ∈ [0,1]^{B×C×H×W}`，按可靠性作归一化加权：

```
base = (r_v · V + r_i · I) / (r_v + r_i + ε)
```

相比全局标量权重，能精确定位到"**哪里、哪个通道**"更可信。

### 3.2 差异补偿残差（Discrepancy-Compensated Residual）— 核心创新
跨模态差异 `δ = V − I` **不应被当作纯噪声丢弃**。我们的核心观察是：
> 两模态分歧强烈的位置，往往正是**互补信号**所在 ——
> 例如暗光下 RGB 看不到、但 IR 中清晰的热目标；或 IR 被热背景淹没、但 RGB 保留的纹理。

因此用 **3×3 深度可分离卷积**学一个局部门控 `g = σ(DWConv3×3(δ))`，**仅在与分歧大的位置**注入补偿残差：

```
out = proj( base + λ · g · δ )
```

把"另一模态丢掉的"信号补回来。`λ` 为可学习标量（初始化 0.1）。

## 4. 与现有模块的区别（定位）
- **vs QualityWeightedFusion**：后者是全局标量 softmax 权重且无残差；DCAF 是局部逐通道可靠性 + 差异补偿残差。
- **vs DMAF**：后者对差分图做 GAP 全局池化丢空间；DCAF 在**全分辨率**上做局部门控，保留位置判别力。
- **vs EdgeBlendFusion**：后者仅做 Sobel 边缘 + 固定混合，无可靠性建模、无差异补偿。
- **vs ICAFusion / C2Former**：无注意力、无 Transformer，参数量与 QWF 同量级。

## 5. 复杂度与效率
- 仅含 1×1 瓶颈、深度可分离 3×3 与 1×1 投影，**无 softmax / 注意力 / Transformer**。
- **~399k 参数 / 融合点**（×3 ≈ 1.2M），与 QualityWeightedFusion **同量级**，远轻于所有 Transformer 类融合。

## 6. 论文贡献声明（建议表述）
1. 提出 **DCAF**，首次将"**差异补偿**"作为互补信号恢复机制用于 RGB-T 融合，用局部门控在分歧处注入跨模态互补信息。
2. 提出**局部逐通道可靠性重加权**，取代全局标量质量权重，实现位置/通道自适应的融合。
3. 在统一的 YOLOv26-RGBT 框架与大规模受控基准上验证 DCAF 的有效性（见实验章节，指标待训练后填充 FLIR / LLVIP / M3FD / KAIST）。

> ⚠️ 写作提示：DCAF 尚未训练，论文实验章节的 DCAF 数值需训练后填充；可先以基线 midfusion（mAP50-95=0.492 @ 34.4G, 80.6 FPS）与高分复现模块（DWR 0.5262 / DASI 0.5232 / COI 0.5216）作对照基线。
