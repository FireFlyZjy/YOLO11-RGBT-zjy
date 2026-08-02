# 模块参数速查表

> 本文件从 `指令.md` 拆分，包含所有已注册模块的 YAML args 格式。
> 写 YAML 配置时直接查此表。每次新增模块后同步更新。

---

## Conv 替代类

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| FDConv | `[c2, k, s]` | Conv | `FDConv, [64, 3, 2]` |
| FADC | `[c2, k, s]` | Conv | `FADC, [256, 3, 2]` |
| SFSConv | `[c2, k, s]` | Conv | `SFSConv, [64, 3, 2]` |
| Conv_Coord | `[c2, k, s]` | Conv | `Conv_Coord, [128, 3, 2]` |
| PConv | `[c2, k, s]` | Conv | `PConv, [256, 3, 2]` |
| Dynamic_conv2d | `[c2, k, s]` | Conv | `Dynamic_conv2d, [256, 3, 2]` |
| Conv_DE | `[c2]` | Conv | `Conv_DE, [64]` |
| GSConv | `[c2]` | Conv | `GSConv, [128]` |
| Involution | `[c2, kernel_size, stride]` | Conv | `Involution, [128, 3, 1]` |
| SPDConv | `[c2]` | Conv/C3k2 | `SPDConv, [256]` |
| **REM** | `[c2]` | Conv | `REM, [256]` |
| **COI** | `[c2]` | Conv | `COI, [256]` |
| **RFBBranch** | `[c2]` | Conv | `RFBBranch, [256]` |
| **DASI** | `[c2, s]` | Conv | `DASI, [64, 2]` |

## C3k2 替代类

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| C3_RFEM | `[c2, False, e]` | C3k2 | `C3_RFEM, [256, False, 0.25]` |
| StarBlock | `[c2]` | C3k2 | `StarBlock, [512]` |
| Faster_Block | `[c2]` | C3k2 | `Faster_Block, [256]` |
| Att_SFEM | `[c2, shortcut, g, e]` | C3k2(RGB) | `Att_SFEM, [256, False, 1, 0.25]` |
| Att_IRAFAB | `[c2, c3k, e, g, shortcut]` | C3k2(IR) | `Att_IRAFAB, [256, False, 0.25, 1, True]` |
| VoVGSCSP | `[c2, e]` | C3k2 | `VoVGSCSP, [256, 0.5]` |
| CSPOmniKernel | `[c2]` | C3k2 | `CSPOmniKernel, [512]` |
| **C3k2_CKConv** | `[c2, c3k, e]` | C3k2 | `C3k2_CKConv, [512, True, 0.25]` |

## SPPF 替代类（全局上下文）

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| vHeat | `[c2]` | SPPF | `vHeat, [1024]` |
| **vHeat_Block** | `[c2, t, mlp_ratio, drop_path]` | SPPF/C3k2 | `vHeat_Block, [1024, 1.0, 4.0, 0.1]` |
| TOST | `[c2]` | SPPF | `TOST, [1024]` |
| EfficientViM_Block | `[c2]` | SPPF | `EfficientViM_Block, [1024]` |
| WTE_Mamba | `[c2]` | SPPF | `WTE_Mamba, [1024]` |
| EVCBlock | `[c2]` | SPPF | `EVCBlock, [1024]` |
| SPPCSPC | `[c2, e]` | SPPF | `SPPCSPC, [1024, 0.5]` |
| SPPFCSPC | `[c2, e]` | SPPF | `SPPFCSPC, [1024, 0.5]` |
| **AFFN** | `[c2, hidden_features]` | SPPF | `AFFN, [1024, 512]` |
| **C2PSA_AFFN** | `[c2]` | C2PSA | `C2PSA_AFFN, [1024]` |
| **PAM** | `[c2]` | SPPF | `PAM, [1024]` |
| **C2PSA_PAM** | `[c2]` | C2PSA | `C2PSA_PAM, [1024]` |
| **FSAS** | `[c2]` | SPPF | `FSAS, [1024]` |
| **C2PSA_FSAS** | `[c2]` | C2PSA | `C2PSA_FSAS, [1024]` |
| **SCAM** | `[c2, num_heads]` | SPPF | `SCAM, [1024, 8]` |
| **C2PSA_SCAM** | `[c2, num_heads]` | C2PSA | `C2PSA_SCAM, [1024, 8]` |
| **PolyAttention** | `[c2, num_heads]` | C2PSA | `PolyAttention, [512, 8]` |
| **C2PSA_PolyAttention** | `[c2, num_heads]` | C2PSA | `C2PSA_PolyAttention, [1024, 8]` |
| **PolyConv** | `[c2]` | C3k2 | `PolyConv, [256]` |
| **C3k2_PolyConv** | `[c2, c3k]` | C3k2 | `C3k2_PolyConv, [512, True]` |
| **WDAM** | `[c2, num_heads, window_size, shift_size]` | C2PSA | `WDAM, [1024, 8, 5, 2]` |
| **C2PSA_WDAM** | `[c2]` | C2PSA | `C2PSA_WDAM, [1024]` |
| **C2PSA_C2Former** | `[c2, num_heads, n_groups, offset_range_factor]` | C2PSA | `C2PSA_C2Former, [1024, 8, 1, 2]` |
| **C2PSA_RADM** | `[c2]` | C2PSA | `C2PSA_RADM, [1024]` |
| **C2PSA_μPCAD** | `[c2, heads]` | C2PSA | `C2PSA_μPCAD, [1024, 4]` |
| **MemoryAttention** | `[c2, num_mem, dim_head]` | SPPF/C2PSA | `MemoryAttention, [1024, 64, 64]` |

## 融合替代类（from 是列表 `[vis, ir]`）

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| Att_CIFusion | `[c2]` | Concat | `Att_CIFusion, [512]` |
| Att_AFF | `[c2]` | Concat | `Att_AFF, [512]` |
| Att_iAFF | `[c2]` | Concat | `Att_iAFF, [512]` |
| Att_ICAFusion | `[c2]` | Concat | `Att_ICAFusion, [512]` |
| **vHeat_Fusion** | `[c2, t]` | Concat | `vHeat_Fusion, [512, 1.0]` |
| **FCMMFusion** | `[c2, ratio]` | Concat | `FCMMFusion, [512, 4]` |
| **FCMBlockFusion** | `[c2, ratio]` | Concat | `FCMBlockFusion, [512, 4]` |
| **DMAF** | `[c2]` | Concat | `DMAF, [512]` |
| **QualityWeightedFusion** | `[c2]` | Concat | `QualityWeightedFusion, [512]` |
| **EdgeBlendFusion** | `[c2]` | Concat | `EdgeBlendFusion, [512]` |
| **EI2Fusion** | `[c2]` | Concat | `EI2Fusion, [512]` |
| **ASFFFusion** | `[c2]` | Concat | `ASFFFusion, [512]` |
| **Att_HAFFormer** | `[c2, num_heads]` | Concat | `Att_HAFFormer, [512, 8]` |
| **Att_CrossAttention_M** | `[c2, num_heads]` | Concat | `Att_CrossAttention_M, [512, 8]` |
| **PhaseGuidedFilter** | `[c2]` | Concat | `PhaseGuidedFilter, [512]` |
| **C2Former_Fusion** | `[c2, num_heads, n_groups, offset_range_factor]` | Concat | `C2Former_Fusion, [512, 8, 1, 2]` |
| **RFAF_Fusion** | `[c2]` | Concat | `RFAF_Fusion, [512]` |
| **SCA** | `[c2, head_num, window_size]` | Concat | `SCA, [512, 4, 7]` |
| **GFM** | `[c2, expend_ratio]` | Concat | `GFM, [512, 2]` |
| **EDS** | `[c2]` | Concat | `EDS, [512]` |

## Neck 替代类（from 是列表 `[P3, P4, P5]` 等多尺度）

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| **ProgressiveAgg** | `[c2, use_lcar]` | Neck | `ProgressiveAgg, [256, True]` |
| **Aggregation** | `[c2]` | Neck | `Aggregation, [256]` |

## 上采样替代类

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| CARAFE_Upsample | `[scale]` | nn.Upsample | `CARAFE_Upsample, [2]` |

## 检测头替代类（from 是列表 `[P3, P4, P5]`）

| 模块 | YAML args | 替换对象 | 示例 |
|------|-----------|---------|------|
| DecoupledHead | `[nc, hidc]` | Detect | `DecoupledHead, [nc, 256]` |
| Detect_ATAH | `[nc, hidc]` | Detect | `Detect_ATAH, [nc, 256]` |

## 注意力插入类（from 是单个 layer，插入 Concat 之后）

| 模块 | YAML args | 示例 |
|------|-----------|------|
| Att_CBAM | `[c2]` | `Att_CBAM, [512]` |
| Att_Coord | `[c2]` | `Att_Coord, [512]` |
| Att_SimAM | `[c2]` | `Att_SimAM, [512]` |
| Att_CPCA | `[c2]` | `Att_CPCA, [512]` |
| Att_EMA | `[c2]` | `Att_EMA, [512]` |
| Att_ECA | `[c2]` | `Att_ECA, [512]` |
| Att_Shuffle | `[c2]` | `Att_Shuffle, [512]` |
| Att_LSKA | `[c2]` | `Att_LSKA, [512]` |
| Att_Triplet | `[c2]` | `Att_Triplet, [512]` |
| Att_GAM | `[c2]` | `Att_GAM, [512]` |
| Att_GatherExcite | `[c2, extent, reduction]` | `Att_GatherExcite, [512, 0, 16]` (全局) / `Att_GatherExcite, [512, 4, 16]` (局部) |
| Att_ELA | `[c2]` | `Att_ELA, [512]` |
| Att_ULSAM | `[c2]` | `Att_ULSAM, [512]` |
| **SRA** | `[c2, head_num, window_size]` | `SRA, [512, 4, 7]` |
| **HA** | `[c2]` | `HA, [512]` |
| **MemoryAttention** | `[c2, num_mem, dim_head]` | `MemoryAttention, [1024, 64, 64]` |
| Att_StripPool | `[c2]` | `Att_StripPool, [1024]` |
| Att_AIFI | `[c2]` | `Att_AIFI, [1024]` |
| Att_ScalSeq | `[c2]` | `Att_ScalSeq, [512]` |
| ContextAggregation | `[c2]` | `ContextAggregation, [1024]` |
| CrissCrossAttention | `[c2]` | `CrissCrossAttention, [1024]` |
| SOCA | `[c2, reduction]` | `SOCA, [1024, 8]` |
| SKAttention | `[c2, reduction]` | `SKAttention, [1024, 16]` |
| NAMAttention | `[c2]` | `NAMAttention, [1024]` |
| S2Attention | `[c2]` | `S2Attention, [1024]` |
| ACmix | `[c2]` | `ACmix, [1024]` |
| BoT3 | `[c2, e, e2, w, h]` | `BoT3, [1024, 0.5, 1, 10, 10]` |
| EBlock | `[c2]` | `EBlock, [256]` |
| DBlock | `[c2]` | `DBlock, [256]` |
| **BinaryAttention** | `[c2, num_heads, attn_drop]` | `BinaryAttention, [512, 8, 0.0]` |
| **BinaryChannelAttention** | `[c2, reduction]` | `BinaryChannelAttention, [512, 16]` |
| **C3k2_BinaryAttention** | `[c2, c3k]` | `C3k2_BinaryAttention, [512, True]` |
| **RADE** | `[c2, reduction]` | `RADE, [512, 4]` |
| **RADM** | `[c2]` | `RADM, [1024]` |

## 颈部融合类（from 是列表）

| 模块 | YAML args | 示例 |
|------|-----------|------|
| ASF_fusion | `[c2]` | `ASF_fusion, [512]` |
| Zoom_cat | `[c2]` | `Zoom_cat, [512]` |
| attention_model | `[c2]` | `attention_model, [512]` |
