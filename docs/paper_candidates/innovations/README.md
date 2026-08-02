# 创新方案 — 3 个 YAML 组合配置

> 基于 mAP50 > 0.865 的 4 个最优模块（DASI、COI、Aggregation、ProgressiveAgg），
> 从非对称分支、双层组合、选择性深度三个维度创建创新配置。
> 日期: 2026-08-02

## 方案速查

| 方案 | 名称 | YAML 路径 | 核心创新 | 层号位移 |
|------|------|-----------|---------|----------|
| 1 | COI-DASI-Asymmetric | `2026-08-02/conv/...-COI-DASI-Asymmetric.yaml` | 模态特异性非对称 backbone | 无 |
| 2 | DASI-ProgAgg | `2026-08-02/neck/...-DASI-ProgAgg.yaml` | Backbone+Neck 双层组合 | +2 |
| 3 | COI-Lite | `2026-08-02/conv/...-COI-Lite.yaml` | 选择性深度（仅最浅层） | 无 |

## 设计动机

分析 89 组实验后发现高分模块的 4 个共同点：
1. **多分支并行**（非单路径加权）
2. **多尺度感受野**（dilation/金字塔）
3. **残差/跳跃连接**（信息保留）
4. **高分辨率位置增强**（浅层/neck P3）

基于此，3 个方案分别从不同维度探索创新空间：
- 方案 1: 从**分支对称性**维度——打破 RGB/IR 对称设计
- 方案 2: 从**层级组合**维度——backbone+neck 叠加
- 方案 3: 从**深度选择性**维度——消融浅层增强的边际收益

## 代码改动

仅 1 处代码修改: `DASI.py` 添加 `s=1` stride 参数（向后兼容）

## 验证结果 (model.info)

| 方案 | Layers | Params(M) | GFLOPs | vs 基线(34.41G) |
|------|--------|-----------|--------|-----------------|
| COI-DASI-Asymmetric | 504 | 15.28 | **33.21** | **-1.20 (更轻!)** |
| DASI-ProgAgg | 572 | 16.76 | 175.36 | +140.95 |
| COI-Lite | 462 | 15.24 | **34.84** | **+0.43 (几乎持平)** |
| *基线* | — | 15.24 | 34.41 | — |

亮点: 方案1 FLOPs 反而低于基线; 方案3 几乎零成本增强。

## 待验证

- [x] YAML model.info() 验证（3 个全部通过）
- [ ] 2 epochs 冒烟测试（用户暂不执行，待后续确认）
- [ ] 完整训练 + 结果对比

## 文件结构

```
docs/paper_candidates/innovations/
├── README.md              ← 本文件
├── plan1-asymmetric.md     ← 方案1 详细说明
├── plan2-combination.md    ← 方案2 详细说明
└── plan3-selective.md      ← 方案3 详细说明
```

## 相关文档

- [创新点.md](../../创新点.md) — `## 2026-08-02` 段
- [改进位置.md](../../改进位置.md) — `## 2026-08-02` 段
- [comparison/summary.md](../comparison/summary.md) — 4 模块横向对比
