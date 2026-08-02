# Paper Candidates — 论文候选模块集合

> 从 FLIR 数据集 89 组实验中筛选出 mAP50 > 0.865 的 4 个模块（5 组配置），
> 包含源码、YAML 配置、模块原理卡片和实验结果，供学术论文撰写使用。

## 目录结构

```
docs/paper_candidates/
├── README.md                  ← 本文件（总览索引）
├── COI/
│   ├── source.py              # 模块源码
│   ├── config.yaml            # YAML 配置
│   ├── module_card.md         # 模块原理卡片
│   └── results.csv            # 实验记录（COI-100 + COI-150）
├── Aggregation/
│   ├── source.py / config.yaml / module_card.md / results.csv
├── ProgressiveAgg/
│   ├── source.py / config.yaml / module_card.md / results.csv
├── DASI/
│   ├── source.py / config.yaml / module_card.md / results.csv
└── comparison/
    └── summary.md             # 4 模块横向对比 + 基线对照
```

## 模块速查

| 模块 | 类别 | 来源 | mAP50 | mAP50-95 | Params(M) | GFLOPs | FPS | Epochs |
|------|------|------|-------|----------|-----------|--------|-----|--------|
| **DASI** | attention | 红外小目标(2024) | **0.8860** | **0.5232** | 15.57 | 142.46 | 22.84 | 100 |
| **COI-150** | conv | HVPNet | 0.8799 | 0.5216 | 14.96 | 108.96 | 16.97 | 150 |
| **COI** | conv | HVPNet | 0.8693 | 0.5088 | 14.96 | 108.96 | 12.84 | 100 |
| **Aggregation** | neck | HVPNet | 0.8674 | 0.5058 | 33.67 | 232.24 | 20.69 | 100 |
| **ProgressiveAgg** | neck | ProGRess+HVPNet | 0.8654 | 0.4998 | 16.43 | 42.64 | **88.14** | 100 |
| *基线* | — | — | 0.8495 | 0.4900 | 15.24 | 34.41 | 52.67 | 100 |

## 各模块定位

| 模块 | 核心价值 | 适用场景 |
|------|---------|---------|
| **DASI** | mAP50/mAP50-95 双榜第 1，Recall 最高 | 红外小目标检测，精度优先 |
| **COI** | 极轻量三重残差，延长训练可继续涨点 | 浅层 backbone 增强，训练时长充裕时 |
| **Aggregation** | 逐元素乘跨尺度交互，Precision 最高 | 多尺度特征深度融合，显存充裕时 |
| **ProgressiveAgg** | FPS 最高，精度-速度均衡最优 | 实时检测，速度优先场景 |

## 数据集与训练配置

| 项目 | 配置 |
|------|------|
| 数据集 | FLIR-noDog (3 类: person/car/bicycle) |
| 预训练权重 | weights/yolo26s.pt |
| 输入 | RGBT 4 通道 (RGB 3ch + IR 1ch) |
| 架构 | YOLO26s 双模态 Mid-Fusion |
| 优化器 | SGD |
| 图像尺寸 | 640×640 |
| GPU | NVIDIA GeForce GTX 1650 Ti (4GB) |
