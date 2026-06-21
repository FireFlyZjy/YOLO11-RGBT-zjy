---
comments: true
description: 学习如何使用 Ultralytics YAML 配置文件构建和自定义模型架构。掌握模块定义、连接方式和缩放参数。
keywords: Ultralytics, YOLO, 模型架构, YAML 配置, 神经网络, 深度学习, 骨干网络, 检测头, 模块, 自定义模型
---

# 模型 YAML 配置指南

模型 YAML 配置文件是 Ultralytics 神经网络的架构蓝图。它定义了层的连接方式、每个模块使用的参数，以及整个网络如何在不同模型尺寸之间缩放。

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yaml-configuration-guide.avif" alt="模型 YAML 配置工作流程">

## 配置结构

模型 YAML 文件组织为三个主要部分，它们协同工作以定义架构。

### Parameters（参数）部分

**parameters** 部分指定模型的全局特性和缩放行为：

```yaml
# Parameters
nc: 80 # 类别数量
scales: # 复合缩放常数 [深度, 宽度, 最大通道数]
    n: [0.50, 0.25, 1024] # nano: 浅层, 窄通道
    s: [0.50, 0.50, 1024] # small: 浅深度, 标准宽度
    m: [0.50, 1.00, 512]  # medium: 中等深度, 完整宽度
    l: [1.00, 1.00, 512]  # large: 完整深度和宽度
    x: [1.00, 1.50, 512]  # extra-large: 最大性能
kpt_shape: [17, 3] # 仅用于姿态模型
```

- `nc` 设置模型预测的类别数量。
- `scales` 定义复合缩放因子，用于调整模型深度、宽度和最大通道数，以生成不同尺寸变体（从 nano 到 extra-large）。
- `kpt_shape` 用于姿态模型。可以是 `[N, 2]` 表示 `(x, y)` 关键点，或 `[N, 3]` 表示 `(x, y, visibility)`。

!!! tip "使用 `scales` 减少冗余"

    `scales` 参数允许你从单个基础 YAML 生成多个模型尺寸。例如，当你加载 `yolo26n.yaml` 时，Ultralytics 读取基础 `yolo26.yaml` 并应用 `n` 缩放因子（`depth=0.50`, `width=0.25`）来构建 nano 变体。

!!! note "`nc` 和 `kpt_shape` 依赖于数据集"

    如果你的数据集指定了不同的 `nc` 或 `kpt_shape`，Ultralytics 会在运行时自动覆盖模型配置以匹配数据集 YAML。

### Backbone（骨干网络）和 Head（检测头）架构

模型架构由 backbone（特征提取）和 head（任务特定）部分组成：

```yaml
backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]]      # 0: 初始卷积
    - [-1, 1, Conv, [128, 3, 2]]     # 1: 下采样
    - [-1, 3, C2f, [128, True]]      # 2: 特征处理

head:
    - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 6: 上采样
    - [[-1, 2], 1, Concat, [1]]                 # 7: 跳跃连接
    - [-1, 3, C2f, [256]]                       # 8: 处理特征
    - [[8], 1, Detect, [nc]]                    # 9: 检测层
```

## 层定义格式

每一层都遵循统一的模式：**`[from, repeats, module, args]`**

| 组件       | 用途             | 示例                                                       |
| ---------- | ---------------- | ---------------------------------------------------------- |
| **from**   | 输入连接         | `-1`（上一层）, `6`（第6层）, `[4, 6, 8]`（多输入）         |
| **repeats**| 重复次数         | `1`（单次）, `3`（重复3次）                                 |
| **module** | 模块类型         | `Conv`, `C2f`, `TorchVision`, `Detect`                     |
| **args**   | 模块参数         | `[64, 3, 2]`（通道数, 卷积核, 步长）                        |

### 连接模式

`from` 字段在网络中创建灵活的数据流模式：

=== "顺序流"

    ```yaml
    - [-1, 1, Conv, [64, 3, 2]]    # 从前一层获取输入
    ```

=== "跳跃连接"

    ```yaml
    - [[-1, 6], 1, Concat, [1]]    # 将当前层与第6层合并
    ```

=== "多输入融合"

    ```yaml
    - [[4, 6, 8], 1, Detect, [nc]] # 使用3种特征尺度的检测头
    ```

!!! note "层索引"

    层从0开始索引。负数索引引用前面的层（`-1` = 上一层），正数索引引用特定位置的层。

### 模块重复

`repeats` 参数创建更深的网络段：

```yaml
- [-1, 3, C2f, [128, True]] # 创建3个连续的 C2f 模块
- [-1, 1, Conv, [64, 3, 2]] # 单个卷积层
```

实际重复次数会乘以模型尺寸配置中的深度缩放因子。

## 可用模块

模块按功能组织，定义在 [Ultralytics 模块目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules) 中。以下表格按类别列出了常用模块，源代码中还有更多可用模块：

### 基本操作

| 模块          | 用途                         | 源码                                                                                             | 参数                                    |
| ------------- | ---------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------- |
| `Conv`        | 卷积 + BatchNorm + 激活函数   | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)   | `[out_ch, kernel, stride, pad, groups]` |
| `nn.Upsample` | 空间上采样                    | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html)                 | `[size, scale_factor, mode]`            |
| `nn.Identity` | 直通操作                      | [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html)                 | `[]`                                    |

### 复合模块

| 模块     | 用途                           | 源码                                                                                             | 参数                            |
| -------- | ------------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------- |
| `C2f`    | 带2个卷积的 CSP 瓶颈模块        | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, shortcut, expansion]` |
| `SPPF`   | 空间金字塔池化（快速版）         | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, kernel_size]`         |
| `Concat` | 通道维度拼接                    | [conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py)   | `[dimension]`                   |

### 专用模块

| 模块          | 用途                       | 源码                                                                                             | 参数                                                      |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| `TorchVision` | 加载任意 torchvision 模型   | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, model_name, weights, unwrap, truncate, split]`  |
| `Index`       | 从列表中提取特定张量         | [block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) | `[out_ch, index]`                                         |
| `Detect`      | YOLO 检测头                 | [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)   | `[nc]`                                                    |

!!! info "完整模块列表"

    以上仅列出了部分可用模块。要查看完整的模块列表及其参数，请浏览 [modules 目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules)。

## 高级特性

### TorchVision 集成

TorchVision 模块可以将任意 [TorchVision 模型](https://docs.pytorch.org/vision/stable/models.html) 无缝集成为骨干网络：

=== "Python"

    ```python
    from ultralytics import YOLO

    # 使用 ConvNeXt 骨干网络的模型
    model = YOLO("convnext_backbone.yaml")
    results = model.train(data="coco8.yaml", epochs=100)
    ```

=== "YAML 配置"

    ```yaml
    backbone:
      - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, False]]
    head:
      - [-1, 1, Classify, [nc]]
    ```

    **参数详解：**

    - `768`：期望输出的通道数
    - `convnext_tiny`：模型架构（[可用模型](https://docs.pytorch.org/vision/stable/models.html)）
    - `DEFAULT`：使用预训练权重
    - `True`：移除分类头
    - `2`：截断最后2层
    - `False`：返回单个张量（非列表）

!!! tip "多尺度特征"

    将最后一个参数设为 `True` 可获取中间特征图用于多尺度检测。

### Index 模块用于特征选择

当使用输出多个特征图的模型时，Index 模块可以选择特定的输出：

```yaml
backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]] # 多输出
head:
    - [0, 1, Index, [192, 4]] # 选择第4个特征图（192通道）
    - [0, 1, Index, [384, 6]] # 选择第6个特征图（384通道）
    - [0, 1, Index, [768, 8]] # 选择第8个特征图（768通道）
    - [[1, 2, 3], 1, Detect, [nc]] # 多尺度检测
```

## 模块解析系统

了解 Ultralytics 如何定位和导入模块对于自定义开发至关重要：

### 模块查找流程

Ultralytics 在 [`parse_model`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中使用三层解析机制：

```python
# 核心解析逻辑
m = (
    getattr(torch.nn, m[3:])
    if "nn." in m
    else getattr(torchvision.ops, m[16:])
    if "torchvision.ops." in m
    else globals()[m]
)
```

1. **PyTorch 模块**：以 `'nn.'` 开头的名称 -> `torch.nn` 命名空间
2. **TorchVision 操作**：以 `'torchvision.ops.'` 开头的名称 -> `torchvision.ops` 命名空间
3. **Ultralytics 模块**：所有其他名称 -> 通过导入进入全局命名空间

### 模块导入链

标准模块通过 [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 中的导入变为可用：

```python
from ultralytics.nn.modules import (  # noqa: F401
    SPPF,
    C2f,
    Conv,
    Detect,
    # ... 更多模块
    Index,
    TorchVision,
)
```

## 自定义模块集成

### 修改源代码

修改源代码是集成自定义模块最灵活的方式，但需要谨慎操作。要定义和使用自定义模块，请按以下步骤操作：

1. **以开发模式安装 Ultralytics**，使用 [快速入门指南](https://docs.ultralytics.com/quickstart#git-clone) 中的 Git clone 方式。

2. **在 [`ultralytics/nn/modules/block.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py) 中定义你的模块**：

    ```python
    class CustomBlock(nn.Module):
        """自定义模块：Conv-BatchNorm-ReLU 序列。"""

        def __init__(self, c1, c2):
            """使用输入和输出通道数初始化 CustomBlock。"""
            super().__init__()
            self.layers = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU())

        def forward(self, x):
            """模块的前向传播。"""
            return self.layers(x)
    ```

3. **在包级别暴露你的模块**，修改 [`ultralytics/nn/modules/__init__.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/__init__.py)：

    ```python
    from .block import CustomBlock  # noqa 使 CustomBlock 可作为 ultralytics.nn.modules.CustomBlock 使用
    ```

4. **添加到导入中**，修改 [`ultralytics/nn/tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py)：

    ```python
    from ultralytics.nn.modules import CustomBlock  # noqa
    ```

5. **处理特殊参数**（如需），在 `ultralytics/nn/tasks.py` 的 [`parse_model()`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 函数中：

    ```python
    # 在 parse_model() 函数中添加此条件
    if m is CustomBlock:
        c1, c2 = ch[f], args[0]  # 输入通道, 输出通道
        args = [c1, c2, *args[1:]]
    ```

6. **在模型 YAML 中使用该模块**：

    ```yaml
    # custom_model.yaml
    nc: 1
    backbone:
        - [-1, 1, CustomBlock, [64]]
    head:
        - [-1, 1, Classify, [nc]]
    ```

7. **检查 FLOPs** 以确保前向传播正常工作：

    ```python
    from ultralytics import YOLO

    model = YOLO("custom_model.yaml", task="classify")
    model.info()  # 如果工作正常，应打印非零 FLOPs
    ```

## 配置示例

### 基础检测模型

```yaml
# 简单 YOLO 检测模型
nc: 80
scales:
    n: [0.33, 0.25, 1024]

backbone:
    - [-1, 1, Conv, [64, 3, 2]]   # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
    - [-1, 3, C2f, [128, True]]   # 2
    - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
    - [-1, 6, C2f, [256, True]]   # 4
    - [-1, 1, SPPF, [256, 5]]     # 5

head:
    - [-1, 1, Conv, [256, 3, 1]]   # 6
    - [[6], 1, Detect, [nc]]       # 7
```

### TorchVision 骨干网络模型

```yaml
# ConvNeXt 骨干网络 + YOLO 检测头
nc: 80

backbone:
    - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]]

head:
    - [0, 1, Index, [192, 4]] # P3 特征
    - [0, 1, Index, [384, 6]] # P4 特征
    - [0, 1, Index, [768, 8]] # P5 特征
    - [[1, 2, 3], 1, Detect, [nc]] # 多尺度检测
```

### 分类模型

```yaml
# 简单分类模型
nc: 1000

backbone:
    - [-1, 1, Conv, [64, 7, 2, 3]]
    - [-1, 1, nn.MaxPool2d, [3, 2, 1]]
    - [-1, 4, C2f, [64, True]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 8, C2f, [128, True]]
    - [-1, 1, nn.AdaptiveAvgPool2d, [1]]

head:
    - [-1, 1, Classify, [nc]]
```

## 最佳实践

### 架构设计建议

**从简单开始**：在自定义之前，先从经过验证的架构入手。使用现有 YOLO 配置作为模板，逐步修改而非从零构建。

**增量测试**：逐步验证每次修改。一次只添加一个自定义模块，确认其工作正常后再进行下一步修改。

**监控通道数**：确保连接层之间的通道维度匹配。一层的输出通道数（`c2`）必须与序列中下一层的输入通道数（`c1`）一致。

**使用跳跃连接**：利用 `[[-1, N], 1, Concat, [1]]` 模式实现特征复用。这些连接有助于梯度流动，并允许模型组合不同尺度的特征。

**合理缩放**：根据计算资源选择模型规模。边缘设备使用 nano（`n`），平衡性能使用 small（`s`），追求最高精度使用更大的规模（`m`、`l`、`x`）。

### 性能考量

**深度 vs 宽度**：深层网络通过多个变换层捕获复杂的层次化特征，而宽网络在每层并行处理更多信息。根据任务复杂度平衡两者。

**跳跃连接**：改善训练期间的梯度流动，并在整个网络中实现特征复用。在较深的架构中尤其重要，可防止梯度消失。

**瓶颈模块**：在保持模型表达能力的同时降低计算成本。像 `C2f` 这样的模块比标准卷积使用更少的参数，同时保持特征学习能力。

**多尺度特征**：对于在同一图像中检测不同大小的物体至关重要。使用特征金字塔网络（FPN）模式，在不同尺度上配置多个检测头。

## 故障排除

### 常见问题

| 问题                                            | 原因                 | 解决方案                                                                                                   |
| ----------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `KeyError: 'ModuleName'`                        | 模块未导入            | 添加到 [`tasks.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py) 的导入中 |
| 通道维度不匹配                                   | `args` 参数错误       | 验证输入/输出通道兼容性                                                                                     |
| `AttributeError: 'int' object has no attribute` | 参数类型错误          | 查阅模块文档确认正确的参数类型                                                                               |
| 模型构建失败                                     | `from` 引用无效       | 确保引用的层存在                                                                                            |

### 调试技巧

开发自定义架构时，系统化的调试有助于尽早发现问题：

**使用 Identity 检测头进行测试**

用 `nn.Identity` 替换复杂的检测头以隔离骨干网络问题：

```yaml
nc: 1
backbone:
    - [-1, 1, CustomBlock, [64]]
head:
    - [-1, 1, nn.Identity, []] # 调试用直通
```

这样可以直接检查骨干网络输出：

```python
import torch

from ultralytics import YOLO

model = YOLO("debug_model.yaml")
output = model.model(torch.randn(1, 3, 640, 640))
print(f"输出形状: {output.shape}")  # 应与预期维度匹配
```

**模型架构检查**

检查 FLOPs 计数并打印每一层也有助于调试自定义模型配置。有效模型的 FLOPs 计数应为非零值。如果为零，则前向传播可能存在问题。运行简单的前向传播应显示遇到的具体错误。

```python
from ultralytics import YOLO

# 使用详细输出构建模型以查看层详情
model = YOLO("debug_model.yaml", verbose=True)

# 检查模型 FLOPs。前向传播失败会导致 FLOPs 为 0。
model.info()

# 检查各层
for i, layer in enumerate(model.model.model):
    print(f"第 {i} 层: {layer}")
```

**逐步验证**

1. **最小化开始**：先用最简单的架构进行测试
2. **增量添加**：逐层增加复杂度
3. **检查维度**：验证通道和空间尺寸的兼容性
4. **验证缩放**：使用不同模型规模（`n`、`s`、`m`）进行测试

## 常见问题

### 如何更改模型中的类别数量？

在 YAML 文件顶部将 `nc` 参数设置为与数据集类别数量匹配的值。

```yaml
nc: 5 # 5个类别
```

### 可以在模型 YAML 中使用自定义骨干网络吗？

可以。你可以使用任何支持的模块，包括 TorchVision 骨干网络，或者按照 [自定义模块集成](#自定义模块集成) 中的说明定义并导入自己的自定义模块。

### 如何为不同尺寸（nano、small、medium 等）缩放模型？

在 YAML 的 [`scales` 部分](#parameters参数部分) 中定义深度、宽度和最大通道数的缩放因子。当你加载附加了缩放后缀的基础 YAML 文件时（例如 `yolo26n.yaml`），模型将自动应用这些因子。

### `[from, repeats, module, args]` 格式的含义是什么？

此格式指定每一层的构建方式：

- `from`：输入来源
- `repeats`：模块重复次数
- `module`：层类型
- `args`：模块参数

### 如何排查通道不匹配错误？

检查一层的输出通道是否与下一层的期望输入通道匹配。使用 `print(model.model.model)` 检查模型架构。

### 在哪里可以找到可用模块及其参数的列表？

查看 [`ultralytics/nn/modules` 目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/nn/modules) 中的源代码，获取所有可用模块及其参数。

### 如何在 YAML 配置中添加自定义模块？

在源代码中定义你的模块，按照 [修改源代码](#修改源代码) 中的方式导入，然后在 YAML 文件中按名称引用它。

### 可以在自定义 YAML 中使用预训练权重吗？

可以，使用 `model.load("path/to/weights")` 从预训练检查点加载权重。但只有匹配的层的权重才会成功加载。

### 如何验证模型配置？

使用 `model.info()` 检查 FLOPs 计数是否为非零。有效模型的 FLOPs 应为非零值。如果为零，请按照 [调试技巧](#调试技巧) 中的建议查找问题。
