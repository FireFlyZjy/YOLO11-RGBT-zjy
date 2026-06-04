"""
scSE — 空间和通道注意力 (Spatial + Channel Squeeze-and-Excitation)

原理（Concurrent Spatial and Channel SE, 源自 "Concurrent Spatial and Channel SE"）：
    cSE (Channel SE):
        - 对空间做 GlobalAvgPool → Conv2d(1x1, c→c/r) → ReLU → Conv2d(1x1, c/r→c) → Sigmoid
        - 学习每个通道的重要性权重，对通道维度重新校准
        - 与标准 SE 的区别：使用 Conv2d 保持 4D 张量，与 YOLO 的卷积流兼容
    sSE (Spatial SE):
        - Conv2d(1x1, c→1) → Sigmoid → 逐元素乘
        - 仅包含一个 1×1 卷积，极其轻量（c 个参数）
        - 学习每个空间位置的重要性，相当于空间注意力
    scSE (Concurrent cSE + sSE):
        - cSE 和 sSE 并行计算后相加
        - 同时获得通道重校准和空间注意力两种增强

与 RGBT 关联：
    cSE 可以学习哪些通道对当前模态（RGB/IR）更重要，实现通道级的模态自适应。
    sSE 可以聚焦空间上目标所在区域，对多模态特征的对齐和融合有益。
    二者结合（scSE）为多模态特征提供全面的通道+空间注意力增强。

用法（在 yaml 中配置）：
    - [-1, 1, Att_cSE, [out_channels]]       # 仅通道注意力
    - [-1, 1, Att_sSE, [out_channels]]       # 仅空间注意力（极轻量）
    - [-1, 1, Att_scSE, [out_channels, 16]]  # 通道+空间并行注意力, r=16
"""

import torch
import torch.nn as nn


class Att_cSE(nn.Module):
    """通道注意力 (Channel Squeeze-and-Excitation)。

    通过全局平均池化获取通道描述符，经两层 1×1 Conv 降维-升维后
    用 Sigmoid 生成 [0,1] 间的通道权重，对输入进行通道重校准。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数（用于 YOLO 接口兼容，实际与输入相同）
        r (int):  缩减率 (reduction ratio)，默认 16

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)  — 当 c1≠c2 时自动 1×1 投影对齐

    参数量: ~2 * c^2 / r (含 1×1 投影)
    """

    def __init__(self, c1: int, c2: int, r: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(c2, max(c2 // r, 4), 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.excitation = nn.Conv2d(max(c2 // r, 4), c2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：通道重校准。"""
        x = self.proj(x)
        # (B, c2, 1, 1)
        y = self.avgpool(x)
        y = self.squeeze(y)
        y = self.relu(y)
        y = self.excitation(y)
        y = self.sigmoid(y)
        return x * y


class Att_sSE(nn.Module):
    """空间注意力 (Spatial Squeeze-and-Excitation)。

    仅用一个 1×1 卷积将 C 个通道压缩为 1 个空间权重图，
    经 Sigmoid 后对输入进行空间维度加权。超轻量级注意力。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数（用于 YOLO 接口兼容）

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)

    参数量: c2 + (c1≠c2 时额外 c1*c2)
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        self.conv1x1 = nn.Conv2d(c2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：空间加权。"""
        x = self.proj(x)
        # (B, 1, H, W)
        y = self.sigmoid(self.conv1x1(x))
        return x * y


class Att_scSE(nn.Module):
    """空间和通道并行注意力 (Concurrent Spatial and Channel SE)。

    cSE 与 sSE 并行计算，结果逐元素相加。
    同时获得通道维度的重校准和空间维度的聚焦。

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        r (int):  cSE 的缩减率，默认 16

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)

    Example:
        >>> att = Att_scSE(64, 64, r=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = att(x)  # (2, 64, 32, 32)
    """

    def __init__(self, c1: int, c2: int, r: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        self.cse = Att_cSE(c2, c2, r)
        self.sse = Att_sSE(c2, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：cSE + sSE 并行计算后相加。"""
        x = self.proj(x)
        return self.cse(x) + self.sse(x)
