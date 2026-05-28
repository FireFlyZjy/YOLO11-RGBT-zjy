"""
StarBlock — 星操作卷积块

原理（Star Operation / Rewrite the Stars, CVPR 2024）：
    核心操作是 element-wise multiplication（逐元素乘法）：
        out = activation(branch1) * branch2
    这种"星操作"可以在不增加参数的情况下隐式扩展特征维度，
    等价于在更高维空间进行特征变换，显著提升表达能力。

    来源: Rewrite the Stars (CVPR 2024) — https://arxiv.org/abs/2403.19967

机制：
    StarBlock:
        - DWConv (7x7, depthwise): 空间编码
        - f1 / f2: 两个并行的 1x1 卷积分支
        - star_op: ReLU6(f1(x)) * f2(x) 逐元素乘法
        - g: 1x1 卷积投影回原通道
        - DWConv2 (7x7): 二次深度卷积
        - 残差连接

与 RGBT 关联：
    星操作通过隐式高维特征变换，能在不显著增加计算量的情况下增强特征表示，
    适用于双流网络中对 RGB 和 Thermal 特征分别进行增强后再融合。

用法（在 yaml 中配置）：
    - [-1, 1, StarBlock, [dim, mlp_ratio, drop_path]]
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class ConvBN(nn.Sequential):
    """Conv2d + BatchNorm 的快捷组合。

    Args:
        in_planes (int):    输入通道
        out_planes (int):   输出通道
        kernel_size (int):  卷积核大小，默认 1
        stride (int):       步长，默认 1
        padding (int):      填充，默认 0
        dilation (int):     空洞率，默认 1
        groups (int):       分组数，默认 1
        with_bn (bool):     是否包含 BN，默认 True
    """

    def __init__(self, in_planes: int, out_planes: int,
                 kernel_size: int = 1, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, with_bn: bool = True):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, dilation, groups, bias=False)
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class StarBlock(nn.Module):
    """星操作卷积块 — element-wise multiplication 增强特征表达。

    结构:
        x → DWConv (7x7) → f1 (1x1) → ReLU6 ──┐
                          → f2 (1x1) ──────────→ * → g (1x1) → DWConv2 (7x7) → + x → out

    Args:
        dim (int):          通道数
        mlp_ratio (float):  中间通道扩张比，默认 3.0
        drop_path (float):  DropPath 率，默认 0.0

    Shape:
        Input:  (B, dim, H, W)
        Output: (B, dim, H, W)
    """

    def __init__(self, c1: int, c2: int, mlp_ratio: float = 3.0, drop_path: float = 0.0):
        super().__init__()
        dim = c2
        self.proj = nn.Conv2d(c1, dim, 1) if c1 != dim else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)

        self.dwconv = ConvBN(dim, dim, kernel_size=7, stride=1,
                             padding=(7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, hidden_dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, hidden_dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(hidden_dim, dim, kernel_size=1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, kernel_size=7, stride=1,
                              padding=(7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

        # DropPath (随机深度)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：星操作 (ReLU6(f1) * f2) + 残差。"""
        x = self.proj(x)
        identity = x

        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        # 星操作: 逐元素乘法
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))

        return identity + self.drop_path(x)


class DropPath(nn.Module):
    """DropPath / Stochastic Depth — 按样本随机丢弃路径。

    参考: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return mask * x / keep_prob
