"""
PConv + Faster_Block — 部分卷积与轻量前馈模块

原理（Partial Convolution, PConv）：
    仅对输入通道的 1/4 执行 3x3 卷积，其余通道保持不变。
    相比常规 3x3 卷积可减少约 3/4 的 FLOPs，在保持感受野的同时大幅降低计算量。

机制：
    PConv (Partial_conv3):
        - 将输入沿通道维度切分为两部分
        - 仅对前 1/4 通道执行 3x3 depth-wise 风格卷积
        - 与剩余通道拼接后输出
    Faster_Block:
        - PConv → MLP (1x1→1x1) 串联
        - 包含残差连接与可选的 layer scale
        - 支持输入/输出通道数不一致时的自适应通道对齐

与 RGBT 关联：
    轻量级设计适合多模态场景下的高效特征提取，可在保持精度的同时降低计算开销，
    便于在双流网络（RGB + Thermal）中并行部署。

用法（在 yaml 中配置）：
    - [-1, 1, Faster_Block, [dim]]
    - [-1, 1, Faster_Block, [dim, n_div, mlp_ratio, drop_path]]
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class PConv(nn.Module):
    """Partial Convolution — 仅对 1/n_div 通道执行 3x3 卷积。

    Args:
        dim (int):    输入通道数
        n_div (int):  划分份数，仅前 1 份被卷积。默认 4（即 1/4 通道被处理）

    forward 策略:
        - 'split_cat' (默认): 训练/推理通用，先 split 后 conv → cat
        - 'slicing' (仅推理): 原地修改前 1/n_div 通道，效率更高

    Shape:
        Input:  (B, dim, H, W)
        Output: (B, dim, H, W)
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, g: int = 1, act: bool = True, n_div: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, s, 0, bias=False) if c1 != c2 or s != 1 else nn.Identity()
        dim = c2
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv, self.dim_conv, kernel_size=3,
            stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """proj → split → partial_conv → cat"""
        x = self.proj(x)
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat([x1, x2], dim=1)


class Faster_Block(nn.Module):
    """Faster Block — PConv + MLP 轻量模块。

    结构：
        x → PConv (3x3 partial) → MLP (1x1 → 1x1) → + shortcut → out

    Args:
        inc (int):              输入通道数
        dim (int):              输出/隐层通道数（当 inc != dim 时自动添加 1x1 对齐）
        n_div (int):            PConv 的划分份数，默认 4
        mlp_ratio (float):      MLP 隐藏层通道倍数，默认 2.0
        drop_path (float):      DropPath 率，默认 0.0
        layer_scale_init_value (float): Layer Scale 初始值，<=0 表示不使用。默认 0.0

    Shape:
        Input:  (B, inc, H, W)
        Output: (B, dim, H, W)
    """

    def __init__(self, inc: int, dim: int, n_div: int = 4,
                 mlp_ratio: float = 2.0, drop_path: float = 0.0,
                 layer_scale_init_value: float = 0.0):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        # 通道对齐
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, k=1)

        # PConv
        self.spatial_mixing = PConv(dim, n_div)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden, k=1),
            nn.Conv2d(mlp_hidden, dim, kernel_size=1, bias=False),
        )

        # DropPath
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

        # Layer Scale
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True
        ) if layer_scale_init_value > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)

        shortcut = x
        x = self.spatial_mixing(x)

        if self.layer_scale is not None:
            x = self.drop_path(
                self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
            )
        else:
            x = self.drop_path(self.mlp(x))

        return shortcut + x


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
