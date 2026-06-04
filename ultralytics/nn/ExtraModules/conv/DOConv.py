# coding=utf-8
"""
DOConv2d / Conv_DO — Over-parameterized Convolution for YOLO

论文: DO-Conv (https://arxiv.org/abs/2006.12030)
原始实现: https://github.com/yangyanli/DO-Conv

机制:
    DOConv2d 在训练时将标准卷积核分解为两个可学习矩阵 D (depth-wise) 和 W (point-wise):
        DoW = einsum('ims,ois->oim', D, W) -> reshape 为 (O, I//G, M, N)
    这种过参数化 (over-parameterization) 增加了训练时的模型容量和表达能力。
    推理时调用 fuse() 将 D 和 W 合并为单个标准 Conv2d, 零额外计算开销。

Conv_DO 是 YOLO 兼容的封装, 将 DOConv2d + BN + SiLU 包装为标准 Conv 接口,
可直接在 YOLO YAML 配置文件中替换 Conv 使用。

RGBT 价值:
    多模态 (RGB + Thermal) 融合中, 过参数化卷积可以更好地建模跨模态特征交互,
    增强对光照变化和热辐射差异的鲁棒性, 同时推理时零额外成本。

用法 (YAML):
    backbone:
      - [-1, 1, Conv_DO, [64, 3, 2]]   # 替代标准 Conv
      - [-1, 1, Conv_DO, [128, 3, 2]]  # 替代标准 Conv
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from collections.abc import Iterable


def autopad(k, p=None, d=1):
    """自动计算填充使得输出尺寸与输入相同 (YOLO 标准 autopad)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _ntuple(n):
    """将标量扩展为 n 元组."""
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(x for _ in range(n))
    return parse


_pair = _ntuple(2)


class DOConv2d(nn.Module):
    """
    DOConv2d — Over-parameterized Conv2d

    训练时使用 D (depth-wise) + W (point-wise) 两个参数矩阵过参数化,
    推理时 fuse() 合并为单 Conv2d, 零额外开销.

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int | tuple): 卷积核大小, 默认 3
        s (int | tuple): 步长, 默认 1
        p (int | tuple | None): 填充, None 时自动计算
        g (int): 分组卷积组数, 默认 1
        act (bool): 仅用于接口兼容, DOConv2d 内部不使用
        D_mul (int | None): 深度乘数, 默认 kernel_size[0] * kernel_size[1]

    Shape:
        - 输入: (N, C_in, H, W)
        - 输出: (N, C_out, H_out, W_out)
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, D_mul=None):
        super().__init__()

        kernel_size = _pair(k)
        stride = _pair(s)
        dilation = _pair(1)
        padding = _pair(autopad(k, p))

        if c1 % g != 0:
            raise ValueError('in_channels must be divisible by groups')
        if c2 % g != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = g

        M, N = kernel_size
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul

        # ====== W: point-wise 权重 ======
        self.W = Parameter(torch.Tensor(c2, c1 // g, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        # ====== D: depth-wise 权重 ======
        if M * N > 1:
            # D 初始化为 0, D_diag 初始化为单位对角 (保证初始时 DOConv ≈ 标准 Conv)
            self.D = Parameter(torch.zeros(c1, M * N, self.D_mul))

            eye = torch.eye(M * N, dtype=torch.float32).reshape(1, M * N, M * N)
            D_diag = eye.repeat(c1, 1, self.D_mul // (M * N))
            if self.D_mul % (M * N) != 0:
                zeros = torch.zeros(c1, M * N, self.D_mul % (M * N))
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:
                self.D_diag = Parameter(D_diag, requires_grad=False)

    def forward(self, x):
        """
        前向传播: 动态计算 DoW 卷积核并执行卷积.
        """
        M, N = self.kernel_size
        out_shape = (self.out_channels, self.in_channels // self.groups, M, N)

        if M * N > 1:
            # D: (C_in, M*N, D_mul), W_reshaped: (C_out//G, C_in, D_mul)
            D = self.D + self.D_diag  # (C_in, M*N, D_mul)
            W_reshaped = self.W.reshape(self.out_channels // self.groups,
                                        self.in_channels, self.D_mul)
            # einsum: (i, m, s), (o, i, s) -> (o, i, m)
            kernel = torch.einsum('ims,ois->oim', D, W_reshaped)
            kernel = kernel.reshape(out_shape)
        else:
            # 1x1 卷积无需 D
            kernel = self.W.reshape(out_shape)

        return F.conv2d(x, kernel, None, self.stride, self.padding,
                        self.dilation, self.groups)

    def fuse(self):
        """
        融合 D 和 W 为单个标准 Conv2d 权重.

        Returns:
            nn.Conv2d: 融合后的标准卷积层, bias 初始为 0.
        """
        M, N = self.kernel_size
        out_shape = (self.out_channels, self.in_channels // self.groups, M, N)

        if M * N > 1:
            D = self.D + self.D_diag
            W_reshaped = self.W.reshape(self.out_channels // self.groups,
                                        self.in_channels, self.D_mul)
            fused_weight = torch.einsum('ims,ois->oim', D, W_reshaped)
            fused_weight = fused_weight.reshape(out_shape)
        else:
            fused_weight = self.W.reshape(out_shape)

        conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups,
            bias=True
        )
        conv.weight.data = fused_weight
        conv.bias.data.zero_()
        return conv


class Conv_DO(nn.Module):
    """
    Conv_DO — YOLO 兼容的 DOConv 封装

    将 DOConv2d + BatchNorm + SiLU 包装为标准 YOLO Conv 接口,
    可直接在 YAML 配置文件中作为 Conv 的替代使用.

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核大小, 默认 3
        s (int): 步长, 默认 1
        p (int | None): 填充, None 时自动计算
        g (int): 分组数, 默认 1
        act (bool): 是否使用 SiLU 激活, 默认 True

    YAML 用法:
        # 直接替换标准 Conv
        backbone:
          - [-1, 1, Conv_DO, [64, 3, 2]]   # 替代 Conv(3, 64, 3, 2)
          - [-1, 1, Conv_DO, [128, 3, 2]]  # 替代 Conv(64, 128, 3, 2)

        # 或与其它模块组合
        head:
          - [-1, 1, Conv_DO, [256, 3, 1]]

    推理融合:
        model.fuse() 会自动调用 Conv_DO.fuse(), 将 DOConv2d + BN 合并为
        标准 Conv2d, 实现零额外推理开销.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = DOConv2d(c1, c2, k=k, s=s, p=p, g=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        """
        融合 DOConv2d + BN 为单个标准 Conv2d + 激活.

        两步融合:
        1. fuse D + W -> 标准 Conv2d 权重
        2. absorb BN scale/shift -> conv weight/bias

        Returns:
            self: 融合后的模块, self.conv 为标准 Conv2d, self.bn 为 Identity.
        """
        fused_conv = self.conv.fuse()

        if isinstance(self.bn, nn.BatchNorm2d):
            w = fused_conv.weight
            mean = self.bn.running_mean
            var = self.bn.running_var
            gamma = self.bn.weight
            beta = self.bn.bias
            eps = self.bn.eps

            scale = gamma / torch.sqrt(var + eps)
            fused_conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
            fused_conv.bias.data = (fused_conv.bias - mean) * scale + beta

        self.conv = fused_conv
        self.bn = nn.Identity()
        return self
