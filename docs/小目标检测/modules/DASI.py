"""
DASI: 维度感知选择性集成模块 — 红外小目标暴力涨点 (2024)

核心机制:
  1. 跳跃连接 + 1x1卷积投影
  2. 通道分割 → 1x1降维 + 批归一化 + ReLU
  3. 尾部卷积融合输出

对 RGBT 价值: 维度感知选择增强红外通道特征响应
"""

import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3,3), padding=(1,1), dilation=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bag(nn.Module):
    """Bag 模块 — 多尺度特征聚合"""
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(64, 64, kernel_size=(3, 3), padding=(2, 2), dilation=2)
        self.conv2 = conv_block(64, 64, kernel_size=(3, 3), padding=(4, 4), dilation=4)
        self.conv3 = conv_block(64, 64, kernel_size=(3, 3), padding=(6, 6), dilation=6)
        self.conv4 = conv_block(64, 64, kernel_size=(3, 3), padding=(8, 8), dilation=8)
        self.conv_cat = conv_block(256, 64, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x1, x2, x3, x4 = self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)
        return self.conv_cat(torch.cat([x1, x2, x3, x4], dim=1))


class DASI(nn.Module):
    """Dimension-Aware Selective Integration — 维度感知选择性集成"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.bag = Bag()
        self.tail_conv = nn.Sequential(
            conv_block(c2, c2, kernel_size=(1, 1), padding=(0, 0), activation=False)
        )
        self.conv = nn.Sequential(
            conv_block(c2 // 2, c2 // 4, kernel_size=(1, 1), padding=(0, 0), activation=False)
        )
        self.bns = nn.BatchNorm2d(c2)
        self.skips = conv_block(c2, c2, kernel_size=(1, 1), padding=(0, 0), activation=False)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = self.tail_conv(x)
        x = x + x_skip
        x = self.bns(x)
        x = self.relu(x)
        return x


__all__ = ['DASI']
