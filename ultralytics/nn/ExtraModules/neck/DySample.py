"""
DySample: 超轻量高效动态上采样算子 (2024)

核心机制: 基于内容感知的特征重排实现动态上采样
简化版: 使用 pixel shuffle 方式，保持稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """DySample — 动态上采样 (简化版)"""
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.scale = scale
        self.conv = nn.Conv2d(c2, c2 * scale * scale, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.proj(x)
        return self.pixel_shuffle(self.conv(x))


__all__ = ['DySample']
