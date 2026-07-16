"""
HS-FPN: High-level Screening Feature Pyramid Network — 多级特征融合金字塔 (2024)

核心机制: 通道注意力筛选 + 逐元素乘/加融合
"""

import torch
import torch.nn as nn


class ChannelAttention_HSFPN(nn.Module):
    """HS-FPN 通道注意力筛选模块"""
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x


__all__ = ['ChannelAttention_HSFPN']
