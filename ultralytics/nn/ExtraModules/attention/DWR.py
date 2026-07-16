"""
DWR: Dilated-Wise Residual attention — 可扩张残差注意力模块

核心机制:
  1. 3x3卷积降维 → 3分支空洞卷积(dilation=1,3,5)
  2. 多尺度拼接 → 1x1融合 → 残差加回

对 RGBT 价值: 多空洞率捕获不同尺度目标特征
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, d if d > 1 else k // 2, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWR(nn.Module):
    """Dilated-Wise Residual attention — 可扩张残差注意力"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.conv_3x3 = Conv(c2, c2 // 2, 3)
        self.conv_d1 = Conv(c2 // 2, c2 // 2, 3, d=1)
        self.conv_d3 = Conv(c2 // 2, c2 // 2, 3, d=3)
        self.conv_d5 = Conv(c2 // 2, c2 // 2, 3, d=5)
        self.conv_1x1 = Conv(c2 // 2 * 3, c2, k=1)

    def forward(self, x):
        x = self.proj(x)
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_d1(conv_3x3), self.conv_d3(conv_3x3), self.conv_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out)
        return x_out + x


__all__ = ['DWR']
