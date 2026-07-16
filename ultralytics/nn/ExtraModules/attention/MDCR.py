"""
MDCR: 多膨胀通道精炼模块 — 红外小目标暴力涨点 (2024)

核心机制:
  1. 将输入通道分为4组，每组用不同膨胀率(dilation=1,2,4,8)处理
  2. 跨组通道重排: 每组取对应通道拼接 → 1x1卷积融合
  3. 输出融合各尺度膨胀特征

对 RGBT 价值: 多膨胀率同时捕获RGB纹理(小膨胀)和Thermal轮廓(大膨胀)
"""

import torch
import torch.nn as nn


class conv_block(nn.Module):
    """Conv + BN + ReLU 基础块"""
    def __init__(self, in_c, out_c, kernel_size=(3,3), padding=(1,1), dilation=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MDCR(nn.Module):
    """Multi-Dilated Channel Refinement — 多膨胀通道精炼"""
    def __init__(self, c1, c2, rate=None):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        c = c2

        rate = rate or [1, 2, 4, 8]
        self.block1 = conv_block(c // 4, c // 4, padding=rate[0], dilation=rate[0], groups=8)
        self.block2 = conv_block(c // 4, c // 4, padding=rate[1], dilation=rate[1], groups=8)
        self.block3 = conv_block(c // 4, c // 4, padding=rate[2], dilation=rate[2], groups=8)
        self.block4 = conv_block(c // 4, c // 4, padding=rate[3], dilation=rate[3], groups=8)
        self.out_s = conv_block(4, 4, kernel_size=(1, 1), padding=(0, 0))
        self.out = conv_block(c, c, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.proj(x)
        # 4 组按通道拆分
        xs = torch.chunk(x, 4, dim=1)
        x1, x2, x3, x4 = self.block1(xs[0]), self.block2(xs[1]), self.block3(xs[2]), self.block4(xs[3])

        # 跨组通道重排
        split_tensors = []
        for channel in range(x1.size(1)):
            channel_tensors = [
                tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]
            ]
            concat_channel = self.out_s(torch.cat(channel_tensors, dim=1))
            split_tensors.append(concat_channel)

        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x


__all__ = ['MDCR']
