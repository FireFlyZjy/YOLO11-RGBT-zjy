"""
RFBBranch: 4分支非对称感受野增强模块

论文: HVPNet (RGB-T 显著目标检测)
与现有 BasicRFB/BasicRFB_small 的区别:
  BasicRFB:    3分支, dilation=[1,2,3], 普通3x3卷积
  BasicRFB_small: 4分支, dilation=[1,3,3,5], 带1x3/3x1非对称
  RFBBranch:   4分支, dilation=[3,5,7], 每分支先1x3+3x1非对称再3x3空洞

核心机制:
  4 分支分别使用 dilation=3,5,7 捕获大感受野,
  每分支先用 1x3/3x1 非对称卷积增强边缘方向感知

用法 (YAML — 单输入, 替换 Conv):
  - [-1, 1, RFBBranch, [c2]]
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RFBBranch(nn.Module):
    """RFBBranch — 4分支非对称感受野增强"""

    def __init__(self, c1, c2, s=1):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, stride=s) if c1 != c2 or s > 1 else nn.Identity()

        self.relu = nn.ReLU(True)

        # 4 分支: 空洞率 0(无空洞), 3, 5, 7
        self.branch0 = nn.Sequential(
            BasicConv2d(c2, c2, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(c2, c2, 1),
            BasicConv2d(c2, c2, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(c2, c2, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(c2, c2, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(c2, c2, 1),
            BasicConv2d(c2, c2, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(c2, c2, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(c2, c2, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(c2, c2, 1),
            BasicConv2d(c2, c2, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c2, c2, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c2, c2, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * c2, c2, 3, padding=1)
        self.conv_res = BasicConv2d(c2, c2, 1)

    def forward(self, x):
        x = self.proj(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


__all__ = ['RFBBranch']
