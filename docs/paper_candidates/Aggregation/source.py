"""
Aggregation: 多尺度密集注意力聚合

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. 3 个尺度金字塔特征 P3/P4/P5
  2. 上采样后逐元素乘进行跨尺度交互:
     - x2_1 = upsample(x1) * x2
     - x3_1 = upsample²(x1) * upsample(x2) * x3
  3. 密集拼接 + 卷积 → 单通道 attention map (或 c2 通道输出)

对 RGBT 价值:
  逐元素乘跨尺度交互能有效融合不同分辨率的模态特征,
  生成的 attention map 可用于后续特征加权

用法 (YAML — 单输入多尺度, 接收 3 个特征图):
  - [-1, 1, Aggregation, [c2]]

参数:
  c1: 输入通道 (或通道列表)
  c2: 输出通道数
"""

import torch
import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Aggregation(nn.Module):
    """Aggregation — 多尺度密集注意力聚合 (3尺度)"""

    def __init__(self, c1, c2):
        super().__init__()
        # c1 可能是 list [c_p3, c_p4, c_p5] 或 int
        if isinstance(c1, (list, tuple)):
            chs = list(c1)
        else:
            chs = [c1, c1, c1]

        self.proj = nn.ModuleList([
            nn.Conv2d(ch, c2, 1) if ch != c2 else nn.Identity()
            for ch in chs
        ])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(c2, c2, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(c2, c2, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(c2, c2, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(c2, c2, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * c2, 2 * c2, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * c2, 2 * c2, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * c2, 3 * c2, 3, padding=1)
        self.conv4 = BasicConv2d(3 * c2, 3 * c2, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * c2, c2, 1)

    def forward(self, x):
        """
        x: 列表, 按 [最小分辨率, 中分辨率, 最大分辨率] 顺序
           即 [P5(深层), P4, P3(浅层)]
        """
        if isinstance(x, torch.Tensor):
            x = [x, x, x]

        x1, x2, x3 = x[0], x[1], x[2]  # x1=最小, x2=中, x3=最大

        # 投影到统一通道 (各尺度独立投影)
        x1 = self.proj[0](x1)
        x2 = self.proj[1](x2)
        x3 = self.proj[2](x3)

        x1_1 = x1

        # 逐元素乘跨尺度交互
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (self.conv_upsample2(self.upsample(self.upsample(x1))) *
                self.conv_upsample3(self.upsample(x2)) * x3)

        # 密集拼接
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


__all__ = ['Aggregation']
