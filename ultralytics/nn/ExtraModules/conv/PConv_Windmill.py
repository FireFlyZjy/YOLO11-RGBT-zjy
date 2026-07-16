"""
PConv: Pinwheel-shaped Convolution — 风车形卷积 (AAAI 2025)

论文: AAAI 2025 — PConv: 一种新颖的风车形卷积，符合微弱小目标的
      像素高斯空间分布，增强特征提取，显著增加接受野

核心机制:
  1. 使用非对称填充(Asymmetric Padding)将标准卷积变为风车形
  2. 4个方向分支: 上/下/左/右条带卷积 → 拼接 → 融合
  3. 对高斯分布的小目标像素有更好的覆盖

对 RGBT 价值: 风车形卷积在空间上更贴合小目标的分布模式，
              Thermal 模态中小目标尤为适用

用法:
  - [-1, 1, PConv, [c2, k, s]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PConv(nn.Module):
    """Pinwheel-shaped Convolution using Asymmetric Padding"""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        c2 = c2
        # 4 方向非对称填充: (left, right, top, bottom)
        paddings = [(k, 0, 0, 0), (0, k, 0, 0), (0, 0, k, 0), (0, 0, 0, k)]
        self.pads = nn.ModuleList([nn.ZeroPad2d(p) for p in paddings])

        self.cw = nn.Conv2d(c2, c2 // 4, (1, k), stride=s, padding=0)
        self.ch = nn.Conv2d(c2, c2 // 4, (k, 1), stride=s, padding=0)
        self.cat = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.proj(x)
        yw0 = self.cw(self.pads[0](x))  # (B, C/4, H, W+1)
        yw1 = self.cw(self.pads[1](x))  # (B, C/4, H, W+1)
        yh0 = self.ch(self.pads[2](x))  # (B, C/4, H+1, W)
        yh1 = self.ch(self.pads[3](x))  # (B, C/4, H+1, W)

        # 对齐所有分支到相同空间尺寸
        h, w = x.shape[2:]
        yw0 = F.interpolate(yw0, size=(h, w), mode='bilinear', align_corners=False)
        yw1 = F.interpolate(yw1, size=(h, w), mode='bilinear', align_corners=False)
        yh0 = F.interpolate(yh0, size=(h, w), mode='bilinear', align_corners=False)
        yh1 = F.interpolate(yh1, size=(h, w), mode='bilinear', align_corners=False)

        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class APBottleneck(nn.Module):
    """Bottleneck with PConv — 风车形卷积瓶颈块"""
    def __init__(self, c1, c2, shortcut=True, k=3, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = PConv(c_, c2, k, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_PConv(nn.Module):
    """C3k2 with PConv — 集成风车形卷积的C3k2"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_ * 2, c2, 1)
        self.m = nn.Sequential(*(APBottleneck(c_, c_, shortcut, g=g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


__all__ = ['PConv', 'APBottleneck', 'C3k2_PConv']
