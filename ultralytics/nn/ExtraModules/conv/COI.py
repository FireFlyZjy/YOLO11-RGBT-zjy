"""
COI: Conv-One-Identity — 三重残差卷积

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  三条并行路径:
    1. Shortcut (恒等映射 + BN)
    2. Depthwise Conv (保持通道独立)
    3. 1x1 Conv (跨通道线性组合)
  三条输出相加 → GELU 激活

对 RGBT 价值:
  极轻量基础组件, 深度可分离卷积 + 残差连接,
  适合嵌入 RGB-T 检测网络 backbone

用法 (YAML — 单输入, 替换 Conv):
  - [-1, 1, COI, [c2]]
"""

import torch
import torch.nn as nn


class COI(nn.Module):
    """Conv-One-Identity — 三重残差卷积"""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.dw = nn.Conv2d(c2, c2, kernel_size=k, padding=k // 2, stride=s, groups=c2)
        self.conv1_1 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)

        shortcut = self.bn1(x)
        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))

        return self.act(shortcut + x_dw + x_conv1_1)


__all__ = ['COI']
