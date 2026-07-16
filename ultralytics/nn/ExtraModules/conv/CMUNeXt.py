"""
CMUNeXt: 大内核倒瓶颈设计 — 医学图像检测骨干

核心机制: 大核深度可分离卷积 + 倒瓶颈(4x扩展) + GELU
"""

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """CMUNeXt Block — 大核深度可分离卷积"""
    def __init__(self, c1, c2, depth=1, k=7):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        layers = []
        for i in range(depth):
            layers.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(c2, c2, kernel_size=k, groups=c2, padding=k // 2),
                    nn.GELU(), nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(c2, c2 * 4, kernel_size=1), nn.GELU(), nn.BatchNorm2d(c2 * 4),
                nn.Conv2d(c2 * 4, c2, kernel_size=1), nn.GELU(), nn.BatchNorm2d(c2)
            ))
        self.block = nn.Sequential(*layers)
        self.up = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.block(x)
        x = self.up(x)
        return x


__all__ = ['CMUNeXtBlock']
