"""
DAB: Dual Attention Block — 双注意力块 (遥感去雾 2024.12)

核心机制: 通道注意力 + 空间注意力 + 局部通道注意力 + 全局通道注意力
"""

import torch
import torch.nn as nn


class LocalChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, padding_mode='reflect')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, H, W = x.shape
        att = self.gap(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att = att.reshape(N, C, 1, 1)
        return (x * att + x) + (self.local(x) * x)


class GlobalChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.ca(x) * x


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial(x) * x


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(self.act(self.fc1(x)))
        return x.transpose(1, 2).reshape(B, -1, H, W)


class DualAttentionBlock(nn.Module):
    """Dual Attention Block — 双注意力块"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.norm1 = nn.BatchNorm2d(c2)
        self.norm2 = nn.BatchNorm2d(c2)
        self.conv1 = nn.Conv2d(c2, c2, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=5, padding=2, groups=c2, padding_mode='reflect')
        self.gp = LocalChannelAttention(c2)
        self.cam = GlobalChannelAttention(c2)
        self.pam = SpatialAttention(c2)
        self.mlp = Mlp(1, c2, hidden_features=c2 * 4, out_features=c2)
        self.mlp2 = Mlp(1, c2 * 3, hidden_features=c2 * 4, out_features=c2)

    def forward(self, x):
        x = self.proj(x)
        identity = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.gp(x), self.cam(x), self.pam(x)], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class C3k2_DAB(nn.Module):
    """C3k2 with DualAttentionBlock"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DualAttentionBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ['DualAttentionBlock', 'C3k2_DAB']
