"""
PPA: 并行化注意力设计 — 红外小目标暴力涨点 (2024)

核心机制:
  1. 三重卷积分支 (3x3 × 3) + 跳跃连接
  2. LocalGlobalAttention: 局部-全局注意力 (2x2 和 4x4 窗口)
  3. ECA 通道注意力 + SpatialAttention 空间注意力
  4. Dropout + BN + ReLU 输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3,3), padding=(1,1), dilation=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, b=1, gamma=2):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class LocalGlobalAttention(nn.Module):
    """局部-全局注意力"""
    def __init__(self, channels, window_size=2):
        super().__init__()
        self.window_size = window_size
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        # 局部窗口划分
        q = self.query(x).reshape(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 1, 3, 5)
        k = self.key(x).reshape(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 1, 3, 5)
        v = self.value(x).reshape(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 1, 3, 5)

        attn = self.softmax((q * (c ** -0.5)) @ k.transpose(-2, -1))
        out = (attn @ v).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return out


class PPA(nn.Module):
    """Parallelized Attention — 并行化注意力设计"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        filters = c2

        self.skip = conv_block(c2, filters, kernel_size=(1, 1), padding=(0, 0), activation=False)
        self.c1 = conv_block(c2, filters, kernel_size=(3, 3), padding=(1, 1))
        self.c2 = conv_block(filters, filters, kernel_size=(3, 3), padding=(1, 1))
        self.c3 = conv_block(filters, filters, kernel_size=(3, 3), padding=(1, 1))

        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)

        x1, x2, x3 = self.c1(x), self.c2(x), self.c3(x)
        x_out = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4

        x_out = self.cn(x_out)
        x_out = self.sa(x_out)
        x_out = self.drop(x_out)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        return x_out


__all__ = ['PPA', 'LocalGlobalAttention', 'SpatialAttentionModule', 'ECA']
