"""
CAMixing: 卷积-注意融合模块 + CAFMAttention + MSFN

论文: CAMixing — 卷积-注意融合模块和多尺度提取能力 (2024)

核心机制:
  1. CAFMAttention: 卷积和注意力融合注意力模块
  2. MSFN: 多尺度前馈网络
  3. 组合为 CAMixingTransformerBlock

对 RGBT 价值: 红外小目标专用设计，全局上下文+局部特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm for 2D features"""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = nn.Sequential(
                nn.GroupNorm(1, dim, eps=1e-6),
            )
        else:
            self.body = nn.Sequential(
                nn.GroupNorm(1, dim, eps=1e-6),
            )

    def forward(self, x):
        return self.body(x)


class CAFMAttention(nn.Module):
    """卷积和注意力融合注意力模块"""
    def __init__(self, dim, num_heads, bias=False, sr_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.sr = nn.Identity()

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 对 k/v 做空间下采样，降低注意力矩阵大小
        if self.sr_ratio > 1:
            k = self.sr(k)
            v = self.sr(v)
        _, _, h_k, w_k = k.shape

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h_k * w_k)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h_k * w_k).permute(0, 1, 3, 2)

        attn = (q @ k) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class MSFN(nn.Module):
    """多尺度前馈网络"""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                   kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                   kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv3x3(x).chunk(2, dim=1)
        x3, x4 = self.dwconv5x5(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2 + F.gelu(x3) * x4
        x = self.project_out(x)
        return x


class CAMixingTransformerBlock(nn.Module):
    """CAMixing Transformer Block — 卷积-注意融合模块"""
    def __init__(self, c1, c2, num_heads=4, ffn_expansion_factor=2.66, sr_ratio=8):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.norm1 = LayerNorm(c2)
        self.attn = CAFMAttention(c2, num_heads, sr_ratio=sr_ratio)
        self.norm2 = LayerNorm(c2)
        self.ffn = MSFN(c2, ffn_expansion_factor)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


__all__ = ['CAMixingTransformerBlock', 'CAFMAttention', 'MSFN']
