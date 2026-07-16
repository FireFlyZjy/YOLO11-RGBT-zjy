"""
MSDA: Multi-Scale Dilated Attention — 多尺度空洞注意力 (中科院一区)

核心机制:
  1. 多头注意力的每个头使用不同膨胀率的空洞采样
  2. nn.Unfold 实现空洞邻域采样
  3. 多尺度空洞覆盖不同感受野

对 RGBT 价值: 多尺度膨胀同时捕获细粒度纹理和粗粒度热辐射轮廓
"""

import torch
import torch.nn as nn


class DilateAttention(nn.Module):
    """Dilated Attention — 单个空洞率的注意力"""
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape(B, d // self.head_dim, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(B, d // self.head_dim, self.head_dim,
                                    self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(B, d // self.head_dim, self.head_dim,
                                    self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    """MSDA — Multi-Scale Dilated Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, kernel_size=3,
                 dilation=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation or [1, 2, 3, 4]
        self.kernel_size = kernel_size
        self.scale = head_dim ** -0.5
        self.num_dilation = len(self.dilation)
        assert num_heads % self.num_dilation == 0

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, None, attn_drop, kernel_size, self.dilation[i])
            for i in range(self.num_dilation)
        ])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.clone()
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)

        for i in range(self.num_dilation):
            y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])

        y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        y3 = self.proj(y2)
        y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
        return y4


class C2PSA_MSDA(nn.Module):
    """C2PSA with MSDA"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MultiDilatelocalAttention(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ['MultiDilatelocalAttention', 'C2PSA_MSDA']
