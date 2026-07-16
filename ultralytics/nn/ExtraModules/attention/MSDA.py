"""
MSDA: Multi-Scale Dilated Attention — 多尺度空洞注意力 (中科院一区)
"""

import torch
import torch.nn as nn


class DilateAttention(nn.Module):
    def __init__(self, head_dim, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        pad = dilation * (kernel_size - 1) // 2
        self.unfold = nn.Unfold(kernel_size, dilation, pad, 1)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        num_heads = d // self.head_dim
        q = q.reshape(B, num_heads, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(B, num_heads, self.head_dim, -1, H * W).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        v = self.unfold(v).reshape(B, num_heads, self.head_dim, -1, H * W).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    """MSDA — Multi-Scale Dilated Attention (c1, c2 compatible)"""
    def __init__(self, c1, c2, num_heads=8, kernel_size=3, dilation=None):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        dim = c2
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation or [1, 2, 3, 4]
        self.num_dilation = len(self.dilation)
        assert num_heads % self.num_dilation == 0, \
            f"num_heads({num_heads}) must be multiple of num_dilation({self.num_dilation})"

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.dilate_attention = nn.ModuleList([
            DilateAttention(head_dim, kernel_size, self.dilation[i])
            for i in range(self.num_dilation)
        ])
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} != dim {self.dim}"

        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Split into dilation groups
        qs = q.reshape(B, self.num_dilation, C // self.num_dilation, H, W).unbind(1)
        ks = k.reshape(B, self.num_dilation, C // self.num_dilation, H, W).unbind(1)
        vs = v.reshape(B, self.num_dilation, C // self.num_dilation, H, W).unbind(1)

        out = []
        for i in range(self.num_dilation):
            out.append(self.dilate_attention[i](qs[i], ks[i], vs[i]))

        out = torch.cat([o.permute(0, 3, 1, 2) for o in out], dim=1)
        return self.proj_out(out)


__all__ = ['MultiDilatelocalAttention']
