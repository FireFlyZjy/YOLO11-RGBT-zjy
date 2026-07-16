"""
SHViT: Single-Head Vision Transformer — 单头注意力模块 (CVPR2024)

核心机制:
  1. SHSA: 单头自注意力 — 通道拆分为pdim(做注意力)+剩余(跳跃)
  2. SHSABlock: DWConv + SHSA + FFN 残差组合
  3. 分组归一化 + SiLU 激活

对 RGBT 价值: 极轻量自注意力，适合嵌入双模态检测网络
"""

import torch
import torch.nn as nn


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)


class SHSA_GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(1, num_channels)


class SHSABlock_FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class SHSA(nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.pre_norm = SHSA_GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.SiLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x


class SHSABlock(nn.Module):
    """Single-Head Self-Attention Block"""
    def __init__(self, c1, c2, qk_dim=16, pdim=32):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        dim = c2
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))

    def forward(self, x):
        x = self.proj(x)
        return self.ffn(self.mixer(self.conv(x)))


class C2f_SHViT(nn.Module):
    """C2f with SHViT block"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SHSABlock(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ['SHSA', 'SHSABlock', 'C2f_SHViT']
