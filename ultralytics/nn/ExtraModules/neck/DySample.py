"""
DySample: 超轻量高效动态上采样算子 (2024)

核心机制: 基于内容感知的特征重排实现动态上采样
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """DySample — 动态上采样"""
    def __init__(self, c1, c2, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.groups = groups
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        assert c2 % groups == 0
        self.c = c2 // groups
        self.offset = nn.Conv2d(c2, self.c * scale * scale * 2, 1)
        self.scope = nn.Conv2d(c2, self.c * scale * scale * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        offset = self.offset(x).reshape(B, self.groups, self.c * 2, self.scale, self.scale, H, W)
        scope = self.scope(x).reshape(B, self.groups, self.c * 2, self.scale, self.scale, H, W)
        scope = self.softmax(scope)

        x_g = x.reshape(B, self.groups, self.c, H, W)
        out = F.grid_sample(x_g, self._grid(H, W, offset, scope, x.device), mode='bilinear', align_corners=False)
        return out.reshape(B, C, H * self.scale, W * self.scale)

    def _grid(self, H, W, offset, scope, device):
        h, w = H * self.scale, W * self.scale
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                                        torch.linspace(-1, 1, w, device=device))
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


__all__ = ['DySample']
