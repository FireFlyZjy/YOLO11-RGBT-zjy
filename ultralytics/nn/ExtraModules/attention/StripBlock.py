"""
StripBlock: 大型条带卷积模块 — StripR-CNN (2025)

论文: StripR-CNN — 大型条带卷积遥感目标检测新SOTA (2025)

核心机制:
  1. Strip_Attention: 条带注意力 (1xk 和 kx1 条带卷积)
  2. StripMlp: 条带MLP
  3. StripBlock: 组合条带注意力和MLP
  4. 层缩放 (LayerScale) 稳定训练

对 RGBT 价值: 条带卷积对细长目标(车辆/行人等)特别有效
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    """DropPath (Stochastic Depth)"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Strip_Attention(nn.Module):
    """条带注意力 — 水平/垂直条带卷积"""
    def __init__(self, dim, k1=1, k2=19):
        super().__init__()
        self.dim = dim
        self.k1, self.k2 = k1, k2

        # H 方向条带 (1xk)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv1d(dim, dim, kernel_size=k2, padding=k2 // 2, groups=dim)
        # W 方向条带 (kx1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w = nn.Conv1d(dim, dim, kernel_size=k2, padding=k2 // 2, groups=dim)

        # 标准 1x1 卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # H 方向: 先池化到 (C, H, 1), 再转 (C, H) 做Conv1d
        x_h = self.pool_h(x).squeeze(-1)  # (B, C, H)
        x_h = self.conv_h(x_h).unsqueeze(-1)  # (B, C, H, 1)

        # W 方向: 池化到 (C, 1, W), 转 (C, W)
        x_w = self.pool_w(x).squeeze(-2)  # (B, C, W)
        x_w = self.conv_w(x_w).unsqueeze(-2)  # (B, C, 1, W)

        # 融合条带注意力
        attn = x_h * x_w  # (B, C, H, W)
        attn = self.conv1x1(attn)
        attn = self.sigmoid(self.proj(attn))

        return x * attn


class StripMlp(nn.Module):
    """条带 MLP — 深度可分离卷积 + 线性层"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StripBlock(nn.Module):
    """StripBlock — 大型条带卷积块"""
    def __init__(self, c1, c2, k1=1, k2=19, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.norm1 = nn.BatchNorm2d(c2)
        self.norm2 = nn.BatchNorm2d(c2)
        self.attn = Strip_Attention(c2, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(c2 * mlp_ratio)
        self.mlp = StripMlp(in_features=c2, hidden_features=mlp_hidden_dim)

        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(c2), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(c2), requires_grad=True)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


__all__ = ['StripBlock', 'Strip_Attention']
