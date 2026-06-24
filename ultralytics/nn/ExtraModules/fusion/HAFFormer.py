"""
HAFFormer: Hierarchical Attention Fusion Transformer (层级注意力融合Transformer)

来源: LCAFNet (RGB-T dual-stream detection)
核心机制:
  1. CrossAttention_S (单向交叉注意力):
     - QK 来自模态A: 1x1 Conv → 3x3 Depthwise Conv (局部特征增强)
     - V 来自模态B: 1x1 Conv → 3x3 Depthwise Conv
     - 多头注意力: attn = softmax(Q @ K^T * temperature) @ V
  2. 双向交叉: RGB→IR 和 IR→RGB 各做一次 CrossAttention_S
  3. 门控融合: concat → 1x1 Conv → GELU → 3x3 DWConv → sigmoid 门控
     → w * rgb_enhanced + (1-w) * ir_enhanced

与现有模块的区别:
  - ICAFusion: Transformer (Linear) QKV, 需要自适应池化到固定网格
  - CIFusion: 通道注意力, 无空间建模
  - HAFFormer: Depthwise Conv QKV, 保留空间分辨率, 局部+全局建模

用法: [[vis_layer, ir_layer], 1, Att_HAFFormer, [c2, num_heads]]
"""

import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention_S(nn.Module):
    """单向交叉注意力: QK从模态A, V从模态B
    Depthwise Conv 增强局部特征, 多头注意力实现跨模态交互
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [query_feat, value_feat] — QK来自x[0], V来自x[1]
        """
        fea_0, fea_1 = x[0], x[1]
        b, c, h, w = fea_0.shape

        qk = self.qk_dwconv(self.qk(fea_0))
        q, k = qk.chunk(2, dim=1)

        v = self.v_dwconv(self.v(fea_1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class HAFFormerCore(nn.Module):
    """HAFFormer 核心: 双向 CrossAttention_S + 门控融合"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        bias = False
        self.mhca_rgb = CrossAttention_S(dim, num_heads, bias)
        self.mhca_ir = CrossAttention_S(dim, num_heads, bias)

        self.concat = nn.Identity()  # 简化: 直接 cat
        self.conv = nn.Sequential(nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias), nn.GELU())
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        rgb_fea, ir_fea = x[0], x[1]

        # 双向交叉注意力
        out_rgb = self.mhca_rgb([rgb_fea, ir_fea]) + rgb_fea
        out_ir = self.mhca_ir([ir_fea, rgb_fea]) + ir_fea

        # 门控融合
        fea_cat = torch.cat([out_rgb, out_ir], dim=1)
        w = self.dwconv(self.conv(fea_cat)).sigmoid()
        fused = w * out_rgb + (1 - w) * out_ir
        return fused


class Att_HAFFormer(nn.Module):
    """HAFFormer YOLO 包装器: 多输入通道投影 + HAFFormerCore

    用法: [[vis_layer, ir_layer], 1, Att_HAFFormer, [c2, num_heads]]
    """
    def __init__(self, c1, c2, num_heads=8):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]
        else:
            c_vis = c_ir = c1
        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()
        self.fusion = HAFFormerCore(c2, num_heads)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            a, b = self.proj_vis(x[0]), self.proj_ir(x[1])
        else:
            a = b = self.proj_vis(x)
        return self.fusion([a, b])
