"""
CrossAttention_M: 双向深度可分离卷积交叉注意力 (Bidirectional DW-Conv Cross Attention)

来源: LCAFNet (RGB-T dual-stream detection)
核心机制:
  1. 共享 QKV 投影: 1x1 Conv → 3x3 Depthwise Conv (局部特征增强)
     两个模态共用同一组 QKV 投影权重, 实现参数共享
  2. 双向交叉注意力:
     - attn_ir = softmax(rgb_q @ ir_k^T * temperature) @ ir_v
     - attn_rgb = softmax(ir_q @ rgb_k^T * temperature) @ rgb_v
  3. 输出: 各自经过 1x1 Conv 投影

与现有模块的区别:
  - ICAFusion: Transformer Linear QKV, 需要池化到固定网格, O(N^2)
  - CIFusion: 通道注意力, 无空间建模
  - CrossAttention_M: Depthwise Conv QKV, 保留空间分辨率, 局部+全局

用法: [[vis_layer, ir_layer], 1, Att_CrossAttention_M, [c2, num_heads]]
"""

import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention_MCore(nn.Module):
    """双向深度可分离卷积交叉注意力核心"""
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat]
        Returns:
            fused: 融合后的单张量 (B, C, H, W)
        """
        rgb_fea, ir_fea = x[0], x[1]
        b, c, h, w = rgb_fea.shape

        # 共享 QKV 投影 (各模态独立计算, 共享权重)
        rgb_qkv = self.qkv_dwconv(self.qkv(rgb_fea))
        rgb_q, rgb_k, rgb_v = rgb_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        # Reshape 为多头
        rgb_q = rearrange(rgb_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_k = rearrange(rgb_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        rgb_v = rearrange(rgb_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 归一化
        rgb_q = torch.nn.functional.normalize(rgb_q, dim=-1)
        rgb_k = torch.nn.functional.normalize(rgb_k, dim=-1)
        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        # 双向交叉注意力
        attn_ir = (rgb_q @ ir_k.transpose(-2, -1)) * self.temperature
        attn_rgb = (ir_q @ rgb_k.transpose(-2, -1)) * self.temperature

        attn_ir = attn_ir.softmax(dim=-1)
        attn_rgb = attn_rgb.softmax(dim=-1)

        out_ir = attn_ir @ ir_v
        out_rgb = attn_rgb @ rgb_v

        # Reshape 回空间维度
        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_rgb = rearrange(out_rgb, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 投影 + 融合 (两个方向相加)
        out = self.project_out(out_rgb + out_ir)
        return out


class Att_CrossAttention_M(nn.Module):
    """CrossAttention_M YOLO 包装器: 多输入通道投影 + 双向交叉注意力

    用法: [[vis_layer, ir_layer], 1, Att_CrossAttention_M, [c2, num_heads]]
    """
    def __init__(self, c1, c2, num_heads=8):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]
        else:
            c_vis = c_ir = c1
        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()
        self.cross_attn = CrossAttention_MCore(c2, num_heads)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            a, b = self.proj_vis(x[0]), self.proj_ir(x[1])
        else:
            a = b = self.proj_vis(x)
        return self.cross_attn([a, b])
