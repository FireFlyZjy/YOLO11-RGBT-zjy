"""
GFM: Global Fusion Module — 全局融合模块

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. Concat (RGB + T) 或 Concat (RGB + T + RGB×T) — expend_ratio 控制
  2. DWPWConv (Depthwise + Pointwise) 降维
  3. SAttention (简化的 Q-K-V 自注意力) 全局特征交互
  4. 残差连接 + GELU

对 RGBT 价值:
  全局自注意力建模跨模态长程依赖，DWPWConv 保持轻量

用法 (YAML — 多输入, from为列表 [vis, ir], 替换 Concat):
  - [[vis_layer, ir_layer], 1, GFM, [c2, expend_ratio]]
  expend_ratio=2: concat(RGB, IR), 输出通道 c2
  expend_ratio=3: concat(RGB, IR, RGB×IR), 输出通道 c2

参数:
  c1: [c_vis, c_ir] 或 int
  c2: 输出通道数
  expend_ratio: 2 或 3 (默认 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAttention(nn.Module):
    """
    简化版自注意力: QKV 自注意力 + 局部深度可分离卷积增强
    输入 (B, C, H, W), 输出 (B, C, H, W)
    """
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.sa_num_heads = sa_num_heads
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        head_dim = dim // sa_num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)

        q = self.q(x_flat).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_flat).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 局部深度卷积增强
        x_local = self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W))
        x_local = x_local.view(B, C, N).transpose(1, 2)

        x_out = self.proj(x_attn + x_local)
        x_out = self.proj_drop(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)  # 回到 (B, C, H, W)
        return x_out


class DWPWConv(nn.Module):
    """Depthwise + Pointwise 卷积"""
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class GFM(nn.Module):
    """Global Fusion Module — 全局融合模块"""

    def __init__(self, c1, c2, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        # 多输入投影

        self.proj_vis = nn.Conv2d(c1[0], c2, 1) if c1[0] != c2 else nn.Identity()
        self.proj_ir  = nn.Conv2d(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()

        # 融合路径
        cat_dim = c2 * expend_ratio
        self.dw_pw = DWPWConv(cat_dim, c2)
        self.sa = SAttention(dim=c2)
        self.act = nn.GELU()

    def forward(self, x):
        """x: [vis, ir] 列表, 或单张量"""
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            vis = ir = x

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        B, C, H, W = vis.shape

        if self.expend_ratio == 2:
            cat = torch.cat((vis, ir), dim=1)
        else:
            multi = vis * ir
            cat = torch.cat((vis, ir, multi), dim=1)

        x_rc = self.dw_pw(cat)  # 降维到 c2
        x_ = self.sa(x_rc)      # 自注意力
        x_ = x_ + x_rc          # 残差
        return self.act(x_)


__all__ = ['GFM']
