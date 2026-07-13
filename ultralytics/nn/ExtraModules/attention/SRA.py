"""
SRA: Strip Recurrent Attention — 条带循环注意力

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. 空间注意力优先: 对 H / W 方向分别做条带池化 → 分组卷积(核3,5,7,9) → 门控
  2. 通道自注意力: 下采样后做 QKV 自注意力 → 通道门控
  3. 输出 = 输入 × 空间注意力 × 通道注意力

对 RGBT 价值:
  条带注意力在 H/W 方向上保持全局感知，适合捕捉 RGB-T 水平/垂直方向的模态差异

用法 (YAML — 单输入, 替换 Conv):
  - [-1, 1, SRA, [c2, head_num, window_size]]

参数:
  c1: 输入通道数
  c2: 输出通道数
  head_num: 自注意力头数 (默认 4)
  window_size: 下采样窗口大小 (默认 7, -1=全局)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SRA(nn.Module):
    """Strip Recurrent Attention — 条带循环注意力模块"""

    def __init__(self, c1, c2, head_num=4, window_size=7,
                 group_kernel_sizes=None, qkv_bias=False,
                 down_sample_mode='avg_pool', attn_drop_ratio=0.,
                 gate_layer='sigmoid'):
        super().__init__()

        # 通道对齐投影
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        dim = c2

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes or [3, 5, 7, 9]
        self.window_size = window_size
        self.down_sample_mode = down_sample_mode

        assert dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = dim // 4

        # 四个不同核大小的分组深度可分离卷积 (H/W 方向)
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=self.group_kernel_sizes[0],
                                   padding=self.group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=self.group_kernel_sizes[1],
                                      padding=self.group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=self.group_kernel_sizes[2],
                                      padding=self.group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=self.group_kernel_sizes[3],
                                      padding=self.group_kernel_sizes[3] // 2, groups=group_chans)

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        # 通道自注意力
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        # 下采样策略
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size),
                                              stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size),
                                              stride=window_size)
            else:
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size),
                                              stride=window_size)

    def forward(self, x):
        """输入 (B, C, H, W), 输出 (B, C, H, W)"""
        x = self.proj(x)
        b, c, h_, w_ = x.size()

        # ===== 空间注意力优先 (条带方向) =====
        # (B, C, H) — 沿 W 方向平均池化
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W) — 沿 H 方向平均池化
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # H 方向注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        # W 方向注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # ===== 通道自注意力 =====
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # (B, head_num, head_dim, N)
        q = rearrange(q, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        k = rearrange(k, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        v = rearrange(v, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b hn hd (h w) -> b (hn hd) h w', h=int(h_), w=int(w_))
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x


__all__ = ['SRA']
