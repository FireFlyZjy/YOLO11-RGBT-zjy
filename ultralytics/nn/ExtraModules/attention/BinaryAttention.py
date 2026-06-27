"""
BinaryAttention: CVPR2026 1-bit QK-Attention
来源: https://arxiv.org/abs/2509.25164 (The Hong Kong Polytechnic University & OPPO Research Institute)

核心机制: 将query和key二值化(1-bit)，用位运算替代浮点点积，实现极低比特注意力计算
理论支撑: 定理1证明二值注意力保留原始协方差结构
性能: 在A100 GPU上比FlashAttention2快2倍以上

适配YOLO: 提取核心量化逻辑，适配CNN架构的通道注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any


# ====================== 核心量化函数 ======================

class STESign(Function):
    """
    二值化函数 + Straight-Through Estimator (STE) 梯度估计
    前向: sign(x) -> {-1, 1}
    反向: STE近似梯度 (|x| <= 1 时传递梯度)
    """
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # sign函数: 正数->1, 负数->-1, 零->1
        return x.sign().clamp(min=0) * 2 - 1  # 确保零值映射到1

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # STE: 仅在 |x| <= 1 时传递梯度
        grad_input[x.abs() > 1] = 0
        return grad_input


class SymQuantizer(Function):
    """
    对称量化函数
    将浮点数量化到指定比特数的整数范围
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise=False):
        ctx.save_for_backward(input, clip_val)
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            # 逐通道量化
            if input.dim() == 4:
                max_input = (
                    torch.max(torch.abs(input), dim=-2, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            else:
                max_input = (
                    torch.max(torch.abs(input), dim=-2, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        output = torch.round(input * s).div(s + 1e-6)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


# 全局函数引用
binarize = STESign.apply
symquantize = SymQuantizer.apply


def round_ste(z):
    """带STE的舍入函数"""
    zhat = z.round()
    return z + (zhat - z).detach()


# ====================== BinaryAttention 核心模块 ======================

class BinaryAttention(nn.Module):
    """
    BinaryAttention: 1-bit QK注意力机制 (CVPR2026)

    核心思想:
    1. 将query和key二值化为{-1, 1}，用位运算(XNOR+popcount)替代浮点点积
    2. 引入可学习偏置减轻量化信息损失
    3. 注意力系数和value使用8-bit量化实现端到端加速

    适配YOLO:
    - 输入: (B, C, H, W) 特征图
    - 输出: (B, C, H, W) 注意力加权特征图
    - 作为通道注意力使用，可插入C3k2等模块

    用法: [-1, 1, BinaryAttention, [c2]]
    """
    def __init__(self, c1, c2, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.num_heads = num_heads
        self.head_dim = c2 // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1投影处理c1 != c2 (必须在qkv之前)
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # QKV投影 (使用1x1卷积适配CNN，输入已经是c2)
        self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.proj = nn.Conv2d(c2, c2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

        # 可学习偏置 (增强二值注意力的判别能力)
        self.bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # 量化参数
        self.act_clip_val = nn.Parameter(torch.tensor([-2.0, 2.0]))

    def _quantize(self, x):
        """二值化: 缩放二值表示"""
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign

    def _quantize_p(self, x):
        """8-bit量化注意力系数 (无符号)"""
        qmax = 255
        s = 1.0 / qmax
        q = round_ste(x / s).clamp(0, qmax)
        return s * q

    def _quantize_v(self, x, bits=8):
        """8-bit量化value (对称量化)"""
        return symquantize(x, self.act_clip_val, bits, False)

    def forward(self, x):
        # 通道对齐 (如果c1 != c2)
        x = self.channel_align(x)

        B, C, H, W = x.shape
        N = H * W

        # QKV投影
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, num_heads, head_dim, N)

        # 二值化Q和K
        q_bin = self._quantize(q)
        k_bin = self._quantize(k)

        # 计算注意力 (位运算等效)
        # q_bin: (B, num_heads, head_dim, N) -> transpose -> (B, num_heads, N, head_dim)
        # k_bin: (B, num_heads, head_dim, N)
        attn = (q_bin.transpose(-2, -1) @ k_bin) * self.scale  # (B, num_heads, N, N)

        # 添加可学习偏置
        attn = attn + self.bias

        # softmax + dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 量化注意力系数和value
        attn_q = self._quantize_p(attn)
        v_q = self._quantize_v(v, 8)

        # 注意力加权
        # attn_q: (B, num_heads, N, N)
        # v_q: (B, num_heads, head_dim, N) -> transpose -> (B, num_heads, N, head_dim)
        out = (attn_q @ v_q.transpose(-2, -1))  # (B, num_heads, N, head_dim)

        # 重塑回空间维度
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)  # (B, num_heads, head_dim, N) -> (B, C, H, W)

        # 输出投影
        out = self.proj(out)

        return out


class BinaryChannelAttention(nn.Module):
    """
    轻量级二值通道注意力 (适配YOLO CNN架构)

    简化版: 仅对通道维度进行二值化注意力计算
    参数量更少，更适合替换SE/CBAM等轻量注意力

    用法: [-1, 1, BinaryChannelAttention, [c2]]
    """
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1, bias=False),
        )

        # 二值化参数
        self.scale = nn.Parameter(torch.ones(1, c2, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x):
        # 通道统计
        y = self.avg_pool(x)
        y = self.fc(y)

        # 二值化通道权重
        y_bin = binarize(y) * self.scale + self.bias
        y_bin = torch.sigmoid(y_bin)

        return x * y_bin


# ====================== C3k2_BinaryAttention ======================

from ultralytics.nn.modules.block import C3k2, Bottleneck, C3k


class BinaryBottleneck(Bottleneck):
    """
    集成BinaryAttention的Bottleneck

    在标准Bottleneck基础上添加BinaryAttention通道注意力
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.attn = BinaryChannelAttention(c2, c2)

    def forward(self, x):
        return self.attn(x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)))


class C3k2_BinaryAttention(C3k2):
    """
    C3k2_BinaryAttention: 集成BinaryAttention的C3k2模块

    将C3k2中的Bottleneck替换为BinaryBottleneck
    在保持C3k2结构的同时添加二值注意力机制

    用法: [-1, 2, C3k2_BinaryAttention, [c2, c3k]]
    示例: [-1, 2, C3k2_BinaryAttention, [512, True]]
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # 替换Bottleneck为BinaryBottleneck
        self.m = nn.ModuleList(
            C3k_BinaryAttention(self.c, self.c, 2, shortcut, g) if c3k
            else BinaryBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k_BinaryAttention(C3k):
    """
    C3k_BinaryAttention: 集成BinaryAttention的C3k模块

    用法: [-1, 2, C3k_BinaryAttention, [c2]]
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BinaryBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
