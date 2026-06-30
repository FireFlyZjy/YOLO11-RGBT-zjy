"""
PolyNeXt模块集合
来源: PolyNeXt (ICLR2024 多项式神经网络)

提取的模块:
1. PolyAttention - 多项式注意力
2. PolyConv - 多项式卷积
3. C2PSA_PolyAttention - 集成PolyAttention的C2PSA
4. C3k2_PolyConv - 集成PolyConv的C3k2

核心创新:
- PolyAttention: ((q @ k.T) * scale + 1) ** 4 替代softmax，多项式近似更高效
- PolyConv: conv1(x) * conv2(x).flip(dims=[1])，多项式交互增强特征表达

理论支撑:
- 多项式可以近似任意连续函数（Stone-Weierstrass定理）
- 多项式注意力避免了exp操作，计算更友好
- 多项式卷积通过元素级乘法和通道翻转实现高阶特征交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================== PolyAttention ======================

class PolyAttention(nn.Module):
    """
    PolyAttention: 多项式注意力 (ICLR2024)

    核心思想: 使用多项式 ((q @ k.T) * scale + 1) ** 4 替代 softmax
    理论: 多项式可以近似softmax，且无需exp操作，计算更高效

    结构: QKV投影 → 多项式注意力计算 → 输出投影

    用法: [-1, 1, PolyAttention, [c2, num_heads]]
    示例: [-1, 1, PolyAttention, [512, 8]]
    """
    def __init__(self, c1, c2, num_heads=8, head_dim=32, qkv_bias=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_dim = num_heads * head_dim

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # QKV投影 (使用Conv2d适配CNN)
        self.qkv = nn.Conv2d(c2, self.attention_dim * 2, 1, 1, bias=qkv_bias)

        # 深度可分离卷积增强局部性
        self.q_conv = nn.Conv2d(self.attention_dim, self.attention_dim, 5, 1, 2,
                                groups=self.attention_dim, bias=False)
        self.k_conv = nn.Conv2d(self.attention_dim, self.attention_dim, 5, 1, 2,
                                groups=self.attention_dim, bias=False)
        self.v_conv = nn.Conv2d(self.attention_dim, self.attention_dim, 3, 1, 1,
                                groups=self.attention_dim, bias=False)

        # 输出投影
        self.final_conv = nn.Conv2d(self.attention_dim, self.attention_dim, 3, 1, 1,
                                    groups=self.attention_dim, bias=False)
        self.proj = nn.Conv2d(self.attention_dim, c2, 1, 1, bias=False)

        # 可学习缩放参数 (多项式关键)
        # 初始化为 sigmoid(scale) ≈ head_dim^{-0.5}
        self.scale = nn.Parameter(
            torch.tensor([-(head_dim ** -0.5) / ((head_dim ** -0.5) - 1)] * num_heads)
            .log().view(1, -1, 1, 1)
        )

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)
        B, C, H, W = x.shape
        N = H * W

        # QKV投影
        qkv = self.qkv(x).reshape(B, -1, H, W)
        qk, v = qkv.split(self.attention_dim, 1)

        # 局部性增强
        q = self.q_conv(qk).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.k_conv(qk).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.v_conv(v).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 多项式注意力: ((q @ k.T) * scale + 1) ** 4
        # 这是PolyNeXt的核心创新，替代softmax
        attn = ((q @ k.transpose(-2, -1)) * self.scale.sigmoid() + 1) ** 4

        # L1归一化（替代softmax）
        attn = F.normalize(attn, p=1, dim=-1)

        # 注意力加权
        x = attn @ v

        # reshape回空间维度
        x = x.transpose(1, 2).reshape(B, self.attention_dim, H, W)
        x = self.final_conv(x)
        x = self.proj(x)

        return x


# ====================== PolyConv ======================

class PolyConv(nn.Module):
    """
    PolyConv: 多项式卷积

    核心思想: conv1(x) * conv2(x).flip(dims=[1]) 实现多项式交互
    理论: 元素级乘法+通道翻转实现高阶特征交互，增强表达能力

    结构: 1x1 Conv → 3x3 DWConv × 3x3 DWConv(flip) → 残差

    用法: [-1, 1, PolyConv, [c2]]
    示例: [-1, 1, PolyConv, [256]]
    """
    def __init__(self, c1, c2, kernel_size=3, expand_ratio=0.75):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_inner = int(c2 * expand_ratio)

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 1x1投影
        self.proj_in = nn.Conv2d(c2, c_inner, 1, bias=False)

        # 双路深度可分离卷积
        self.conv1 = nn.Conv2d(c_inner, c_inner, kernel_size, 1, kernel_size // 2,
                               groups=c_inner, bias=False)
        self.conv2 = nn.Conv2d(c_inner, c_inner, kernel_size, 1, kernel_size // 2,
                               groups=c_inner, bias=False)

        # 输出投影
        self.proj_out = nn.Sequential(
            nn.Conv2d(c_inner, c_inner, 3, 1, 1, groups=c_inner, bias=False),
            nn.Conv2d(c_inner, c2, 1, bias=False)
        )

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)

        # 1x1投影
        x_inner = self.proj_in(x)

        # 多项式交互: conv1(x) * conv2(x).flip(dims=[1])
        # flip(dims=[1]) 沿通道维度翻转，实现跨通道交互
        out1 = self.conv1(x_inner)
        out2 = self.conv2(x_inner).flip(dims=[1])
        x_poly = out1 * out2

        # 输出投影 + 残差
        x = self.proj_out(x_poly) + x

        return x


# ====================== PolyMLP ======================

class PolyMLP(nn.Module):
    """
    PolyMLP: 多项式MLP

    核心思想: high * low 实现门控交互
    结构: 1x1 Conv → split → high * low → 1x1 Conv → 残差

    用法: [-1, 1, PolyMLP, [c2, expansion]]
    示例: [-1, 1, PolyMLP, [256, 2]]
    """
    def __init__(self, c1, c2, expansion=2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_inner = int(c2 * expansion)
        self.c_half = c_inner // 2

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 扩展投影
        self.proj_expand = nn.Conv2d(c2, c_inner, 1, bias=False)

        # 输出投影
        self.proj_out = nn.Sequential(
            nn.Conv2d(self.c_half, c2, 1, bias=False)
        )

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)

        # 扩展投影
        x_exp = self.proj_expand(x)

        # 分割 + 门控交互
        high, low = x_exp.split(self.c_half, dim=1)
        x_gated = high * low

        # 输出投影 + 残差
        x = self.proj_out(x_gated) + x

        return x


# ====================== C2PSA包装器 ======================

class C2PSA_PolyAttention(nn.Module):
    """
    C2PSA_PolyAttention: 集成PolyAttention的C2PSA模块

    用法: [-1, 2, C2PSA_PolyAttention, [c2, num_heads]]
    示例: [-1, 2, C2PSA_PolyAttention, [1024, 8]]
    """
    def __init__(self, c1, c2, n=1, num_heads=8, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[PolyAttention(self.c, self.c, num_heads) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))


# ====================== C3k2包装器 ======================

class Bottleneck_PolyConv(nn.Module):
    """集成PolyConv的Bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0]//2, bias=False)
        self.cv2 = PolyConv(c_, c2, kernel_size=k[1])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k_PolyConv(nn.Module):
    """集成PolyConv的C3k"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.m = nn.Sequential(*(Bottleneck_PolyConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class C3k2_PolyConv(nn.Module):
    """
    C3k2_PolyConv: 集成PolyConv的C3k2模块

    用法: [-1, 2, C3k2_PolyConv, [c2, c3k]]
    示例: [-1, 2, C3k2_PolyConv, [512, True]]
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        if c3k:
            self.m = nn.ModuleList([C3k_PolyConv(self.c, self.c, 2, shortcut, g) for _ in range(n)])
        else:
            self.m = nn.ModuleList([Bottleneck_PolyConv(self.c, self.c, shortcut, g) for _ in range(n)])

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        for m in self.m:
            b = m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
