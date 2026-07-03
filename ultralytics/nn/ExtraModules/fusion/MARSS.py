"""
MARSS: Modular Attention and State Space Modules
=================================================
论文: MARSS: Radar Semantic Segmentation via Modular Attention and State Space Models
来源: CVPR 2026
论文链接: https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.pdf

核心模块:
    1. REM (Radar Enhancement Module) - 小波增强模块，轴向卷积模拟四子带分解
    2. RADE (Radar-Aware Denoising Encoder) - 雷达感知去噪编码器，CBAM式通道+空间注意力
    3. RFAF (Radar Feature Adaptive Fusion) - 雷达特征自适应融合，多尺度+区域级注意力
    4. RADM (Radar State Space Decoder) - 状态空间解码模块，轴向自注意力+DWConv

对 RGBT 的价值:
    - REM: 轴向卷积(1×3, 3×1)天然适合处理红外图像的方向性特征
    - RADE: 通道+空间注意力级联去噪，适合低信噪比的红外图像
    - RFAF: 十字形感受野+多尺度空洞卷积，适合跨模态特征融合
    - RADM: 轴向自注意力显式建模不同方向的长程依赖

与已有模块的区别:
    - CBAM: 标准通道+空间注意力，无去噪设计
    - WDAM: 小波域窗口注意力，无轴向卷积
    - C2Former: 可变形交叉注意力，无区域级注意力
    - MARSS: 专为雷达/红外图像设计，轴向卷积+区域注意力+轴向自注意力

用法:
    REM: [-1, 1, REM, [c2]]
    RADE: [-1, 1, RADE, [c2, reduction]]
    RFAF_Fusion: [[vis, ir], 1, RFAF_Fusion, [c2]]
    RADM: [-1, 1, RADM, [c2]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 模块1: REM - 小波增强模块 (Radar Enhancement Module)
# ============================================================

class REM(nn.Module):
    """
    REM: Radar Enhancement Module - 基于小波变换的四子带特征增强

    机制: 用轴向卷积模拟 DWT 的四子带分解
        - LL子带: 1×1卷积 (低频全局信息)
        - LH子带: 1×3 → 3×1 → 3×3 (水平高频)
        - HL子带: 3×1 → 1×3 → 3×3 (垂直高频)
        - HH子带: 3×3 → 3×3 (对角高频)

    对 RGBT 的价值: 轴向卷积天然适合处理红外图像的方向性特征
    用法: [-1, 1, REM, [c2]]

    参数:
        c1: 输入通道数
        c2: 输出通道数 (必须能被4整除)
    """
    def __init__(self, c1, c2):
        super().__init__()
        assert c2 % 4 == 0, f"REM requires c2 % 4 == 0, got c2={c2}"
        mid_c = c2 // 4

        # 通道对齐
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        # LL子带: 低频全局信息 (1×1卷积)
        self.branch_LL = nn.Sequential(
            nn.Conv2d(c2, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )

        # LH子带: 水平高频 (先1×3垂直轴向卷积 → 再3×1水平轴向卷积 → 3×3聚合)
        self.branch_LH = nn.Sequential(
            nn.Conv2d(c2, mid_c, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )

        # HL子带: 垂直高频 (与LH反向: 先3×1 → 再1×3 → 3×3)
        self.branch_HL = nn.Sequential(
            nn.Conv2d(c2, mid_c, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )

        # HH子带: 对角高频 (3×3卷积)
        self.branch_HH = nn.Sequential(
            nn.Conv2d(c2, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU()
        )

        # 最终融合
        self.fuse = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.proj(x)
        ll = self.branch_LL(x)
        lh = self.branch_LH(x)
        hl = self.branch_HL(x)
        hh = self.branch_HH(x)
        out = torch.cat([ll, lh, hl, hh], dim=1)
        return self.bn(self.fuse(out)) + x


# ============================================================
# 模块2: RADE - 雷达感知去噪编码器
# ============================================================

class ChannelAttention_RAD(nn.Module):
    """
    RADE的通道注意力 - 论文公式(1)
    Xc = sigmoid(f_MLP(GAP(X))) ⊙ X

    关键改进点(相比标准SE):
    - 使用AdaptiveAvgPool提取全局频谱统计
    - MLP使用1×1卷积实现，更轻量
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.gap(x))
        return x * weight


class SpatialAttention_RAD(nn.Module):
    """
    RADE的空间注意力 - 论文公式(2)
    Xs = sigmoid(Conv7x7([AvgPool; MaxPool](Xc))) ⊙ Xc

    7×7大核卷积捕获雷达图中的空间结构(目标边界、微动条纹)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * mask


class RADE(nn.Module):
    """
    RADE: Radar-Aware Denoising Encoder - 雷达感知去噪编码器

    机制: CBAM式级联注意力
        1. 1×1卷积预处理
        2. 通道注意力: GAP → MLP → Sigmoid (抑制噪声通道)
        3. 空间注意力: AvgPool+MaxPool → 7×7 Conv → Sigmoid (增强目标区域)
        4. 残差连接

    对 RGBT 的价值: 通道+空间注意力级联去噪，适合低信噪比的红外图像
    用法: [-1, 1, RADE, [c2, reduction]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        reduction: 通道注意力的缩减比例 (默认 4)
    """
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        # 通道对齐
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        self.channel_attn = ChannelAttention_RAD(c2, reduction)
        self.spatial_attn = SpatialAttention_RAD()
        self.post_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        x = self.proj(x)
        identity = x
        out = self.pre_conv(x)
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        out = self.post_conv(out)
        return out + identity


# ============================================================
# 模块3: RFAF - 雷达特征自适应融合
# ============================================================

class MultiScaleFusion(nn.Module):
    """
    RFAF Stage I: 多尺度融合注意力
    - 通道注意力分支: GAP → 1×1 bottleneck → ReLU → 1×1 → Sigmoid
    - 空间注意力分支: 7×7 Conv → Sigmoid
    - 两路相加后1×1融合
    """
    def __init__(self, channels):
        super().__init__()
        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.sp_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        ch_weight = self.ch_attn(x)
        x_ch = x * ch_weight
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_weight = self.sp_attn(torch.cat([avg_out, max_out], dim=1))
        x_sp = x * sp_weight
        return self.fuse(x_ch + x_sp)


class RegionAttention(nn.Module):
    """
    RFAF Stage II: 区域级注意力 (论文公式3)

    关键创新: 十字形感受野，分别处理水平和垂直邻域
    - 水平分支: 1×k 卷积 → 捕获水平方向模式
    - 垂直分支: k×1 卷积 → 捕获垂直方向模式
    - 多尺度空洞卷积分支: 不同dilation rate的3×3卷积

    迁移到双模态: 十字形感受野同样适用于RGB-IR融合，
    因为红外特征的方向性在融合后仍然保留
    """
    def __init__(self, channels):
        super().__init__()
        mid_c = channels // 2

        # 区域注意力分支 (十字形)
        self.region_branch = nn.Sequential(
            nn.Conv2d(channels, mid_c, kernel_size=(1, 7), padding=(0, 3), groups=mid_c),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=(7, 1), padding=(3, 0), groups=mid_c),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
        )

        # 多尺度空洞卷积分支
        self.ms_branch = nn.Sequential(
            nn.Conv2d(channels, mid_c, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
        )

        # 融合投影回原通道数
        self.project = nn.Conv2d(mid_c * 2, channels, 1)

    def forward(self, x):
        region = self.region_branch(x)
        ms = self.ms_branch(x)
        out = torch.cat([region, ms], dim=1)
        return self.project(out)


class RFAF_Fusion(nn.Module):
    """
    RFAF_Fusion: 雷达特征自适应融合 (跨模态融合版本)

    机制: 多尺度融合注意力 + 区域级注意力 + 残差连接
        1. Concat两模态特征 → 1×1压缩
        2. 多尺度融合注意力 → 学习模态重要性
        3. 区域级注意力(十字形感受野) → 在不同空间区域自适应选择主导模态
        4. 残差连接 → 保留原始信息

    对 RGBT 的价值: 十字形感受野+多尺度空洞卷积，适合跨模态特征融合
    用法: [[vis, ir], 1, RFAF_Fusion, [c2]]

    参数:
        c1: 输入通道数列表 (由 parse_model 自动注入)
        c2: 输出通道数
    """
    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]
        else:
            c_vis = c_ir = c1

        # 通道对齐
        self.proj_vis = nn.Conv2d(c_vis, c2, 1) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1) if c_ir != c2 else nn.Identity()

        # 模态压缩: 将2C通道压缩回C
        self.compress = nn.Conv2d(c2 * 2, c2, 1, bias=False)

        # Stage I: 多尺度融合注意力
        self.multi_scale_fusion = MultiScaleFusion(c2)

        # Stage II: 区域级注意力
        self.region_attention = RegionAttention(c2)

        # 最终融合
        self.final_fuse = nn.Conv2d(c2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis_x, ir_x = x[0], x[1]
        else:
            vis_x = ir_x = x

        # 通道对齐
        vis_x = self.proj_vis(vis_x)
        ir_x = self.proj_ir(ir_x)

        # 1. 拼接 + 压缩
        x_fused = self.compress(torch.cat([vis_x, ir_x], dim=1))

        # 2. 多尺度融合注意力
        x_ms = self.multi_scale_fusion(x_fused)

        # 3. 区域级注意力
        x_region = self.region_attention(x_ms)

        # 4. 融合 + 残差
        out = self.final_fuse(x_ms + x_region)
        out = self.bn(out) + x_fused

        return out


# ============================================================
# 模块4: RADM - 状态空间解码模块
# ============================================================

class AxialSelfAttention(nn.Module):
    """
    轴向自注意力 - 将2D注意力分解为沿单一轴的1D注意力

    论文核心思想: 雷达图的各向异性使得range和Doppler轴
    上的特征模式截然不同，分别建模更高效

    迁移到双模态检测:
    - 在YOLO Neck中对H轴和W轴分别做注意力
    - 比标准2D Self-Attention计算量小得多
    """
    def __init__(self, channels, axis='H'):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        if self.axis == 'H':
            x_perm = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        else:
            x_perm = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        x_norm = self.norm(x_perm)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = self.proj(out)

        # 残差
        out = out + x_perm

        # 恢复形状
        if self.axis == 'H':
            out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)
        else:
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return out


class RADM(nn.Module):
    """
    RADM: Radar State Space Decoder Module - 状态空间解码模块

    机制: 三分支并行
        - H-axis分支: 沿高度轴做轴向自注意力
        - W-axis分支: 沿宽度轴做轴向自注意力
        - Global分支: Depth-wise Conv + Self-Attention
        最后三分支拼接 + 融合 + 残差

    对 RGBT 的价值: 轴向自注意力显式建模不同方向的长程依赖
    用法: [-1, 1, RADM, [c2]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
    """
    def __init__(self, c1, c2):
        super().__init__()
        # 通道对齐
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        c1_quarter = c2 // 4
        c1_half = c2 // 2

        # H-axis分支
        self.h_proj = nn.Conv2d(c2, c1_quarter, 1)
        self.h_local = nn.Sequential(
            nn.Conv2d(c1_quarter, c1_quarter, 3, padding=1, groups=c1_quarter),
            nn.BatchNorm2d(c1_quarter),
            nn.GELU(),
            nn.Conv2d(c1_quarter, c1_quarter, 1),
            nn.BatchNorm2d(c1_quarter),
        )
        self.h_axial = AxialSelfAttention(c1_quarter, axis='H')

        # W-axis分支
        self.w_proj = nn.Conv2d(c2, c1_half, 1)
        self.w_local = nn.Sequential(
            nn.Conv2d(c1_half, c1_half, 3, padding=1, groups=c1_half),
            nn.BatchNorm2d(c1_half),
            nn.GELU(),
            nn.Conv2d(c1_half, c1_half, 1),
            nn.BatchNorm2d(c1_half),
        )
        self.w_axial = AxialSelfAttention(c1_half, axis='W')

        # Global分支
        self.g_proj = nn.Conv2d(c2, c1_quarter, 1)
        self.g_dwconv = nn.Conv2d(c1_quarter, c1_quarter, 3, padding=1, groups=c1_quarter)
        self.g_attn = nn.Sequential(
            nn.LayerNorm(c1_quarter),
            nn.Linear(c1_quarter, c1_quarter),
        )

        # 三分支融合
        self.fuse = nn.Sequential(
            nn.Conv2d(c1_quarter + c1_half + c1_quarter, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        x = self.proj(x)
        identity = x
        B, C, H, W = x.shape

        # H-axis分支
        h = self.h_proj(x)
        h = self.h_local(h) + h
        h = self.h_axial(h)

        # W-axis分支
        w = self.w_proj(x)
        w = self.w_local(w) + w
        w = self.w_axial(w)

        # Global分支
        g = self.g_proj(x)
        g = self.g_dwconv(g) + g
        g_flat = g.permute(0, 2, 3, 1)
        g_flat = self.g_attn(g_flat) + g_flat
        g = g_flat.permute(0, 3, 1, 2)

        # 三分支拼接 + 融合 + 残差
        out = torch.cat([h, w, g], dim=1)
        out = self.fuse(out) + identity

        return out


# ============================================================
# C2PSA 包装器
# ============================================================

class C2PSA_RADM(nn.Module):
    """
    C2PSA_RADM: 集成 RADM 的 C2PSA 模块

    机制: Split-Concat 结构 + RADM 处理一个分支
    用法: [-1, 1, C2PSA_RADM, [c2]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        n: RADM 重复次数 (默认 1)
        e: 扩展比例 (默认 0.5)
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, f"C2PSA_RADM requires c1 == c2, got c1={c1}, c2={c2}"
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[RADM(self.c, self.c) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
