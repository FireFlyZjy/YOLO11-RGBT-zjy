"""
CSPOmniKernel — CSP多尺度全方向卷积块

来源: GeoFuse-YOLO (SOEP: Small Object Enhance Pyramid)

机制:
    CSP结构 + OmniKernel(多尺度深度可分离卷积) + FGM(频域门控)
    - OmniKernel: 多个水平/垂直分离的深度卷积(1×5, 5×1等)捕获不同方向特征
    - FGM: 全局池化门控 + 局部深度卷积, 模拟频域高低频分离
    - CSP分路: 一路OmniKernel处理, 一路恒等映射, 拼接后恢复
    - 激进降维(reduction=2/4/8) + 极小扩展比(e=0.0625)保持轻量

对RGBT的价值:
    多尺度多方向卷积能同时捕获RGB的纹理和IR的热分布,
    轻量级设计(激进降维+小e值)适合双分支架构

用法:
    [-1, 1, CSPOmniKernel, [c2]]
"""

import torch
import torch.nn as nn


class FGM(nn.Module):
    """频域门控模块 (Frequency Gating Module)

    通过全局池化+卷积生成通道门控, 与局部深度卷积相乘,
    模拟频域高低频分离的效果。
    """

    def __init__(self, dim):
        super().__init__()
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.local_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=min(dim, 8))
        self.alpha = 0.1
        self.beta = 0.9

    def forward(self, x):
        global_feat = self.global_conv(x)
        local_feat = self.local_conv(x)
        fused = global_feat * local_feat
        return fused * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    """多尺度全方向深度可分离卷积

    使用水平(1×k)和垂直(k×1)分离的深度卷积捕获多方向多尺度特征,
    配合通道注意力自适应加权, 可选FGM频域门控增强。
    """

    def __init__(self, dim, kernel_sizes=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5]

        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.convs.append(nn.Conv2d(dim, dim, (1, k), padding=(0, pad), groups=dim))
            self.convs.append(nn.Conv2d(dim, dim, (k, 1), padding=(pad, 0), groups=dim))

        self.base_conv = nn.Conv2d(dim, dim, 1)

        # 轻量通道注意力 (无BatchNorm, P5安全)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 16, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 16, 8), dim, 1),
            nn.Sigmoid(),
        )

        # 低维时启用FGM
        self.fgm = FGM(dim) if dim <= 128 else None
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = 0.02

    def forward(self, x):
        identity = x
        out = self.base_conv(x)

        # 收集多尺度多方向特征
        features = [out]
        for conv in self.convs:
            features.append(conv(out))
        fused = torch.stack(features, dim=0).mean(dim=0)

        # 通道注意力
        attn = self.attention(fused)
        attended = fused * attn

        # 可选频域门控
        if self.fgm is not None:
            attended = self.fgm(attended)

        out = out + attended
        out = self.act(out)
        out = self.out_conv(out)

        return identity + out * self.res_scale


class CSPOmniKernel(nn.Module):
    """CSPOmniKernel: CSP多尺度全方向卷积块

    机制: CSP分路(OmniKernel+恒等) + 激进降维 + 分组卷积
    对RGBT的价值: 多尺度多方向卷积同时捕获RGB纹理和IR热分布
    用法: [-1, 1, CSPOmniKernel, [c2]]

    Args:
        c1: 输入通道数 (由parse_model自动注入)
        c2: 输出通道数
        e: 内部扩展比例 (默认0.0625, 非常小以保持轻量)
    """

    def __init__(self, c1, c2, e=0.0625):
        super().__init__()
        self.e = e

        # 通道对齐 (c1 != c2时)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        # 激进降维
        if c2 >= 256:
            reduction = 8
        elif c2 >= 128:
            reduction = 4
        else:
            reduction = 2
        self.reduction = reduction

        dim_reduced = c2 // reduction

        # 降维 + 升维
        self.cv1 = nn.Conv2d(c2, dim_reduced, 1)
        self.cv2 = nn.Conv2d(dim_reduced, c2, 1) if reduction > 1 else nn.Identity()

        # 分组卷积进一步降低计算量
        self.group_conv = nn.Conv2d(dim_reduced, dim_reduced, 1, groups=min(8, dim_reduced))

        # OmniKernel维度 (极小)
        ok_dim = max(int(dim_reduced * e), 16)
        self.m = OmniKernel(ok_dim)

    def forward(self, x):
        identity = self.proj(x)

        # 降维 + 分组卷积
        x_reduced = self.cv1(identity)
        x_reduced = self.group_conv(x_reduced)

        # CSP分支分割
        split_dim = max(int(x_reduced.shape[1] * self.e), 16)
        split_dim = min(split_dim, x_reduced.shape[1])
        ok_branch, identity_branch = torch.split(
            x_reduced,
            [split_dim, x_reduced.shape[1] - split_dim],
            dim=1,
        )

        # OmniKernel处理
        processed = self.m(ok_branch)

        # 合并 + 升维
        out = torch.cat((processed, identity_branch), 1)
        out = self.cv2(out)

        # 残差连接 (小scale避免破坏特征)
        return identity + out * 0.05
