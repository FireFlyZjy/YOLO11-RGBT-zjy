"""
ProgressiveAgg: 渐进式特征聚合 (ProGRess 思想 + HVPNet 密集融合)

论文灵感:
  - ProGRess (rmae-progress): Progressive Leapwise Fusion (PLF) +
    Lightweight Channel Attention Residual (LCAR) + Bottleneck Fusion
  - HVPNet: 多尺度密集上采样 + 逐元素乘聚合

核心机制:
  1. PLF: 从深到浅渐进融合 (P5→P4→P3), 每个阶段将深层上采样后
     与当前层特征拼接 → 1x1 融合
  2. LCAR: 每层融合后经轻量通道注意力 + 可选残差连接
  3. 所有层上采样到 P3 分辨率 → 拼接 → Bottleneck 融合 → 输出

对 RGBT 价值:
  渐进式融合保留多尺度信息, 轻量通道注意力在不增加大量参数下
  增强跨模态特征的通道选择性

用法 (YAML — 单输入多尺度, 替换 Concat/SPPF 等 neck, 3尺度):
  - [-1, 1, ProgressiveAgg, [c2]]

参数:
  c1: 输入通道数 (通常来自 P3/P4/P5)
  c2: 内部融合通道数, 输出 = c2
  use_lcar: 是否使用轻量通道注意力残差 (默认 True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightChannelAttention(nn.Module):
    """轻量通道注意力 (LCA) — 原 ProGRess LCAR 模块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class FusionBlock(nn.Module):
    """渐进融合块: 深层特征上采样 → 与当前层拼接 → 1x1 融合"""
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch1 + in_ch2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x1, x2):
        # x1: 当前层, x2: 深层 (需要上采样)
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x1, x2], dim=1))


class ProgressiveAgg(nn.Module):
    """ProgressiveAgg — 渐进式多尺度特征聚合 (Neck)"""

    def __init__(self, c1, c2, use_lcar=True):
        super().__init__()
        # c1 可能是 list (从多个层来的通道)
        if isinstance(c1, (list, tuple)):
            chs = list(c1)
        else:
            chs = [c1, c1, c1]  # 默认 3 尺度

        n_scales = len(chs)
        self.n_scales = n_scales
        self.use_lcar = use_lcar

        # 各尺度投影对齐
        self.proj = nn.ModuleList([
            nn.Conv2d(ch, c2, 1) if ch != c2 else nn.Identity()
            for ch in chs
        ])

        # 渐进融合块: 从深到浅
        self.fusion_blocks = nn.ModuleList()
        for i in range(n_scales - 1):
            self.fusion_blocks.append(FusionBlock(c2, c2, c2))

        # LCAR: 轻量通道注意力
        if use_lcar:
            self.gates = nn.ModuleList([
                LightweightChannelAttention(c2) for _ in range(n_scales)
            ])

        # Bottleneck 融合: 所有尺度上采样后拼接
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2 * n_scales, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: 多尺度特征列表 [P3, P4, P5] 或 [P3, P4, P5, P6] ...
           按分辨率从大到小排列
        """
        if isinstance(x, torch.Tensor):
            x = [x]

        # 1. 投影所有尺度到统一通道 c2
        feats = [proj(f) for proj, f in zip(self.proj, x)]

        # 2. 渐进融合 (从深到浅)
        n = len(feats)
        fused = [None] * n
        # 最深层
        fused[-1] = feats[-1]

        for i in range(n - 2, -1, -1):
            fused[i] = self.fusion_blocks[i](feats[i], fused[i + 1])

        # 3. LCAR (轻量通道注意力 + 可选残差)
        if self.use_lcar:
            for i in range(n):
                residual = fused[i]
                fused[i] = self.gates[i](fused[i])
                fused[i] = fused[i] + residual  # 残差

        # 4. 上采样到最大分辨率 + Bottleneck 融合
        target_hw = fused[0].shape[2:]
        upsampled = [fused[0]]
        for i in range(1, n):
            if fused[i].shape[2:] != target_hw:
                upsampled.append(F.interpolate(fused[i], size=target_hw,
                                               mode='bilinear', align_corners=False))
            else:
                upsampled.append(fused[i])

        out = self.bottleneck(torch.cat(upsampled, dim=1))
        return out


__all__ = ['ProgressiveAgg']
