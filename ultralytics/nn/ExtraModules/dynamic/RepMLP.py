"""
RepMLP — 重参数化 MLP 的简化版 (Simplified RepMLP Block)

原理（RepMLP: Re-parameterizing Convolutions into Fully-connected Layers，
      原论文：https://arxiv.org/abs/2105.01883）：
    将特征图分割为 h×w 的 patches，对每个 patch 独立进行全连接变换（Partition Perceptron），
    同时引入全局感知器（Global Perceptron）在 patch 间传播上下文信息。
    - 全局感知器：AvgPool 将 (H,W) 降采样到 (h_parts, w_parts) → 1×1 Conv 跨 patch 混合
    - 分区感知器：将每个 patch 展平 → Conv2d(1×1, groups=c) 相当于 per-patch FC → BN

简化说明：
    完整 RepMLP 包含复杂的重参数化逻辑（训练时的卷积分支在推理时融合进 FC），
    此处提供简化版本，保留核心的 patch 级 FC 处理 + 全局上下文混合机制，
    省略重参数化部分（训练时额外的卷积分支），使模块更轻量易用。

与 RGBT 关联：
    Patch 级 FC 可以在每个局部 patch 上独立学习 RGB-thermal 的模态交互模式。
    全局感知器在不同模态之间传播上下文，增强多模态特征的一致性表达。

用法（在 yaml 中配置）：
    - [-1, 1, RepMLP_Block, [out_channels, 7, 7]]     # 7×7 patches
    - [-1, 1, RepMLP_Block, [out_channels, 4, 4, 2]]   # 4×4 patches, 降维=2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepMLP_Block(nn.Module):
    """RepMLP 简化块 — 基于 patch 的全连接处理 + 全局上下文混合。

    将特征图在空间上分割为 h×w 的 patches，对每个 patch 应用
    per-patch FC（分区感知器），并通过全局感知器在 patch 间传播信息。

    Args:
        c1 (int):            输入通道数
        c2 (int):            输出通道数
        h (int):             patch 高度（空间分割尺寸），默认 7
        w (int):             patch 宽度（空间分割尺寸），默认 7
        fc1_fc2_reduction (int): 全局感知器中间层压缩比，默认 1（不压缩）
        act (bool):          是否使用 GELU 激活，默认 True

    Shape:
        Input:  (B, C_in, H, W) — H, W 需分别能被 h, w 整除
        Output: (B, C_out, H, W)

    处理流程:
        1. 1×1 投影对齐通道（c1≠c2 时）
        2. 全局感知器：AvgPool(h,w) 降采样 → 1×1 Conv 跨 patch 混合 → 广播回每个 patch
        3. 分区感知器：将特征分割为 (h_parts, w_parts) 个 patches → per-patch FC → BN
        4. 重组回原始空间尺寸

    Example:
        >>> block = RepMLP_Block(64, 128, h=8, w=8)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)  # (2, 128, 32, 32)
    """

    def __init__(self, c1: int, c2: int, h: int = 7, w: int = 7,
                 fc1_fc2_reduction: int = 1, act: bool = True):
        super().__init__()
        self.C = c2
        self.h = h  # patch 高度
        self.w = w  # patch 宽度

        # 1×1 通道投影
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 全局感知器 (Global Perceptron)
        #   用 1×1 Conv 替代 FC，空间尺寸无关，天然支持任意输入大小
        internal = max(c2 // fc1_fc2_reduction, 32)
        self.gp = nn.Sequential(
            nn.Conv2d(c2, internal, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal, c2, 1, bias=False),
        )

        # 分区感知器 (Partition Perceptron / FC3)
        #   用 Conv2d(1×1, groups=c2) 实现 per-patch FC
        #   每个通道独立处理其 patch 展平后的向量
        self.fc3 = nn.Conv2d(c2 * h * w, c2 * h * w, 1,
                             groups=c2, bias=False)
        self.fc3_bn = nn.BatchNorm1d(c2 * h * w)

        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 (B, C_in, H, W)

        Returns:
            输出张量 (B, C_out, H, W)
        """
        B, C_in, H, W = x.shape

        # 1. 通道对齐
        x = self.proj(x)  # (B, C, H, W)

        # 自动计算 patch 数量
        h_parts = H // self.h
        w_parts = W // self.w

        # 2. 全局感知器
        #    将每个 patch 池化为一个点 → 1×1 Conv 混合跨 patch 信息 → 广播回 patches
        v = F.avg_pool2d(x, kernel_size=(self.h, self.w),
                          stride=(self.h, self.w))  # (B, C, h_parts, w_parts)
        v = self.gp(v)                              # (B, C, h_parts, w_parts)

        # 广播到每个 patch 内：v: (B, C, h_parts, 1, w_parts, 1)
        v = v.reshape(B, self.C, h_parts, 1, w_parts, 1)

        # 3. 分割为 patches
        #    (B, C, h_parts, h, w_parts, w)
        x = x.reshape(B, self.C, h_parts, self.h, w_parts, self.w)
        x = x + v  # 全局上下文注入

        # 4. 分区感知器 (Partition Perceptron)
        #    重组为 (B, h_parts*w_parts, C*h*w) 对每个 patch 独立做 FC
        x = x.permute(0, 2, 4, 1, 3, 5)            # (B, h_parts, w_parts, C, h, w)
        x = x.reshape(-1, self.C * self.h * self.w, 1, 1)  # (B*h_parts*w_parts, C*h*w, 1, 1)

        # per-patch FC (用 1×1 conv+groups 实现)
        x = self.fc3(x)                             # (B*h_parts*w_parts, C*h*w, 1, 1)
        x = x.reshape(-1, self.C * self.h * self.w)  # (B*h_parts*w_parts, C*h*w)
        x = self.fc3_bn(x)                          # per-patch BN

        # 5. 重组回原空间尺寸
        #    (B, h_parts, w_parts, C, h, w)
        x = x.reshape(B, h_parts, w_parts, self.C, self.h, self.w)
        #    (B, C, h_parts, h, w_parts, w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        #    (B, C, H, W)
        x = x.reshape(B, self.C, H, W)

        x = self.act(x)
        return x
