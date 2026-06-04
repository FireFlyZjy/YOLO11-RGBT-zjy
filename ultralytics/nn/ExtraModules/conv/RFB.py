# coding=utf-8
"""
BasicRFB / BasicRFB_small — Receptive Field Block for YOLO

论文: Receptive Field Block Net for Accurate and Fast Object Detection (ECCV 2018)
      https://arxiv.org/abs/1711.07767

机制:
    RFB (Receptive Field Block) 通过多分支空洞卷积模拟人类视觉皮层中
    不同尺寸的感受野 (Receptive Field). 核心思想:
    - 多个并行分支, 每个分支包含 1x1 降维 + 3x3 空洞卷积
    - 各分支使用不同 dilation rate 以捕获多尺度特征
    - 多分支输出拼接 + 1x1 投影 + Shortcut 连接
    - 有效增强小目标和多尺度目标的特征表达

    BasicRFB:   3 分支, dilation = [1, 2, 3], 通道压缩比 = 1/8
    BasicRFB_small: 4 分支, 额外非对称卷积 (1x3, 3x1), dilation = [1, 3, 3, 5], 通道压缩比 = 1/4

RGBT 价值:
    多模态 (RGB + Thermal) 检测中, 多分支空洞卷积可以同时捕获不同模态
    下的多尺度特征 —— 例如 RGB 中的纹理细节 (小感受野) 和 Thermal 中的
    热辐射轮廓 (大感受野), 有效提升小目标检测和跨模态匹配能力.

用法 (YAML):
    backbone:
      - [-1, 1, BasicRFB, [256, 3, 1]]       # 3-branch RFB
      - [-1, 1, BasicRFB_small, [128, 3, 1]]  # 4-branch RFB
"""

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """自动计算填充使得输出尺寸与输入相同 (YOLO 标准 autopad)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """RFB 内部使用的 Conv-BN-SiLU 基础块, 与 YOLO Conv 操作一致."""

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), d, g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-5, momentum=0.01, affine=True)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """不使用 BN 的前向 (融合后使用)."""
        return self.act(self.conv(x))


class BasicRFB(nn.Module):
    """
    BasicRFB — 3 分支感受野增强模块

    包含 3 个并行分支, 每个分支使用 1x1 降维 + 3x3 空洞卷积:
        - branch0: dilation = 1  (标准 3x3)
        - branch1: dilation = 2
        - branch2: dilation = 3
    各分支输出拼接后经 1x1 投影, 与 Shortcut 相加后激活.

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 保留参数, 内部使用固定核大小
        s (int): 步长, 默认 1
        p (int | None): 保留参数, 填充自动计算
        act (bool): 是否使用激活, 默认 True

    Shape:
        - 输入: (N, C_in, H, W)
        - 输出: (N, C_out, H_out, W_out)
          其中 H_out = H / s, W_out = W / s
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.scale = 0.1           # shortcut 缩放因子
        self.out_channels = c2
        inter_planes = c1 // 8     # 中间通道压缩比
        visual = 1                 # 基础空洞率

        # ---- branch0: dilation=1 ----
        self.branch0 = nn.Sequential(
            Conv(c1, 2 * inter_planes, k=1, s=s),                    # 1x1 降维 + 下采样
            Conv(2 * inter_planes, 2 * inter_planes, k=3, d=visual,
                 p=visual, act=False),                               # 3x3 dilation=1, 无激活
        )

        # ---- branch1: dilation=2 ----
        self.branch1 = nn.Sequential(
            Conv(c1, inter_planes, k=1, s=1),                        # 1x1 降维
            Conv(inter_planes, 2 * inter_planes, k=3, s=s, p=1),     # 3x3 下采样
            Conv(2 * inter_planes, 2 * inter_planes, k=3,
                 d=visual + 1, p=visual + 1, act=False),             # 3x3 dilation=2, 无激活
        )

        # ---- branch2: dilation=3 ----
        self.branch2 = nn.Sequential(
            Conv(c1, inter_planes, k=1, s=1),                        # 1x1 降维
            Conv(inter_planes, (inter_planes // 2) * 3, k=3, p=1),   # 3x3 扩展
            Conv((inter_planes // 2) * 3, 2 * inter_planes, k=3, s=s, p=1),  # 3x3 下采样
            Conv(2 * inter_planes, 2 * inter_planes, k=3,
                 d=2 * visual + 1, p=2 * visual + 1, act=False),     # 3x3 dilation=3, 无激活
        )

        # 拼接所有分支 (6 * inter_planes) -> 投影到 c2
        self.ConvLinear = Conv(6 * inter_planes, c2, k=1, act=False)

        # Shortcut 连接
        self.shortcut = Conv(c1, c2, k=1, s=s, act=False)

        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), dim=1)
        out = self.ConvLinear(out)

        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.act(out)

        return out


class BasicRFB_small(nn.Module):
    """
    BasicRFB_small — 4 分支感受野增强模块 (细粒度版本)

    包含 4 个并行分支, 在 BasicRFB 基础上增加非对称卷积分支:
        - branch0: dilation=1 标准 3x3
        - branch1: 3x1 + 3x3 dilation=3 (水平条纹)
        - branch2: 1x3 + 3x3 dilation=3 (垂直条纹)
        - branch3: 1x3 -> 3x1 + 3x3 dilation=5 (大感受野)
    通道压缩比为 1/4, 更细粒度.

    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 保留参数, 内部使用固定核大小
        s (int): 步长, 默认 1
        p (int | None): 保留参数, 填充自动计算
        act (bool): 是否使用激活, 默认 True

    Shape:
        - 输入: (N, C_in, H, W)
        - 输出: (N, C_out, H_out, W_out)
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.scale = 0.1
        self.out_channels = c2
        inter_planes = c1 // 4      # 中间通道压缩比 (比 BasicRFB 更大)

        # ---- branch0: 标准 3x3 ----
        self.branch0 = nn.Sequential(
            Conv(c1, inter_planes, k=1, s=1),
            Conv(inter_planes, inter_planes, k=3, p=1, act=False),
        )

        # ---- branch1: 3x1 -> 3x3 dilation=3 (水平) ----
        self.branch1 = nn.Sequential(
            Conv(c1, inter_planes, k=1, s=1),
            Conv(inter_planes, inter_planes, k=(3, 1), p=(1, 0)),
            Conv(inter_planes, inter_planes, k=3, p=3, d=3, act=False),
        )

        # ---- branch2: 1x3 -> 3x3 dilation=3 (垂直) ----
        self.branch2 = nn.Sequential(
            Conv(c1, inter_planes, k=1, s=1),
            Conv(inter_planes, inter_planes, k=(1, 3), s=s, p=(0, 1)),
            Conv(inter_planes, inter_planes, k=3, p=3, d=3, act=False),
        )

        # ---- branch3: 1x3 -> 3x1 -> 3x3 dilation=5 ----
        self.branch3 = nn.Sequential(
            Conv(c1, inter_planes // 2, k=1, s=1),
            Conv(inter_planes // 2, (inter_planes // 4) * 3, k=(1, 3), p=(0, 1)),
            Conv((inter_planes // 4) * 3, inter_planes, k=(3, 1), s=s, p=(1, 0)),
            Conv(inter_planes, inter_planes, k=3, p=5, d=5, act=False),
        )

        # 拼接所有分支 (4 * inter_planes) -> 投影到 c2
        self.ConvLinear = Conv(4 * inter_planes, c2, k=1, act=False)

        # Shortcut 连接
        self.shortcut = Conv(c1, c2, k=1, s=s, act=False)

        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.ConvLinear(out)

        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.act(out)

        return out
