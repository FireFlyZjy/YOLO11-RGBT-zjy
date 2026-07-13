"""
EDS: Edge-aware Dynamic Sampling Fusion — 边缘感知动态采样融合

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. RGB × IR 乘积路径 + RGB + IR 求和路径 → 双路径融合
  2. 4 分支空洞卷积 (dilation=1,3,5,7) 捕获多尺度边缘/轮廓
  3. 拼接 + 融合卷积输出

对 RGBT 价值:
  不同空洞率的卷积分支可以分别捕获 RGB 纹理边缘(小空洞)和
  Thermal 热辐射轮廓(大空洞), 实现跨模态边缘互补

用法 (YAML — 多输入, from为列表 [vis, ir], 替换 Concat):
  - [[vis_layer, ir_layer], 1, EDS, [c2]]

参数:
  c1: [c_vis, c_ir] 或 int
  c2: 输出通道数
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    """Conv + BN + LeakyReLU 基础块"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EDS(nn.Module):
    """Edge-aware Dynamic Sampling — 边缘感知动态采样融合"""

    def __init__(self, c1, c2):
        super().__init__()

        # 多输入投影

        self.proj_vis = nn.Conv2d(c1[0], c2, 1) if c1[0] != c2 else nn.Identity()
        self.proj_ir  = nn.Conv2d(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()

        # 双路径融合
        self.conv1 = BasicConv2d(c2, c2, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(c2, c2, kernel_size=3, padding=1)

        # 4 分支空洞卷积 (dilation=1,3,5,7)
        d = c2 // 4  # 每个分支通道数
        self.dconv1 = BasicConv2d(c2, d, kernel_size=3, padding=1, dilation=1)
        self.dconv2 = BasicConv2d(c2, d, kernel_size=3, padding=3, dilation=3)
        self.dconv3 = BasicConv2d(c2, d, kernel_size=3, padding=5, dilation=5)
        self.dconv4 = BasicConv2d(c2, d, kernel_size=3, padding=7, dilation=7)

        self.fuse_dconv = nn.Conv2d(c2, c2, kernel_size=3, padding=1)

    def forward(self, x):
        """x: [vis, ir] 列表, 或单张量"""
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            vis = ir = x

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        # 乘积 + 求和双路径
        multiplication = self.conv1(vis * ir)
        summation = self.conv2(vis + ir)
        fusion = summation + multiplication

        # 4 分支空洞卷积
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)

        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        return out


__all__ = ['EDS']
