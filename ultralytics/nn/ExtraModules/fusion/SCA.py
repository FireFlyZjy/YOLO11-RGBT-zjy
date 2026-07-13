"""
SCA: Spatial-Channel Attention Fusion — 空间-通道注意力跨模态融合

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. RGB × IR 元素乘 → SRA 空间注意力 → attention map
  2. RGB + IR 元素和 → 卷积对齐
  3. 和特征 × attention map → SRA 通道自注意力 → 融合输出

对 RGBT 价值:
  乘积路径捕捉模态间共现响应，和路径保留各自独立特征，
  SRA 条带注意力在 H/W 方向保持全局感知，适合 RGB-T 检测

用法 (YAML — 多输入, from为列表 [vis, ir], 替换 Concat):
  - [[vis_layer, ir_layer], 1, SCA, [c2, head_num, window_size]]

参数:
  c1: [c_vis, c_ir] 或 int (单输入时)
  c2: 输出通道数
  head_num: SRA 注意力头数 (默认 4)
  window_size: SRA 下采样窗口 (默认 7)
"""

import torch
import torch.nn as nn
from ultralytics.nn.ExtraModules.attention.SRA import SRA


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


class SCA(nn.Module):
    """Spatial-Channel Attention — 跨模态融合模块"""

    def __init__(self, c1, c2, head_num=4, window_size=7):
        super().__init__()

        # 多输入处理: 分别投影两个模态

        self.proj_vis = nn.Conv2d(c1[0], c2, 1) if c1[0] != c2 else nn.Identity()
        self.proj_ir  = nn.Conv2d(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()

        self.sra = SRA(c2, c2, head_num=head_num, window_size=window_size)
        self.conv2 = BasicConv2d(c2, c2, kernel_size=3, padding=1)

    def forward(self, x):
        """x: [vis, ir] 列表, 或单张量"""
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            vis = ir = x

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        # 乘积路径: 捕捉模态间共现
        multiplication = vis * ir
        # 求和路径: 保留各自独立特征
        summation = self.conv2(vis + ir)

        # SRA 空间注意力
        sa = self.sra(multiplication)
        # 用 attention 加权和路径
        summation_sa = summation * sa

        # 最终 SRA 通道自注意力
        sc_feat = self.sra(summation_sa)

        return sc_feat


__all__ = ['SCA']
