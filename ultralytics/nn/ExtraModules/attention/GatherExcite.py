"""
GatherExcite: Gather-Excite 注意力模块

来源: timm (Gather-Excite Networks, NeurIPS 2018)
参考: Fracture_Detection_Improved_YOLOv8
说明: timm 1.0.26 的 GatherExcite 有 ConvMlp 参数冲突 bug, 此处用纯 PyTorch 重写

核心机制:
  1. Gather (聚合): 通过 AvgPool 聚合空间信息
     - extent=0: 全局 GAP (整张特征图)
     - extent>0: 局部 AvgPool (kernel=2*extent-1, stride=extent), 保留空间分辨率
  2. Excite (激发): MLP → Sigmoid 生成通道门控
  3. 输出: x * gate

与 SE 的区别:
  - SE 只用全局 GAP, 丢失所有空间信息
  - GatherExcite 可用 extent>0 做局部聚合, 保留空间上下文
  - extent=0 时退化为 SE 的变体

用法: [-1, 1, Att_GatherExcite, [c2, extent, reduction]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatherExciteCore(nn.Module):
    """Gather-Excite 核心: 局部/全局聚合 + MLP 通道门控"""
    def __init__(self, channels, extent=0, reduction=16, use_mlp=True, add_maxpool=False):
        super().__init__()
        self.extent = extent
        self.add_maxpool = add_maxpool

        if extent == 0:
            self.gk = 0
            self.gs = 0
        else:
            assert extent % 2 == 0, f"extent must be even, got {extent}"
            self.gk = extent * 2 - 1
            self.gs = extent

        if use_mlp:
            rd_channels = max(channels // reduction, 4)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, rd_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(rd_channels, channels, 1, bias=False),
            )
        else:
            self.mlp = nn.Identity()

        self.gate = nn.Sigmoid()

    def forward(self, x):
        size = x.shape[-2:]
        if self.extent == 0:
            x_ge = x.mean(dim=(2, 3), keepdim=True)
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
        else:
            x_ge = F.avg_pool2d(x, kernel_size=self.gk, stride=self.gs,
                                padding=self.gk // 2, count_include_pad=False)
            if self.add_maxpool:
                x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


class Att_GatherExcite(nn.Module):
    """GatherExcite YOLO 包装器

    用法: [-1, 1, Att_GatherExcite, [c2, extent, reduction]]
    参数:
        c2: 输出通道数
        extent: 聚合范围 (0=全局, 2/4/8=局部), default=0
        reduction: MLP 降维比, default=16
    """
    def __init__(self, c1, c2, extent=0, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
        self.ge = GatherExciteCore(c2, extent=extent, reduction=reduction)

    def forward(self, x):
        return self.ge(self.conv(x))
