"""
CKConv: Cross-shaped Kernel Convolution
========================================
论文: IEEE (https://ieeexplore.ieee.org/document/11320952)
来源: wechat.md

核心机制:
    1. 多尺度十字形卷积核: 支持 [3, 5, 7, 9] 多种核大小
    2. 轴向卷积: 每个核大小使用 1×k (水平) + k×1 (垂直) 实现十字形感受野
    3. 残差连接: body 分支(3×3 DWConv) + head 分支(轴向卷积)
    4. 多尺度融合: 所有尺度分支拼接后 1×1 融合

对 RGBT 的价值:
    - 轴向卷积天然适合处理红外图像的方向性特征
    - 多尺度设计可以捕获不同大小的目标
    - 十字形感受野减少计算量，保持感受野范围

与已有模块的区别:
    - 标准 Conv: 方形卷积核，计算量大
    - DWConv: 深度可分离卷积，但只有单一尺度
    - REM: 轴向卷积模拟小波分解，但只有单一尺度
    - CKConv: 多尺度十字形卷积核，更灵活

用法:
    CKConv: [-1, 1, CKConv, [c2, kk, s]]
    C3k2_CKConv: [-1, 2, C3k2_CKConv, [c2, c3k, e]]

参数:
    c1: 输入通道数
    c2: 输出通道数
    kk: 卷积核大小列表，默认 [3, 5, 7]
    s: 步长，默认 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class CKConv(nn.Module):
    """
    CKConv: Cross-shaped Kernel Convolution - 多尺度十字形卷积核

    机制:
        1. 1×1卷积通道对齐
        2. 对每个核大小 ki:
           - body: 3×3 DWConv (局部特征)
           - head: 1×ki → ki×1 轴向卷积 (十字形感受野)
           - output = body + head
        3. 所有尺度分支拼接后 1×1 融合

    对 RGBT 的价值: 轴向卷积天然适合处理红外图像的方向性特征
    用法: [-1, 1, CKConv, [c2, kk, s]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        kk: 卷积核大小列表，默认 [3, 5, 7]
        s: 步长，默认 1
    """
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):
        super().__init__()
        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("kk must be a list containing 3, 5, 7, and/or 9")
        self.kk = kk
        self.c1 = c1
        self.c2 = c2
        self.s = s

        # 通道对齐
        self.conv_1x1 = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()

        # 多尺度十字形卷积分支
        self.branches = nn.ModuleDict()
        for ki in kk:
            # body: 3×3 DWConv (局部特征)
            self.branches[f'k{ki}_body'] = Conv(c2, c2 // 2, (3, 3), s=1, g=c2 // 2)
            # head: 1×ki → ki×1 轴向卷积 (十字形感受野)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2 // 2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2 // 2)
            self.branches[f'k{ki}_head_v'] = Conv(c2 // 2, c2 // 2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2 // 2)
            # 输出投影
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2 // 2, c2, 1, groups=c2 // 2)

        # 多尺度融合
        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1)

    def forward(self, x):
        outputs = []
        x = self.conv_1x1(x)

        for ki in self.kk:
            # head: 十字形感受野
            y = self.branches[f'k{ki}_head_h'](x)
            y = self.branches[f'k{ki}_head_v'](y)
            # body: 局部特征
            ys = self.branches[f'k{ki}_body'](x)
            # 残差连接
            out = ys + y
            out = self.branches[f'k{ki}_conv2'](out)
            outputs.append(out)

        # 多尺度融合
        out = torch.cat(outputs, dim=1)
        out = self.conv_fuse(out)
        return out


class C3k2_CKConv(nn.Module):
    """
    C3k2_CKConv: 集成 CKConv 的 C3k2 模块

    机制: C3k2 结构 + CKConv 作为 Bottleneck
    用法: [-1, 2, C3k2_CKConv, [c2, c3k, e]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        c3k: 是否使用 CKConv (默认 True)
        e: 扩展比例 (默认 0.5)
        n: 重复次数 (默认 2)
    """
    def __init__(self, c1, c2, c3k=True, e=0.5, n=2):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        if c3k:
            self.m = nn.ModuleList([
                nn.Sequential(
                    CKConv(self.c, self.c),
                    CKConv(self.c, self.c)
                ) for _ in range(n)
            ])
        else:
            self.m = nn.ModuleList([
                nn.Sequential(
                    Conv(self.c, self.c, 3),
                    Conv(self.c, self.c, 3)
                ) for _ in range(n)
            ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
