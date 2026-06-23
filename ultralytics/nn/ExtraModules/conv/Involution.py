import torch
import torch.nn as nn


class Involution(nn.Module):
    """Involution: 逆卷积算子, 位置感知的动态卷积核
    机制: 每个位置生成专属卷积核(基于1x1→展开→分组乘法), 兼顾位置特异性和通道交互
    对RGBT的价值: 位置感知特征提取, 红外/可见光不同位置的自适应核
    来源: Involution (CVPR 2021), yoloair-main
    用法: [-1, 1, Involution, [c2, kernel_size, stride]]
    """
    def __init__(self, c1, c2, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        c = c2
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = c // self.group_channels
        self.conv1 = nn.Conv2d(c, c // reduction_ratio, 1)
        self.conv2 = nn.Conv2d(c // reduction_ratio, kernel_size ** 2 * self.groups, 1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        x = self.conv(x)
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
        return out


class InvolutionBottleneck(nn.Module):
    """InvolutionBottleneck: 含Involution的标准瓶颈"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = Involution(c_, c_, 3, 1)
        self.cv3 = nn.Conv2d(c_, c2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        out = self.cv2(out)
        out = self.bn3(self.cv3(out))
        return self.act(x + out) if self.add else self.act(out)
