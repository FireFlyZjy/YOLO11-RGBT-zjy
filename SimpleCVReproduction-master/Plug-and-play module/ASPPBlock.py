import torch.nn as nn
import torch


class SeparableConv2d(nn.Module):
    # 深度可分离卷积模块
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积 (Depthwise Convolution)，通过设定 groups=in_channels 实现各个通道独立进行空间卷积
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               groups=in_channels,
                               bias=bias)
        # 逐点卷积 (Pointwise Convolution)，使用 1x1 卷积将分离的通道特征进行跨通道融合
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ASPP(nn.Module):
    # 空洞空间金字塔池化 (ASPP) 的单个分支实现
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate = rate
        if rate == 1:
            # 当膨胀率 rate 为 1 时，等价于标准的 1x1 卷积
            kernel_size = 1
            padding = 0
        else:
            # 当膨胀率 rate 大于 1 时，使用 3x3 卷积，padding 需与 rate 对应以保持特征图尺寸不变
            kernel_size = 3
            padding = rate
            # self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1) # 原始实现可能使用的是标准卷积
            # 使用 3x3 深度可分离卷积作为额外特征处理层
            self.conv1 = SeparableConv2d(planes, planes, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()

            # self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False) # 原始的空洞卷积实现
        # 使用带有膨胀率的深度可分离卷积来代替标准空洞卷积，进一步降低参数量
        self.atrous_convolution = SeparableConv2d(inplanes, planes,
                                                  kernel_size, 1, padding,
                                                  rate)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        # x = self.relu(x) # 此处激活被注释掉，可能为了保持线性特征或留到后面操作
        if self.rate != 1:
            # 如果是具有较大感受野的分支 (rate > 1)，则额外经过一次 3x3 可分离卷积进行特征提纯
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        # 参数初始化：卷积层使用 Kaiming 正态分布初始化，BN 层权重设为 1，偏置设为 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

