from ultralytics.nn.ExtraModules.SE import SEAttention
import math
import torch
import torch.nn as nn
from ..modules.conv import Conv, autopad
from ultralytics.nn.ExtraModules.ACBlock import ACBlock
from ultralytics.nn.ExtraModules.BlazeBlock import BlazeBlock
from ultralytics.nn.ExtraModules.GatingContext import GatingContext


# -------------------------------------------------- GatingContext start------------------------------------------------------
class GCBlock(nn.Module):
    # Context Gating YOLO 适配包装器
    def __init__(self, c1, c2):
        super().__init__()
        # 如果输入输出通道不一致，先用 1x1 卷积对齐
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        # 初始化底层的 Context Gating，维度传入 c2
        self.cg = GatingContext(c2, add_batch_norm=True)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape

        # 魔法转换：把 (B, C, H, W) 变成 (B*H*W, C)
        # 这样就能完美适配 GatingContext 内部的 matmul 和 BatchNorm1d
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)

        # 传入 Context Gating 获取加权后的特征
        out_flat = self.cg(x_flat)

        # 把形状变回 YOLO 需要的四维特征图 (B, C, H, W)
        out = out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out
# -------------------------------------------------- GatingContext end------------------------------------------------------

# -------------------------------------------------- BlazeBlock start------------------------------------------------------
class Conv_Blaze(nn.Module):
    """
    BlazeBlock 的 YOLO 兼容包装器
    将 YOLO 标准的 c1, c2 参数对齐到 BlazeBlock 的 inp, oup1, oup2
    """

    def __init__(self, c1, c2, use_double=False, stride=1, kernel_size=5):
        # c1: 输入通道 (YOLO 自动传递)
        # c2: 最终输出通道 (YOLO 自动传递)
        # use_double: 是否启用双层 Block
        super().__init__()

        # 在 YOLO 中，c2 必须是该层最终输出的通道数。
        if use_double:
            # 如果使用双层结构，将中间通道和最终输出通道都设为 c2
            # (你也可以根据需要修改中间层通道数)
            self.block = BlazeBlock(inp=c1, oup1=c2, oup2=c2, stride=stride, kernel_size=kernel_size)
        else:
            # 单层结构：oup2 设为 None
            self.block = BlazeBlock(inp=c1, oup1=c2, oup2=None, stride=stride, kernel_size=kernel_size)

    def forward(self, x):
        return self.block(x)
# -------------------------------------------------- BlazeBlock end------------------------------------------------------

# -------------------------------------------------- SeparableConv2d深度可分离卷积 + ASPP空洞空间金字塔池化 start------------------------------------------------------
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


# 将 ASPP 参数适配为 YOLO 标准的 (c1, c2, ...)
class ASPP_Branch(nn.Module):
    # 将原来的 inplanes, planes 改为 c1, c2
    def __init__(self, c1, c2, rate):
        super(ASPP_Branch, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            self.conv1 = SeparableConv2d(c2, c2, 3, 1, 1) # 注意这里用 c2 (planes)
            self.bn1 = nn.BatchNorm2d(c2)
            self.relu1 = nn.ReLU()

        self.atrous_convolution = SeparableConv2d(c1, c2, kernel_size, 1, padding, rate)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        if self.rate != 1:
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
# -------------------------------------------------- SeparableConv2d深度可分离卷积 + ASPP空洞空间金字塔池化 end------------------------------------------------------

# -------------------------------------------------- ACBlock start------------------------------------------------------
class Conv_AC(nn.Module):
    # 标准卷积的非对称卷积平替
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        # 只有在 3x3 卷积时才触发多分支结构，如果是 1x1 卷积直接用普通 Conv
        if k == 3:
            self.conv = ACBlock(in_channels=c1, out_channels=c2, kernel_size=k,
                                stride=s, padding=autopad(k, p), groups=g)
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)

        self.bn = nn.BatchNorm2d(c2) if k != 3 else nn.Identity()  # ACBlock 内部已经有 BN 了
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
# -------------------------------------------------- ACBlock end------------------------------------------------------

# -------------------------------------------------- C2f_DCN start------------------------------------------------------
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        # 初始化卷积核权重，形状为 (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        # 初始化偏置参数
        self.bias = nn.Parameter(torch.empty(out_channels))

        # 计算偏移和掩码所需的输出通道数：每个变形组需要 3 个参数 (2个方向的偏移 + 1个掩码权重) * 卷积核高 * 卷积核宽
        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        # 用于学习偏移量 (offset) 和掩码 (mask) 的标准卷积层
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels) # 批归一化层
        self.act = Conv.default_act # 使用默认激活函数 (通常为 SiLU)
        self.reset_parameters()

    '''
    def forward(self, x):
        # 计算偏移量和掩码
        offset_mask = self.conv_offset_mask(x)
        # 沿通道维度分为三部分：y方向偏移 (o1), x方向偏移 (o2), 以及掩码 (mask)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        # 拼接 x 和 y 方向的偏移量
        offset = torch.cat((o1, o2), dim=1)
        # 使用 sigmoid 将掩码权重限制在 0 到 1 之间
        mask = torch.sigmoid(mask)
        # 调用 torchvision 原生的形变卷积算子进行前向计算
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x) # 归一化
        x = self.act(x) # 激活
        return x
        '''

    def forward(self, x):
        # 1. 确保输入 x 是连续的（因为它可能来自 Concat 操作）
        x = x.contiguous()

        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)

        # 2. 确保拼接后的 offset 是连续的
        offset = torch.cat((o1, o2), dim=1).contiguous()

        # 3. 确保经过 sigmoid 的 mask 是连续的（这一步最容易导致 0xC0000005）
        mask = torch.sigmoid(mask).contiguous()

        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        # 自定义参数初始化策略
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        # 初始化偏移和掩码卷积的权重和偏置为 0，使初始状态等同于标准卷积
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCN(nn.Module):
    # 带有 DCN 的标准瓶颈层
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # 输入通道, 输出通道, 是否使用残差连接, 分组数, 卷积核尺寸, 膨胀系数
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        # 第一个卷积：如果指定核大小为3则用 DCNv2，否则用普通 Conv
        if k[0] == 3:
            self.cv1 = DCNv2(c1, c_, k[0], 1)
        else:
            self.cv1 = Conv(c1, c_, k[0], 1)
        # 第二个卷积：如果指定核大小为3则用 DCNv2，否则用普通 Conv
        if k[1] == 3:
            self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g)
        else:
            self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # 当且仅当 shortcut 为 True 且输入输出通道数相同时，才启用残差相加
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 如果启用了残差连接则返回 x + 卷积结果，否则直接返回卷积结果
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCN(nn.Module):
    # 包含 2 个卷积层的 CSP 瓶颈层
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # 输入通道, 输出通道, Bottleneck重复次数, 残差连接, 分组数, 膨胀系数
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1) # 起始 1x1 卷积，通道数翻倍以便后续 split
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 末尾的 1x1 融合卷积
        # 生成 n 个带有 DCN 的 Bottleneck 模块列表
        self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        # 将输入通过 1x1 卷积后沿通道维度均分为两部分
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 依次将上一层/分支的输出送入下一个 Bottleneck_DCN，并将结果追加到列表 y 中
        y.extend(m(y[-1]) for m in self.m)
        # 将原有的两个分支以及所有 Bottleneck_DCN 的输出拼接在一起，经过 cv2 融合后输出
        return self.cv2(torch.cat(y, 1))
# -------------------------------------------------- C2f_DCN end------------------------------------------------------

# -------------------------------------------------- SEAttention begin------------------------------------------------------
class Conv_SE(nn.Module):
    # 标准卷积层 + SE 注意力机制
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道, 输出通道, 卷积核, 步长, 填充, 分组, 激活函数
        super(Conv_SE, self).__init__()
        # 自动计算 padding 保证尺寸符合预期
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) # 批归一化
        # 如果 act 为 True 使用 SiLU，否则使用指定的激活函数或恒等映射
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SEAttention(c2) # 在激活后进行通道注意力加权

    def forward(self, x):
        # 前向传播：卷积 -> 归一化 -> 激活 -> SE 注意力
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        # 融合模式下的前向传播（通常用于推理时加速，省去 BN 层）
        return self.att(self.act(self.conv(x)))


class CSP_ATT(nn.Module):
    # CSP 跨阶段局部网络 https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=False, g=1, e=0.5):
        super(CSP_ATT, self).__init__()
        c_ = int(2 * c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个分支的起始 1x1 卷积
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二个分支（捷径分支）的 1x1 卷积
        self.cv3 = Conv(c_, c_, 3, 1)  # 3x3 卷积，用于进一步特征提取
        self.cv4 = Conv(c_, c_, 1, 1)  # 1x1 卷积，用于调整通道
        # 空间金字塔池化：使用不同核大小进行最大池化，padding 保持特征图尺寸不变
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)  # 处理池化拼接后结果的卷积（4倍通道是因为 1个原始+3个池化）
        self.cv6 = Conv(c_, c_, 3, 1)  # 进一步加强空间特征提取
        self.cv7 = Conv(2 * c_, c2, 1, 1)  # 融合两个主分支并调整到输出通道 c2
        self.att = SEAttention(c2)  # SE 注意力机制层

    def forward(self, x):
        # 第一路径：卷积 -> SPP 多尺度池化 -> 卷积
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))

        # 第二路径：简单的 1x1 卷积
        y2 = self.cv2(x)

        # 拼接两个路径，融合后通过注意力机制
        return self.att(self.cv7(torch.cat((y1, y2), dim=1)))

# -------------------------------------------------- SEAttention end------------------------------------------------------

