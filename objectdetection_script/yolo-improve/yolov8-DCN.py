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