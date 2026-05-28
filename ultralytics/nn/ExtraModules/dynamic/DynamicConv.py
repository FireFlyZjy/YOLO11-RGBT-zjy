"""
DynamicConv — 动态多专家卷积

原理（Dynamic Convolution）：
    使用 K 组并行卷积核（K=4 experts），通过注意力机制根据输入动态聚合。
    每组卷积核对应一个"专家"，注意力权重由 squeeze-and-excitation 风格的
    轻量网络生成。相当于每个输入样本拥有定制化的卷积参数。

机制：
    attention2d:
        - Global AvgPool → FC (降维) → ReLU → FC (K 维) → Softmax (温度控制)
        - temperature 参数控制 softmax 的平滑程度，训练中可逐渐退火
    Dynamic_conv2d:
        - 存储 K 组可学习卷积权重 (K, C_out, C_in, k, k)
        - 前向时用 attention 权重对 K 组参数加权求和
        - 将 batch 维度合并到 groups 中实现分组动态卷积

与 RGBT 关联：
    动态卷积可根据输入模态自适应调整卷积参数，对 RGB 和 Thermal 特征
    分别生成最合适的卷积核，提升多模态特征的提取质量。

用法（在 yaml 中配置）：
    - [-1, 1, Dynamic_conv2d, [out_channels, kernel_size, K]]
    - [-1, 1, Dynamic_conv2d, [out_channels, kernel_size, K, stride, padding]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2d(nn.Module):
    """动态卷积的注意力路由模块。

    通过全局平均池化和两层 MLP 为每个样本生成 K 个专家的权重。

    Args:
        in_planes (int):    输入通道数
        K (int):            专家（卷积核组）数量
        temperature (int):  Softmax 温度，必须满足 temperature % 3 == 1。默认 34
        ratio (float):      中间层通道压缩比，默认 0.25

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, K) — Softmax 归一化的注意力权重
    """

    def __init__(self, in_planes: int, K: int, temperature: int = 34,
                 ratio: float = 0.25):
        super().__init__()
        assert temperature % 3 == 1, "temperature 必须满足 temperature % 3 == 1"

        hidden = max(int(in_planes * ratio), K)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, K, kernel_size=1, bias=False)
        self.temperature = temperature

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def update_temperature(self):
        """退火降低温度，使注意力分布更锐利。"""
        if self.temperature != 1:
            self.temperature -= 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成 K 维注意力权重。"""
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, dim=1)


class Dynamic_conv2d(nn.Module):
    """动态卷积 — K 组专家卷积核的注意力加权聚合。

    Args:
        in_planes (int):    输入通道数
        out_planes (int):   输出通道数
        kernel_size (int):  卷积核大小
        K (int):            专家数量，默认 4
        ratio (float):      注意力模块压缩比，默认 0.25
        stride (int):       步长，默认 1
        padding (int):      填充，默认 0
        dilation (int):     空洞率，默认 1
        groups (int):       分组卷积数，默认 1
        bias (bool):        是否使用偏置，默认 True
        temperature (int):  Softmax 温度，默认 34

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H_out, W_out)

    Example:
        >>> conv = Dynamic_conv2d(64, 128, 3, K=4)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = conv(x)  # (2, 128, 30, 30)

    注意:
        - 训练初期温度较高，注意力分布较均匀；可逐步调用 update_temperature() 退火
        - 由于动态聚合权重，计算量约为 K 倍静态卷积
    """

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int,
                 K: int = 4, ratio: float = 0.25,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, temperature: int = 34):
        super().__init__()
        assert in_planes % groups == 0

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.K = K

        # 注意力路由
        self.attention = Attention2d(in_planes, K, temperature, ratio)

        # K 组可学习卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(K, out_planes, in_planes // groups, kernel_size, kernel_size),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.Tensor(K, out_planes)) if bias else None

        self._init_weights()

    def _init_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
        if self.bias is not None:
            fan = self.in_planes * self.kernel_size ** 2
            bound = 1 / (fan ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def update_temperature(self):
        """退火降低注意力温度。"""
        self.attention.update_temperature()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：注意力加权聚合 K 组卷积参数后执行卷积。"""
        B, C_in, H, W = x.shape

        # (B, K)
        attn = self.attention(x)

        # 合并 batch 维度到 groups: (1, B*C_in, H, W)
        x = x.view(1, -1, H, W)

        # 注意力加权聚合卷积权重
        # weight: (K, C_out, C_in//g, k, k) → (K, -1)
        w = self.weight.view(self.K, -1)
        agg_weight = torch.mm(attn, w).view(
            -1, self.in_planes, self.kernel_size, self.kernel_size
        )

        if self.bias is not None:
            b = self.bias.view(self.K, -1)
            agg_bias = torch.mm(attn, b).view(-1)
            out = F.conv2d(x, weight=agg_weight, bias=agg_bias,
                           stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(x, weight=agg_weight, bias=None,
                           stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups * B)

        # 恢复 batch 维度
        _, _, H_out, W_out = out.shape
        return out.view(B, self.out_planes, H_out, W_out)
