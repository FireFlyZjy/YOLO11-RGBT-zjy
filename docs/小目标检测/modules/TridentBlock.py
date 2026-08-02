"""
TridentBlock + C3_RFEM — 三分支共享权重的空洞卷积模块

原理（Trident / Receptive Field Enhancement, RFEM）：
    使用共享权重的 1x1 + 3x3 卷积，通过三种不同的空洞率 (dilation=1,2,3)
    提取多尺度感受野特征，再将三路结果融合。共享权重策略避免了参数量随分支数线性增长。

机制：
    TridentBlock:
        - share_weightconv1: 共享 1x1 降维卷积权重
        - share_weightconv2: 共享 3x3 卷积权重，以不同 dilation 与 padding 执行
        - 三个前向分支分别对应 dilation=1 (small), 2 (middle), 3 (big)
        - 每个分支包含 residual 连接
    C3_RFEM:
        类似 C3 结构的封装，内部堆叠多个 RFEM（= TridentBlock + 多路融合）

与 RGBT 关联：
    多尺度感受野对多模态目标检测尤为重要——RGB 与 Thermal 模态中目标尺度分布不同，
    空洞卷积可在不降低分辨率的情况下捕获更大范围上下文。

用法（在 yaml 中配置）：
    - [-1, 1, C3_RFEM, [out_channels, n]]
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad


class TridentBlock(nn.Module):
    """三分支共享权重的空洞卷积块。

    三个分支共享同一组卷积参数，仅 dilation/padding 不同，
    分别感受 small / middle / big 三种尺度的上下文。

    Args:
        c1 (int):    输入通道数
        c2 (int):    输出通道数
        stride (int):     步长，默认 1
        e (float):        隐藏层通道扩张比，默认 0.5
        dilate (list):    三种空洞率，默认 [1, 2, 3]
        bias (bool):      是否使用偏置，默认 False

    Shape:
        Input:  (B, C_in, H, W)
        Output: list of 3 tensors, each (B, C_out, H_out, W_out)
    """

    def __init__(self, c1: int, c2: int, stride: int = 1,
                 e: float = 0.5, dilate=None, bias: bool = False):
        super().__init__()
        if dilate is None:
            dilate = [1, 2, 3]
        self.stride = stride
        self.dilate = dilate
        c_ = int(c2 * e)

        # 共享权重（可学习参数，非 nn.Conv2d 实例）
        self.share_weight_conv1 = nn.Parameter(torch.Tensor(c_, c1, 1, 1))
        self.share_weight_conv2 = nn.Parameter(torch.Tensor(c2, c_, 3, 3))

        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        self.bias = nn.Parameter(torch.Tensor(c2)) if bias else None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.share_weight_conv1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.share_weight_conv2, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def _conv_op(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """执行单分支共享权重卷积。"""
        residual = x
        out = nn.functional.conv2d(x, self.share_weight_conv1, bias=self.bias)
        out = self.bn1(out)
        out = self.act(out)

        padding = dilation  # 保证输出尺寸不变
        out = nn.functional.conv2d(
            out, self.share_weight_conv2, bias=self.bias,
            stride=self.stride, padding=padding, dilation=dilation
        )
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out

    def forward(self, x: torch.Tensor):
        """
        Returns:
            list: [small_output, middle_output, big_output]
        """
        return [
            self._conv_op(x, self.dilate[0]),
            self._conv_op(x, self.dilate[1]),
            self._conv_op(x, self.dilate[2]),
        ]


class RFEM(nn.Module):
    """Receptive Field Enhancement Module — 多尺度感受野增强。

    堆叠 TridentBlock 并将三个尺度的输出融合：
        out = BN(SiLU(x + sum(trident_outputs)))

    Args:
        c1 (int):       输入通道数
        c2 (int):       输出通道数
        n (int):        TridentBlock 堆叠层数，默认 1
        e (float):      隐藏层扩张比，默认 0.5
        stride (int):   步长，默认 1

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H_out, W_out)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, stride: int = 1):
        super().__init__()
        layers = [TridentBlock(c1, c2, stride=stride, e=e)]
        for _ in range(1, n):
            layers.append(TridentBlock(c2, c2, e=e))
        self.trident_blocks = nn.ModuleList(layers)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.trident_blocks:
            s, m, b = block(out)
            out = s + m + b + out
        return self.act(self.bn(out))


class C3_RFEM(nn.Module):
    """C3 风格的多尺度感受野增强模块。

    类似 C3 结构：1x1 降维 → RFEM 堆叠 → 1x1 升维 + 残差连接。

    Args:
        c1 (int):       输入通道数
        c2 (int):       输出通道数
        n (int):        RFEM 堆叠层数，默认 1
        e (float):      中间层通道扩张比，默认 0.5

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=1)
        self.m = nn.Sequential(*[RFEM(c_, c_, n=1, e=e) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.m(self.cv1(x)))
