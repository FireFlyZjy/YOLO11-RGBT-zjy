import torch
import torch.nn as nn


class Channel_Att(nn.Module):
    """BN统计通道注意力内部模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual
        return x


class NAMAttention(nn.Module):
    """NAM: No-Internal-Covariance-Shift Attention, 极轻量BN统计通道注意力
    机制: 利用BatchNorm的scale参数作为通道重要性代理, 零额外学习参数
    对RGBT的价值: 极轻量通道重标定, 参数预算紧张时的baseline对比
    来源: NAM (arXiv:2111.12419), yoloair-main
    用法: [-1, 1, NAMAttention, [c2]]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.Channel_Att = Channel_Att(c2)

    def forward(self, x):
        x = self.conv(x)
        return self.Channel_Att(x)
