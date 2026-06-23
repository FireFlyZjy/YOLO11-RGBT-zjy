import torch
import torch.nn as nn
from collections import OrderedDict


class SKAttention(nn.Module):
    """SK: Selective Kernel Attention, 选择性核注意力
    机制: 多尺度卷积核(1,3,5,7)并行→全局描述符→FC瓶颈→softmax跨核选择
    对RGBT的价值: 自适应多尺度感受野, 热红外目标多尺度特性
    来源: SKNet (CVPR 2019), yoloair-main
    用法: [-1, 1, SKAttention, [c2, reduction]]
    """
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        c = c2
        kernels = [1, 3, 5, 7]
        L = 32
        d = max(L, c // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(c, c, kernel_size=k, padding=k // 2, groups=1)),
                    ('bn', nn.BatchNorm2d(c)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(c, d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(d, c))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.conv(x)
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k, bs, c, h, w
        U = sum(conv_outs)  # bs, c, h, w
        S = U.mean(-1).mean(-1)  # bs, c
        Z = self.fc(S)  # bs, d
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))
        attention_weights = torch.stack(weights, 0)  # k, bs, c, 1, 1
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V
