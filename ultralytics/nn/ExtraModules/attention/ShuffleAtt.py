import torch
import torch.nn as nn

class ShuffleAttention(nn.Module):
    """Shuffle Attention: 分组内半通道注意力半空间注意力 + channel shuffle"""
    def __init__(self, channel=512, G=8):
        super().__init__()
        self.G = G
        half_c = channel // (2 * G)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(half_c, half_c)
        self.cweight = nn.Parameter(torch.zeros(1, half_c, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, half_c, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, half_c, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, half_c, 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(b, -1, h, w)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        return self.channel_shuffle(out, 2)
