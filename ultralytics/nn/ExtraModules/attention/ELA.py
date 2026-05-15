import torch.nn as nn

class ELA(nn.Module):
    """ELA: Efficient Local Attention, H/W双向1D卷积+GroupNorm方向局部注意力"""
    def __init__(self, channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        gn_groups = min(16, channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(gn_groups, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape(b, c, h)).reshape(b, c, h, 1)
        x_w = self.conv1x1(self.pool_w(x).reshape(b, c, w)).reshape(b, c, 1, w)
        return x * x_h * x_w
