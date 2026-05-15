import torch
import torch.nn as nn

class GAM_Attention(nn.Module):
    """GAM: Global Attention Mechanism, 通道MLP+空间7×7卷积串行注意力"""
    def __init__(self, in_channels, rate=4):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // rate)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // rate), in_channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // rate), 7, 1, 3),
            nn.BatchNorm2d(max(1, in_channels // rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // rate), in_channels, 7, 1, 3),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        x_channel_att = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_channel_att.permute(0, 3, 1, 2).contiguous().sigmoid()
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        return x * x_spatial_att
