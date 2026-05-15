import torch.nn as nn

class AFF(nn.Module):
    """AFF: Attentional Feature Fusion, 局部+全局注意力学习双特征图逐通道融合权重"""
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, inter_channels, bias=False), nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_pool(xa).flatten(1)
        xg = self.global_fc(xg).unsqueeze(-1).unsqueeze(-1)
        wei = self.sigmoid(xl + xg)
        return 2 * x * wei + 2 * residual * (1 - wei)


class iAFF(nn.Module):
    """iAFF: iterative AFF, 两次局部+全局注意力精炼融合权重"""
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, inter_channels, bias=False), nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels, bias=False))
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True), nn.Conv2d(inter_channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.global_pool2 = nn.AdaptiveAvgPool2d(1)
        self.global_fc2 = nn.Sequential(
            nn.Linear(channels, inter_channels, bias=False), nn.ReLU(inplace=True),
            nn.Linear(inter_channels, channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xg = self.global_pool(xa).flatten(1)
        xg = self.global_fc(xg).unsqueeze(-1).unsqueeze(-1)
        wei = self.sigmoid(self.local_att(xa) + xg)
        xi = x * wei + residual * (1 - wei)
        xg2 = self.global_pool2(xi).flatten(1)
        xg2 = self.global_fc2(xg2).unsqueeze(-1).unsqueeze(-1)
        wei2 = self.sigmoid(self.local_att2(xi) + xg2)
        return x * wei2 + residual * (1 - wei2)
