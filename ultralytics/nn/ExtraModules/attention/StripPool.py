import torch
import torch.nn as nn
import torch.nn.functional as F

class StripPooling(nn.Module):
    """StripPooling: 1D条带池化长程空间上下文, 水平和垂直方向长程依赖"""
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = max(1, in_channels // 4)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv2_0 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(inter_channels))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False), nn.BatchNorm2d(inter_channels))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False), nn.BatchNorm2d(inter_channels))
        self.pool1 = nn.AdaptiveAvgPool2d((20, 12))
        self.pool2 = nn.AdaptiveAvgPool2d((20, 12))
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x); x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode='bilinear', align_corners=True)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode='bilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode='bilinear', align_corners=True)
        x1_out = self.conv2_5(torch.relu(x2_1 + x2_2 + x2_3))
        x2_out = self.conv2_6(torch.relu(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1_out, x2_out], dim=1))
        return torch.relu(x + out)
