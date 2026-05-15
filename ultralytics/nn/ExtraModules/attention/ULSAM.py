import torch
import torch.nn as nn

class ULSAM(nn.Module):
    """ULSAM: Ultra-Lightweight Subspace Attention, 分组通道独立空间softmax注意力"""
    def __init__(self, nin, num_splits=4):
        super().__init__()
        self.num_splits = num_splits
        c_per_group = nin // num_splits
        self.dws_conv = nn.Conv2d(c_per_group, c_per_group, 1, 1, 0, groups=c_per_group)
        self.dws_bn = nn.BatchNorm2d(c_per_group)
        self.dws_relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.point_conv = nn.Conv2d(c_per_group, 1, 1, 1, 0)
        self.point_bn = nn.BatchNorm2d(1)
        self.point_relu = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        group_size = x.shape[1] // self.num_splits
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out_list = []
        for feat in sub_feat:
            out = self.dws_relu(self.dws_bn(self.dws_conv(feat)))
            out = self.maxpool(out)
            out = self.point_relu(self.point_bn(self.point_conv(out)))
            b, _, h, w = out.shape
            out = self.softmax(out.view(b, 1, -1)).view(b, 1, h, w)
            out_list.append(out * feat + feat)
        return torch.cat(out_list, dim=1)
