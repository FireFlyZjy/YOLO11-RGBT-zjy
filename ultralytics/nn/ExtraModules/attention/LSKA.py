import torch.nn as nn

class LSKA(nn.Module):
    """LSKA: Large Separable Kernel Attention, 大核分解为级联1D深度卷积"""
    def __init__(self, dim, k_size=7):
        super().__init__()
        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 1), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (3, 1), (1, 1), (1, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (3, 1), (1, 1), (2, 0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 1), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (3, 1), (1, 1), (1, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 5), (1, 1), (0, 4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (5, 1), (1, 1), (4, 0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, (1, 5), (1, 1), (0, 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (5, 1), (1, 1), (2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 7), (1, 1), (0, 9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (7, 1), (1, 1), (9, 0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, (1, 5), (1, 1), (0, 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (5, 1), (1, 1), (2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 11), (1, 1), (0, 15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (11, 1), (1, 1), (15, 0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, (1, 5), (1, 1), (0, 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (5, 1), (1, 1), (2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 17), (1, 1), (0, 24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (17, 1), (1, 1), (24, 0), groups=dim, dilation=3)
        else:
            self.conv0h = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 1), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, (3, 1), (1, 1), (1, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, (3, 1), (1, 1), (2, 0), groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn
