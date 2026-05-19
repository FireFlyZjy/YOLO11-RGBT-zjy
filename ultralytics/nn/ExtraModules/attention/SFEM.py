import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from ...modules.conv import Conv
from ...modules.block import C2f


class ScharrConv(nn.Module):
    """Scharr边缘检测卷积 (不可训练), 提取空间梯度信息"""
    def __init__(self, channel):
        super().__init__()
        scharr_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
        scharr_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)
        scharr_x = torch.tensor(scharr_x).unsqueeze(0).unsqueeze(0).expand(channel, 1, 3, 3)
        scharr_y = torch.tensor(scharr_y).unsqueeze(0).unsqueeze(0).expand(channel, 1, 3, 3)
        self.scharr_x = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.scharr_y = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.scharr_x.weight.data = scharr_x.clone()
        self.scharr_y.weight.data = scharr_y.clone()
        self.scharr_x.requires_grad = False
        self.scharr_y.requires_grad = False

    def forward(self, x):
        return self.scharr_x(x) * 0.5 + self.scharr_y(x) * 0.5


class FreqSpatial(nn.Module):
    """FreqSpatial: 频域+空间双路径特征增强
    - 空间路径: Scharr边缘检测 + 2层Conv
    - 频域路径: FFT -> Conv -> iFFT
    - 两路径相加后1x1 Conv融合
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sed = ScharrConv(in_channels)
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)
        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()
        fft_feat = self.fft_conv(fft_feat)
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')
        fft_feat = self.fft_conv2(fft_feat)

        return self.final_conv(spatial_feat + fft_feat)


class SFEM(C2f):
    """SFEM: Spatial-Frequency Enhanced Module, C2f结构 + FreqSpatial瓶颈
    专为RGB分支设计, 同时提取空间边缘和频域纹理信息
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FreqSpatial(self.c) for _ in range(n))
