import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ...modules.block import C3k2, C3k

try:
    from timm.models.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.): super().__init__(); self.drop_prob = drop_prob
        def forward(self, x): return x


class AdaptiveMultiKernelConv2d(nn.Module):
    """自适应多核卷积: 方核+水平条带+垂直条带, 动态加权融合"""
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size,
                      padding=square_kernel_size // 2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (1, band_kernel_size),
                      padding=(0, band_kernel_size // 2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (band_kernel_size, 1),
                      padding=(band_kernel_size // 2, 0), groups=in_channels),
        ])
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1),
        )

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(3)]).sum(0)
        return self.act(self.bn(x))


class AdaptiveMultiKernelDWConv(nn.Module):
    """分组多核深度卷积: 通道分两组, 分别用不同核尺寸"""
    def __init__(self, channel=256, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [3, 5]
        self.convs = nn.ModuleList([
            AdaptiveMultiKernelConv2d(channel // 2, ks, ks * 3 + 2) for ks in kernels
        ])
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.SiLU(),
        )

    def forward(self, x):
        c = x.size(1)
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        return self.conv_1x1(x_group)


class ConvolutionalGLU(nn.Module):
    """卷积门控线性单元: 深度卷积 + 门控机制"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features),
            act_layer(),
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class AdaptiveMixerBlock(nn.Module):
    """自适应混合块: AdaptiveMultiKernelDWConv + ConvolutionalGLU, 带LayerScale"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = AdaptiveMultiKernelDWConv(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class IR_AFAB(C3k2):
    """IR_AFAB: Infrared Adaptive Feature Aggregation Block
    专为红外分支设计, 使用自适应多核深度卷积 + 门控线性单元替代标准瓶颈
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else AdaptiveMixerBlock(self.c)
            for _ in range(n)
        )
