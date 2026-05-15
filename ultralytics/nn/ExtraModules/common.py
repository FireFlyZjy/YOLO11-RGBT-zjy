from ultralytics.nn.ExtraModules.attention.SE import SEAttention
import math
import torch
import torch.nn as nn
from ..modules.conv import Conv, autopad
from ultralytics.nn.ExtraModules.conv.ACBlock import ACBlock
from ultralytics.nn.ExtraModules.conv.BlazeBlock import BlazeBlock
from ultralytics.nn.ExtraModules.attention.GatingContext import GatingContext


# -------------------------------------------------- GatingContext start------------------------------------------------------
class GCBlock(nn.Module):
    # Context Gating YOLO 适配包装器
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.cg = GatingContext(c2, add_batch_norm=True)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        out_flat = self.cg(x_flat)
        out = out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out
# -------------------------------------------------- GatingContext end------------------------------------------------------

# -------------------------------------------------- BlazeBlock start------------------------------------------------------
class Conv_Blaze(nn.Module):
    def __init__(self, c1, c2, use_double=False, stride=1, kernel_size=5):
        super().__init__()
        if use_double:
            self.block = BlazeBlock(inp=c1, oup1=c2, oup2=c2, stride=stride, kernel_size=kernel_size)
        else:
            self.block = BlazeBlock(inp=c1, oup1=c2, oup2=None, stride=stride, kernel_size=kernel_size)
    def forward(self, x):
        return self.block(x)
# -------------------------------------------------- BlazeBlock end------------------------------------------------------

# -------------------------------------------------- SeparableConv2d深度可分离卷积 + ASPP空洞空间金字塔池化 start------------------------------------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class ASPP_Branch(nn.Module):
    def __init__(self, c1, c2, rate):
        super(ASPP_Branch, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            self.conv1 = SeparableConv2d(c2, c2, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(c2)
            self.relu1 = nn.ReLU()
        self.atrous_convolution = SeparableConv2d(c1, c2, kernel_size, 1, padding, rate)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        if self.rate != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
# -------------------------------------------------- SeparableConv2d深度可分离卷积 + ASPP空洞空间金字塔池化 end------------------------------------------------------

# -------------------------------------------------- ACBlock start------------------------------------------------------
class Conv_AC(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if k == 3:
            self.conv = ACBlock(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, padding=autopad(k, p), groups=g)
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) if k != 3 else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
# -------------------------------------------------- ACBlock end------------------------------------------------------

# -------------------------------------------------- C2f_DCN start------------------------------------------------------
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        out_channels_offset_mask = (self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(self.in_channels, out_channels_offset_mask, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.reset_parameters()

    def forward(self, x):
        x = x.contiguous()
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1).contiguous()
        mask = torch.sigmoid(mask).contiguous()
        x = torch.ops.torchvision.deform_conv2d(
            x, self.weight, offset, mask, self.bias,
            self.stride[0], self.stride[1], self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1], self.groups, self.deformable_groups, True)
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCN(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DCNv2(c1, c_, k[0], 1) if k[0] == 3 else Conv(c1, c_, k[0], 1)
        self.cv2 = DCNv2(c_, c2, k[1], 1, groups=g) if k[1] == 3 else Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f_DCN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DCN(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
# -------------------------------------------------- C2f_DCN end------------------------------------------------------

# -------------------------------------------------- SEAttention begin------------------------------------------------------
class Conv_SE(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv_SE, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SEAttention(c2)

    def forward(self, x):
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.att(self.act(self.conv(x)))


class CSP_ATT(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=False, g=1, e=0.5):
        super(CSP_ATT, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
        self.att = SEAttention(c2)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.att(self.cv7(torch.cat((y1, y2), dim=1)))
# -------------------------------------------------- SEAttention end------------------------------------------------------

# ============================================================================
# 以下为 YOLO 兼容包装器 (2026-05-13 ~ 2026-05-14)
# 每个 Att_XXX 是对 ExtraModules 中独立模块的薄包装:
#   1. 接受 YOLO 标准 (c1, c2) 参数
#   2. 若 c1 != c2 用 1×1 卷积对齐通道
#   3. 调用底层独立模块
# ============================================================================

from .attention.CBAM import CBAMBlock
from .attention.CoordAtt import CoordAtt
from .attention.SimAM import SimAM as SimAMCore
from .attention.CPCA import CPCA as CPCACore
from .attention.EMA import EMA as EMACore
from .attention.ECA import ECAAttention
from .attention.ShuffleAtt import ShuffleAttention
from .attention.LSKA import LSKA as LSKACore
from .attention.TripletAtt import TripletAttention
from .attention.GAM import GAM_Attention
from .attention.ELA import ELA as ELACore
from .attention.AIFI import AIFI as AIFICore
from .fusion.AFF import AFF as AFFCore, iAFF as iAFFCore
from .attention.ULSAM import ULSAM as ULSAMCore
from .attention.StripPool import StripPooling

# -------------------------------------------------- Att_CBAM (2026-05-13) ------------------------------------------------------
class Att_CBAM(nn.Module):
    def __init__(self, c1, c2, reduction=16, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.cbam = CBAMBlock(channel=c2, reduction=reduction, kernel_size=kernel_size)
    def forward(self, x):
        return self.cbam(self.conv(x))

# -------------------------------------------------- Att_Coord (2026-05-13) ------------------------------------------------------
class Att_Coord(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.coord = CoordAtt(inp=c2, reduction=reduction)
    def forward(self, x):
        return self.coord(self.conv(x))

# -------------------------------------------------- Att_SimAM (2026-05-13) ------------------------------------------------------
class Att_SimAM(nn.Module):
    def __init__(self, c1, c2, e_lambda=1e-4):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.simam = SimAMCore(e_lambda=e_lambda)
    def forward(self, x):
        return self.simam(self.conv(x))

# -------------------------------------------------- Att_CPCA (2026-05-13) ------------------------------------------------------
class Att_CPCA(nn.Module):
    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.cpca = CPCACore(channels=c2, reduction=reduction)
    def forward(self, x):
        return self.cpca(self.conv(x))

# -------------------------------------------------- Att_EMA (2026-05-13) ------------------------------------------------------
class Att_EMA(nn.Module):
    def __init__(self, c1, c2, factor=8):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.ema = EMACore(channels=c2, factor=factor)
    def forward(self, x):
        return self.ema(self.conv(x))

# -------------------------------------------------- Att_ECA (2026-05-13) ------------------------------------------------------
class Att_ECA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.eca = ECAAttention(c=c2)
    def forward(self, x):
        return self.eca(self.conv(x))

# -------------------------------------------------- Att_Shuffle (2026-05-13) ------------------------------------------------------
class Att_Shuffle(nn.Module):
    def __init__(self, c1, c2, G=8):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.sa = ShuffleAttention(channel=c2, G=G)
    def forward(self, x):
        return self.sa(self.conv(x))

# -------------------------------------------------- Att_LSKA (2026-05-13) ------------------------------------------------------
class Att_LSKA(nn.Module):
    def __init__(self, c1, c2, k_size=7):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.lska = LSKACore(dim=c2, k_size=k_size)
    def forward(self, x):
        return self.lska(self.conv(x))

# -------------------------------------------------- Att_Triplet (2026-05-13) ------------------------------------------------------
class Att_Triplet(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.triplet = TripletAttention()
    def forward(self, x):
        return self.triplet(self.conv(x))

# -------------------------------------------------- Att_GAM (2026-05-13) ------------------------------------------------------
class Att_GAM(nn.Module):
    def __init__(self, c1, c2, rate=4):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.gam = GAM_Attention(in_channels=c2, rate=rate)
    def forward(self, x):
        return self.gam(self.conv(x))

# -------------------------------------------------- Att_ELA (2026-05-13) ------------------------------------------------------
class Att_ELA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.ela = ELACore(channels=c2)
    def forward(self, x):
        return self.ela(self.conv(x))

# -------------------------------------------------- Att_AIFI (2026-05-14) ------------------------------------------------------
class Att_AIFI(nn.Module):
    def __init__(self, c1, c2, num_heads=8, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.aifi = AIFICore(c1=c2, num_heads=num_heads, dropout=dropout)
    def forward(self, x):
        return self.aifi(self.conv(x))

# -------------------------------------------------- Att_ScalSeq (2026-05-14) ------------------------------------------------------
class Att_ScalSeq(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_p3, c_p4, c_p5 = c1
        else:
            c_p3 = c_p4 = c_p5 = c1
        self.conv1 = Conv(c_p4, c2, 1)
        self.conv2 = Conv(c_p5, c2, 1)
        self.conv3 = Conv(c_p3, c2, 1)
        self.conv3d = nn.Conv3d(c2, c2, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))
    def forward(self, x):
        p3, p4, p5 = x[0], x[1], x[2]
        p4_up = nn.functional.interpolate(self.conv1(p4), p3.size()[2:], mode='nearest')
        p5_up = nn.functional.interpolate(self.conv2(p5), p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(self.conv3(p3), -3)
        p4_3d = torch.unsqueeze(p4_up, -3)
        p5_3d = torch.unsqueeze(p5_up, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        x = self.pool_3d(self.act(self.bn(self.conv3d(combine))))
        return torch.squeeze(x, 2)

# -------------------------------------------------- Att_AFF (2026-05-14) ------------------------------------------------------
class Att_AFF(nn.Module):
    def __init__(self, c1, c2, r=4):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_a, c_b = c1[0], c1[1]
        else:
            c_a = c_b = c1
        self.conv_a = nn.Conv2d(c_a, c2, 1, 1, 0, bias=False) if c_a != c2 else nn.Identity()
        self.conv_b = nn.Conv2d(c_b, c2, 1, 1, 0, bias=False) if c_b != c2 else nn.Identity()
        self.aff = AFFCore(channels=c2, r=r)
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            a, b = self.conv_a(x[0]), self.conv_b(x[1])
        else:
            a = b = self.conv_a(x)
        return self.aff(a, b)

# -------------------------------------------------- Att_iAFF (2026-05-14) ------------------------------------------------------
class Att_iAFF(nn.Module):
    def __init__(self, c1, c2, r=4):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_a, c_b = c1[0], c1[1]
        else:
            c_a = c_b = c1
        self.conv_a = nn.Conv2d(c_a, c2, 1, 1, 0, bias=False) if c_a != c2 else nn.Identity()
        self.conv_b = nn.Conv2d(c_b, c2, 1, 1, 0, bias=False) if c_b != c2 else nn.Identity()
        self.iaff = iAFFCore(channels=c2, r=r)
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            a, b = self.conv_a(x[0]), self.conv_b(x[1])
        else:
            a = b = self.conv_a(x)
        return self.iaff(a, b)

# -------------------------------------------------- Att_ULSAM (2026-05-14) ------------------------------------------------------
class Att_ULSAM(nn.Module):
    def __init__(self, c1, c2, num_splits=4):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.ulsam = ULSAMCore(nin=c2, num_splits=num_splits)
    def forward(self, x):
        return self.ulsam(self.conv(x))

# -------------------------------------------------- Att_StripPool (2026-05-14) ------------------------------------------------------
class Att_StripPool(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.strip = StripPooling(in_channels=c2)
    def forward(self, x):
        return self.strip(self.conv(x))
