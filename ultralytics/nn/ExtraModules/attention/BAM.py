import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BAM Internal Implementations (完全自包含, 无mmcv依赖)
# 来源: https://github.com/Jongchan/attention-module
# ============================================================

class Flatten(nn.Module):
    """展平层: (B, C, H, W) -> (B, C*H*W)"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """BAM通道注意力门: GAP -> MLP(FC->BN->ReLU->...->FC) -> 通道权重

    与SE稍有不同: BAM的通道门在输出后 expand_as(x) 得到 (B,C,H,W) 权重图,
    然后与空间门逐元素相乘, 而非像SE一样直接乘以特征。
    """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super().__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(f'gate_c_fc_{i}', nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module(f'gate_c_bn_{i + 1}', nn.LayerNorm(gate_channels[i + 1]))
            self.gate_c.add_module(f'gate_c_relu_{i + 1}', nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # Global average pooling to 1x1
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        # MLP -> expand to (B, C, 1, 1) -> (B, C, H, W)
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    """BAM空间注意力门: 1x1降维 -> 级联空洞卷积 -> 1x1 -> 空间权重

    使用空洞卷积(默认rate=4, 2层)扩大感受野,
    等效于更大的卷积核但参数更少, 对IR小目标更友好。
    """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.gate_s = nn.Sequential()
        # 1x1降维: C -> C/r
        self.gate_s.add_module('gate_s_conv_reduce0',
                                nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.GroupNorm(4,gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        # 级联空洞卷积: 扩大感受野
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                f'gate_s_conv_di_{i}',
                nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                          kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module(f'gate_s_bn_di_{i}', nn.GroupNorm(4,gate_channel // reduction_ratio))
            self.gate_s.add_module(f'gate_s_relu_di_{i}', nn.ReLU())
        # 1x1输出: C/r -> 1 (单通道空间权重图)
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, x):
        # (B, 1, H, W) -> expand -> (B, C, H, W)
        return self.gate_s(x).expand_as(x)


# ============================================================
# YOLO-Compatible Wrapper (c1, c2 签名, 支持通道投影)
# ============================================================

class Att_BAM(nn.Module):
    """[BAM] 瓶颈注意力模块 (Bottleneck Attention Module)

    机制:
        通道注意力(ChannelGate) + 空洞空间注意力(SpatialGate)并行计算后融合。
        通道分支: GAP->MLP(FC->BN->ReLU->FC) -> (B,C,1,1) -> expand通道权重
        空间分支: 1x1降维(C->C/r) -> 级联空洞卷积(rate=4,2层) -> 1x1 -> 单通道空间权重
        两分支逐元素相乘 -> sigmoid(1+att) -> 与原特征逐元素相乘

    与CBAM的关键区别:
        1. CBAM串行(通道->空间), BAM并行
        2. BAM空间支路使用空洞卷积(默认rate=4,2层,等效9x9感受野)
        3. BAM使用 sigmoid(1+att) 而非直接 sigmoid(att) 门控, 保留更多原特征
        4. BAM的通道门包含BN, CBAM没有

    RGBT价值:
        - 空洞卷积感受野: 对IR小目标更友好(大范围背景参考)
        - 并行结构: 通道/空间相互独立, 更适合双模态融合后特征
        - 相比CBAM更强的全局建模能力, 同时参数量相当
        - 1+sigmoid门控保留原始信号强度, 防止过度抑制

    复杂度:
        通道分支: O(C^2/r)   (全连接层)
        空间分支: O(HWC/r)   (1x1 + 空洞卷积)
        总参数量约: 2*C^2/r + 9*C/r + C

    YAML用法 (单输入, from=-1):
        - [-1, 1, Att_BAM, [c2]]
        - [-1, 1, Att_BAM, [c2, 16]]        # 自定义reduction
        - [-1, 1, Att_BAM, [c2, 16, 2, 4]]  # 自定义reduction/空洞层数/空洞率

    注册需求: 需在 tasks.py 的 attention 模块列表中添加 Att_BAM
        参考 CBAM, BAMBlock(已存在), CoordAtt 的注册方式:
        elif m in {..., Att_BAM, ...}:
            c2 = ch[f]
            args = [c2, *args]
    """
    def __init__(self, c1, c2, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.channel_att = ChannelGate(c2, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialGate(c2, reduction_ratio=reduction_ratio,
                                        dilation_conv_num=dilation_conv_num,
                                        dilation_val=dilation_val)

    def forward(self, x):
        x = self.proj(x)
        # 通道+空间注意力并行相乘, sigmoid(1+att)门控
        att = 1 + torch.sigmoid(self.channel_att(x) * self.spatial_att(x))
        return att * x
