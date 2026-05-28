# ASF: Adaptive Spatial Fusion (自适应空间融合)
# 来源: https://github.com/PatrickLi/objectdetection_script (yolov5-asf.py)
# 说明:
#   本文件实现了跨尺度特征融合所需的各种模块。核心思想是将 P3/P4/P5
#   三个尺度的特征通过池化/插值对齐到同一空间大小, 然后融合。
#   包含两个注意机制:
#     - channel_att: ECA 风格的一维卷积通道注意力
#     - local_att: 坐标注意力 (CoordAttention 的简化版)
#   最终通过 ASF_fusion 将 Zoom_cat + attention_model 组合为完整融合模块。
# 用法 (YOLO YAML):
#   [[P3_layer, P4_layer, P5_layer], 1, Zoom_cat, []]
#   [[feat1, feat2], 1, attention_model, []]
#   [[P3_layer, P4_layer, P5_layer], 1, ASF_fusion, []]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# ===================== 跨尺度拼接 =====================

class Zoom_cat(nn.Module):
    """跨尺度 Zoom + Concat

    将输入的三尺度特征 [l, m, s] 都对齐到 m 的分辨率后拼接。
    其中:
      - l (大尺度, 分辨率最低): 使用 MaxPool + AvgPool 降采样到 m 大小
      - m: 保持不动
      - s (小尺度, 分辨率最高): 使用 nearest 插值下采样到 m 大小

    用法 (YOLO YAML): [[P3, P4, P5], 1, Zoom_cat, []]
      c2 = sum(ch[x] for x in f)   (由 parse_model 自动计算)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """l, m, s 分别表示大中小三个尺度, 最终全部对齐到 m 的分辨率"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]  # (h, w)

        # 大尺度特征: MaxPool + AvgPool 降采样到目标分辨率
        l_pooled = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        # 小尺度特征: nearest 插值缩小
        s_interp = F.interpolate(s, tgt_size, mode='nearest')

        return torch.cat([l_pooled, m, s_interp], dim=1)


# ===================== 3D卷积跨尺度融合 (来自yolov5-asf.py的ScalSeq) =====================

class ScalSeq(nn.Module):
    """3D 卷积跨尺度特征序列融合

    将 P3/P4/P5 三个层通过 1×1 卷积对齐到统一通道 → unsqueeze 为 3D → 3D 卷积 → 池化

    用法 (YOLO YAML):
      [[P3_layer, P4_layer, P5_layer], 1, ScalSeq, [channel]]
      其中 channel 为期望的统一通道数
    """

    def __init__(self, inc, channel):
        """参数:
            inc (list): 三个输入层的通道数 [c_p3, c_p4, c_p5]
            channel (int): 统一的目标通道数
        """
        super().__init__()
        self.conv1 = Conv(inc[1], channel, 1)   # P4 通道对齐
        self.conv2 = Conv(inc[2], channel, 1)   # P5 通道对齐
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        """x = [p3, p4, p5]"""
        p3, p4, p5 = x[0], x[1], x[2]

        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.shape[2:], mode='nearest')

        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.shape[2:], mode='nearest')

        # 在 depth 维度上拼接 → 3D 卷积 → 池化取回
        p3_3d = torch.unsqueeze(p3, -3)         # (b, c, 1, h, w)
        p4_3d = torch.unsqueeze(p4_2, -3)       # (b, c, 1, h, w)
        p5_3d = torch.unsqueeze(p5_2, -3)       # (b, c, 1, h, w)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)  # (b, c, 3, h, w)

        out = self.conv3d(combine)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool_3d(out)                 # (b, c, 1, h, w)
        out = out.squeeze(2)                     # (b, c, h, w)
        return out


# ===================== 注意力模块 =====================

class channel_att(nn.Module):
    """ECA 风格通道注意力 (一维卷积)

    自适应确定一维卷积核大小: k = max(3, odd(|log2(c) + b| / gamma))
    通过全局平均池化 → 1D Conv → Sigmoid → 逐通道缩放
    """

    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)         # (b, c, 1, 1)
        y = y.squeeze(-1)            # (b, c, 1)
        y = y.transpose(-1, -2)      # (b, 1, c)
        y = self.conv(y)             # (b, 1, c)
        y = y.transpose(-1, -2)      # (b, c, 1)
        y = y.unsqueeze(-1)          # (b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    """坐标注意力 (CoordAttention 简化版)

    沿 H 和 W 方向分别做全局平均池化 → 拼接 → 1×1 卷积 → 分开 → 1×1 卷积 → Sigmoid
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1,
                                  stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1,
                             stride=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1,
                             stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        """保持与原始 yolov5-asf.py 完全一致的实现"""
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)  # (b,c,1,w) -> (b,c,w,1)
        x_w = torch.mean(x, dim=2, keepdim=True)                      # (b,c,h,1)

        x_cat_conv_relu = self.relu(
            self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        )                                                             # (b,c/r,h+w,1)

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(
            self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2))
        )                                                             # (b,c,1,w) -> (b,c,1,w)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))           # (b,c,h,1)

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# ===================== ECA+Coord 融合注意 =====================

class attention_model(nn.Module):
    """通道注意力 + 坐标注意力融合

    对两个输入:
      - input1 先经过 channel_att (ECA)
      - 然后与 input2 逐元素相加
      - 结果再经过 local_att (坐标注意力)

    YOLO YAML: [[feat1, feat2], 1, attention_model, []]
      c2 = ch[f[-1]]  (parse_model 自动取最后一个输入的通道数)
    """

    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


# ===================== 简易 Add 模块 =====================

class Add(nn.Module):
    """逐元素相加 (两个输入)"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] + x[1]


# ===================== 完整 ASF 融合模块 =====================

class ASF_fusion(nn.Module):
    """ASF (Adaptive Spatial Fusion) 完整融合模块

    将 Zoom_cat 跨尺度拼接与 attention_model 通道-坐标注意力融合集成为一个模块。
    流程:
      1) 三个尺度 (P3, P4, P5) 通过 Zoom_cat 对齐到 P4 分辨率并拼接
      2) 拼接结果经 1×1 卷积 + BN + SiLU 降维
      3) 与另一个输入特征通过 attention_model 融合 (通道注意 + 坐标注意)

    YOLO YAML:
      [[P3_layer, P4_layer, P5_layer, other_feat], 1, ASF_fusion, [out_ch]]

    参数:
        in_ch (int): Zoom_cat 拼接后的通道数 (P3_ch + P4_ch + P5_ch)
        out_ch (int): 目标输出通道数
        other_ch (int): other_feat 的通道数 (未明确传入时由 parse_model 决定)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.zoom_cat = Zoom_cat()
        self.reduce = Conv(in_ch, out_ch, k=1)
        self.att = attention_model(out_ch)

    def forward(self, x):
        """x = [p3, p4, p5, other_feat]"""
        p3, p4, p5, other = x[0], x[1], x[2], x[3]
        zoomed = self.zoom_cat([p3, p4, p5])
        zoomed = self.reduce(zoomed)
        return self.att([zoomed, other])
