# ATAH: Adaptive Task-aware Alignment Head
# 来源: CMFADet (Cross-Modality Feature Adaptive Interaction for RGB-IR Detection)
# 原始依赖 mmcv (ModulatedDeformConv2d + build_norm_layer), 已改写为纯 PyTorch + torchvision
# 修改日期: 2026-05-26
# 改动: 1x1投影对齐输入通道 + torchvision.ops.deform_conv2d 替代 mmcv

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops
from ultralytics.nn.modules import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


class Scale(nn.Module):
    """可学习缩放因子, 初始值 1.0"""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Conv_GN(nn.Module):
    """Conv2d + GroupNorm(16) + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class TaskawareFeatureModulator(nn.Module):
    """任务感知特征调制: 全局平均池化 → 降维 → 升维 → Sigmoid → 逐通道加权"""
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        in_channels = feat_channels * stacked_convs
        self.la_conv1 = nn.Conv2d(in_channels, in_channels // la_down_rate, 1)
        self.la_conv2 = nn.Conv2d(in_channels // la_down_rate, stacked_convs, 1)

    def forward(self, x, avg_feat):
        b, c, h, w = x.shape
        y = self.la_conv2(F.relu(self.la_conv1(avg_feat)))
        y = y.sigmoid()  # 保持4D, unsqueeze会导致广播出5D
        feat_list = [x[:, i * self.feat_channels:(i + 1) * self.feat_channels] * y[:, i:i + 1]
                     for i in range(self.stacked_convs)]
        return torch.cat(feat_list, dim=1)


class DyDCNv2(nn.Module):
    """Deformable Conv v2 with GroupNorm, 使用 torchvision.ops.deform_conv2d (无需 mmcv)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = 3
        self.padding = 1
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.norm = nn.GroupNorm(16, out_channels)
        self._init_weight()

    def _init_weight(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, offset, mask):
        x = tv_ops.deform_conv2d(
            x.contiguous(), offset, self.weight, self.bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            mask=mask,
        )
        return self.norm(x)


class Detect_ATAH(nn.Module):
    """Adaptive Task-aware Alignment Head — 替换标准Detect的检测头
    用法: [[P3_layer, P4_layer, P5_layer], 1, Detect_ATAH, [nc, hidc]]

    改动 vs 原始CMFADet:
    - 新增1x1投影层对齐P3/P4/P5不同通道到统一 hidc
    - DyDCNv2 使用 torchvision.ops.deform_conv2d 替代 mmcv.ModulatedDeformConv2d
    - 新增 self.proj 处理输入通道对齐

    参数:
        nc   : 类别数
        hidc : 内部隐藏通道数 (256 for scale 's')
        ch   : 输入通道列表 [c_p3, c_p4, c_p5] (由 parse_model 自动注入)
    """

    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 1x1投影: 将P3/P4/P5不同通道对齐到统一 hidc
        self.proj = nn.ModuleList([
            nn.Conv2d(c, hidc, 1, bias=False) if c != hidc else nn.Identity()
            for c in ch
        ])

        # 共享特征提取
        self.share_conv = nn.Sequential(
            Conv_GN(hidc, hidc // 2, 3),
            Conv_GN(hidc // 2, hidc // 2, 3),
        )

        # 分类与回归分支 (任务感知)
        self.cls_branch = TaskawareFeatureModulator(hidc // 2, 2, 16)
        self.reg_branch = TaskawareFeatureModulator(hidc // 2, 2, 16)

        # 通道缩减: TaskawareFeatureModulator输出hidc, 下游(DyDCNV2/cv2/cv3)需要hidc//2
        self.cls_reduce = nn.Conv2d(hidc, hidc // 2, 1, bias=False)
        self.reg_reduce = nn.Conv2d(hidc, hidc // 2, 1, bias=False)

        # 回归对齐: 可变形卷积 + 空间偏移
        self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
        self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
        self.offset_dim = 2 * 3 * 3  # 18

        # 分类对齐: 空间概率调制
        self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
        self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)

        # 输出头
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        # 通道对齐
        x = [self.proj[i](x[i]) for i in range(self.nl)]

        for i in range(self.nl):
            # 共享卷积
            stack_res_list = [self.share_conv[0](x[i])]
            stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
            feat = torch.cat(stack_res_list, dim=1)

            # 任务感知特征调制
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_reduce(self.cls_branch(feat, avg_feat))
            reg_feat = self.reg_reduce(self.reg_branch(feat, avg_feat))

            # 回归对齐: 可变形卷积空间校正
            offset_and_mask = self.spatial_conv_offset(feat)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
            reg_feat = self.DyDCNV2(reg_feat, offset, mask)

            # 分类对齐: 空间概率图约束
            cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()

            # 拼接: reg(4*reg_max) + cls(nc)
            x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)

        if self.training:
            return x

        # --- 推理路径 ---
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and getattr(self, 'format', '') in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and getattr(self, 'format', '') in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化偏置, 需要 stride 可用后调用"""
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[:self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
