# DecoupledHead: 解耦检测头
# 来源: https://github.com/PatrickLi/objectdetection_script (yolov5-DecoupledHead.py)
# 说明:
#   标准 YOLO 检测头使用单个卷积同时预测分类和回归结果。
#   解耦头将共享 stem 后的特征分流到两个独立分支:
#     - cls branch: 预测类别概率
#     - reg branch: 预测边界框 (含 objectness)
#   这种设计避免了分类与回归任务之间的特征竞争, 提升收敛速度与检测精度。
#
#   本实现适配 YOLOv11+ 的训练流程 (TAL, anchor-free), 兼容 3 个检测层 (P3/P4/P5)。
#   参考 Detect_ATAH 的接口约定:
#     - dynamic / export / format / end2end / max_det / shape / anchors / strides 类属性
#     - bias_init() 方法
#     - decode_bboxes() 方法
#     - 训练时返回 list, 推理时返回 (y, x) 元组
#
# 用法 (YOLO YAML):
#   [[P3_layer, P4_layer, P5_layer], 1, DecoupledHead, [nc, hidc]]
#   示例:
#     nc: 80
#     hidc: 256
#     [[17, 20, 23], 1, DecoupledHead, [nc, hidc]]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import dist2bbox, make_anchors


class DecoupledHead(nn.Module):
    """解耦检测头: 共享 stem → 分离 cls/reg 分支

    参数:
        nc (int): 类别数
        hidc (int): 隐藏层通道数 (中间特征通道)
        ch (tuple): 输入通道列表 [c_p3, c_p4, c_p5] (由 parse_model 自动注入)
        reg_max (int): DFL 回归的最大索引, 默认 16
    """

    # ---- YOLO 兼容属性 ----
    dynamic = False          # 是否动态重建 grid
    export = False           # 导出模式
    format = None            # 导出格式 (onnx, pb, tflite, ...)
    end2end = False          # 端到端模式
    max_det = 300            # 最大检测数
    shape = None             # 缓存输入 shape
    anchors = torch.empty(0)  # anchor 点 (由 make_anchors 生成)
    strides = torch.empty(0)  # 每个检测层的步长

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc                     # 类别数
        self.nl = len(ch)                # 检测层数 (通常为 3: P3, P4, P5)
        self.reg_max = 16                # DFL 回归参数
        self.no = nc + self.reg_max * 4  # 每个 anchor 的输出通道数
        self.stride = torch.zeros(self.nl)  # 步长 (由 parse_model 填充)

        # ---- 1×1 投影: 将各层不同通道对齐到统一 hidc ----
        self.proj = nn.ModuleList([
            nn.Conv2d(c, hidc, 1, bias=False) if c != hidc else nn.Identity()
            for c in ch
        ])

        # ---- 共享 stem: 1×1 降维 + 3×3 特征提取 ----
        self.stem = nn.Sequential(
            Conv(hidc, hidc, k=3),
            Conv(hidc, hidc, k=3),
        )

        # ---- 分类分支: 3×3 卷积 + 1×1 输出 ----
        self.cls_branch = nn.Sequential(
            Conv(hidc, hidc, k=3),
            nn.Conv2d(hidc, self.nc, 1),
        )

        # ---- 回归分支: 3×3 卷积 + 1×1 输出 (含 bbox + obj) ----
        self.reg_branch = nn.Sequential(
            Conv(hidc, hidc, k=3),
        )
        self.reg_conv = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.obj_conv = nn.Conv2d(hidc, 1, 1)

        # ---- DFL (Distribution Focal Loss) ----
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """前向传播

        参数:
            x (list[torch.Tensor]): 三个检测层的特征, [(b,c3,h3,w3), (b,c4,h4,w4), (b,c5,h5,w5)]

        返回:
            (list | tuple): training → list; inference → (y, x)
        """
        # 1×1 投影对齐通道
        x = [self.proj[i](x[i]) for i in range(self.nl)]

        for i in range(self.nl):
            # 共享 stem
            feat = self.stem(x[i])

            # 分类分支
            cls_out = self.cls_branch(feat)

            # 回归分支
            reg_feat = self.reg_branch(feat)
            reg_out = self.reg_conv(reg_feat)   # (b, 4*reg_max, h, w)
            obj_out = self.obj_conv(reg_feat)   # (b, 1, h, w)

            # 拼接: [reg(4*reg_max), cls(nc)]
            x[i] = torch.cat((reg_out, cls_out), dim=1)

        # ---- 训练模式: 直接返回 list 供 TAL 计算损失 ----
        if self.training:
            return x

        # ---- 推理模式: 解码边界框 ----
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        dbox = self.decode_bboxes(box)

        y = torch.cat((dbox, cls.sigmoid()), dim=1)
        return y if self.export else (y, x)

    def decode_bboxes(self, bboxes):
        """将 DFL 分布解码为边界框坐标

        参数:
            bboxes (torch.Tensor): DFL 输出的分布, shape (b, 4*reg_max, n)

        返回:
            torch.Tensor: 解码后的边界框, shape (b, 4, n), 格式 (cx, cy, w, h)
        """
        return dist2bbox(
            self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1
        ) * self.strides

    def bias_init(self):
        """初始化偏置 (需要在 stride 可用后调用)

        回归分支偏置设为 1.0, 分类分支偏置根据类别数设置初始值
        """
        # 回归头最后一层偏置初始化
        self.reg_conv.bias.data[:] = 1.0
        # 分类头最后一层偏置初始化
        self.cls_branch[-1].bias.data[:self.nc] = math.log(
            5 / self.nc / (640 / 16) ** 2
        )
