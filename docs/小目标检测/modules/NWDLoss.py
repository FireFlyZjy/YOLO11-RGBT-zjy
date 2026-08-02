# NWD Loss: Normalized Wasserstein Distance Loss
# 论文: "Enhancing Geometric Factors into Model Learning and Inference
#        for Object Detection and Instance Segmentation"
#        https://arxiv.org/abs/2005.03572
# 说明:
#   NWD (Normalized Wasserstein Distance) 是一种基于最优传输理论的边界框
#   相似度度量。它将边界框建模为二维高斯分布, 然后计算两个高斯分布之间的
#   Wasserstein 距离, 最终通过指数函数归一化到 (0, 1] 区间。
#
#   相比 IoU 系列损失, NWD 对小目标更友好, 因为它对位置偏差的惩罚更平滑,
#   没有 IoU 的 "梯度消失" 问题 (当两个框完全不重叠时 IoU=0, 梯度为 0)。
#
# 使用方式 (在 ultralytics/utils/loss.py 中修改):
#   见下方 integrate_into_yolo_loss() 函数的文档字符串。

import torch
import torch.nn as nn


def bbox_to_gaussian(box):
    """将边界框转换为二维高斯分布的均值与协方差对角元素

    参数:
        box (torch.Tensor): 边界框坐标, shape (..., 4), 格式 (cx, cy, w, h)

    返回:
        (torch.Tensor, torch.Tensor, torch.Tensor):
          - mu: 均值, shape (..., 2), 即 (cx, cy)
          - var_w: w 方向方差, shape (...), 即 w^2/4
          - var_h: h 方向方差, shape (...), 即 h^2/4
    """
    mu = box[..., :2]          # (cx, cy)
    w = box[..., 2]
    h = box[..., 3]
    var_w = (w ** 2) / 4       # 高斯分布在 w 方向的方差
    var_h = (h ** 2) / 4       # 高斯分布在 h 方向的方差
    return mu, var_w, var_h


def wasserstein_distance(mu1, var_w1, var_h1, mu2, var_w2, var_h2, eps=1e-7):
    """计算两个二维高斯分布之间的 Wasserstein 距离的平方 (W2^2)

    参数:
        mu1, mu2 (torch.Tensor): 均值, shape (..., 2)
        var_w1, var_w2 (torch.Tensor): w 方向方差, shape (...)
        var_h1, var_h2 (torch.Tensor): h 方向方差, shape (...)
        eps (float): 防止除零

    返回:
        torch.Tensor: Wasserstein 距离的平方, shape (...), 值 ≥ 0
    """
    center_dist = (mu1[..., 0] - mu2[..., 0]) ** 2 + \
                  (mu1[..., 1] - mu2[..., 1]) ** 2  # 中心点欧氏距离平方

    wh_dist = ((var_w1.sqrt() - var_w2.sqrt()) ** 2 +
               (var_h1.sqrt() - var_h2.sqrt()) ** 2) * 4  # 等价于 ((w1 - w2)^2 + (h1 - h2)^2) / 4

    return center_dist + wh_dist + eps


def normalized_wasserstein_distance(pred, target, eps=1e-7, constant=12.8):
    """计算归一化 Wasserstein 距离 (Normalized Wasserstein Distance)

    公式: NWD = exp(-sqrt(W2^2) / constant)
    其中 W2^2 是两个边界框对应高斯分布的 Wasserstein-2 距离平方。
    结果范围: (0, 1], 越接近 1 表示两个框越相似。

    参数:
        pred (torch.Tensor): 预测框, shape (n, 4), 格式 (cx, cy, w, h)
        target (torch.Tensor): 目标框, shape (n, 4), 格式 (cx, cy, w, h)
        eps (float): 防止除零
        constant (float): 归一化常数, 默认 12.8 (约等于数据集的平均框尺寸)

    返回:
        torch.Tensor: NWD 相似度, shape (n,), 范围 (0, 1]
    """
    mu1, vw1, vh1 = bbox_to_gaussian(pred)
    mu2, vw2, vh2 = bbox_to_gaussian(target)

    w2_sq = wasserstein_distance(mu1, vw1, vh1, mu2, vw2, vh2, eps)
    return torch.exp(-torch.sqrt(w2_sq) / constant)


class NWDLoss(nn.Module):
    """NWD Loss 模块

    在训练过程中, 可以用 NWD Loss 部分替代 IoU Loss 来提升小目标检测性能。

    使用方法 (在 ultralytics/utils/loss.py 中):
      见 integrate_into_yolo_loss() 的文档字符串。

    参数:
        constant (float): 归一化常数, 默认 12.8
        eps (float): 防止除零
    """

    def __init__(self, constant=12.8, eps=1e-7):
        super().__init__()
        self.constant = constant
        self.eps = eps

    def forward(self, pred, target):
        """计算 NWD Loss

        NWD Loss = 1 - NWD (类似 IoU Loss = 1 - IoU)

        参数:
            pred (torch.Tensor): 预测框, shape (n, 4), 格式 (cx, cy, w, h)
            target (torch.Tensor): 目标框, shape (n, 4), 格式 (cx, cy, w, h)

        返回:
            torch.Tensor: Loss 值, shape (n,)
        """
        nwd = normalized_wasserstein_distance(
            pred, target, eps=self.eps, constant=self.constant
        )
        return 1.0 - nwd


# ====================================================================
# 集成指南: 如何在 YOLOv11+ 训练中使用 NWD Loss
# ====================================================================
#
# 1. 打开 ultralytics/utils/loss.py
#
# 2. 在文件顶部导入:
#    from loss.NWDLoss import normalized_wasserstein_distance
#
# 3. 在 BboxLoss 类的 forward() 方法中找到计算 iou 损失的位置
#    (通常在损失函数的后半部分, 格式类似):
#
#      iou = bbox_iou(pred, target, xywh=True, CIoU=True)
#      loss_iou = (1.0 - iou).mean()
#
#    将其替换为:
#
#      iou = bbox_iou(pred, target, xywh=True, CIoU=True)
#      nwd = normalized_wasserstein_distance(pred, target)
#
#      iou_ratio = 0.5  # IoU 与 NWD 的混合比例 (可自行调整)
#      loss_iou = (1 - iou_ratio) * (1.0 - nwd).mean() + iou_ratio * (1.0 - iou).mean()
#
# 4. (可选) 在 objectness 损失部分, 也可以混合 NWD 来软化正样本目标值:
#
#      iou = (iou.detach() * iou_ratio + nwd.detach() * (1 - iou_ratio)).clamp(0, 1)
#
# 参数调节建议:
#   - constant=12.8 适用于大多数通用目标检测数据集
#   - 对于小目标密集场景 (如 VisDrone, UAVDT), 建议增大 constant (例如 20~30)
#   - iou_ratio 在 0.3~0.7 之间调整, 小目标场景推荐 0.3~0.5 (即 NWD 权重更大)
# ====================================================================

def integrate_into_yolo_loss():
    """集成指南 (仅作文档用途, 不可直接调用)

    请参照上面的文档字符串手动修改 ultralytics/utils/loss.py。
    """
    raise NotImplementedError(
        "请手动修改 ultralytics/utils/loss.py, "
        "详情见本文件底部的集成指南文档字符串。"
    )
