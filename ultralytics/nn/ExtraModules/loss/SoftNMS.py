# Soft-NMS: 软化非极大值抑制
# 论文: "Soft-NMS — Improving Object Detection With One Line of Code"
#        https://arxiv.org/abs/1704.04503
# 说明:
#   传统的 NMS (Non-Maximum Suppression) 会将所有与最高分框 IoU 超过阈值
#   的框直接置零 (即完全删除), 这可能导致相邻的同类目标被误删。
#   Soft-NMS 改为根据 IoU 大小对分数进行高斯衰减:
#     s_i = s_i * exp(-IoU^2 / sigma)
#   这样与最高分框高度重叠的框分数会大幅降低 (而不是直接归零),
#   若该框确实对应另一个目标, 其分数仍可能保留在阈值以上。
#
# 使用方式:
#   在推理或后处理阶段, 将原 NMS 调用替换为 soft_nms。
#   例如在 val_zjy_full.py 或 ultralytics/utils/ops.py 中:
#
#     from loss.SoftNMS import soft_nms
#     keep = soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25)
#
#   也可以与 torchvision.ops.nms 组合使用进行两阶段过滤。

import math
import torch


def box_iou_for_nms(box1, box2, GIoU=False, DIoU=False, CIoU=False,
                    SIoU=False, EIou=False, eps=1e-7):
    """计算 box1 (1,4) 与 box2 (n,4) 之间的 IoU (支持多种变体)

    与 torchvision.ops.box_iou 的区别:
     - 支持 GIoU / DIoU / CIoU / SIoU / EIoU
     - 纯 torch 实现, 无外部依赖
     - Soft-NMS 中默认使用标准 IoU

    参数:
        box1 (torch.Tensor): 单个边界框, shape (1, 4), 格式 (x1, y1, x2, y2)
        box2 (torch.Tensor): N 个边界框, shape (N, 4), 格式 (x1, y1, x2, y2)
        GIoU, DIoU, CIoU, SIoU, EIou (bool): 是否使用对应变体
        eps (float): 防止除零

    返回:
        torch.Tensor: IoU 值, shape (N,)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)

    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # 交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # 并集面积
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU or EIou:
        # 最小外接矩形
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)

        if CIoU or DIoU or EIou:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

            if CIoU:
                v = (4 / math.pi ** 2) * \
                    (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            elif EIou:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                return iou - (rho2 / c2 + rho_w2 / cw ** 2 + rho_h2 / ch ** 2)
            return iou - rho2 / c2  # DIoU

        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area  # GIoU

    elif SIoU:
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + \
                     torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return iou - 0.5 * (distance_cost + shape_cost)

    return iou


def soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
    """Soft-NMS: 软化非极大值抑制

    与传统 NMS 不同, Soft-NMS 不会直接删除与高分框 IoU 超过阈值的框,
    而是对它们的分数进行高斯衰减:
      s_i = s_i * exp(-IoU(i, max_box)^2 / sigma)

    算法流程:
      1) 找到当前最高分框, 加入保留列表 keep
      2) 计算该框与其余所有框的 IoU
      3) 对 IoU 超过阈值的框进行分数衰减 (高斯惩罚)
      4) 删除分数低于 score_threshold 的框
      5) 对剩余框按分数排序, 重复步骤 1~4

    参数:
        bboxes (torch.Tensor): 边界框坐标, shape (N, 4), 格式 (x1, y1, x2, y2)
        scores (torch.Tensor): 检测分数 (已包含类别置信度), shape (N,)
        iou_thresh (float): IoU 阈值, 超过此阈值的框会受惩罚, 默认 0.5
        sigma (float): 高斯衰减系数, sigma 越小则衰减越剧烈, 默认 0.5
        score_threshold (float): 分数阈值, 低于此值的框被删除, 默认 0.25

    返回:
        torch.LongTensor: 保留框的索引, shape (K,)
    """
    order = torch.arange(0, scores.size(0), device=bboxes.device)
    keep = []

    while order.numel() > 1:
        # 取当前最高分框 (第一个)
        i = order[0]
        keep.append(i)

        # 计算该框与其余框的 IoU
        iou = box_iou_for_nms(bboxes[i], bboxes[order[1:]]).squeeze()

        # 对 IoU 超过阈值的框进行高斯衰减
        idx = (iou > iou_thresh).nonzero(as_tuple=False).squeeze()
        if idx.numel() > 0:
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            iou_vals = iou[idx]
            decay = torch.exp(-torch.pow(iou_vals, 2) / sigma)
            scores[order[idx + 1]] *= decay

        # 筛选分数仍高于阈值的框
        new_order_mask = (scores[order[1:]] > score_threshold).nonzero(as_tuple=False).squeeze()
        if new_order_mask.numel() == 0:
            break
        if new_order_mask.dim() == 0:
            new_order_mask = new_order_mask.unsqueeze(0)

        # 找到最高分框并移到最前面
        max_score_idx = torch.argmax(scores[order[new_order_mask + 1]])
        if max_score_idx != 0:
            new_order_mask[[0, max_score_idx]] = new_order_mask[[max_score_idx, 0]]

        order = order[new_order_mask + 1]

    return torch.LongTensor(keep)


def soft_nms_batch(bboxes, scores, class_ids, iou_thresh=0.5, sigma=0.5,
                   score_threshold=0.25):
    """批处理 Soft-NMS (按类别分别执行)

    参数:
        bboxes (torch.Tensor): 边界框, shape (N, 4), 格式 (x1, y1, x2, y2)
        scores (torch.Tensor): 分数, shape (N,)
        class_ids (torch.Tensor): 类别 ID, shape (N,)
        iou_thresh (float): IoU 阈值
        sigma (float): 高斯衰减系数
        score_threshold (float): 分数阈值

    返回:
        torch.LongTensor: 保留框的索引, shape (K,)
        torch.Tensor: 更新后的分数, shape (N,)
    """
    keep_all = []
    unique_classes = class_ids.unique()

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_indices = cls_mask.nonzero(as_tuple=False).squeeze()

        if cls_indices.dim() == 0:
            cls_indices = cls_indices.unsqueeze(0)

        cls_bboxes = bboxes[cls_indices]
        cls_scores = scores[cls_indices]

        if cls_bboxes.shape[0] == 0:
            continue

        cls_keep = soft_nms(cls_bboxes, cls_scores, iou_thresh, sigma, score_threshold)
        keep_all.append(cls_indices[cls_keep])

    if len(keep_all) == 0:
        return torch.LongTensor([])

    return torch.cat(keep_all)
