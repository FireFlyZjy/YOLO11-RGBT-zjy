"""
RT-SFOD Training Strategies
============================
论文: RT-SFOD (ECCV 2026)
来源: RT-SFOD/scripts/YOLO26/stage2_rtsfod_yolo26.py

提取的核心训练策略 (改造为可复用组件):
    1. DHF (Dual-Head pseudo-label Fusion) - 双头伪标签融合
    2. MARD (Multi-scale Adaptive Representation Diversification) - 多尺度自适应表示多样化
    3. Variance Loss - 方差损失，鼓励特征多样性
    4. Covariance Loss - 协方差损失，减少特征冗余

对 RGBT 的价值:
    - DHF: 利用双头输出提升伪标签质量，适合半监督双模态学习
    - MARD: 在多尺度特征上鼓励多样性，提升特征区分能力
    - Variance/Covariance Loss: 可作为正则化损失，提升特征质量

与已有损失函数的区别:
    - 标准 YOLO Loss: CIoU + BCE_cls + BCE_conf
    - Cross Modal Consistency Loss: 跨模态一致性损失
    - RT-SFOD Loss: 特征多样性正则化损失

用法:
    variance_loss: 方差损失
    covariance_loss: 协方差损失
    mard_loss: MARD 多尺度损失 (组合方差+协方差)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def variance_loss(tokens: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Variance Loss: 方差损失，鼓励特征多样性

    机制: 计算特征的标准差，如果低于阈值 gamma 则惩罚
    对 RGBT 的价值: 鼓励不同模态的特征保持多样性，避免坍缩

    参数:
        tokens: 特征 tokens [N, C]
        gamma: 方差阈值 (默认 1.0)
        eps: 数值稳定性 (默认 1e-4)
    """
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    std = torch.sqrt(tokens.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(tokens: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Covariance Loss: 协方差损失，减少特征冗余

    机制: 计算特征的协方差矩阵，惩罚非对角线元素
    对 RGBT 的价值: 减少特征冗余，提升特征效率

    参数:
        tokens: 特征 tokens [N, C]
        eps: 数值稳定性 (默认 1e-4)
    """
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    n, c = tokens.shape
    z = tokens - tokens.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + eps)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / (c * (c - 1) + 1e-6)


class MARD_Loss(nn.Module):
    """
    MARD_Loss: Multi-scale Adaptive Representation Diversification Loss

    机制:
        1. 在多尺度特征上采样前景和背景 token
        2. 计算方差损失 (鼓励多样性)
        3. 计算协方差损失 (减少冗余)
        4. 加权求和

    对 RGBT 的价值:
        - 在 P3/P4/P5 多尺度上应用特征多样性正则化
        - 提升特征的区分能力，有利于检测

    用法: 作为辅助损失，在训练时添加到总损失中

    参数:
        alpha: 方差损失权重 (默认 1.0)
        beta: 协方差损失权重 (默认 0.1)
        gamma: 方差阈值 (默认 1.0)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, feats: list, boxes: list = None) -> dict:
        """
        计算 MARD 损失

        参数:
            feats: 多尺度特征列表 [P3, P4, P5]，每个 [B, C, H, W]
            boxes: 检测框列表 (可选，用于采样前景/背景 token)

        返回:
            dict: 包含 total, variance, covariance 损失
        """
        total_var = torch.tensor(0.0, device=feats[0].device)
        total_cov = torch.tensor(0.0, device=feats[0].device)

        for fmap in feats:
            B, C, H, W = fmap.shape
            # 将特征 reshape 为 tokens [B*H*W, C]
            tokens = fmap.permute(0, 2, 3, 1).reshape(-1, C)

            # 随机采样 (如果 token 太多)
            if tokens.shape[0] > 1024:
                idx = torch.randperm(tokens.shape[0], device=tokens.device)[:1024]
                tokens = tokens[idx]

            total_var = total_var + variance_loss(tokens, self.gamma)
            total_cov = total_cov + covariance_loss(tokens)

        loss = self.alpha * total_var + self.beta * total_cov
        return {
            'mard_loss': loss,
            'variance_loss': total_var,
            'covariance_loss': total_cov
        }


class DualHeadFusion(nn.Module):
    """
    DualHeadFusion: 双头伪标签融合 (DHF)

    机制: 利用 YOLO 的 one2one 和 one2many 双头输出
        1. one2one 头: 高置信度锚点
        2. one2many 头: 补充检测
        3. 去重融合: 移除重叠框，保留互补检测

    对 RGBT 的价值:
        - 提升伪标签质量，适合半监督双模态学习
        - 双头互补，提升检测召回率

    参数:
        tau_o2o: one2one 头置信度阈值 (默认 0.5)
        tau_o2m: one2many 头置信度阈值 (默认 0.5)
        tau_no: 新框 IoU 阈值 (默认 0.2)
        tau_dup: 去重 IoU 阈值 (默认 0.7)
    """
    def __init__(self, tau_o2o: float = 0.5, tau_o2m: float = 0.5,
                 tau_no: float = 0.2, tau_dup: float = 0.7):
        super().__init__()
        self.tau_o2o = tau_o2o
        self.tau_o2m = tau_o2m
        self.tau_no = tau_no
        self.tau_dup = tau_dup

    def classwise_nms(self, boxes: torch.Tensor) -> torch.Tensor:
        """类别内 NMS"""
        if boxes.numel() == 0:
            return boxes
        from torchvision.ops import nms
        kept = []
        classes = boxes[:, 5].long().unique(sorted=True)
        for cls in classes:
            idx = torch.where(boxes[:, 5].long() == cls)[0]
            keep = nms(boxes[idx, :4], boxes[idx, 4], self.tau_dup)
            kept.append(boxes[idx[keep]])
        return torch.cat(kept, dim=0) if kept else boxes.new_zeros((0, 6))

    def forward(self, boxes_one2one: torch.Tensor, boxes_one2many: torch.Tensor) -> torch.Tensor:
        """
        融合双头输出

        参数:
            boxes_one2one: one2one 头输出 [N, 6] (x1, y1, x2, y2, conf, cls)
            boxes_one2many: one2many 头输出 [M, 6]

        返回:
            fused: 融合后的检测框 [K, 6]
        """
        device = boxes_one2one.device if boxes_one2one.numel() > 0 else boxes_one2many.device

        # 处理 one2one 头
        if boxes_one2one is None or boxes_one2one.numel() == 0:
            anchors = torch.zeros((0, 6), device=device)
        else:
            anchors = boxes_one2one.to(device)
            anchors = anchors[anchors[:, 4] >= self.tau_o2o]

        # 处理 one2many 头
        if boxes_one2many is None or boxes_one2many.numel() == 0:
            candidates = torch.zeros((0, 6), device=device)
        else:
            candidates = boxes_one2many.to(device)
            candidates = candidates[candidates[:, 4] >= self.tau_o2m]

        # 融合策略
        if candidates.numel() == 0:
            fused = anchors
        elif anchors.numel() == 0:
            extras = self.classwise_nms(candidates)
            fused = extras
        else:
            # 计算 IoU，找出新框
            from torchvision.ops import box_iou
            max_iou = box_iou(candidates[:, :4], anchors[:, :4]).max(dim=1).values
            extras = candidates[max_iou <= self.tau_no]
            extras = self.classwise_nms(extras)
            fused = torch.cat([anchors, extras], dim=0) if extras.numel() else anchors

        # 按置信度排序
        if fused.numel():
            fused = fused[torch.argsort(fused[:, 4], descending=True)]
        return fused
