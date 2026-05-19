import torch
import torch.nn as nn

class CIFusion(nn.Module):
    """CIFusion: Cross-modality Interaction Fusion (跨模态交叉交换通道注意力融合)
    论文: CMFADet - A Cross-Modality Feature Adaptive Interaction Approach for RGB-IR Detection
    核心机制: 对拼接的RGB+IR特征做通道注意力 + 交叉交换残差,
             让RGB注意力门控后的特征放到IR位置(反之亦然),实现双向信息传递
    """
    def __init__(self, c1, r=16):
        """c1: 单个模态的通道数 (两个模态通道数必须相同)"""
        super().__init__()
        self.c1_single = c1
        self.c_total = c1 * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.c_total, self.c_total // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c_total // r, self.c_total, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: concatenated [RGB, IR], shape (B, c1*2, H, W)
        b = x.size(0)
        y = self.avg_pool(x).view(b, self.c_total)
        y = self.fc(y).view(b, self.c_total, 1, 1)
        x1 = x * y
        # 交叉交换: RGB注意力特征 -> IR位置, IR注意力特征 -> RGB位置
        return x + torch.cat((x1[:, self.c1_single:, ...], x1[:, :self.c1_single, ...]), dim=1)
