"""
SPDConv — 空间到深度卷积 (Space-to-Depth Conv)

来源: GeoFuse-YOLO (SOEP: Small Object Enhance Pyramid)

机制:
    空间到深度转换将 H×W 的特征图分解为 4 个 (H//2)×(W//2) 的子特征,
    拼接后通道变为 4C, 经过 3×3 卷积处理, 再上采样恢复原分辨率,
    最后与 1×1 残差连接相加。保留了更多空间细节信息。

对RGBT的价值:
    小目标(远距离行人/车辆)在IR模态中特征稀疏, SPD通过空间到深度
    转换保留了更丰富的空间位置信息, 有助于小目标检测。

用法:
    [-1, 1, SPDConv, [c2]]
"""

import torch
import torch.nn as nn

from ..modules.conv import Conv


class SPDConv(nn.Module):
    """SPDConv: 空间到深度卷积, 保持分辨率的小目标增强模块

    机制: 空间到深度转换(4x通道) → 3x3卷积 → 上采样恢复 → 残差连接
    对RGBT的价值: 小目标在IR中特征稀疏, SPD保留更多空间细节
    用法: [-1, 1, SPDConv, [c2]]

    Args:
        c1: 输入通道数 (由parse_model自动注入)
        c2: 输出通道数
    """

    def __init__(self, c1, c2):
        super().__init__()
        # 空间到深度路径: 4*c1 → c2
        self.spd_conv = nn.Sequential(
            Conv(c1 * 4, c2, k=3, s=1)
        )
        # 残差路径: c1 → c2
        self.residual = Conv(c1, c2, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        B, C, H, W = x.shape
        residual = self.residual(x)

        # 空间太小时直接返回残差 (P5层1x1保护)
        if H < 2 or W < 2:
            return residual

        # 确保偶数尺寸
        if H % 2 != 0 or W % 2 != 0:
            H = H - (H % 2)
            W = W - (W % 2)
            x = x[:, :, :H, :W]

        # 空间到深度: 4个子特征拼接, 通道变为4C
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        spd_features = torch.cat([x0, x1, x2, x3], dim=1)

        # 3x3卷积处理 + 上采样恢复分辨率
        spd_out = self.spd_conv(spd_features)
        spd_out = self.upsample(spd_out)

        return spd_out + residual
