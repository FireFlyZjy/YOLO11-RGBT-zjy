# CARAFE: Content-Aware ReAssembly of FEatures
# 论文: https://arxiv.org/abs/1905.02188
# 说明:
#   CARAFE 是一种内容感知的上采样模块, 会为每个目标像素预测一个独立的
#   上采样核 (而不是固定的双线性/最近邻插值), 从而实现更精细的特征重建。
#   包含两个阶段:
#     1) 核预测 (Kernel Prediction): 对输入特征图压缩 → 编码 → PixelShuffle → Softmax
#     2) 特征重组 (Feature Reassembly): 对输入进行最近邻上采样 → unfold 取邻域 → 加权求和
# 用法 (YOLO YAML):
#   [[-1, 1, CARAFE, [c2, k_enc, k_up, c_mid, scale]]]
#   默认: [[-1, 1, CARAFE, [c2, 3, 5, 64, 2]]]
#
# 注意事项:
#   - c_mid 控制核预测阶段的通道压缩量, 越小则计算量越小 (但表示能力下降)
#   - scale 控制上采样倍率 (推荐 2, 目前仅支持等比例上采样)
#   - 输入输出通道数相同 (c → c)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class CARAFE(nn.Module):
    """CARAFE: 内容感知的特征重组上采样模块

    参数:
        c (int): 输入输出通道数
        k_enc (int): 核预测阶段的编码器卷积核大小, 默认 3
        k_up (int): 重组核大小, 默认 5
        c_mid (int): 核预测阶段的中间通道数, 默认 64
        scale (int): 上采样倍率, 默认 2
    """

    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        super().__init__()
        self.scale = scale

        # 阶段1: 核预测
        # 压缩通道: c → c_mid
        self.comp = Conv(c, c_mid, k=1)
        # 编码核权重: c_mid → (scale * k_up)^2
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        # PixelShuffle: 将空间维度扩大 scale 倍, 通道数变为 k_up^2
        self.pix_shf = nn.PixelShuffle(scale)

        # 阶段2: 特征重组
        # 先对输入做最近邻上采样到目标分辨率
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        #  unfold: 提取每个像素的 k_up×k_up 邻域
        self.unfold = nn.Unfold(
            kernel_size=k_up,
            dilation=scale,
            padding=k_up // 2 * scale,
        )

    def forward(self, X):
        """前向传播

        参数:
            X (torch.Tensor): 输入特征图, shape (b, c, h, w)

        返回:
            torch.Tensor: 上采样后的特征图, shape (b, c, h*scale, w*scale)
        """
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        # ---- 核预测 ----
        W = self.comp(X)                           # (b, c_mid, h, w)
        W = self.enc(W)                            # (b, (scale*k_up)^2, h, w)
        W = self.pix_shf(W)                        # (b, k_up^2, h*scale, w*scale)
        W = torch.softmax(W, dim=1)                # 沿通道维做 Softmax 得到权重

        # ---- 特征重组 ----
        X_up = self.upsmp(X)                       # (b, c, h*scale, w*scale)
        X_unf = self.unfold(X_up)                  # (b, c*k_up^2, h*scale, w*scale)
        X_unf = X_unf.view(b, c, k_up ** 2, h_, w_)  # (b, c, k_up^2, h_, w_)

        # 加权求和: einsum('bkhw,bckhw->bchw', [W, X_unf])
        # 即每个目标位置 (h_, w_) 与对应核 W[:, :, h_, w_] 对邻域加权
        X = torch.einsum('bkhw,bckhw->bchw', W, X_unf)
        return X


class CARAFE_Upsample(nn.Module):
    """CARAFE 的 YOLO YAML 友好包装器

    在 YAML 中可用作 nn.Upsample 的替代:
      [[-1, 1, CARAFE_Upsample, [None, 2, 'nearest']]]
    该包装器会自动推断输入通道 c, 然后创建 CARAFE 实例并调用。

    参数:
        c (int): 输入通道数 (由 YOLO parse_model 自动注入)
        scale_factor (int): 上采样倍率, 默认 2
        mode (str): 占位参数 (与 nn.Upsample 接口对齐, 实际使用 CARAFE 方式)
    """

    def __init__(self, c=None, scale_factor=2, mode='nearest'):
        super().__init__()
        self.c = c
        self.scale_factor = scale_factor

    def forward(self, x):
        if not hasattr(self, 'carafe'):
            self.carafe = CARAFE(self.c or x.shape[1], scale=self.scale_factor).to(x.device)
        return self.carafe(x)
