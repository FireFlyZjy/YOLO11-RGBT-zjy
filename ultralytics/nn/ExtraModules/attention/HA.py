"""
HA: Holistic Attention — 全局注意力精炼模块

论文: HVPNet (RGB-T 显著目标检测)
核心机制:
  1. 用高斯核对 attention map 做平滑卷积 (模拟视觉皮层中心-环绕抑制)
  2. min-max 归一化到 [0,1]
  3. 用平滑后的 attention 加权特征图

对 RGBT 价值:
  高斯平滑消除注意力图中的噪声/孤立点，使注意力响应更平滑，
  更适合 RGB-T 中热辐射边缘的模糊过渡区域

用法 (YAML — 单输入, 在 aggregation 后使用):
  - [-1, 1, HA, [c2]]

参数:
  c1: 输入通道数 (同时作为 attention 通道数)
  c2: 输出通道数
  kernel_size: 高斯核大小 (默认 31)
  sigma: 高斯核标准差 (默认 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HA(nn.Module):
    """Holistic Attention — 全局注意力精炼"""

    def __init__(self, c1, c2, kernel_size=31, sigma=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        # 用 PyTorch 生成高斯核 (替代 scipy.stats.norm)
        gk = self._gkern(kernel_size, sigma)
        # 注册为 buffer (非参数, 不参与梯度)
        self.register_buffer('gaussian_kernel', gk.unsqueeze(0).unsqueeze(0))

    def _gkern(self, kernlen=31, nsig=4):
        """生成 2D 高斯核 (纯 PyTorch)"""
        interval = (2 * nsig + 1.) / kernlen
        x = torch.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        # 用 torch.distributions.Normal 的 CDF 替代 scipy.stats.norm.cdf
        from torch.distributions import Normal
        norm = Normal(0, 1)
        kern1d = norm.cdf(x[1:]) - norm.cdf(x[:-1])
        kernel_raw = torch.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _min_max_norm(self, in_):
        """逐样本 min-max 归一化到 [0,1]"""
        b, c, h, w = in_.shape
        max_ = in_.reshape(b, c, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_ = in_.reshape(b, c, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        in_ = in_ - min_
        return in_ / (max_ - min_ + 1e-8)

    def forward(self, x):
        """
        x 是单输入张量 (B, C, H, W), 其中前 c_attn 个通道作为 attention map
        或 x 是单输入张量, HA 作用于 x 整体
        """
        x = self.proj(x)
        # 高斯平滑
        soft_attention = F.conv2d(x, self.gaussian_kernel.expand(x.size(1), -1, -1, -1),
                                  padding=self.gaussian_kernel.size(-1) // 2,
                                  groups=x.size(1))
        soft_attention = self._min_max_norm(soft_attention)
        return x * soft_attention


__all__ = ['HA']
