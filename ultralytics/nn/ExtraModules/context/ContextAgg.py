"""
ContextAggregation — 全局上下文聚合模块

原理（Non-local / Global Context）：
    通过 Non-local 风格的自注意力机制捕获全局上下文信息。
    将输入特征映射到 query (a)、key (k)、value (v) 三个空间，
    通过 key-query 相似度加权聚合全局信息，再叠加回原始特征。

机制：
    ContextAggregation:
        - a (attn):  1x1 Conv → Sigmoid，作为空间注意力图
        - k (key):   1x1 Conv → Softmax 沿空间维度，作为全局注意力权重
        - v (value): 1x1 Conv 降维，作为待聚合的值
        - matmul(v @ k): 全局上下文聚合 → 1x1 Conv 恢复通道
        - 输出: x + m(agg) * a（残差 + 空间门控）

    相比标准 Non-local 的优势：
        - 计算量更小（O(C^2) 而非 O(N^2)）
        - 无需 mmcv 依赖，使用标准 Conv-BN

与 RGBT 关联：
    全局上下文建模对多模态检测中理解场景整体分布至关重要，
    可帮助模型对齐 RGB 和 Thermal 模态的全局语义信息。

用法（在 yaml 中配置）：
    - [-1, 1, ContextAggregation, [reduction]]

    reduction 控制中间通道数 = max(in_channels // reduction, 1)，默认 1。
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class ContextAggregation(nn.Module):
    """全局上下文聚合模块（Non-local 风格，无 mmcv 依赖）。

    将输入特征图通过注意力机制聚合全局上下文，再以残差形式加回。

    Args:
        in_channels (int):  输入/输出通道数
        reduction (int):    中间通道压缩比，默认 1（即不压缩）

    Shape:
        Input:  (B, C, H, W)
        Output: (B, C, H, W)

    Example:
        >>> block = ContextAggregation(256, reduction=2)
        >>> x = torch.randn(4, 256, 32, 32)
        >>> out = block(x)  # (4, 256, 32, 32)
    """

    def __init__(self, in_channels: int, reduction: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 1)

        # 使用普通 Conv2d (无BN), 避免 stride 计算时 batch=1 报错
        self.a = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.k = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.v = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.m = nn.Conv2d(self.inter_channels, in_channels, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """初始化权重：a, k, v 层使用 kaiming_normal, m 层初始化为 0。"""
        for m in [self.a, self.k, self.v]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.m.weight, 0)
        if self.m.bias is not None:
            nn.init.constant_(self.m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # 空间注意力: (N, 1, H, W) → sigmoid
        a = self.a(x).sigmoid()

        # Key: (N, 1, H, W) → (N, 1, HW, 1) → softmax 沿 HW
        k = self.k(x).view(N, 1, -1, 1).softmax(dim=2)

        # Value: (N, C_inter, H, W) → (N, 1, C_inter, HW)
        v = self.v(x).view(N, 1, self.inter_channels, -1)

        # 全局上下文聚合: (N, 1, C_inter, HW) @ (N, 1, HW, 1) = (N, C_inter, 1, 1)
        y = torch.matmul(v, k).view(N, self.inter_channels, 1, 1)

        # 投影 + 空间门控 + 残差
        y = self.m(y) * a

        return x + y
