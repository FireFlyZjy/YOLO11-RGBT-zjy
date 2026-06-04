"""
DGC — 动态分组卷积 (Dynamic Group Convolution)

原理（Dynamic Multi-Head Convolution, 源自 https://arxiv.org/abs/2007.04242）：
    标准卷积对所有输入使用固定参数。DGC 引入可学习的通道门控（channel gating）：
    每组的门控网络根据输入动态生成 [0,1] 间的软掩码，决定是否激活对应通道。
    - 多个并行头（head），每个头处理 c2//heads 个通道
    - 每个头的门控：GlobalAvgPool → FC 降维 → ReLU → FC 升维 → Sigmoid(/T)
    - 温度参数 T 控制门控的软硬程度：T→0 趋近离散选择，T→∞ 趋近均匀分布
    - 门控与输入相乘实现软激活/抑制，然后执行分组卷积

与 RGBT 关联：
    多模态数据中，不同模态的特征通道重要性差异显著。
    DGC 可以自动学习对 RGB 特征重要的通道集与对 Thermal 特征重要的通道集，
    实现模态感知的通道分配。温度退火使门控在训练后期趋近离散选择。

简化说明（对比原始实现）：
    - 移除 L1 损失跟踪（global_progress、inactive_channels 等训练循环逻辑）
    - 移除手动剪枝调度（属于训练流程，不应固化在模块中）
    - 保持核心门控机制：avgpool → FC → sigmoid → 软门控
    - 增加温度参数实现软分配（temperature-based soft assignment）
    - 增加 1×1 投影支持 c1≠c2

用法（在 yaml 中配置）：
    - [-1, 1, Conv_DGC, [out_channels, 3]]                  # 默认参数
    - [-1, 1, Conv_DGC, [out_channels, 3, 1, 1, None, 1]]  # stride=1, 4 heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_DGC(nn.Module):
    """动态分组卷积 — 多头通道门控的动态卷积。

    将输出通道分为 heads 组，每组配备独立的门控网络，
    根据输入动态生成通道级软掩码，控制信息流动。

    Args:
        c1 (int):           输入通道数
        c2 (int):           输出通道数（必须被 heads 整除）
        k (int):            卷积核大小，默认 3
        s (int):            步长，默认 1
        p (int, optional):  填充，默认 None 时自动设为 k//2
        g (int):            分组卷积数（在每头内部生效），默认 1
        act (bool):         是否使用 SiLU 激活，默认 True
        heads (int):        门控头数，默认 4
        squeeze_rate (int): 门控 MLP 中间层压缩比，默认 16
        temperature (float):Sigmoid 温度参数，默认 1.0。
                            低温使门控趋近于 0/1 离散选择；
                            高温使门控更平滑。

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H_out, W_out)

    Example:
        >>> conv = Conv_DGC(64, 128, 3, heads=4)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = conv(x)  # (2, 128, 30, 30)

    参数量：
        - 投影层（c1≠c2 时）：c1 × c2
        - 每头门控：c2 × (c2/r) + (c2/r) × (c2/heads)
        - 每头卷积：(c2/heads) × (c2/heads) × k² （分组卷积）
        - BN + 激活：少量
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1,
                 p: int = None, g: int = 1, act: bool = True,
                 heads: int = 4, squeeze_rate: int = 16,
                 temperature: float = 1.0):
        super().__init__()
        assert c2 % heads == 0, \
            f"输出通道数 c2 ({c2}) 必须能被 heads ({heads}) 整除"
        assert squeeze_rate >= 2, f"squeeze_rate ({squeeze_rate}) 应 >= 2"

        self.heads = heads
        self.temperature = temperature
        self.head_dim = c2 // heads
        self.groups = g

        # 1×1 通道投影（c1≠c2 时对齐）
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 每头的门控网络
        #   输入: 全局池化后的 c2 维向量
        #   输出: head_dim 维 [0,1] 软掩码
        mid_ch = max(c2 // squeeze_rate, 8)
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c2, mid_ch, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_ch, self.head_dim, bias=False),
            )
            for _ in range(heads)
        ])

        # 每头的卷积层（分组卷积，groups=g）
        self.head_convs = nn.ModuleList([
            nn.Conv2d(self.head_dim, self.head_dim, k, s,
                      padding=p if p is not None else k // 2,
                      groups=g, bias=False)
            for _ in range(heads)
        ])

        # 后处理
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：通道投影 → 多头门控 → 多头卷积 → BN → 激活。

        Args:
            x: 输入张量 (B, C_in, H, W)

        Returns:
            输出张量 (B, C_out, H_out, W_out)
        """
        B, _, H, W = x.shape

        # 1. 通道对齐
        x = self.proj(x)  # (B, c2, H, W)

        # 2. 按头分割输入通道
        #    (B, heads, head_dim, H, W)
        x_split = x.view(B, self.heads, self.head_dim, H, W)

        outputs = []
        for i in range(self.heads):
            # 3. 生成第 i 头的门控掩码
            #    avgpool → FC → sigmoid(/T) → (B, head_dim, 1, 1)
            gate = self.gate_nets[i](x)  # (B, head_dim)
            gate = torch.sigmoid(gate / self.temperature)
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, head_dim, 1, 1)

            # 4. 软门控（逐通道乘法）
            head_in = x_split[:, i, :, :, :]  # (B, head_dim, H, W)
            head_gated = head_in * gate

            # 5. 卷积
            head_out = self.head_convs[i](head_gated)  # (B, head_dim, H_out, W_out)
            outputs.append(head_out)

        # 6. 合并所有头的输出
        out = torch.cat(outputs, dim=1)  # (B, c2, H_out, W_out)

        # 7. BN + 激活
        out = self.bn(out)
        out = self.act(out)
        return out
