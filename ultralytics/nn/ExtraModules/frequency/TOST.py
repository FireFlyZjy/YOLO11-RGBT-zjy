import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TOST(nn.Module):
    """TOST (TSSA): Token Statistics Self-Attention / 令牌统计自注意力

    论文: ICLR2025 - Token Statistics Transformer: Linear Attention via
          Second-Order Statistics

    核心机制:
      标准 Self-Attention:  softmax(Q @ K^T / sqrt(d)) @ V  →  O(N^2)
      TOST (统计注意力):    phi(Q) @ (phi(K)^T @ V)         →  O(N)

      - 使用 element-wise ReLU 作为核函数 phi (Linear Attention 技巧)
      - 将 Key-Value 的交互从 NxN 降低到 DxD (维度固定, 与 N 无关)
      - 计算步骤:
        1. Q, K, V 通过 1x1 Conv 投影到低维多头空间
        2. phi = ReLU (非负特征映射, 保证线性注意力数值稳定)
        3. KV_agg = phi(K)^T @ V:  先聚合 K 和 V (O(N*D^2))
        4. O = phi(Q) @ KV_agg:    再用 Q 查询聚合结果 (O(N*D^2))
        5. 整体复杂度 O(N), 实际上与 N 呈线性关系

    为什么适用于 RGBT 检测:
      - 全局上下文建模能力类似 Self-Attention, 但计算量线性
      - 适合 YOLO 中的多尺度特征 (80x80 也毫无压力)
      - 可以替换 AIFI 作为 Transformer 层, 速度更快

    YAML 使用示例:
      - [26, TOST, [256, 256, 8, 0.25]]   # 8头, 压缩比0.25
      - [26, TOST, [512, 256, 4]]          # 替代 C2f 的 bottleneck
    """

    def __init__(self, c1, c2, num_heads=8, ratio=0.25):
        """Initialize TOST.
        Args:
            c1: 输入通道数
            c2: 输出通道数
            num_heads: 注意力头数 (default=8), c2 必须可被 num_heads 整除
            ratio: QK 投影维度压缩比 (default=0.25)
        """
        super().__init__()
        assert c2 % num_heads == 0, f"c2 ({c2}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = c2 // num_heads
        self.scale = self.head_dim ** -0.5

        # QK 投影到低维空间 (降低 D 减小 D² 开销)
        self.qk_dim = max(int(c1 * ratio), num_heads * 2)  # 确保可分给各头

        # 多头 QK 投影
        self.qk_conv = nn.Conv2d(c1, self.qk_dim * 2, 1, bias=False)
        # 多头 V 投影
        self.v_conv = nn.Conv2d(c1, c2, 1, bias=False)

        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        """前向传播: O(n) 线性统计注意力.

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            out: 输出特征 (B, C2, H, W)
        """
        B, C, H, W = x.shape
        N = H * W  # 令牌数

        # ===== 1. QKV 投影 =====
        qk = self.qk_conv(x)  # (B, qk_dim*2, H, W)
        v = self.v_conv(x)    # (B, c2, H, W)

        d = self.qk_dim
        q = qk[:, :d]   # (B, d, H, W)
        k = qk[:, d:]   # (B, d, H, W)

        # ===== 2. Reshape 为多头 =====
        # q/k: (B, num_heads, d_per_head, N)
        q = q.view(B, self.num_heads, d // self.num_heads, N)
        k = k.view(B, self.num_heads, d // self.num_heads, N)
        # v: (B, num_heads, head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        # ===== 3. ReLU 特征映射 (核函数) =====
        # 保证非负, 使线性注意力在数值上稳定
        q_phi = F.relu(q)   # (B, H, D_k, N)
        k_phi = F.relu(k)   # (B, H, D_k, N)

        # ===== 4. O(n) 线性注意力 =====
        # 不计算 NxN 注意力矩阵
        # Step A: KV_agg = phi(K)^T @ V  (先聚合 K 和 V)
        #   k_phi: (B, H, D_k, N), v: (B, H, D_v, N)
        #   k_phi @ v^T = (B, H, D_k, N) @ (B, H, N, D_v) = (B, H, D_k, D_v)
        kv_agg = torch.matmul(k_phi, v.transpose(-2, -1))  # (B, H, D_k, D_v)

        # Step B: O = phi(Q) @ KV_agg / N (除以 N 做均值归一化)
        #   q_phi: (B, H, D_k, N), kv_agg: (B, H, D_k, D_v)
        #   q_phi^T @ kv_agg = (B, H, N, D_k) @ (B, H, D_k, D_v) = (B, H, N, D_v)
        # 但 q_phi^T @ kv_agg 在内存中不小, 改用 einsum 更清晰
        out = torch.einsum("bhdn,bhdv->bhnv", q_phi, kv_agg)  # (B, H, N, D_v)

        # Normalization: 防止值过大
        # Z = phi(Q) @ sum(phi(K), dim=-1)  逐位置归一化因子
        z = torch.einsum("bhdn,bhd->bhn", q_phi, k_phi.sum(dim=-1))  # (B, H, N)
        out = out / (z.unsqueeze(-1) + 1e-6)  # (B, H, N, D_v)

        # ===== 5. Reshape 回图像格式 =====
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, C)
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)  # (B, C, H, W)

        # ===== 6. 输出投影 =====
        out = self.proj(out)
        return out

    def __repr__(self):
        return (f"TOST(c_out={self.v_conv.out_channels}, "
                f"heads={self.num_heads}, qk_dim={self.qk_dim})")
