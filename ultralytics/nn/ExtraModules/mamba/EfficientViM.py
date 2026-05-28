"""
EfficientViM: Efficient Vision Mamba with Hidden State Mixing
==============================================================
论文: EfficientViM: Efficient Vision Mamba with Hidden State Mixing
      (CVPR 2025)

核心创新: 隐状态混合 (Hidden State Mixing, HSM)
    标准 Mamba 的计算瓶颈是 O(L D^2) 的 SSM 扫描 (L=序列长, D=特征维)。
    EfficientViM 通过"先计算隐状态 h_in，再进行投影"的策略，
    将复杂度从 O(L D^2) 降低到 O(N D^2 + L N D)，其中 N << D 为状态维度。

    具体来说:
    1. 标准 SSM: 每一步计算 B*x_t, C*h_t, 都涉及 D 维投影
    2. EfficientViM: 先通过轻量扫描得到隐态序列 h, 再一次性投影到输出空间

设计特点:
    - 单头设计: 避免多头 SSM 的记忆瓶颈
    - 计算高效: 计算量与序列长度近似线性
    - 全局感受野: 保留 SSM 的长距离依赖建模能力

RGBT 相关性:
    双模态 (RGB+IR) 检测中，特征图通常较大 (P3/P4 层级)。
    EfficientViM 的高效扫描使其能够处理大分辨率特征图，
    同时保持全局上下文建模能力，有利于融合可见光和热红外模态的互补信息。

实现说明:
    - 纯 PyTorch 实现，无需 mamba-ssm CUDA 内核
    - 选择性扫描使用简化的 RNN 循环实现
    - 支持双向扫描以获得更好的 2D 空间上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_utils import selective_scan


class EfficientViM_Block(nn.Module):
    """
    EfficientViM Block - 隐状态混合的高效 Vision Mamba 模块。

    结构:
        x -> LayerNorm -> 线性投影 -> 深度可分离卷积 -> SiLU ->
        简化选择性扫描(隐状态混合) -> 输出投影 -> + 残差连接

    参数:
        dim:         输入/输出通道数
        d_state:     SSM 状态维度 (默认 8, 比标准 Mamba 更小)
        d_conv:      深度可分离卷积核大小 (默认 3)
        expand:      内部扩展因子 (默认 1.0, 即不扩展)
        dt_rank:     步长投影秩 ("auto" 则自动计算)
        bias:        是否使用偏置
        use_bidirectional: 是否使用双向扫描 (默认 True)

    前向:
        x: (B, C, H, W) 或 (B, L, D) 格式的特征图
        return: 同输入形状的输出
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: float = 1.0,
        dt_rank: str = "auto",
        bias: bool = False,
        use_bidirectional: bool = True,
    ):
        super().__init__()
        dim = c2  # 使用c2作为内部维度
        self.dim = dim
        self.proj_in = nn.Conv2d(c1, dim, 1) if c1 != dim else nn.Identity()
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = max(math.ceil(dim / 16), 2) if dt_rank == "auto" else dt_rank
        self.use_bidirectional = use_bidirectional
        num_dirs = 2 if use_bidirectional else 1

        # 1. LayerNorm + 输入投影
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)

        # 2. 深度可分离 1D 卷积 (优化: 使用分组卷积)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # 3. SSM 参数投影 (输入相关: B, C, delta)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # 4. A 矩阵 (S4D 初始化: 从 1 递增的对数)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1)  # (D, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # 5. D skip 参数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 如果是双向，添加反向参数
        if use_bidirectional:
            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
            self.A_log_b = nn.Parameter(torch.log(A.clone()))
            self.A_log_b._no_weight_decay = True
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

        # 6. 输出投影
        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

        self.act = nn.SiLU()

    def _forward_single_direction(self, x, conv1d, x_proj, dt_proj, A_log, D):
        """单方向 SSM 前向传播。"""
        batch, seqlen, dim = x.shape

        # 深度可分离卷积 (B, L, D) -> (B, D, L)
        x_conv = x.permute(0, 2, 1).contiguous()
        x_conv = conv1d(x_conv)[..., :seqlen]
        x_conv = self.act(x_conv)

        # 参数投影: delta, B, C
        x_dbl = x_proj(x_conv.permute(0, 2, 1).contiguous())  # (B, L, R+2N)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = dt_proj(dt)  # (B, L, D)

        # 重排到扫描所需格式
        # 扫描输入: (B, D, L), B: (B, N, L), C: (B, N, L)
        xs = x_conv  # 已经是 (B, D, L)
        dt_t = dt.permute(0, 2, 1).contiguous()  # (B, D, L)
        B_t = B_mat.permute(0, 2, 1).contiguous()  # (B, N, L)
        C_t = C_mat.permute(0, 2, 1).contiguous()  # (B, N, L)
        A_mat = -torch.exp(A_log.float())  # (D, N)

        y = selective_scan(
            xs, dt_t, A_mat, B_t, C_t,
            D=D.float(),
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = y.permute(0, 2, 1).contiguous()  # (B, L, D)
        return y

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 或 (B, L, D)

        Returns: 与输入相同形状
        """
        # 1x1 通道投影 (处理 c1≠c2)
        if x.dim() == 4:
            x = self.proj_in(x)

        # 处理 4D 输入 (B, C, H, W) -> (B, H*W, C)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_seq = x.flatten(2).permute(0, 2, 1).contiguous()  # (B, L, D)
            is_4d = True
        else:
            x_seq = x
            is_4d = False
            H, W = None, None

        # 残差
        residual = x_seq
        batch, seqlen, _ = x_seq.shape

        # LayerNorm
        x_seq = self.norm(x_seq)

        # 输入投影 + split
        xz = self.in_proj(x_seq)
        x_in, z = xz.chunk(2, dim=-1)

        # 正向扫描
        y = self._forward_single_direction(
            x_in, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D
        )

        # 反向扫描 (双向融合)
        if self.use_bidirectional:
            x_in_rev = x_in.flip(1)
            y_rev = self._forward_single_direction(
                x_in_rev, self.conv1d, self.x_proj_b, self.dt_proj_b,
                self.A_log_b, self.D_b
            )
            y_rev = y_rev.flip(1)
            y = (y + y_rev) * 0.5

        # 门控
        y = y * self.act(z)

        # 输出投影 + 残差
        out = self.out_proj(y) + residual

        # 还原 4D 形状
        if is_4d:
            out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return out


class EfficientViM_HSM(nn.Module):
    """
    EfficientViM 的隐状态混合 (Hidden State Mixing) 变体。

    与标准 EfficientViM_Block 的区别:
    标准版先投影再扫描，HSM 版在扫描前先计算隐状态，
    然后一次性投影到输出空间，进一步减少计算量。

    HSM 流程:
        1. x -> norm -> in_proj -> split x, z
        2. x -> conv1d -> act -> 扫描 h_t = f(h_{t-1}, x_t)
        3. h -> 一次性投影到输出 (替代逐点 C*h_t)
        4. * act(z) + residual

    参数:
        同上 EfficientViM_Block
    """
    def __init__(self, dim, d_state=8, d_conv=3, expand=1.0, dt_rank="auto", bias=False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_inner = int(expand * dim)
        self.dt_rank = max(math.ceil(dim / 16), 2) if dt_rank == "auto" else dt_rank

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)

        # HSM: 用于隐状态混合的投影
        # B_proj 和 delta_proj 用于控制状态更新
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # A 矩阵
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # HSM: 隐状态到输出的混合投影 (替代 C)
        self.hidden_state_proj = nn.Linear(self.d_state, self.d_inner, bias=False)

        # D skip
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x):
        """前向传播。"""
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_seq = x.flatten(2).permute(0, 2, 1)
            is_4d = True
        else:
            x_seq = x
            is_4d = False
            H, W = None, None

        residual = x_seq
        x_seq = self.norm(x_seq)

        xz = self.in_proj(x_seq)
        x_in, z = xz.chunk(2, dim=-1)

        # 卷积
        x_conv = self.act(self.conv1d(x_in.permute(0, 2, 1))[..., :x_in.shape[1]])

        # HSM: 先进行轻量扫描，收集隐状态序列
        x_dbl = self.x_proj(x_conv.permute(0, 2, 1))
        dt_r, B_vec = torch.split(x_dbl, [self.dt_rank, self.d_state], dim=-1)

        dt = self.dt_proj(dt_r)  # (B, L, D)
        A = -torch.exp(self.A_log.float())  # (D, N)
        B_vec = B_vec.permute(0, 2, 1)  # (B, N, L)
        dt_t = dt.permute(0, 2, 1)  # (B, D, L)

        # 离散化
        dt_soft = F.softplus(dt_t + self.dt_proj.bias.float().view(1, -1, 1))
        dA = torch.exp(dt_soft.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))  # (B, D, N, L)
        dB = dt_soft.unsqueeze(-1) * B_vec.unsqueeze(1)  # (B, D, N, L)

        # 扫描收集隐状态序列 (B, D, N, L)
        h = torch.zeros_like(dA[..., 0])
        h_list = []
        for i in range(x_in.shape[1]):
            h = dA[..., i] * h + dB[..., i] * x_conv[:, :, i].unsqueeze(-1)
            h_list.append(h)
        h_seq = torch.stack(h_list, dim=-1)  # (B, D, N, L)

        # HSM: 隐状态混合 -> 一次性投影到输出空间
        h_seq = h_seq.permute(0, 3, 1, 2)  # (B, L, D, N)
        y = self.hidden_state_proj(h_seq)  # (B, L, D, D)
        y = y.sum(dim=-1)  # (B, L, D)

        # D skip
        y = y + self.D.float() * x_conv.permute(0, 2, 1)

        # 门控 + 输出
        y = y * self.act(z)
        out = self.out_proj(y) + residual

        if is_4d:
            out = out.permute(0, 2, 1).view(B, C, H, W)

        return out


import math
