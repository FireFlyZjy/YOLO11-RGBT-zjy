"""
Mamba/SSM 纯 PyTorch 工具函数
==============================
提供不依赖 mamba-ssm CUDA 内核的简单选择性扫描实现，
以及 S6 风格的状态空间模型构建块。

核心机制说明 (SSM / S6):
    Mamba 的核心是选择性状态空间模型 (Selective SSM, S6):
        h_t = A * h_{t-1} + B * x_t          (状态更新)
        y_t = C * h_t + D * x_t              (输出投影)
    其中 A, B, C 是输入相关的 (input-dependent)，实现了"选择性"信息过滤。
    与传统 RNN 不同的是，Mamba 通过并行扫描算法加速训练。

RGBT 相关性:
    红外和可见光图像具有不同的纹理和热辐射特征。
    SSM 的选择性机制可以自适应地关注两种模态中的关键信息，
    在长距离空间依赖建模上比 CNN 更高效，比 Transformer 计算量更小。

参考:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)
    EfficientViM: Efficient Vision Mamba with Hidden State Mixing (CVPR2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 简化选择性扫描 (纯 PyTorch, 无 CUDA 依赖)
# ============================================================

def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """
    简化的选择性扫描操作 (纯 PyTorch RNN 循环实现)。

    这是 S6 模型的串行等价实现，采用显式 for 循环进行递推。
    训练中比 CUDA 并行扫描慢，但功能等价且无需编译。

    Args:
        u:      输入序列, shape (B, D, L)  — 经卷积和激活后的特征
        delta:  步长参数, shape (B, D, L)
        A:      状态转移矩阵, shape (D, N)  — N = d_state
        B:      输入投影矩阵, shape (B, N, L)
        C:      输出投影矩阵, shape (B, N, L)
        D:      skip 连接参数, shape (D,) 或 None
        delta_bias: delta 偏置, shape (D,) 或 None
        delta_softplus: 是否对 delta 应用 softplus 激活

    Returns:
        y: 输出序列, shape (B, D, L)
    """
    batch, dim, seqlen = u.shape
    d_state = A.shape[-1]

    # discretization: softplus(delta + bias)
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta.to(torch.float32))

    # A_bar = exp(delta * A)   [离散化后的状态矩阵]
    # B_bar = delta * B        [离散化后的输入矩阵]
    delta = delta.to(torch.float32)
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    C = C.to(torch.float32)

    # delta: (B, D, L), A: (D, N) -> dA: (B, D, N, L)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(-1))  # (B, D, N, L)
    dB = delta.unsqueeze(-1) * B.unsqueeze(1)  # (B, D, N, L)

    # 串行扫描: h_t = dA_t * h_{t-1} + dB_t * u_t
    h = torch.zeros(batch, dim, d_state, device=u.device, dtype=torch.float32)
    ys = []
    for i in range(seqlen):
        h = dA[..., i] * h + dB[..., i] * u[:, :, i].unsqueeze(-1)  # (B, D, N)
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
    y = torch.stack(ys, dim=-1)  # (B, D, L)

    # skip 连接
    if D is not None:
        y = y + D.to(torch.float32).unsqueeze(0).unsqueeze(-1) * u.to(torch.float32)

    return y.to(u.dtype)


def bidirectional_selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """
    双向选择性扫描。

    在序列的正向和反向分别进行 SSM 扫描，然后将两个方向的结果相加。
    这使得每个位置都能同时利用前后上下文信息，对二维图像特征尤为重要。

    Args:
        参数同上。

    Returns:
        y: 双向融合输出, shape (B, D, L)
    """
    fwd = selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus)
    rev = selective_scan(
        u.flip(-1), delta.flip(-1), A,
        B.flip(-1), C.flip(-1),
        D, delta_bias, delta_softplus
    )
    return fwd + rev.flip(-1)


# ============================================================
# SSM 核心构建块 (S6 风格)
# ============================================================

class SSMCore(nn.Module):
    """
    S6 风格的状态空间模型核心模块 (纯 PyTorch)。

    结构:
        1. 输入线性投影: x -> x 和 z (门控)
        2. 深度可分离 1D 卷积 (局部上下文)
        3. SiLU 激活
        4. 参数投影: A, B, C, delta (输入相关)
        5. 选择性扫描 (RNN 循环)
        6. 门控输出: y * SiLU(z)
        7. 输出投影

    参数:
        d_model:    模型/通道维度
        d_state:    状态维度 (默认 16)
        d_conv:     卷积核大小 (默认 3)
        expand:     内部扩展因子 (默认 2)
        dt_rank:    步长投影秩 ("auto" 或整数)
        bias:       是否使用偏置
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(math.ceil(d_model / 16), 2) if dt_rank == "auto" else dt_rank

        # 输入投影: x -> [x, z]   (split later)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 深度可分离 1D 卷积 (沿序列维度)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # SSM 参数投影: 输入相关 B, C, delta
        # 这里 x 经过投影生成 dt_rank + 2*d_state 维度的向量
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化 dt_proj 权重
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # A 矩阵: 按 S4D 初始化 - 从 1 到 d_state 的对数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D skip 连接
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: (B, L, D)  序列化的特征

        Returns:
            out: (B, L, D)
        """
        batch, seqlen, dim = x.shape

        # 1. 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 2. 深度可分离 1D 卷积 (B, L, D) -> (B, D, L)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv1d(x)[..., :seqlen]
        x = self.act(x)

        # 3. 参数投影
        x_dbl = self.x_proj(x.permute(0, 2, 1).contiguous())  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # dt: (B, L, dt_rank) -> (B, D, L)
        dt = self.dt_proj(dt)  # (B, L, D)
        dt = dt.permute(0, 2, 1).contiguous()  # (B, D, L)

        # B, C: (B, L, N) -> (B, N, L)
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        A = -torch.exp(self.A_log.float())  # (D, N)

        # 4. 选择性扫描
        y = selective_scan(
            x, dt, A, B, C,
            D=self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # 5. 门控
        y = y.permute(0, 2, 1).contiguous()  # (B, L, D)
        y = y * self.act(z)

        # 6. 输出投影
        out = self.out_proj(y)
        return out


class BidirectionalSSMCore(nn.Module):
    """
    双向 S6 状态空间模型核心。

    在序列的正反两个方向分别执行 SSM 扫描，融合双向上下文信息。
    对于图像特征 (2D 展平为 1D)，双向扫描可以同时利用左右邻域信息。

    参数:
        同 SSMCore
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(math.ceil(d_model / 16), 2) if dt_rank == "auto" else dt_rank

        # 共享输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 正反向各一个卷积 (参数独立)
        self.conv1d_fwd = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.conv1d_rev = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)

        # 正反向 SSM 参数
        self.x_proj_fwd = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_rev = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_rev = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A 矩阵初始化
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log_fwd = nn.Parameter(torch.log(A))
        self.A_log_rev = nn.Parameter(torch.log(A))
        self.A_log_fwd._no_weight_decay = True
        self.A_log_rev._no_weight_decay = True

        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))
        self.D_rev = nn.Parameter(torch.ones(self.d_inner))
        self.D_fwd._no_weight_decay = True
        self.D_rev._no_weight_decay = True

        # 输出投影 (融合双向后)
        self.out_proj = nn.Linear(self.d_inner * 2, d_model, bias=bias)
        self.act = nn.SiLU()

    def _forward_one_direction(self, x, z, conv1d, x_proj, dt_proj, A_log, D):
        """单方向的前向传播。"""
        batch, seqlen, dim = x.shape
        A = -torch.exp(A_log.float())

        x = x.permute(0, 2, 1).contiguous()
        x = conv1d(x)[..., :seqlen]
        x = self.act(x)

        x_dbl = x_proj(x.permute(0, 2, 1).contiguous())
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = dt_proj(dt).permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        y = selective_scan(
            x, dt, A, B, C,
            D=D.float(),
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = y.permute(0, 2, 1).contiguous()
        return y * self.act(z)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)

        Returns:
            out: (B, L, D)
        """
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        y_fwd = self._forward_one_direction(x, z,
            self.conv1d_fwd, self.x_proj_fwd, self.dt_proj_fwd,
            self.A_log_fwd, self.D_fwd)

        x_rev = x.flip(1)
        z_rev = z.flip(1)
        y_rev = self._forward_one_direction(x_rev, z_rev,
            self.conv1d_rev, self.x_proj_rev, self.dt_proj_rev,
            self.A_log_rev, self.D_rev)
        y_rev = y_rev.flip(1)

        out = torch.cat([y_fwd, y_rev], dim=-1)
        return self.out_proj(out)


# ============================================================
# 2D 扫描辅助
# ============================================================

def window_partition(x, window_size):
    """将 2D 特征划分为非重叠窗口。"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """从窗口还原为 2D 特征。"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


import math
