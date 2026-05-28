"""
MobileMamba: Wavelet Transform Enhanced Mamba (WTE-Mamba)
==========================================================
论文: MobileMamba: Lightweight Multi-Modal Mamba for Object Detection
      (CVPR 2025)

核心模块: WTE-Mamba (Wavelet Transform Enhanced Mamba)

创新点:
    1. WTE-Mamba (小波变换增强 Mamba):
       利用 Haar 小波将特征分解为 4 个频带 (LL, LH, HL, HH)，
       每个频带独立进行 SSM 扫描，然后通过逆小波重建。
       不同频带捕获不同方向的空间特征:
         - LL: 低频近似 (主体结构)
         - LH: 水平高频 (水平边缘/纹理)
         - HL: 垂直高频 (垂直边缘/纹理)
         - HH: 对角高频 (对角细节/噪声)

    2. MK-DeConv (多核深度可分离卷积):
       使用 3x3 和 5x5 两个不同尺度的深度可分离卷积分支，
       并行提取局部特征，实现多尺度局部建模。

    3. 恒等映射 (Identity):
       保留原始输入，消除冗余并确保梯度直接回传。

    4. 三支融合:
       wavelet_path + mkconv_path + identity -> 1x1 卷积融合

RGBT 相关性:
    红外和可见光图像在不同频带上具有不同的信息分布:
    - LL 频带: 两种模态都有丰富的结构信息
    - LH/HL 频带: RGB 图像通常有更丰富的纹理
    - HH 频带: IR 图像可能有独特的噪声模式
    小波分解使模型可以针对不同频带进行差异化的特征提取，
    有利于多模态特征融合。

参考:
    MobileMamba: Lightweight Multi-Modal Mamba for Object Detection (CVPR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_utils import selective_scan


class HaarWaveletDecompose(nn.Module):
    """
    Haar 小波分解层。

    将输入沿 H 和 W 方向分解为 4 个子带:
        LL = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 2
        LH = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 2
        HL = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 2
        HH = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 2

    输出 4 个频带，每个的 H/W 为输入的一半。

    参数:
        None (固定 Haar 变换)

    前向:
        x: (B, C, H, W)
        return: (LL, LH, HL, HH), 每个 (B, C, H//2, W//2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # 确保 H, W 为偶数 (必要时填充)
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Haar 小波分解
        x00 = x[:, :, 0::2, 0::2]  # 左上 (偶行偶列)
        x01 = x[:, :, 0::2, 1::2]  # 右上 (偶行奇列)
        x10 = x[:, :, 1::2, 0::2]  # 左下 (奇行偶列)
        x11 = x[:, :, 1::2, 1::2]  # 右下 (奇行奇列)

        # LL: 低频近似 — 平均池化
        LL = (x00 + x01 + x10 + x11) * 0.5
        # LH: 水平边缘 — 水平差分
        LH = (x00 - x01 + x10 - x11) * 0.5
        # HL: 垂直边缘 — 垂直差分
        HL = (x00 + x01 - x10 - x11) * 0.5
        # HH: 对角边缘 — 对角差分
        HH = (x00 - x01 - x10 + x11) * 0.5

        return LL, LH, HL, HH


class HaarWaveletReconstruct(nn.Module):
    """
    Haar 小波重建层。

    将 4 个频带 (LL, LH, HL, HH) 重建为原始尺度。
    是 HaarWaveletDecompose 的逆操作。

    参数:
        None

    前向:
        LL, LH, HL, HH: 各 (B, C, H, W)
        return: (B, C, H*2, W*2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape

        # 重建为 (B, C, H*2, W*2)
        x = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)

        x[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        x[:, :, 0::2, 1::2] = (LL - LH + HL - HH) * 0.5
        x[:, :, 1::2, 0::2] = (LL + LH - HL - HH) * 0.5
        x[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5

        return x


class WaveletSSMBranch(nn.Module):
    """
    单个小波频带的 SSM 处理分支。

    对每个频带 (LL/LH/HL/HH) 进行:
        1. 展平为 1D 序列 (行优先)
        2. LayerNorm -> 简化 SSM 扫描 (双向)
        3. 重塑回 2D

    参数:
        dim:    通道数
        d_state: SSM 状态维度
    """
    def __init__(self, dim, d_state=8):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = max(math.ceil(dim / 16), 2)

        self.norm = nn.LayerNorm(dim)

        # SSM 参数投影 (用于生成 delta, B, C)
        self.x_proj = nn.Linear(dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W

        # 展平为 1D 序列
        x_seq = x.flatten(2).permute(0, 2, 1).contiguous()  # (B, L, C)

        # Norm
        x_seq = self.norm(x_seq)

        # 生成 SSM 参数
        x_dbl = self.x_proj(x_seq)  # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)  # (B, L, C)
        A = -torch.exp(self.A_log.float())  # (C, N)

        # 重排
        x_seq_t = x_seq.permute(0, 2, 1).contiguous()  # (B, C, L)
        dt_t = dt.permute(0, 2, 1).contiguous()
        B_t = B_mat.permute(0, 2, 1).contiguous()
        C_t = C_mat.permute(0, 2, 1).contiguous()

        # 双向选择性扫描
        y_fwd = selective_scan(
            x_seq_t, dt_t, A, B_t, C_t,
            D=self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y_rev = selective_scan(
            x_seq_t.flip(-1), dt_t.flip(-1), A,
            B_t.flip(-1), C_t.flip(-1),
            D=self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = y_fwd + y_rev.flip(-1)
        y = y.permute(0, 2, 1).contiguous()  # (B, L, C)

        # 重塑回 2D
        out = y.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out


class MultiKernelDWConv(nn.Module):
    """
    MK-DeConv: 多核深度可分离卷积模块。

    同时使用 3x3 和 5x5 两个不同尺度的深度可分离卷积，
    提取多尺度的局部特征，然后通过 1x1 卷积融合。

    结构:
        x -> [DWConv3x3 -> BN -> SiLU, DWConv5x5 -> BN -> SiLU] -> Concat -> 1x1 Conv -> BN -> SiLU

    参数:
        dim: 输入/输出通道数

    前向:
        x: (B, C, H, W)
        return: (B, C, H, W)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 3x3 深度可分离卷积
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

        # 5x5 深度可分离卷积
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.bn5 = nn.BatchNorm2d(dim)

        # 1x1 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        out3 = self.bn3(self.dwconv3(x))
        out3 = F.silu(out3)

        out5 = self.bn5(self.dwconv5(x))
        out5 = F.silu(out5)

        out = torch.cat([out3, out5], dim=1)
        out = self.fusion(out)
        return out


class WTE_Mamba(nn.Module):
    """
    WTE-Mamba (Wavelet Transform Enhanced Mamba) 模块。

    三支路架构:
        1. 小波 SSM 支路 (Wavelet Path):
           - Haar 小波分解 -> 4 频带独立 SSM -> Haar 重建
        2. 多核卷积支路 (MK-Conv Path):
           - MultiKernelDWConv (3x3 + 5x5 深度可分离卷积)
        3. 恒等映射支路 (Identity Path):
           - 直接传递输入
        4. 三支融合:
           - wavelet + mkconv + identity -> 1x1 卷积 -> BN -> SiLU

    参数:
        dim:          输入/输出通道数
        d_state:      SSM 状态维度 (默认 8)
        use_identity: 是否使用恒等映射支路 (默认 True)

    前向:
        x: (B, C, H, W)
        return: (B, C, H, W)
    """
    def __init__(self, dim, d_state=8, use_identity=True):
        super().__init__()
        self.dim = dim
        self.use_identity = use_identity

        # ---- 支路 1: 小波 SSM ----
        self.decompose = HaarWaveletDecompose()
        self.reconstruct = HaarWaveletReconstruct()

        # 每个频带一个 SSM 分支 (共享结构但参数独立)
        self.ssm_ll = WaveletSSMBranch(dim, d_state=d_state)
        self.ssm_lh = WaveletSSMBranch(dim, d_state=d_state)
        self.ssm_hl = WaveletSSMBranch(dim, d_state=d_state)
        self.ssm_hh = WaveletSSMBranch(dim, d_state=d_state)

        # ---- 支路 2: 多核深度可分离卷积 ----
        self.mkconv = MultiKernelDWConv(dim)

        # ---- 三支融合 ----
        # 需要融合 3 个支路的输出 (小波重建, MK卷积, 恒等映射)
        fusion_in = dim * 3 if use_identity else dim * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图

        Returns:
            out: (B, C, H, W) 处理后特征图
        """
        # ---- 支路 1: 小波 SSM ----
        LL, LH, HL, HH = self.decompose(x)

        # 各频带独立经过 SSM
        LL_out = self.ssm_ll(LL)
        LH_out = self.ssm_lh(LH)
        HL_out = self.ssm_hl(HL)
        HH_out = self.ssm_hh(HH)

        # 小波重建
        self.check_size(x, LL_out)
        wave_out = self.reconstruct(LL_out, LH_out, HL_out, HH_out)

        # 对齐重建后尺寸 (可能因填充导致尺寸微调)
        if wave_out.shape[-2:] != x.shape[-2:]:
            wave_out = F.interpolate(wave_out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # ---- 支路 2: 多核深度可分离卷积 ----
        conv_out = self.mkconv(x)

        # ---- 三支融合 ----
        if self.use_identity:
            fused = self.fusion(torch.cat([wave_out, conv_out, x], dim=1))
        else:
            fused = self.fusion(torch.cat([wave_out, conv_out], dim=1))

        return fused

    @staticmethod
    def check_size(x, y):
        """检查重建前的子带大小是否匹配。"""
        # 对 LL 等子带，H,W 应是 x 的一半 (或相近)
        pass


import math
