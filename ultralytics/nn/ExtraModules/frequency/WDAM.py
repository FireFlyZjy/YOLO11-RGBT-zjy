"""
WDAM: Wavelet-based Dual Attention Module
==========================================
来源: wechat.md 中的 WDAM 模块

核心机制:
    1. Haar 小波分解: 将特征分解为 LL(低频) + LH/HL/HH(高频)
    2. 低频分支: 窗口注意力 (Window Attention) + 相对位置偏置 + Shift 操作
    3. 高频分支: 卷积融合 LH、HL，调制 Value
    4. IDWT 逆小波重建 + 残差连接

对 RGBT 的价值:
    红外和可见光图像在不同频带具有不同的信息分布:
    - LL 频带: 两种模态都有丰富的结构信息
    - LH/HL 频带: RGB 图像通常有更丰富的纹理
    - HH 频带: IR 图像可能有独特的噪声模式
    小波分解使模型可以针对不同频带进行差异化的特征提取，
    窗口注意力增强局部建模能力。

与已有模块的区别:
    - FDConv: FFT 频域解耦卷积 (全局频域)
    - vHeat: DCT 热传导扩散 (全局建模)
    - WDAM: DWT 小波域窗口注意力 (多尺度频域 + 局部注意力)

用法:
    WDAM: [-1, 1, WDAM, [c2, num_heads, window_size, shift_size]]
    C2PSA_WDAM: [-1, 1, C2PSA_WDAM, [c2]]

参考:
    原始代码来自 wechat.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HaarWaveletDecompose(nn.Module):
    """
    Haar 小波分解层 (复用自 MobileMamba.py)。

    将输入沿 H 和 W 方向分解为 4 个子带:
        LL = (x00 + x01 + x10 + x11) / 2  # 低频近似
        LH = (x00 - x01 + x10 - x11) / 2  # 水平高频
        HL = (x00 + x01 - x10 - x11) / 2  # 垂直高频
        HH = (x00 - x01 - x10 + x11) / 2  # 对角高频

    输出 4 个频带，每个的 H/W 为输入的一半。
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
    Haar 小波重建层 (复用自 MobileMamba.py)。

    将 4 个频带 (LL, LH, HL, HH) 重建为原始尺度。
    是 HaarWaveletDecompose 的逆操作。
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


class WDAM(nn.Module):
    """
    WDAM: Wavelet-based Dual Attention Module

    机制: Haar 小波分解 + 窗口注意力(低频) + 卷积(高频) + 逆小波重建
    对 RGBT 的价值: 小波域差异化处理多模态频带信息
    用法: [-1, 1, WDAM, [c2, num_heads, window_size, shift_size]]

    参数:
        c1: 输入通道数 (由 parse_model 自动注入)
        c2: 输出通道数
        num_heads: 注意力头数 (默认 8)
        window_size: 窗口大小 (默认 5)
        shift_size: Shift 操作的偏移量 (默认 2)
        bias: 是否使用偏置 (默认 False)
    """
    def __init__(self, c1, c2, num_heads=8, window_size=5, shift_size=2, bias=False):
        super().__init__()
        self.dim = c2
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 通道对齐 (必须)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        # 小波变换 (使用项目已有的 Haar 实现，无外部依赖)
        self.dwt = HaarWaveletDecompose()
        self.idwt = HaarWaveletReconstruct()

        # 高频分支
        self.high_conv = nn.Sequential(
            nn.Conv2d(c2 * 2, c2 * 2, 3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 * 2, c2, 1, bias=bias),
            nn.ReLU(inplace=True)
        )
        self.high_out = nn.Sequential(
            nn.Conv2d(c2 * 3, c2 * 3, 3, padding=1, groups=3, bias=bias),
            nn.ReLU(inplace=True)
        )

        # 低频注意力 QKV
        self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(c2 * 3, c2 * 3, 3, padding=1, groups=c2 * 3, bias=bias)
        self.project_out = nn.Conv2d(c2, c2, 1, bias=bias)

        # 相对位置偏置 (静态参数，EMA 兼容)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def window_partition(self, x):
        """将特征图分割为窗口"""
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        newH, newW = H + pad_h, W + pad_w
        x = x.view(B, C, newH // ws, ws, newW // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x.view(-1, C, ws, ws)
        return windows, pad_h, pad_w, H, W

    def window_reverse(self, windows, pad_h, pad_w, oriH, oriW):
        """将窗口合并回特征图"""
        ws = self.window_size
        newH = oriH + pad_h
        newW = oriW + pad_w
        B = windows.shape[0] // ((newH // ws) * (newW // ws))
        C = windows.shape[1]
        x = windows.view(B, newH // ws, newW // ws, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, newH, newW)
        if pad_h or pad_w:
            x = x[..., :oriH, :oriW]
        return x

    def shift(self, x, s):
        """Shift 操作增强跨窗口信息交流"""
        if s > 0:
            x = torch.roll(x, shifts=(-s, -s), dims=(2, 3))
        return x

    def rev_shift(self, x, s):
        """逆 Shift 操作"""
        if s > 0:
            x = torch.roll(x, shifts=(s, s), dims=(2, 3))
        return x

    def window_attn(self, q, k, v):
        """窗口内的注意力计算"""
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        attn = torch.matmul(q.transpose(-2, -1), k)
        N = self.window_size * self.window_size
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + rpb
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        return out

    def forward(self, x):
        # 通道对齐
        x = self.proj(x)

        B, C, H, W = x.shape

        # DWT 强制输入为偶数尺寸，补齐原图
        pad_hw = 0
        if H % 2 != 0 or W % 2 != 0:
            pad_hw = 1
            x_pad = F.pad(x, (0, W % 2, 0, H % 2), mode='constant', value=0)
        else:
            x_pad = x

        # DWT 分解
        LL, LH, HL, HH = self.dwt(x_pad)

        # 高频融合权重
        filter_hv = self.high_conv(torch.cat([LH, HL], dim=1))

        # QKV
        qkv = self.qkv_dwconv(self.qkv(LL))
        q, k, v_inp = qkv.chunk(3, dim=1)
        v = v_inp * filter_hv + v_inp

        # 窗口分区 (q/k/v 共用同一套 padding 参数)
        ll_shifted = self.shift(LL, self.shift_size)
        win_q, pad_h, pad_w, llH, llW = self.window_partition(ll_shifted)
        win_k, _, _, _, _ = self.window_partition(ll_shifted)
        win_v, _, _, _, _ = self.window_partition(v)

        B_win, Cq, ws, _ = win_q.shape
        hd = Cq // self.num_heads
        q = win_q.view(B_win, self.num_heads, hd, ws * ws)
        k = win_k.view(B_win, self.num_heads, hd, ws * ws)
        v = win_v.view(B_win, self.num_heads, hd, ws * ws)

        # 窗口注意力
        attn_out = self.window_attn(q, k, v)
        attn_out = attn_out.view(B_win, Cq, ws, ws)
        ll_out = self.window_reverse(attn_out, pad_h, pad_w, llH, llW)
        ll_out = self.rev_shift(ll_out, self.shift_size)
        ll_out = self.project_out(ll_out)

        # 高频重建
        h_all = self.high_out(torch.cat([LH, HL, HH], dim=1))
        LH_new, HL_new, HH_new = h_all.chunk(3, dim=1)

        # IDWT 重建
        recon = self.idwt(ll_out, LH_new, HL_new, HH_new)

        # 还原回原图原始尺寸，保证残差相加维度完全一致
        if pad_hw > 0:
            recon = recon[..., :H, :W]

        # 残差连接
        return recon + x


class C2PSA_WDAM(nn.Module):
    """
    C2PSA_WDAM: 集成 WDAM 的 C2PSA 模块

    机制: Split-Concat 结构 + WDAM 处理一个分支
    用法: [-1, 1, C2PSA_WDAM, [c2]]

    参数:
        c1: 输入通道数 (由 parse_model 自动注入)
        c2: 输出通道数
        n: WDAM 重复次数 (默认 1)
        e: 扩展比例 (默认 0.5)
        num_heads: 注意力头数 (默认 8)
        window_size: 窗口大小 (默认 5)
        shift_size: Shift 操作的偏移量 (默认 2)
    """
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=8, window_size=5, shift_size=2):
        super().__init__()
        assert c1 == c2, f"C2PSA_WDAM requires c1 == c2, got c1={c1}, c2={c2}"
        self.c = int(c1 * e)  # bottleneck width
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[
            WDAM(self.c, self.c, num_heads, window_size, shift_size)
            for _ in range(n)
        ])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
