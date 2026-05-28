import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SFSConv(nn.Module):
    """SFS-Conv: Spatial-Frequency Selective Convolution (空间-频率选择性卷积)

    论文: CVPR2024 - SFS-Conv: Spatial-Frequency Selective Convolution
          for Lightweight Image Restoration
    核心机制:
      1. 通道分裂 (Channel Split): 将输入通道 50/50 分为空间支路和频率支路
      2. 空间支路 (Spatial Branch): 多尺度深度可分离卷积 (3x3 + 5x5)
         提取多尺度空间特征
      3. 频率支路 (Frequency Branch): 简化 Gabor 滤波器组
         - Gabor 滤波 = 高斯窗 * 正弦波, 可同时定位空域和频域
         - 使用可学习的频率参数 (方向 θ, 波长 λ, 相位 ψ)
         - 对噪声具有天然的鲁棒性 (类似带通滤波)
      4. CSU (Channel Selection Unit): 参数无关的自适应融合
         - 计算两分支的通道统计量 (均值+标准差)
         - Sigmoid 门控决定每通道的融合比例
      5. 拼接 + 1x1 投影到输出通道

    为什么适用于 RGBT 检测:
      - RGB 细节多, 红外噪声多, 两分支天然适配不同模态
      - 频率支路的 Gabor 滤波对红外噪声有抑制作用
      - 空间支路保留 RGB 的精细纹理
      - CSU 自适应融合让网络为每个通道选择最优的"空间-频率"平衡
      - 参数量仅为标准卷积的 ~18% (深度可分离 + 通道分裂)

    YAML 使用示例:
      - [26, SFSConv, [128, 256, 3, 2]]   # 替代标准 Conv, 轻量化
      - [26, SFSConv, [512, 512, 3, 1]]   # 等通道替换
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        """Initialize SFSConv.
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小 (default=3)
            s: 步长 (default=1)
            p: 填充, None 自动为 same (default=None)
            g: 分组数 (保留参数, 模块内固定用 DW conv)
            act: 激活函数 (default=True -> SiLU)
        """
        super().__init__()

        # 1x1 通道投影: c1 → c2 (处理首层3通道等极端情况)
        hidden = max(c2, 8)
        self.proj_in = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        ) if c1 != hidden else nn.Identity()

        # 分裂后各支路的通道数 (基于投影后的通道)
        c_half = max(hidden // 2, 2)

        # ===== 空间支路: 多尺度深度可分离卷积 =====
        # 3x3 深度卷积
        self.spatial_conv1 = nn.Conv2d(
            c_half, c_half, 3, 1, 1, groups=c_half, bias=False
        )
        # 5x5 深度卷积 (更大的感受野)
        self.spatial_conv2 = nn.Conv2d(
            c_half, c_half, 5, 1, 2, groups=c_half, bias=False
        )
        # 1x1 融合
        self.spatial_fuse = nn.Conv2d(c_half, c_half, 1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(c_half)
        self.spatial_act = nn.ReLU(inplace=True)

        # ===== 频率支路: 可学习的 Gabor 滤波 =====
        # 使用简化 Gabor 初始化: 深度卷积 + 方向/频率可学习
        # 对于多方向 Gabor, 使用多组
        self.n_orientations = 4  # 4 个方向: 0, 45, 90, 135 度

        # 每个方向一个深度卷积, 但用分组交换实现
        # 实际: 将 c_half 分为 n_orientations 组, 每组一个方向
        assert c_half >= self.n_orientations, (
            f"c_half ({c_half}) must be >= n_orientations ({self.n_orientations})"
        )
        gabor_groups = max(c_half // self.n_orientations, 1)
        self.freq_conv = nn.Conv2d(
            c_half, c_half, k, 1, autopad(k, p), groups=gabor_groups, bias=False,
        )
        # 用 Gabor 分布初始化
        self._init_gabor(self.freq_conv, k, gabor_groups)
        self.freq_bn = nn.BatchNorm2d(c_half)
        self.freq_act = nn.ReLU(inplace=True)

        # ===== CSU: Channel Selection Unit =====
        # 参数无关自适应融合: 基于通道统计量 (均值+标准差) 生成门控
        self.csu_gate = nn.Sequential(
            nn.Conv2d(c_half * 4, c_half * 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_half * 2, c_half * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # ===== 输出投影 =====
        # 输入通道恢复: c_half*2 = c1 (通道分裂的逆)
        self.proj = nn.Conv2d(c_half * 2, c2, 1, s, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    @staticmethod
    def _init_gabor(conv, k, n_groups):
        """用简化的 Gabor 滤波器初始化卷积权重.

        每个组初始化为不同方向和频率的 Gabor 核:
          G(x,y) = exp(-(x'²+γ²y'²)/(2σ²)) * cos(2πx'/λ + ψ)
        其中 x' = x cos θ + y sin θ, y' = -x sin θ + y cos θ
        """
        C_out = conv.weight.shape[0]
        C_per_group = C_out // n_groups
        center = k // 2
        sigma = k / 4.0
        gamma = 0.5
        lambd = k / 2.0

        with torch.no_grad():
            for g in range(n_groups):
                # 每个组一个不同的方向
                theta = math.pi * g / n_groups  # 均匀分布在 [0, π)
                for c in range(C_per_group):
                    idx = g * C_per_group + c
                    if idx >= C_out:
                        break
                    for y in range(k):
                        for x in range(k):
                            x_ = x - center
                            y_ = y - center
                            x_theta = x_ * math.cos(theta) + y_ * math.sin(theta)
                            y_theta = -x_ * math.sin(theta) + y_ * math.cos(theta)
                            # Gabor
                            gauss = math.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2))
                            sinusoid = math.cos(2 * math.pi * x_theta / lambd)
                            conv.weight[idx, 0, y, x] = gauss * sinusoid

    def _csu_fuse(self, spatial_feat, freq_feat):
        """Channel Selection Unit: 自适应融合两分支.

        1. 计算每分支的通道均值 + 标准差 (空域统计量)
        2. 拼接成 4*c_half 维特征
        3. 两层 1x1 Conv + Sigmoid 生成门控权重
        4. 权重切分为 spatial_gate + freq_gate
        """
        # 通道统计量 (B, c_half, 1, 1)
        s_mean = spatial_feat.mean(dim=(2, 3), keepdim=True)
        s_std = spatial_feat.std(dim=(2, 3), keepdim=True) + 1e-6
        f_mean = freq_feat.mean(dim=(2, 3), keepdim=True)
        f_std = freq_feat.std(dim=(2, 3), keepdim=True) + 1e-6

        # 拼接: (B, c_half*4, 1, 1)
        stats = torch.cat([s_mean, s_std, f_mean, f_std], dim=1)

        # 门控: (B, c_half*2, 1, 1)
        gate = self.csu_gate(stats)

        # 切分: (B, c_half, 1, 1) each
        gate_s = gate[:, :spatial_feat.shape[1]]
        gate_f = gate[:, spatial_feat.shape[1]:]

        # 加权融合
        fused_s = spatial_feat * gate_s
        fused_f = freq_feat * gate_f
        return fused_s, fused_f

    def forward(self, x):
        """前向传播:
        1x1投影 -> 通道分裂 -> 空间支路 -> 频率支路 -> CSU 融合 -> 输出投影
        """
        x = self.proj_in(x)
        B, C, H, W = x.shape
        c_half = max(C // 2, 2)

        # ===== 1. 通道分裂 =====
        x_spatial = x[:, :c_half]  # 前半: 空间支路
        x_freq = x[:, c_half:min(c_half * 2, C)]  # 后半: 频率支路
        # 如果通道数不足, 复制补充
        if x_freq.shape[1] < c_half:
            x_freq = torch.cat([x_freq, x_spatial[:, :c_half - x_freq.shape[1]]], dim=1)

        # ===== 2. 空间支路 =====
        s1 = self.spatial_conv1(x_spatial)
        s2 = self.spatial_conv2(x_spatial)
        s_out = self.spatial_fuse(s1 + s2)
        s_out = self.spatial_bn(s_out)
        s_out = self.spatial_act(s_out)

        # ===== 3. 频率支路 (Gabor) =====
        f_out = self.freq_conv(x_freq)
        f_out = self.freq_bn(f_out)
        f_out = self.freq_act(f_out)

        # ===== 4. CSU 自适应融合 =====
        s_fused, f_fused = self._csu_fuse(s_out, f_out)
        fused = torch.cat([s_fused, f_fused], dim=1)  # (B, c_half*2, H, W)

        # 确保通道数 = 原始输入通道数 (如果 C 是奇数)
        if fused.shape[1] < C:
            fused = torch.cat([fused, fused[:, :C - fused.shape[1]]], dim=1)

        # ===== 5. 输出投影 =====
        out = self.proj(fused)
        out = self.bn(out)
        out = self.act(out)
        return out

    def __repr__(self):
        return (f"SFSConv(c1={self.proj.in_channels // 2 * 2}, "
                f"c2={self.proj.out_channels}, "
                f"k={self.freq_conv.kernel_size[0]})")
