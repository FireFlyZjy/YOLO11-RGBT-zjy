import math
import torch
import torch.nn as nn


class EBlock(nn.Module):
    """EBlock: FFT-based Illumination Enhancement Block (傅里叶光照增强模块)

    论文: CVPR2025 - DarkIR: Robust Infrared Image Detection via
          Frequency-Guided Enhancement
    核心机制:
      1. FFT 将特征分解为幅值 |F| 和相位 ∠F
         - 幅值 (Magnitude): 包含光照/亮度信息 (低频大值, 高频小值)
         - 相位 (Phase): 包含结构/边缘信息 (高对比度区域变化剧烈)
      2. 可学习的幅值调制: 通过全局特征预测每个通道和频率位置的增益
         - 对低照度/低对比度区域: 增强幅值 (特别是中高频)
         - 对正常区域: 保持幅值不变 (增益 ≈ 1)
      3. 保持相位不变, 保证图像结构不扭曲
      4. IFFT 回到空域, 残差连接

    为什么适用于 RGBT 检测:
      - 红外图像通常对比度低, 细节模糊, 属于"低照度"退化
      - EBlock 可以在特征级增强红外模态的可判别性
      - 保留相位 = 保留边缘和物体结构
      - 仅调制幅值 = 改善光照/对比度而不改变内容

    YAML 使用示例:
      - [26, EBlock, [256, 256]]   # 增强模块 (通道不变)
      - [26, EBlock, [128, 256]]   # 带通道投影的增强
    """

    def __init__(self, c1, c2):
        """Initialize EBlock.
        Args:
            c1: 输入通道数
            c2: 输出通道数
        """
        super().__init__()

        # 通道匹配投影
        self.proj_in = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 幅值调制权重预测器
        # 输入: GAP 后的全局特征 (B, c2, 1, 1)
        # 输出: 每个通道的频率幅值增益 (B, c2, H, W//2+1) for rfft2
        # 我们用全连接实现: GAP -> FC -> 插值到频率网格大小
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mag_predictor = nn.Sequential(
            nn.Conv2d(c2, max(c2 // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 4), c2, 1, bias=False),
            nn.Sigmoid(),  # 增益 ∈ (0, 1) — 保守增强
        )
        # 实际增益范围 [1-alpha, 1+alpha], 允许双向调节
        self.gain_alpha = nn.Parameter(torch.tensor(0.2))

        # 输出投影
        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """前向传播:
        FFT -> 幅值/相位分离 -> 幅值调制 -> 重组 -> IFFT -> 残差 -> 投影
        """
        B, C, H, W = x.shape
        x = self.proj_in(x)
        dtype = x.dtype  # 保存原始 dtype (可能为 float16 under AMP)

        # FFT 需要 float32 精度 (cuFFT half precision 仅支持 2 的幂次维度)
        x_f32 = x.float()

        # ===== 1. FFT + Shift =====
        x_fft = torch.fft.fft2(x_f32, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # DC 居中

        # ===== 2. 幅值/相位分离 =====
        mag = torch.abs(x_fft)        # (B, C, H, W): 幅值
        phase = torch.angle(x_fft)    # (B, C, H, W): 相位 [-π, π]

        # ===== 3. 幅值调制 =====
        # 从全局特征预测基础增益
        gain_base = self.mag_predictor(self.gap(x.to(dtype))).float()  # (B, C, 1, 1)

        # 构造与 mag 相同大小的实际增益图
        # 基础增益 + 可学习的 alpha 调节
        gain = 1.0 + self.gain_alpha.float() * (2.0 * gain_base - 1.0)  # (B, C, 1, 1)
        # 广播到全图
        mag_mod = mag * gain  # (B, C, H, W)

        # ===== 4. 幅值 + 相位 -> 复数 =====
        x_fft_mod = torch.complex(mag_mod * torch.cos(phase), mag_mod * torch.sin(phase))

        # ===== 5. IFFT =====
        x_fft_mod = torch.fft.ifftshift(x_fft_mod, dim=(-2, -1))
        x_mod = torch.fft.ifft2(x_fft_mod, norm="ortho").real.to(dtype)  # 恢复原始 dtype

        # ===== 6. 残差连接 + 输出 =====
        out = x + x_mod  # 残差连接
        out = self.proj_out(out)
        out = self.norm(out)
        out = self.act(out)
        return out

    def __repr__(self):
        return f"EBlock(c={self.proj_out.out_channels})"


class DBlock(nn.Module):
    """DBlock: Dilated Spatial Attention Block (空洞空间注意力模块)

    论文: CVPR2025 DarkIR — 简化版去模糊/去噪模块
    核心机制:
      - 多分支空洞卷积提取不同感受野的特征
      - 空间注意力 (Spatial Attention) 融合多尺度信息
      - 适合处理红外图像中的模糊退化

    用法: 放在 EBlock 之后或 Neck 中, 进一步增强空间特征
    """

    def __init__(self, c1, c2, dilations=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, c2, 3, 1, d, dilation=d, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True),
                )
            )

        # 空间注意力融合
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(c2 * len(dilations), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(c2, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        stacked = torch.cat(feats, dim=1)
        attn = self.spatial_attn(stacked)
        out = stacked[:, -x.shape[1]:] * attn + self.proj(x)
        return out
