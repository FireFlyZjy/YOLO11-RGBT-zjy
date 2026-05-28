import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class FDConv(nn.Module):
    """FDConv: Frequency Domain Decoupled Convolution (频域解耦卷积)

    论文: CVPR2025 - Frequency Domain Decoupled Convolution
    核心机制:
      1. FFT 将输入特征图转换到频域 (傅里叶域)
      2. 频域解耦 (FDW): 将频谱分解为 N 个同心圆环频带 (低/中/高频), 每
         个频带覆盖不同的频率范围
      3. 频带调制 (FBM): 每个频带使用可学习的复数权重进行独立调制, 从
         而让网络自适应增强或抑制特定频率成分
      4. 核空间调制 (KSM): 通过频带组合后的特征再经过标准卷积, 等价于
         在频域对卷积核进行了分解与重组合
      5. IFFT 回到空域, 送入标准 Conv2d 完成特征提取

    优势:
      - 相比标准卷积, FDConv 可以自适应地强调输入中有判别力的频率分量
      - 低频: 轮廓/结构信息 (红外模态更依赖)
      - 高频: 纹理/细节信息 (RGB 模态更丰富)
      - 在 RGBT 检测中, 不同模态在不同频带具有不同的信噪比, 频带调制
        可以做到"频率级"的模态自适应

    YAML 使用示例:
      - [26, FDConv, [128, 256, 3, 2]]           # 标准用法, 4频带 (默认)
      - [26, FDConv, [128, 256, 3, 1, None, 1, True, 6]]  # 6频带
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, n_bands=4):
        """Initialize FDConv.
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小 (default=3)
            s: 步长 (default=1)
            p: 填充, None 自动为 same (default=None)
            g: 分组卷积 (default=1)
            act: 激活函数 (default=True -> SiLU)
            n_bands: 频带数量 (default=4)
        """
        super().__init__()
        self.n_bands = n_bands

        # --- 可学习的频带复数权重 ---
        # 每个频带: 一个复数 w = w_real + j * w_imag
        # 权重形状: (1, 1, 1, 1, n_bands)
        self.band_weight_real = nn.Parameter(torch.ones(1, 1, 1, 1, n_bands))
        self.band_weight_imag = nn.Parameter(torch.zeros(1, 1, 1, 1, n_bands))

        # --- 可学习的频带边界偏移 (permits adaptive band partition) ---
        # 在 Sigmoid 之后值为 (0,1), 表示归一化频率 [0, max_dist] 的分割点
        self.band_edges = nn.Parameter(
            torch.linspace(0.0, 1.0, n_bands + 1)[1:-1]  # 内部 n-1 个边界
        )

        # --- 标准卷积 (对频域增强后的空域特征做进一步提取) ---
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        """前向传播:
        FFT -> 频带分解 -> 复数调制 -> IFFT -> Conv2d+BN+SiLU
        """
        B, C, H, W = x.shape
        device = x.device

        # ===================== 1. FFT + Shift =====================
        # 全 FFT (不是 rfft), 得到完整频谱
        x_fft = torch.fft.fft2(x, norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))  # DC 移至中心

        # ===================== 2. 频率距离网格 =====================
        # 归一化频率坐标 [-0.5, 0.5]
        fy = torch.linspace(-0.5, 0.5, H, device=device).view(-1, 1)
        fx = torch.linspace(-0.5, 0.5, W, device=device).view(1, -1)
        dist = torch.sqrt(fx ** 2 + fy ** 2)  # (H, W): 到 DC 的距离
        max_dist = dist.max().item()  # 约 0.707

        # ===================== 3. 构建频带掩码 =====================
        # 边界: [0, sigmoid(e1), sigmoid(e2), ..., 1] * max_dist
        edges = torch.sigmoid(self.band_edges)  # (n_bands-1,) 值在 (0,1)
        edges_sorted = torch.sort(edges)[0]
        all_edges = torch.cat([
            torch.zeros(1, device=device),
            edges_sorted,
            torch.ones(1, device=device),
        ]) * max_dist  # 映射到实际频率距离

        # band_masks: (n_bands, H, W), 每个是 0/1 硬掩码 (一环)
        band_masks = []
        for i in range(self.n_bands):
            low = all_edges[i]
            high = all_edges[i + 1]
            mask = ((dist >= low) & (dist < high)).float()
            band_masks.append(mask)
        band_masks = torch.stack(band_masks, dim=0)  # (n_bands, H, W)

        # ===================== 4. 频带复数调制 =====================
        # 用复数权重加权各频带: mod_real = sum_i w_real_i * mask_i
        #                      mod_imag = sum_i w_imag_i * mask_i
        mod_real = torch.zeros_like(dist).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mod_imag = torch.zeros_like(dist).unsqueeze(0).unsqueeze(0)
        for i in range(self.n_bands):
            m = band_masks[i].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mod_real = mod_real + self.band_weight_real[..., i] * m
            mod_imag = mod_imag + self.band_weight_imag[..., i] * m

        weight = torch.complex(mod_real, mod_imag)  # (1, 1, H, W)
        x_fft_mod = x_fft * weight  # (B, C, H, W)

        # ===================== 5. IFFT 回空域 =====================
        x_fft_mod = torch.fft.ifftshift(x_fft_mod, dim=(-2, -1))
        x_mod = torch.fft.ifft2(x_fft_mod, norm="ortho").real

        # ===================== 6. 标准卷积 =====================
        out = self.conv(x_mod)
        out = self.bn(out)
        out = self.act(out)
        return out

    def __repr__(self):
        return (f"FDConv(c1={self.conv.in_channels}, c2={self.conv.out_channels}, "
                f"k={self.conv.kernel_size[0]}, s={self.conv.stride[0]}, "
                f"n_bands={self.n_bands})")
