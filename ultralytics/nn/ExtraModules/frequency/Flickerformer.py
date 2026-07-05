"""
Flickerformer模块集合
来源: Flickerformer (视频去闪烁网络)
论文: 频域/相位注意力机制

提取的模块:
1. PAM - Phase Attention Module (相位注意力)
2. PhaseGuidedFilter - 相位引导滤波器
3. FSAS - Frequency-Spatial Attention Self-attention (频域-空间自注意力)
4. SCAM - Spatial-Channel Attention Module (空间-通道注意力)

核心创新:
- PAM: 只处理相位信息，保留幅度不变（相位包含结构信息）
- PhaseGuidedFilter: 利用频域相位相关性进行跨特征引导
- FSAS: QK在频域计算，V在空间域调制
- SCAM: 三路分割+双向交叉注意力+频域FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================== 基础组件 ======================

class LayerNorm(nn.Module):
    """适配CNN的LayerNorm (支持BiasFree和WithBias)"""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        # x: (B, C, H, W)
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(sigma + 1e-5)
        if self.bias is not None:
            return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x_norm * self.weight.view(1, -1, 1, 1)


# ====================== PAM: Phase Attention Module ======================

class PAM(nn.Module):
    """
    PAM: 相位注意力模块

    核心思想: 只处理相位信息，保留幅度不变
    理论: 相位包含图像结构信息，幅度包含能量信息

    结构: FFT → 提取相位 → MLP处理相位 → 重建频域 → IFFT

    用法: [-1, 1, PAM, [c2]]
    示例: [-1, 1, PAM, [1024]]
    """
    def __init__(self, c1, c2, expand=2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 相位处理MLP
        self.process = nn.Sequential(
            nn.Conv2d(c2, expand * c2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand * c2, c2, 1, 1, 0)
        )

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)
        _, _, H, W = x.shape
        input_dtype = x.dtype  # 保存输入类型，最后恢复

        # FFT（需要 float32，不支持 half）
        x_freq = torch.fft.rfft2(x.float(), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        # 处理相位 — 确保与模型权重的 dtype 一致
        weight_dtype = next(self.process.parameters()).dtype
        pha = self.process(pha.to(dtype=weight_dtype))

        # 重建频域
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        # IFFT
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out.to(dtype=input_dtype)


# ====================== PhaseGuidedFilter: 相位引导滤波器 ======================

class PhaseGuidedFilter(nn.Module):
    """
    PhaseGuidedFilter: 相位引导滤波器

    核心思想: 利用频域相位相关性进行跨特征引导
    理论: 相位相关性反映结构相似性，可用于跨模态引导

    结构: FFT → 提取相位 → 计算相位相关性 → 引导滤波 → IFFT

    用法: [-1, 1, PhaseGuidedFilter, [c2]]
    示例: [-1, 1, PhaseGuidedFilter, [512]]

    特别适合RGBT融合: RGB相位引导IR特征
    注意: 多输入时c1是列表，需要分别处理
    """
    def __init__(self, c1, c2, ffn_expansion_factor=2.66):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        hidden_dim = int(c2 * ffn_expansion_factor)

        # 处理多输入情况
        if isinstance(c1, (list, tuple)):
            # 多输入: 分别处理每个输入
            self.proj_list = nn.ModuleList([
                nn.Conv2d(c, c2, 1, bias=False) for c in c1
            ])
        else:
            # 单输入: 通道对齐
            self.proj_list = nn.ModuleList([
                nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()
            ])

        # 引导滤波网络
        self.guide_net = nn.Sequential(
            nn.Conv2d(c2, hidden_dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, c2, 1, bias=False),
            nn.Sigmoid()
        )

        self.eps = 1e-8

    def forward(self, x):
        # 处理多输入
        if isinstance(x, (list, tuple)):
            # 多输入: 分别投影后相加
            x_proj = [proj(xi) for proj, xi in zip(self.proj_list, x)]
            x = sum(x_proj)
        else:
            # 单输入: 通道对齐
            x = self.proj_list[0](x)

        _, _, H, W = x.shape

        # FFT
        x_freq = torch.fft.rfft2(x.float())
        mag = torch.abs(x_freq)
        phase = x_freq / (mag + self.eps)

        # 计算相位自相关性
        phase_corr = torch.abs(phase * torch.conj(phase))

        # 引导权重 — 确保与模型权重的 dtype 一致
        weight_dtype = next(self.guide_net.parameters()).dtype
        guide = self.guide_net(phase_corr.to(dtype=weight_dtype))

        # 引导滤波
        x_filtered = guide * x_freq

        # IFFT
        x_out = torch.fft.irfft2(x_filtered, s=(H, W))

        return x_out.to(dtype=x.dtype)


# ====================== FSAS: Frequency-Spatial Attention Self-attention ======================

class FSAS(nn.Module):
    """
    FSAS: 频域-空间自注意力

    核心思想: QK在频域计算，V在空间域调制
    理论: 频域点积替代空间点积，O(N)复杂度

    结构: QKV投影 → QK频域交互 → V空间调制 → 输出投影

    用法: [-1, 1, FSAS, [c2]]
    示例: [-1, 1, FSAS, [512]]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # QKV投影
        self.to_hidden = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.to_hidden_dw = nn.Conv2d(c2 * 3, c2 * 3, 3, 1, 1, groups=c2 * 3, bias=False)

        # 输出投影
        self.project_out = nn.Conv2d(c2, c2, 1, bias=False)

        # LayerNorm
        self.norm = LayerNorm(c2, LayerNorm_type='WithBias')

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)
        B, C, H, W = x.shape

        # QKV投影
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # QK频域交互
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        out_fft = q_fft * k_fft
        out = torch.fft.irfft2(out_fft, s=(H, W))

        # LayerNorm — 先恢复输入 dtype
        out = self.norm(out.to(dtype=x.dtype))

        # V空间调制
        output = v * out

        # 输出投影 — 确保与模型权重的 dtype 一致
        weight_dtype = next(self.project_out.parameters()).dtype
        output = self.project_out(output.to(dtype=weight_dtype))

        return output


# ====================== SCAM: Spatial-Channel Attention Module ======================

class DFFN_Simple(nn.Module):
    """简化的频域FFN (去掉einops依赖)"""
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1,
                                groups=hidden_features * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SCAM(nn.Module):
    """
    SCAM: 空间-通道注意力模块

    核心思想: 三路分割 + 双向交叉注意力 + 频域FFN
    理论: 双向信息流（x1→x2, x3→x2）更丰富的特征交互

    结构: 三路分割 → 双向交叉注意力 → 融合 → DFFN

    用法: [-1, 1, SCAM, [c2, num_heads]]
    示例: [-1, 1, SCAM, [512, 8]]

    注意: c2必须能被3整除（三路分割），且c2/3必须能被num_heads整除
    """
    def __init__(self, c1, c2, num_heads=8, ffn_expansion_factor=2.66):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.num_heads = num_heads

        # 确保c2能被3整除
        assert c2 % 3 == 0, f"SCAM requires c2 divisible by 3, got c2={c2}"
        self.c3 = c2 // 3  # 每路的通道数

        # 确保c3能被num_heads整除
        assert self.c3 % num_heads == 0, f"SCAM requires c3 divisible by num_heads, got c3={self.c3}, num_heads={num_heads}"

        # 通道对齐
        self.channel_align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 三路处理 (输入是c3通道，不是c2)
        self.conv1 = nn.Conv2d(self.c3, 2 * self.c3, 3, 1, 1, bias=False)  # k1, v1
        self.conv2 = nn.Conv2d(self.c3, 3 * self.c3, 3, 1, 1, bias=False)  # q1, q2, x2_mid
        self.conv3 = nn.Conv2d(self.c3, 2 * self.c3, 3, 1, 1, bias=False)  # k2, v2

        # 融合 (3 * c3 = c2)
        self.fusion = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)

        # 温度参数
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # FFN
        self.norm = LayerNorm(c2, LayerNorm_type='WithBias')
        self.ffn = DFFN_Simple(c2, ffn_expansion_factor)

    def forward(self, x):
        # 通道对齐
        x = self.channel_align(x)
        B, C, H, W = x.shape

        # 三路分割
        x1, x2, x3 = x.chunk(3, dim=1)

        # 提取QKV
        k1, v1 = self.conv1(x1).chunk(2, dim=1)
        q1, q2, x2_mid = self.conv2(x2).chunk(3, dim=1)
        k2, v2 = self.conv3(x3).chunk(2, dim=1)

        # 多头reshape
        head_dim = self.c3 // self.num_heads
        q1 = q1.view(B, self.num_heads, head_dim, H * W)
        q2 = q2.view(B, self.num_heads, head_dim, H * W)
        k1 = k1.view(B, self.num_heads, head_dim, H * W)
        v1 = v1.view(B, self.num_heads, head_dim, H * W)
        k2 = k2.view(B, self.num_heads, head_dim, H * W)
        v2 = v2.view(B, self.num_heads, head_dim, H * W)

        # 双向交叉注意力
        attn12 = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn23 = (q2 @ k2.transpose(-2, -1)) * self.temperature2

        attn12 = attn12.softmax(dim=-1)
        attn23 = attn23.softmax(dim=-1)

        out1 = (attn12 @ v1)
        out2 = (attn23 @ v2)

        # reshape回空间维度
        out1 = out1.view(B, self.c3, H, W)
        out2 = out2.view(B, self.c3, H, W)

        # 融合
        out = self.fusion(torch.cat([out1, x2_mid, out2], dim=1))

        # FFN
        out = self.ffn(self.norm(out)) + out

        return out


# ====================== C2PSA包装器 ======================

class C2PSA_PAM(nn.Module):
    """
    C2PSA_PAM: 集成PAM的C2PSA模块

    用法: [-1, 2, C2PSA_PAM, [c2]]
    示例: [-1, 2, C2PSA_PAM, [1024]]
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[PAM(self.c, self.c) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))


class C2PSA_FSAS(nn.Module):
    """
    C2PSA_FSAS: 集成FSAS的C2PSA模块

    用法: [-1, 2, C2PSA_FSAS, [c2]]
    示例: [-1, 2, C2PSA_FSAS, [1024]]
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[FSAS(self.c, self.c) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))


class C2PSA_SCAM(nn.Module):
    """
    C2PSA_SCAM: 集成SCAM的C2PSA模块

    用法: [-1, 2, C2PSA_SCAM, [c2, num_heads]]
    示例: [-1, 2, C2PSA_SCAM, [1024, 8]]

    注意: c2*e必须能被3*num_heads整除（SCAM三路分割+多头注意力）
    """
    def __init__(self, c1, c2, n=1, num_heads=8, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        # 确保c能被3*num_heads整除
        divisor = 3 * num_heads
        if self.c % divisor != 0:
            self.c = (self.c // divisor) * divisor
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[SCAM(self.c, self.c, num_heads) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
