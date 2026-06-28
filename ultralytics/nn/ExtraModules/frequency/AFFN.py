"""
AFFN: Autocorrelation Feature Fusion Network
频域自相关融合模块

核心机制:
1. FFT频域变换提取频域特征
2. 自相关功率谱计算 (Xf * conj(Xf))
3. 频域增强: Xf_new = Xf + α * power
4. 空间域增强: x_out = x_out + β * R (自相关逆变换)
5. DWConv + GELU门控

创新点:
- 首次在YOLO中使用自相关功率谱进行特征增强
- 频域和空间域双重增强，捕获全局上下文
- α/β可学习参数自适应融合权重

用法:
- AFFN: [-1, 1, AFFN, [c2, hidden_features]]
- C2PSA_AFFN: [-1, 2, C2PSA_AFFN, [c2]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFFN(nn.Module):
    """
    AFFN: 频域自相关融合网络

    结构: 1x1投影 → 3x3 DWConv → FFT频域处理 → GELU门控 → 1x1输出

    用法: [-1, 1, AFFN, [c2, hidden_features]]
    示例: [-1, 1, AFFN, [1024, 512]]
    """
    def __init__(self, c1, c2, hidden_features=None, bias=False):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        hidden_features = hidden_features or c2

        # 1x1投影: c1 → hidden_features * 2 (用于后续chunk)
        self.project_in = nn.Conv2d(c1, hidden_features * 2, kernel_size=1, bias=bias)

        # 3x3深度可分离卷积
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )

        # 1x1输出投影: hidden_features → c2
        self.project_out = nn.Conv2d(hidden_features, c2, kernel_size=1, bias=bias)

        # 可学习融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 频域增强权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 自相关增强权重

    def forward(self, x):
        # 1x1投影
        x = self.project_in(x)
        original_dtype = x.dtype
        B, C, H, W = x.shape

        # 转为float进行FFT计算 (避免half精度问题)
        x_float = x.float()

        # 全局FFT (不切分patch，彻底解决维度不匹配问题)
        Xf = torch.fft.rfft2(x_float)

        # 自相关功率谱: R(k) = Xf(k) * conj(Xf(k))
        power = Xf * torch.conj(Xf)

        # 自相关逆变换 (空间域)
        R = torch.fft.irfft2(power, s=(H, W))

        # 频域增强: Xf_new = Xf + α * power
        Xf_new = Xf + self.alpha.to(dtype=Xf.real.dtype) * power

        # 逆变换回空间域
        x_out = torch.fft.irfft2(Xf_new, s=(H, W))

        # 空间域自相关增强: x_out = x_out + β * R
        x_out = x_out + self.beta.to(dtype=R.dtype) * R

        # 恢复原始精度
        x = x_out.to(dtype=original_dtype)

        # DWConv + GELU门控
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        # 1x1输出投影
        x = self.project_out(x)

        return x


class C2PSA_AFFN(nn.Module):
    """
    C2PSA_AFFN: 集成AFFN的C2PSA模块

    用AFFN替换C2PSA中的PSABlock，实现频域自相关增强
    结构: Split → AFFN → Concat → 1x1 Conv

    用法: [-1, 2, C2PSA_AFFN, [c2]]
    示例: [-1, 2, C2PSA_AFFN, [1024]]
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, f"C2PSA_AFFN requires c1 == c2, got c1={c1}, c2={c2}"
        self.c = int(c1 * e)

        # Split: c1 → c + c
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()

        # AFFN模块 (n个串联)
        self.m = nn.Sequential(
            *[AFFN(self.c, self.c, hidden_features=self.c) for _ in range(n)]
        )

        # Concat后投影: 2c → c2
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        # Split
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)

        # AFFN处理
        b = self.m(b)

        # Concat + 投影
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
