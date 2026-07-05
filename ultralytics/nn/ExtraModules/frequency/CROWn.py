"""
CROWn: Coset-fibRated micrO-local co-attention Modules
======================================================
论文: CROWn (Coset-fibRated micrO-local co-attention Network)
来源: CROWn.py (3D 医学图像分割网络)

提取的核心模块 (改造为 2D 版本):
    1. μPCAD (Microlocal Polyphase Co-Attentive Decimator) - 微观多相共注意力降采样器
    2. CrossSourceMHA - 跨源多头注意力

对 RGBT 的价值:
    - μPCAD: 小波变换 + 跨源注意力，适合频域特征融合
    - CrossSourceMHA: 支持不同来源的 Q, K, V，适合跨模态交互

与已有模块的区别:
    - WDAM: 小波域窗口注意力，单一输入
    - C2Former: 可变形交叉注意力，无可变波变换
    - CROWn: 小波变换 + 跨源注意力，多尺度特征融合

用法:
    μPCAD_2D: [-1, 1, μPCAD_2D, [c2, heads, sr_ratio]]
    C2PSA_μPCAD: [-1, 1, C2PSA_μPCAD, [c2]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gn(c):
    """GroupNorm: 自适应分组数"""
    return nn.GroupNorm(num_groups=min(32, max(1, c // 4)), num_channels=c)


class DWT2D_Haar(nn.Module):
    """
    DWT2D_Haar: Haar 小波分解 (2D 版本)

    使用卷积实现的小波分解，输出 LL, LH, HL, HH 四个子带
    """
    def __init__(self, channels):
        super().__init__()
        l = torch.tensor([1 / math.sqrt(2), 1 / math.sqrt(2)])
        h = torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)])
        ll = torch.outer(l, l)
        lh = torch.outer(l, h)
        hl = torch.outer(h, l)
        hh = torch.outer(h, h)

        weight = torch.zeros((channels * 4, 1, 2, 2))
        for c in range(channels):
            weight[c * 4 + 0, 0] = ll
            weight[c * 4 + 1, 0] = lh
            weight[c * 4 + 2, 0] = hl
            weight[c * 4 + 3, 0] = hh
        self.register_buffer('weight', weight)
        self.groups = channels

    def forward(self, x):
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, groups=self.groups)
        LL, LH, HL, HH = torch.chunk(y, 4, dim=1)
        return LL, LH, HL, HH


class ChannelMLP(nn.Module):
    """通道 MLP: 1×1 卷积实现"""
    def __init__(self, c, expansion=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c * expansion, 1)
        self.fc2 = nn.Conv2d(c * expansion, c, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SR(nn.Module):
    """Spatial Reduction for K/V"""
    def __init__(self, c, sr_ratio=2):
        super().__init__()
        self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else None
        self.norm = gn(c)

    def forward(self, x):
        if self.sr is None:
            return x
        return self.norm(self.sr(x))


class CrossSourceMHA(nn.Module):
    """
    CrossSourceMHA: 跨源多头注意力

    机制: 支持不同来源的 Q, K, V
    - Q 来自一个源 (如 MaxPool)
    - K 来自另一个源 (如 AvgPool)
    - V 来自第三个源 (如 小波分解)

    对 RGBT 的价值: 可以让不同模态的特征分别作为 Q, K, V
    """
    def __init__(self, c_q, c_k, c_v, c_out, heads=4, sr_ratio=2):
        super().__init__()
        self.h = heads
        self.q = nn.Conv2d(c_q, c_out, 1)
        self.k = nn.Conv2d(c_k, c_out, 1)
        self.v = nn.Conv2d(c_v, c_out, 1)
        self.norm_q = gn(c_q)
        self.norm_k = gn(c_k)
        self.norm_v = gn(c_v)
        self.proj = nn.Conv2d(c_out, c_out, 1)
        self.sr_k = SR(c_k, sr_ratio)
        self.sr_v = SR(c_v, sr_ratio)
        self.scale = (c_out // heads) ** -0.5

    def _reshape(self, x):
        N, C, H, W = x.shape
        x = x.view(N, self.h, C // self.h, H * W)
        return x.permute(0, 1, 3, 2).contiguous()

    def forward(self, q_src, k_src, v_src):
        q = self.q(self.norm_q(q_src))
        k = self.k(self.norm_k(self.sr_k(k_src)))
        v = self.v(self.norm_v(self.sr_v(v_src)))

        N, Cq, Hq, Wq = q.shape
        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.permute(0, 1, 3, 2).contiguous().view(N, Cq, Hq, Wq)
        return self.proj(out)


class μPCAD_2D(nn.Module):
    """
    μPCAD_2D: 微观多相共注意力降采样器 (2D 版本)

    机制:
        1. Max/Avg 池化 + 小波分解 → 多尺度特征
        2. 跨源注意力: Q=Max, K=Avg, V=Wavelet
        3. 局部融合: 6路特征拼接 → 3×3 Conv + MLP
        4. SE 通道注意力
        5. 学习权重融合局部和注意力特征

    对 RGBT 的价值:
        - 小波变换捕获多分辨率特征
        - 跨源注意力让不同池化方式的特征交互
        - 适合频域特征融合

    用法: [-1, 1, μPCAD_2D, [c2, heads, sr_ratio]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        heads: 注意力头数 (默认 4)
        sr_ratio: 空间降维比例 (默认 2)
        mlp_ratio: MLP 扩展比例 (默认 4)
    """
    def __init__(self, c1, c2, heads=4, sr_ratio=2, mlp_ratio=4, drop=0.0):
        super().__init__()
        c_mid = 2 * c1

        # 通道对齐
        self.proj = nn.Conv2d(c1, c1, 1) if c1 != c1 else nn.Identity()

        self.proj_pool = nn.Conv2d(c1, 2 * c1, 1)
        self.proj_wav = nn.Conv2d(c1, c1, 1)
        self.dwt = DWT2D_Haar(c1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        # [Max, Avg, LL, LH, HL, HH] -> 3×3 -> MLP
        self.fuse_local = nn.Sequential(
            nn.Conv2d(6 * c1, c_mid, 3, padding=1),
            gn(c_mid),
            nn.GELU(),
            ChannelMLP(c_mid, expansion=mlp_ratio, drop=drop)
        )

        # Q=Max, K=Avg, V=Wavelet
        self.attn = CrossSourceMHA(
            c_q=c1, c_k=c1, c_v=4 * c1,
            c_out=c_mid, heads=heads, sr_ratio=sr_ratio
        )
        self.attn_mlp = ChannelMLP(c_mid, expansion=mlp_ratio, drop=drop)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid, c_mid // 4, 1),
            nn.GELU(),
            nn.Conv2d(c_mid // 4, c_mid, 1),
            nn.Sigmoid()
        )

        # LL 投影到 c_mid
        self.ll_proj = nn.Conv2d(c1, c_mid, 1, bias=False)

        # 输出投影
        self.out_proj = nn.Conv2d(c_mid, c2, 1)
        self.out_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        B, C, H, W = x.shape

        pool_in = self.proj_pool(x)
        q_in, k_in = torch.chunk(pool_in, 2, dim=1)
        v_in = self.proj_wav(x)

        x_max = self.maxpool(q_in)
        x_avg = self.avgpool(k_in)
        LL, LH, HL, HH = self.dwt(v_in)

        local = torch.cat([x_max, x_avg, LL, LH, HL, HH], dim=1)
        local = self.fuse_local(local)

        wav_cat = torch.cat([LL, LH, HL, HH], dim=1)
        attn = self.attn(q_src=x_max, k_src=x_avg, v_src=wav_cat)
        attn = self.attn_mlp(attn)

        # LL 通道数是 c1，需要投影到 c_mid
        LL_proj = self.ll_proj(LL)
        fused = torch.sigmoid(self.alpha) * local + torch.sigmoid(self.beta) * attn \
                + self.gamma * LL_proj
        fused = fused * self.se(fused)

        return self.out_bn(self.out_proj(fused))


class μPCAD_2D_NoDown(nn.Module):
    """
    μPCAD_2D_NoDown: 微观多相共注意力模块 (无下采样版本)

    与 μPCAD_2D 相同的机制，但不进行空间下采样
    用于 C2PSA 包装器中，保持空间尺寸不变
    """
    def __init__(self, c1, c2, heads=4, sr_ratio=2, mlp_ratio=4, drop=0.0):
        super().__init__()
        c_mid = 2 * c1

        self.proj_pool = nn.Conv2d(c1, 2 * c1, 1)
        self.proj_wav = nn.Conv2d(c1, c1, 1)
        self.dwt = DWT2D_Haar(c1)

        # 不进行空间下采样，只进行通道变换
        self.maxpool = nn.Identity()
        self.avgpool = nn.Identity()

        # [Max, Avg, LL, LH, HL, HH] -> 3×3 -> MLP
        self.fuse_local = nn.Sequential(
            nn.Conv2d(6 * c1, c_mid, 3, padding=1),
            gn(c_mid),
            nn.GELU(),
            ChannelMLP(c_mid, expansion=mlp_ratio, drop=drop)
        )

        # Q=Max, K=Avg, V=Wavelet
        self.attn = CrossSourceMHA(
            c_q=c1, c_k=c1, c_v=4 * c1,
            c_out=c_mid, heads=heads, sr_ratio=sr_ratio
        )
        self.attn_mlp = ChannelMLP(c_mid, expansion=mlp_ratio, drop=drop)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid, c_mid // 4, 1),
            nn.GELU(),
            nn.Conv2d(c_mid // 4, c_mid, 1),
            nn.Sigmoid()
        )

        # LL 投影到 c_mid
        self.ll_proj = nn.Conv2d(c1, c_mid, 1, bias=False)

        # 输出投影
        self.out_proj = nn.Conv2d(c_mid, c2, 1)
        self.out_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        B, C, H, W = x.shape

        pool_in = self.proj_pool(x)
        q_in, k_in = torch.chunk(pool_in, 2, dim=1)
        v_in = self.proj_wav(x)

        x_max = self.maxpool(q_in)
        x_avg = self.avgpool(k_in)
        LL, LH, HL, HH = self.dwt(v_in)

        # 将小波分解的结果上采样回原始尺寸
        LL_up = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=True)
        LH_up = F.interpolate(LH, size=(H, W), mode='bilinear', align_corners=True)
        HL_up = F.interpolate(HL, size=(H, W), mode='bilinear', align_corners=True)
        HH_up = F.interpolate(HH, size=(H, W), mode='bilinear', align_corners=True)

        local = torch.cat([x_max, x_avg, LL_up, LH_up, HL_up, HH_up], dim=1)
        local = self.fuse_local(local)

        wav_cat = torch.cat([LL_up, LH_up, HL_up, HH_up], dim=1)
        attn = self.attn(q_src=x_max, k_src=x_avg, v_src=wav_cat)
        attn = self.attn_mlp(attn)

        LL_proj = self.ll_proj(LL_up)
        fused = torch.sigmoid(self.alpha) * local + torch.sigmoid(self.beta) * attn \
                + self.gamma * LL_proj
        fused = fused * self.se(fused)

        return self.out_bn(self.out_proj(fused))


class C2PSA_μPCAD(nn.Module):
    """
    C2PSA_μPCAD: 集成 μPCAD 的 C2PSA 模块

    机制: Split-Concat 结构 + μPCAD 处理一个分支
    用法: [-1, 1, C2PSA_μPCAD, [c2]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        n: μPCAD 重复次数 (默认 1)
        e: 扩展比例 (默认 0.5)
        heads: 注意力头数 (默认 4)
    """
    def __init__(self, c1, c2, n=1, e=0.5, heads=4):
        super().__init__()
        assert c1 == c2, f"C2PSA_μPCAD requires c1 == c2, got c1={c1}, c2={c2}"
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[μPCAD_2D_NoDown(self.c, self.c, heads=heads) for _ in range(n)])
        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))
