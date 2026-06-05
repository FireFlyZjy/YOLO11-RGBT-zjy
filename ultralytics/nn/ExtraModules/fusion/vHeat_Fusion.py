"""
vHeat_Fusion: 跨模态热传导融合模块 (Cross-modal Heat Conduction Fusion)

灵感来源: vHeat (CVPR 2025) - 将热传导原理扩展到 RGB-T 双模态融合

核心思想:
    将 RGB 和红外视为同一场景的两种"热源"，在频域（DCT）中利用热传导方程
    进行自适应融合。不同频率分量具有不同的物理意义:
        - DC 分量 (最低频): 平均辐射强度 → IR 主导 (热辐射信息)
        - 低频分量: 形状/结构 → 两模态互补
        - 高频分量: 边缘/纹理 → RGB 主导 (视觉细节)

热传导方程: du/dt = k·(d²u/dx² + d²u/dy²)
    频域解: U(ω, t) = U₀(ω) · exp(-k·t·|ω|²)
    物理意义: 高频分量随时间指数衰减, k 控制衰减速率

对 RGBT 的价值:
    - IR 高频噪声多 → 预测较大 k_ir → 强衰减 → 自动去噪
    - RGB 纹理信息在高频 → 预测较小 k_rgb → 弱衰减 → 保留细节
    - 频域交叉门控: 两模态互相指导哪些频率分量该保留/抑制
    - O(N^1.5) 复杂度, 远低于 Cross-Attention 的 O(N²)
    - 论文级创新: 首次将热传导原理用于跨模态融合

YAML 用法:
    替换 Concat (层21/22/23):
        [[vis_layer, ir_layer], 1, vHeat_Fusion, [c2, t]]
    示例:
        [[6, 16], 1, vHeat_Fusion, [512, 1.0]]          # P3 融合
        [[8, 18], 1, vHeat_Fusion, [512, 1.5]]          # P4 融合 (更强平滑)
        [[10, 20], 1, vHeat_Fusion, [1024, 2.0]]        # P5 融合

参数:
    c1: 输入通道列表 [c_rgb, c_ir] 或单 int
    c2: 输出通道数
    t: 热扩散时间 (default 1.0)
"""

import math
import torch
import torch.nn as nn


class vHeat_Fusion(nn.Module):
    """vHeat Fusion: 频域跨模态热传导融合

    在 DCT 域进行 RGB-IR 融合，利用热传导方程的自适应频率衰减
    实现模态互补的去噪与细节保留。
    """

    def __init__(self, c1, c2, t=1.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32), requires_grad=False)

        # ── 解析双模态输入通道 ──
        if isinstance(c1, (list, tuple)):
            c_rgb, c_ir = c1[0], c1[1]
        else:
            c_rgb = c_ir = c1

        # ── 通道投影 ──
        self.proj_rgb = nn.Conv2d(c_rgb, c2, 1, bias=False) if c_rgb != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        # ═══════════════════════════════════════════
        # 模态特定 k 预测器
        # ═══════════════════════════════════════════
        self.gap_rgb = nn.AdaptiveAvgPool2d(1)
        self.gap_ir = nn.AdaptiveAvgPool2d(1)
        k_hidden = max(c2 // 4, 4)

        self.k_predictor_rgb = nn.Sequential(
            nn.Conv2d(c2, k_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(k_hidden, c2, 1, bias=False),
            nn.Sigmoid(),  # k_rgb ∈ (0,1), 倾向较小值保留细节
        )
        self.k_predictor_ir = nn.Sequential(
            nn.Conv2d(c2, k_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(k_hidden, c2, 1, bias=False),
            nn.Sigmoid(),  # k_ir ∈ (0,1), 倾向较大值抑制噪声
        )

        # ═══════════════════════════════════════════
        # 跨模态频率门控 (Cross-Modal Frequency Gate)
        # ═══════════════════════════════════════════
        # 从两模态的频域统计信息预测融合权重
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 在频域做全局池化
            nn.Conv2d(c2 * 2, k_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(k_hidden, c2 * 2, 1, bias=False),
            nn.Sigmoid(),  # 输出模态门控权重
        )

        # ═══════════════════════════════════════════
        # 输出层
        # ═══════════════════════════════════════════
        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.BatchNorm2d(c2)
        self.act_out = nn.SiLU(inplace=True)

        # ── DCT 矩阵缓存 (普通 dict, 不使用 register_buffer 以兼容 EMA) ──
        self._dct_cache = {}

    # ╔══════════════════════════════════════════════╗
    # ║         DCT / IDCT 工具方法                  ║
    # ╚══════════════════════════════════════════════╝

    @staticmethod
    def _create_dct_matrix(n):
        """DCT-II 正交变换矩阵 (n, n)."""
        i = torch.arange(n, dtype=torch.float32).view(1, n)
        k = torch.arange(n, dtype=torch.float32).view(n, 1)
        mat = torch.cos(math.pi * k * (i + 0.5) / n)
        mat[0] *= math.sqrt(1.0 / n)
        mat[1:] *= math.sqrt(2.0 / n)
        return mat

    def _ensure_dct_mat(self, H, W, device, dtype):
        """确保 DCT 矩阵已缓存 (普通 dict, 兼容 EMA).
        缓存的 key 包含 H,W,device,dtype, 防止 FP32/FP16 类型不匹配.
        """
        key = (H, W, device, dtype)
        if key not in self._dct_cache:
            self._dct_cache[key] = (
                self._create_dct_matrix(H).to(device=device, dtype=dtype),
                self._create_dct_matrix(W).to(device=device, dtype=dtype),
            )

    def _dct2d(self, x):
        """2D DCT: M_H @ x @ M_W^T."""
        M_H, M_W = self._dct_cache[(x.shape[2], x.shape[3], x.device, x.dtype)]
        dct_h = torch.einsum("ih,bchw->bciw", M_H, x)
        return torch.matmul(dct_h, M_W.T)

    def _idct2d(self, x):
        """2D IDCT: M_H^T @ x @ M_W."""
        M_H, M_W = self._dct_cache[(x.shape[2], x.shape[3], x.device, x.dtype)]
        idct_h = torch.einsum("ih,bciw->bchw", M_H, x)
        return torch.matmul(idct_h, M_W)

    # ╔══════════════════════════════════════════════╗
    # ║         主 forward                            ║
    # ╚══════════════════════════════════════════════╝

    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat] 或拼接张量
        Returns:
            fused: 融合后的空域特征 (B, c2, H, W)
        """
        # ── 解析双输入 ──
        if isinstance(x, (list, tuple)):
            rgb, ir = x[0], x[1]
        else:
            # 如果是拼接张量，沿通道均分
            C_half = x.shape[1] // 2
            rgb, ir = x[:, :C_half], x[:, C_half:]

        # ── 通道投影 ──
        rgb = self.proj_rgb(rgb)
        ir = self.proj_ir(ir)

        B, C, H, W = rgb.shape

        # ── 准备 DCT 矩阵 ──
        self._ensure_dct_mat(H, W, rgb.device, rgb.dtype)

        # ── Step 1: 空域 → 频域 (DCT) ──
        dct_rgb = self._dct2d(rgb)  # (B, C, H, W)
        dct_ir = self._dct2d(ir)    # (B, C, H, W)

        # ── Step 2: 频率网格 ──
        wx = torch.arange(W, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, W) / W
        wy = torch.arange(H, device=rgb.device, dtype=rgb.dtype).view(1, 1, H, 1) / H
        freq_sq = wx ** 2 + wy ** 2  # (1, 1, H, W)

        # ── Step 3: 模态特定热扩散系数 ──
        k_rgb = self.k_predictor_rgb(self.gap_rgb(rgb))  # (B, C, 1, 1)
        k_ir = self.k_predictor_ir(self.gap_ir(ir))      # (B, C, 1, 1)

        # ── Step 4: 热扩散衰减 ──
        decay_rgb = torch.exp(-k_rgb * self.t * freq_sq)  # (B, C, H, W)
        decay_ir = torch.exp(-k_ir * self.t * freq_sq)

        # ── Step 5: 跨模态频率门控 ──
        # 拼接两模态的频域特征，预测门控权重
        dct_cat = torch.cat([dct_rgb, dct_ir], dim=1)  # (B, 2C, H, W)
        gate = self.cross_gate(dct_cat)                  # (B, 2C, 1, 1)
        gate_rgb, gate_ir = gate.chunk(2, dim=1)        # each (B, C, 1, 1)

        # ── Step 6: 频域加权融合 ──
        # 每个模态: 门控权重 × 热扩散衰减后的频域特征
        dct_fused = gate_rgb * dct_rgb * decay_rgb + gate_ir * dct_ir * decay_ir

        # ── Step 7: 频域 → 空域 (IDCT) ──
        fused = self._idct2d(dct_fused)  # (B, C, H, W)

        # ── Step 8: 输出投影 ──
        fused = self.proj_out(fused)
        fused = self.norm_out(fused)
        fused = self.act_out(fused)

        return fused

    def __repr__(self):
        c = self.proj_out.out_channels
        return f"vHeat_Fusion(c={c}, t={self.t.item():.1f})"
