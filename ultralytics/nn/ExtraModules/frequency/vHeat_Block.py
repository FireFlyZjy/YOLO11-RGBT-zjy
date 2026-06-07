"""
vHeat_Block: 完整热传导块 (Heat Conduction Block)

来源: vHeat (CVPR 2025) - Building Vision Models upon Heat Conduction
https://github.com/MzeroMiko/vHeat

完整 Transformer-style block 结构:
    x → Norm1 → HCO (含dwconv + 自适应热扩散 + SiLU门控) → DropPath → +残差
      → Norm2 → MLP (1×1 Conv扩张4× → GELU → 1×1 Conv压缩) → DropPath → +残差

HCO (Heat Conduction Operator):
    DCT → 自适应 k 预测 → exp(-k·t·freq²) 衰减 → IDCT → 残差 → 投影

机制:
    1. 3×3 dwconv 局部预处理 (捕获局部上下文)
    2. DCT 将特征变换到频域
    3. 全局池化 + FC 预测每通道热扩散系数 k ∈ (0, 1)
    4. 高频分量指数衰减: exp(-k·t·(wx²+wy²))
    5. IDCT 回到空域，SiLU 门控，残差连接
    6. MLP/FFN 通道混合

对 RGBT 的价值:
    - 全局感受野 O(N^1.5), 低于 Self-Attention 的 O(N²)
    - 热扩散自适应抑制红外高频噪声
    - DropPath 正则化对抗红外模态过拟合
    - MLP 增强通道间信息交互

YAML 用法:
    替换 SPPF + C2PSA (层24-25):
        [-1, 1, vHeat_Block, [1024, 1.0, 4.0, 0.1]]  # c2, t, mlp_ratio, drop_path

    替换 C3k2 (层8/10/18/20):
        [-1, 1, vHeat_Block, [512, 1.0, 2.0, 0.0]]   # 轻量版 (mlp_ratio=2)

参数:
    c1: 输入通道数
    c2: 输出通道数
    t: 热扩散时间 (default 1.0), 越大低频保留越多
    mlp_ratio: MLP 扩张比 (default 4.0), 降低可减少参数
    drop_path: DropPath 概率 (default 0.0)
    layer_scale: LayerScale 初始值 (default None, 建议 1e-5)
"""

import math
import torch
import torch.nn as nn


class vHeat_Block(nn.Module):
    """vHeat Block: Heat Conduction Operator + MLP + DropPath

    原论文 CVPR 2025 的完整 HeatBlock，即插即用替换 C3k2/C2PSA/SPPF.
    """

    def __init__(self, c1, c2, t=1.0, mlp_ratio=4.0, drop_path=0.0, layer_scale=None):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32), requires_grad=False)

        # ── 输入投影 ──
        self.proj_in = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # ═══════════════════════════════════════════
        # HCO: Heat Conduction Operator
        # ═══════════════════════════════════════════

        # 1) dwconv 局部预处理 (原论文: 3×3 groups=c2)
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False)

        # 2) 门控线性层: 将通道投影到 2*c2, 然后 split 成 x 和 z (gate)
        self.gate_linear = nn.Conv2d(c2, c2 * 2, 1, bias=False)

        # 3) 自适应 k 预测器
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.k_predictor = nn.Sequential(
            nn.Conv2d(c2, max(c2 // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 4), c2, 1, bias=False),
            nn.Sigmoid(),  # k ∈ (0, 1)
        )

        # 4) 输出层
        self.hco_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.hco_norm = nn.BatchNorm2d(c2)

        # ── DCT 矩阵缓存 (普通 dict, 兼容 EMA, 不参与序列化) ──
        self._dct_cache = {}

        # ═══════════════════════════════════════════
        # Pre-Norm + DropPath
        # ═══════════════════════════════════════════
        self.norm1 = nn.BatchNorm2d(c2)
        self.drop_path1 = nn.Dropout2d(drop_path) if drop_path > 0 else nn.Identity()

        # ═══════════════════════════════════════════
        # MLP / FFN (Channel-First, 原论文风格)
        # ═══════════════════════════════════════════
        self.mlp_enabled = mlp_ratio > 0
        if self.mlp_enabled:
            self.norm2 = nn.BatchNorm2d(c2)
            mlp_hidden = int(c2 * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Conv2d(c2, mlp_hidden, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(mlp_hidden, c2, 1, bias=False),
            )
            self.drop_path2 = nn.Dropout2d(drop_path) if drop_path > 0 else nn.Identity()

        # ═══════════════════════════════════════════
        # LayerScale (可选)
        # ═══════════════════════════════════════════
        self.has_layer_scale = layer_scale is not None
        if self.has_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(1, c2, 1, 1), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(1, c2, 1, 1), requires_grad=True)

    def __getstate__(self):
        """序列化时排除 _dct_cache, 避免 device remap 不一致."""
        state = self.__dict__.copy()
        state['_dct_cache'] = {}
        return state

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

    # ╔══════════════════════════════════════════════╗
    # ║         HCO 核心计算                          ║
    # ╚══════════════════════════════════════════════╝

    def _hco_forward(self, x):
        """HCO 前向: dwconv → DCT → 自适应衰减 → IDCT → 残差 + 门控."""
        B, C, H, W = x.shape

        # 1) dwconv 局部预处理
        x_local = self.dwconv(x)  # (B, C, H, W)

        # 2) 门控投影: 2C 通道 → split x, z
        gate = self.gate_linear(x_local)  # (B, 2C, H, W)
        hco_x, gate_z = gate.chunk(2, dim=1)  # each (B, C, H, W)

        # 3) 确保 DCT 矩阵
        self._ensure_dct_mat(H, W, x.device, x.dtype)
        M_H, M_W = self._dct_cache[(H, W, x.device, x.dtype)]  # (H, H), (W, W)

        # 4) 2D DCT:  M_H @ hco_x @ M_W^T
        dct_h = torch.einsum("ih,bchw->bciw", M_H, hco_x)  # (B, C, H, W)
        dct = torch.matmul(dct_h, M_W.T)                     # (B, C, H, W)

        # 5) 频率网格
        wx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W) / W
        wy = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1) / H
        freq_sq = wx ** 2 + wy ** 2  # (1, 1, H, W)

        # 6) 自适应热扩散系数 k
        k = self.k_predictor(self.gap(x))  # (B, C, 1, 1), k ∈ (0, 1)

        # 7) 指数衰减: exp(-k * t * freq²)
        decay = torch.exp(-k * self.t * freq_sq)  # (B, C, H, W)
        dct_decayed = dct * decay

        # 8) 2D IDCT: M_H^T @ dct_decayed @ M_W
        idct_h = torch.einsum("ih,bciw->bchw", M_H, dct_decayed)  # (B, C, H, W)
        idct = torch.matmul(idct_h, M_W)                            # (B, C, H, W)

        # 9) SiLU 门控 + 残差
        idct = idct * torch.nn.functional.silu(gate_z)
        out = idct + x

        # 10) 输出投影 + 归一化
        out = self.hco_out(out)
        out = self.hco_norm(out)
        return out

    # ╔══════════════════════════════════════════════╗
    # ║         主 forward                            ║
    # ╚══════════════════════════════════════════════╝

    def forward(self, x):
        """Pre-Norm + HCO + MLP 结构."""
        x = self.proj_in(x)

        # ── HCO 分支 ──
        hco_out = self._hco_forward(self.norm1(x))
        if self.has_layer_scale:
            hco_out = self.gamma1 * hco_out
        x = x + self.drop_path1(hco_out)

        # ── MLP 分支 ──
        if self.mlp_enabled:
            mlp_out = self.mlp(self.norm2(x))
            if self.has_layer_scale:
                mlp_out = self.gamma2 * mlp_out
            x = x + self.drop_path2(mlp_out)

        return x

    def __repr__(self):
        c = self.hco_out.out_channels
        return (f"vHeat_Block(c={c}, t={self.t.item():.1f}, "
                f"mlp={'on' if self.mlp_enabled else 'off'}, "
                f"ls={'on' if self.has_layer_scale else 'off'})")
