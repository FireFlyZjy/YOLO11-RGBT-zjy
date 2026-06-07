import math
import torch
import torch.nn as nn


class vHeat(nn.Module):
    """vHeat: Heat Conduction Operator (热传导算子)

    论文: CVPR2025 - vHeat: Back to the Feature: Learning Robust Features
          from Vision Transformer with Heat Conduction Operator
    核心公式: IDCT( DCT(x) * exp(-k * t * (wx**2 + wy**2)) )

    核心机制:
      1. 2D DCT (离散余弦变换) 将空域特征转换到频域
      2. 构建归一化频率网格 (wx, wy) ∈ [0, 1)
      3. 通过全局池化 + FC 预测每个通道的热扩散系数 k
         - k 值大: 频域衰减快 (更多平滑, 去噪)
         - k 值小: 频域保持 (保留细节)
      4. 应用热传导衰减: exp(-k * t * (wx**2 + wy**2))
         - 物理意义: 高频分量随时间 t 指数衰减 (类似低通滤波)
         - 但 k 是内容自适应的, 所以 = 自适应低通滤波
      5. 2D IDCT 回到空域, 残差连接, 通道投影

    为什么适用于 RGBT 检测:
      - 红外模态通常噪声较多, 需要更强的平滑 (较大 k)
      - RGB 模态细节丰富, 需要保留高频 (较小 k)
      - 通道级的自适应 k 允许每个通道独立选择"平滑程度"
      - DCT/IDCT 复杂度 O(N^1.5), 远低于 Self-Attention 的 O(N^2)

    YAML 使用示例:
      - [26, vHeat, [256, 256]]           # 代替 AIFI, 通道不变
      - [26, vHeat, [256, 128, 2.0]]      # t=2.0 更强平滑
    """

    def __init__(self, c1, c2, t=1.0):
        """Initialize vHeat.
        Args:
            c1: 输入通道数
            c2: 输出通道数
            t: 热扩散时间 (default=1.0), 越大低频保留越多
        """
        super().__init__()
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32), requires_grad=False)

        # 输入输出通道投影 (当 c1 != c2 时)
        self.proj_in = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 每通道热扩散系数 k 的自适应预测器
        # k ∈ (0, 1) 通过 Sigmoid 保证
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.k_predictor = nn.Sequential(
            nn.Conv2d(c2, max(c2 // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 4), c2, 1, bias=False),
            nn.Sigmoid(),  # k ∈ (0, 1)
        )

        # 输出投影 + 归一化
        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm = nn.BatchNorm2d(c2)

        # 缓存 DCT 变换矩阵 (普通 dict, 兼容 EMA, 不参与序列化)
        self._dct_cache = {}

    def __getstate__(self):
        """序列化时排除 _dct_cache, 避免 device remap 不一致."""
        state = self.__dict__.copy()
        state['_dct_cache'] = {}
        return state

    def _ensure_dct_mat(self, H, W, device, dtype):
        """确保 DCT 变换矩阵已预计算 (矩阵乘法法, 复杂度 O(N^1.5)).
        缓存的 key 包含 H,W,device,dtype, 防止 FP32/FP16 类型不匹配.
        """
        key = (H, W, device, dtype)
        if key not in self._dct_cache:
            mat_h = self._create_dct_matrix(H).to(device=device, dtype=dtype)
            mat_w = self._create_dct_matrix(W).to(device=device, dtype=dtype)
            self._dct_cache[key] = (mat_h, mat_w)

    @staticmethod
    def _create_dct_matrix(n):
        """创建 DCT-II 正交变换矩阵, 大小 (n, n).

        公式: M[k, i] = alpha_k * cos(pi * k * (i + 0.5) / n)
        其中 alpha_0 = sqrt(1/n), alpha_k = sqrt(2/n) for k>0
        """
        i = torch.arange(n, dtype=torch.float32).view(1, n)   # (1, n): 空域位置
        k = torch.arange(n, dtype=torch.float32).view(n, 1)   # (n, 1): 频率索引
        mat = torch.cos(math.pi * k * (i + 0.5) / n)          # (n, n)
        mat[0] *= math.sqrt(1.0 / n)                          # DC
        mat[1:] *= math.sqrt(2.0 / n)                         # AC
        return mat

    def forward(self, x):
        """前向传播:
        DCT -> 自适应热扩散 -> IDCT -> 残差 -> 投影 -> 归一化 -> 激活
        """
        B, C, H, W = x.shape
        x = self.proj_in(x)

        # ---------- 预计算 DCT 矩阵 ----------
        self._ensure_dct_mat(H, W, x.device, x.dtype)
        M_H, M_W = self._dct_cache[(H, W, x.device, x.dtype)]  # (H, H), (W, W)

        # ---------- 2D DCT ----------
        # DCT = M_H @ x @ M_W^T
        # 沿 H 维:  einsum('ih,bchw->bciw', M_H, x)
        # 沿 W 维:  result @ M_W^T
        dct_h = torch.einsum("ih,bchw->bciw", M_H, x)         # (B, C, H, W)
        dct = torch.matmul(dct_h, M_W.T)                        # (B, C, H, W)

        # ---------- 频率网格 (归一化) ----------
        # wx ∈ [0, 1/W, 2/W, ..., (W-1)/W];  wy 同理
        wx = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W) / W
        wy = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1) / H
        freq_sq = wx ** 2 + wy ** 2  # (1, 1, H, W)

        # ---------- 自适应热扩散系数 k ----------
        k = self.k_predictor(self.gap(x))  # (B, C, 1, 1), k ∈ (0, 1)

        # ---------- 热扩散衰减 ----------
        # decay = exp(-k * t * freq_sq)
        decay = torch.exp(-k * self.t * freq_sq)  # (B, C, H, W)

        # 应用衰减
        dct_decayed = dct * decay  # (B, C, H, W)

        # ---------- 2D IDCT ----------
        # IDCT = M_H^T @ dct_decayed @ M_W
        # 沿 H 维: M_H^T @ dct  => einsum('ih,bciw->bchw', M_H, dct)
        #           M_H[i,h] * dct[b,c,i,w] sum over i = (M_H^T @ dct)[b,c,h,w]
        # 沿 W 维:  idct @ M_W => torch.matmul(idct, M_W)
        #           (B,C,H,W) @ (W,W): contract last dim of idct (W) with
        #           second-to-last of M_W (W) = (idct @ M_W)[b,c,h,w]
        idct_h = torch.einsum("ih,bciw->bchw", M_H, dct_decayed)  # (B, C, H, W)
        idct = torch.matmul(idct_h, M_W)                            # (B, C, H, W)

        # ---------- 残差连接 ----------
        # IDCT 结果 + 原始输入
        out = idct + x

        # ---------- 输出投影 + 归一化 ----------
        out = self.proj_out(out)
        out = self.norm(out)
        return out

    def __repr__(self):
        return f"vHeat(c={self.proj_out.out_channels}, t={self.t.item():.2f})"
