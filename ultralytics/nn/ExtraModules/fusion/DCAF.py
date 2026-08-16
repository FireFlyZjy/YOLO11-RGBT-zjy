import torch
import torch.nn as nn
import torch.nn.functional as F


class DCAF(nn.Module):
    """DCAF: Discrepancy-Compensated Adaptive Fusion (差异补偿自适应融合)

    一种轻量、局部、互补感知的双模态 (RGB-T) 特征融合模块，作为中期融合中
    Concat 的原位替代，插入到双分支骨干的 P3 / P4 / P5 对接处。

    设计动机与创新点
    ----------------
    现有 RGB-T 融合可归为两类，各有缺陷:
      (1) 全局质量加权 (如 QualityWeightedFusion): 给每个模态一个*整图标量*权重，
          对模态的*局部失效*盲目，例如暗角处 RGB 已失效但权重不变；
      (2) 交叉注意力 / Transformer (如 ICAFusion / C2Former / HAFFormer):
          表达力强但参数量大、训练不稳、计算复杂度高 (近似 O(N^2))。

    我们提出两个互补机制，且均为*局部、逐通道*:
      1) 局部可靠性重加权 (Local Reliability Reweighting):
         通过共享的轻量瓶颈卷积 (1x1 -> 1x1) 为 vis / ir 各自预测
         逐位置、逐通道的可靠性图 r_v, r_i ∈ [0,1]^{B×C×H×W}，
         按可靠性作归一化加权: base = (r_v·V + r_i·I) / (r_v + r_i + ε)。
         相比全局标量权重，能定位到"哪里、哪个通道"更可信。
      2) 差异补偿残差 (Discrepancy-Compensated Residual) — 核心创新:
         跨模态差异 δ = V - I 不应被当作纯粹噪声丢弃。我们的核心观察是:
         两模态分歧强烈的位置，往往正是*互补信号*所在 —— 例如暗光下 RGB
         看不到、但 IR 中清晰的热目标；或 IR 被热背景淹没、但 RGB 保留的纹理。
         因此用 3x3 深度可分离卷积学一个局部门控 g = σ(DWConv3x3(δ))，
         仅在与分歧大的位置注入补偿残差 λ·g·δ，把"另一模态丢掉的"信号补回来。

    与现有模块的区别
    -----------------
      - 对比 QualityWeightedFusion: 后者是全局标量 softmax 权重且无残差；
        DCAF 是局部逐通道可靠性 + 差异补偿残差。
      - 对比 DMAF: 后者对差分图做 GAP 全局池化，空间信息被压没；
        DCAF 在*全分辨率*上做局部门控，保留位置判别力。
      - 对比 EdgeBlendFusion: 后者仅做 Sobel 边缘 + 固定混合，无可靠性建模、
        无差异补偿。
      - 对比 ICAFusion / C2Former 等: 无注意力、无 Transformer，参数量同 QWF 量级。

    复杂度
    ------
      仅含 1x1 瓶颈、深度可分离 3x3 与 1x1 投影，无 softmax/注意力/Transformer，
      计算与显存开销小，可直接嵌入口袋级检测器。

    用法
    ----
      [[vis_layer, ir_layer], 1, DCAF, [c2]]
    """

    def __init__(self, c1, c2, reduction=4):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            c_vis = c_ir = int(c1) // 2

        # 通道投影: 仅在输入通道与输出通道不一致时使用 1x1，否则恒等 (省参数)
        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        # (1) 局部可靠性估计: 共享瓶颈 (1x1 -> SiLU -> 1x1)，输出逐通道可靠性
        c_mid = max(c2 // reduction, 8)
        self.rel_net = nn.Sequential(
            nn.Conv2d(c2, c_mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c2, 1, bias=False),
        )

        # (2) 差异补偿门控: 深度可分离 3x3，提供局部空间上下文，逐通道门控
        self.gate_net = nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False)

        # 可学习补偿强度 (初始化较小，避免训练初期残差主导)
        self.lam = nn.Parameter(torch.tensor(0.1))

        # 输出投影 + 稳定归一化
        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        V = self.proj_vis(vis)
        I = self.proj_ir(ir)

        # (1) 局部可靠性重加权
        r_v = torch.sigmoid(self.rel_net(V))
        r_i = torch.sigmoid(self.rel_net(I))
        base = (r_v * V + r_i * I) / (r_v + r_i + 1e-6)

        # (2) 差异补偿残差: 仅在与分歧大的位置注入互补信号
        delta = V - I
        g = torch.sigmoid(self.gate_net(delta))
        compensated = base + self.lam * g * delta

        return self.act(self.norm_out(self.proj_out(compensated)))
