import torch
import torch.nn as nn


class DMAF(nn.Module):
    """DMAF: Differential Modality-Aware Fusion (差分模态感知融合)

    来源: EI2Det-master (Emerging Image Dual-modal Detection)

    核心机制:
        计算两模态的差值图 → GAP生成通道权重 → tanh激活 → 交叉加权差值图。
        即：用IR的差分权重去门控RGB的差分，用RGB的差分权重去门控IR的差分。
        最终将两路加权差值融合为单一输出。

    与已有融合模块的区别:
        - CIFusion: 通道注意力后交叉交换（通道域）
        - vHeat_Fusion: DCT频域热传导融合（频域）
        - FCMMFusion: 空间门控+通道门控交叉（空域门控）
        - DMAF: 差分图+GAP+tanh交叉加权（差分域）
        DMAF 关注的是"两模态之间的差异"，而非各模态自身的特征。

    对RGBT的价值:
        - 差分图天然抑制共同背景，突出模态特异性目标（如IR热点、RGB纹理）
        - 交叉加权让每种模态的差异信息被对方的全局统计指导
        - 零额外参数，纯计算，可作为轻量融合方案

    用法: [[vis_layer, ir_layer], 1, DMAF, [c2]]
    """

    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            c_vis = c_ir = int(c1) // 2

        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Tanh()
        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, c2)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        # 差分图: 突出各模态独有信息
        diff_vis = vis - ir   # RGB独有（IR没有的纹理细节）
        diff_ir = ir - vis    # IR独有（RGB没有的热辐射）

        # GAP + tanh 生成通道权重
        weight_vis = self.act(self.gap(diff_vis))  # (B, C, 1, 1)
        weight_ir = self.act(self.gap(diff_ir))    # (B, C, 1, 1)

        # 交叉加权: IR的差分权重门控RGB差分，反之亦然
        gated_vis = diff_vis * weight_ir
        gated_ir = diff_ir * weight_vis

        # 融合输出
        fused = self.norm_out(self.proj_out(gated_vis + gated_ir))
        return fused
