import torch
import torch.nn as nn


class FCMMFusion(nn.Module):
    """FCMMFusion: 双分支交叉门控融合

    灵感来源: FCM-main 中 FCM 的双分支 Cross-Gating 机制

    核心机制:
        可见光和红外各走一条主分支，然后用对方分支的统计信息生成门控掩码:
        - 用 vis 的空间注意力去门控 ir 特征，抑制红外中冗余或噪声区域
        - 用 ir 的通道注意力去门控 vis 特征，保留对热辐射敏感的重要通道
        最后将两路门控特征残差融合，并通过输出投影稳定通道数。

    对 RGBT 的价值:
        不是简单 Concat / add，而是通过跨模态门控实现互补增强:
        - IR 帮助 RGB 聚焦目标热区域
        - RGB 帮助 IR 保留视觉纹理和边界细节
    """

    def __init__(self, c1, c2, ratio=4):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            # 单张量输入时沿通道均分, 每半为 c1/2
            c_vis = c_ir = int(c1) // 2

        c_mid = max(c2 // ratio, 8)

        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        # vis branch: spatial gate from ir (GroupNorm, P5空间1x1时BN也会崩)
        self.vis_reduce = nn.Conv2d(c2, c_mid, 1, bias=False)
        self.vis_gn = nn.GroupNorm(1, c_mid)
        self.vis_expand = nn.Conv2d(c_mid, c2, 1, bias=False)

        # ir branch: channel gate from vis (GroupNorm, 因为GAP后空间1x1, BN在bs=1时会报错)
        self.ir_gap = nn.AdaptiveAvgPool2d(1)
        self.ir_fc1 = nn.Conv2d(c2, c_mid, 1, bias=False)
        self.ir_gn = nn.GroupNorm(1, c_mid)
        self.ir_fc2 = nn.Conv2d(c_mid, c2, 1, bias=False)

        self.act = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.out_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.out_norm = nn.GroupNorm(1, c2)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        # spatial gate from ir for vis
        s_gate = self.sigmoid(self.vis_expand(self.act(self.vis_gn(self.vis_reduce(ir)))))
        vis_gated = vis * s_gate

        # channel gate from vis for ir
        c_gate = self.sigmoid(self.ir_fc2(self.act(self.ir_gn(self.ir_fc1(self.ir_gap(vis))))))
        ir_gated = ir * c_gate

        fused = self.out_norm(self.out_proj(vis_gated + ir_gated))
        return self.act(fused)


class FCMBlockFusion(nn.Module):
    """FCMBlockFusion: FCMMFusion + 轻量 refine

    在 FCMMFusion 基础上增加一个可选的双路 refine，保持融合结构即插即用。
    """

    def __init__(self, c1, c2, ratio=4):
        super().__init__()
        self.fuse = FCMMFusion(c1, c2, ratio=ratio)
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.GroupNorm(1, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.GroupNorm(1, c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.refine(self.fuse(x))
