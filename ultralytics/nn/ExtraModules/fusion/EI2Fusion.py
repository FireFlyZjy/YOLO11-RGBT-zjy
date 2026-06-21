import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityWeightedFusion(nn.Module):
    """QualityWeightedFusion: 天气/质量感知自适应加权融合

    灵感来源: EI2Det 的 IWM (Image Weather-aware Module)

    核心机制:
        对vis和ir特征图做GAP提取全局统计，拼接后通过小MLP预测
        两个模态的质量权重，softmax归一化后加权融合。

    与IWM的区别:
        - IWM接收原始图像(3ch)，需要CNN提取特征
        - 本模块直接接收特征图，更轻量
        - 用GroupNorm替代BatchNorm，兼容P5(1x1)+bs=1

    用法: [[vis_layer, ir_layer], 1, QualityWeightedFusion, [c2]]
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
        self.quality_mlp = nn.Sequential(
            nn.Conv2d(c2 * 2, max(c2 // 4, 8), 1, bias=False),
            nn.GroupNorm(1, max(c2 // 4, 8)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), 2, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        cat_gap = torch.cat([self.gap(vis), self.gap(ir)], dim=1)
        quality_weights = self.softmax(self.quality_mlp(cat_gap))
        w_vis = quality_weights[:, 0:1]
        w_ir = quality_weights[:, 1:2]

        fused = self.norm_out(self.proj_out(w_vis * vis + w_ir * ir))
        return self.act(fused)


class EdgeBlendFusion(nn.Module):
    """EdgeBlendFusion: 边缘感知融合

    灵感来源: EI2Det 的 EFM (Edge Feature Module)

    核心机制:
        对vis和ir特征图做Sobel边缘检测，通过可学习的混合因子alpha
        将边缘特征和融合特征混合: (1-alpha)*fused + alpha*edge

    与EFM的区别:
        - EFM接收原始图像(3ch)，需要多尺度CNN提取边缘
        - 本模块直接在特征图上做Sobel，更轻量
        - alpha可学习，而非固定0.2

    用法: [[vis_layer, ir_layer], 1, EdgeBlendFusion, [c2]]
    """

    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            c_vis = c_ir = int(c1) // 2

        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.edge_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.2))

        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, c2)
        self.act = nn.SiLU(inplace=True)

    def sobel_edge(self, x):
        B, C, H, W = x.shape
        edge_x = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_x, padding=1).reshape(B, C, H, W)
        edge_y = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_y, padding=1).reshape(B, C, H, W)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        fused = vis + ir
        edge = self.edge_proj(self.sobel_edge(vis) + self.sobel_edge(ir))

        alpha = torch.sigmoid(self.alpha)
        blended = (1 - alpha) * fused + alpha * edge

        return self.act(self.norm_out(self.proj_out(blended)))


class EI2Fusion(nn.Module):
    """EI2Fusion: 天气感知+边缘混合全流程融合

    灵感来源: EI2Det 的 TransformerFusionBlock 完整流水线

    核心机制 (两阶段):
        阶段1 - 质量感知加权 (IWM思想):
            GAP(vis+ir) → MLP → softmax → [w_vis, w_ir]
            weighted = w_vis * vis + w_ir * ir
        阶段2 - 边缘混合 (EFM思想):
            edge = Sobel(vis) + Sobel(ir)
            output = (1-alpha) * weighted + alpha * edge

    与原版TransformerFusionBlock的区别:
        - 原版需要4个输入(rgb, ir, weight, edge)，本模块只需2个(rgb, ir)
        - 原版用CrossTransformer做交叉注意力，本模块用轻量MLP预测权重
        - 原版edge来自原始图像的多尺度CNN，本模块直接在特征图上做Sobel
        - 原版alpha固定0.2，本模块可学习

    用法: [[vis_layer, ir_layer], 1, EI2Fusion, [c2]]
    """

    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            c_vis = c_ir = int(c1) // 2

        self.proj_vis = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        # Stage 1: Quality-aware weighting (IWM)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.quality_mlp = nn.Sequential(
            nn.Conv2d(c2 * 2, max(c2 // 4, 8), 1, bias=False),
            nn.GroupNorm(1, max(c2 // 4, 8)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), 2, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

        # Stage 2: Edge blending (EFM)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.edge_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.2))

        self.proj_out = nn.Conv2d(c2, c2, 1, bias=False)
        self.norm_out = nn.GroupNorm(1, c2)
        self.act = nn.SiLU(inplace=True)

    def sobel_edge(self, x):
        B, C, H, W = x.shape
        edge_x = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_x, padding=1).reshape(B, C, H, W)
        edge_y = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_y, padding=1).reshape(B, C, H, W)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            vis, ir = x[:, :C_half], x[:, C_half:]

        vis = self.proj_vis(vis)
        ir = self.proj_ir(ir)

        # Stage 1: Quality-aware weighting
        cat_gap = torch.cat([self.gap(vis), self.gap(ir)], dim=1)
        quality_weights = self.softmax(self.quality_mlp(cat_gap))
        w_vis = quality_weights[:, 0:1]
        w_ir = quality_weights[:, 1:2]
        weighted_fused = w_vis * vis + w_ir * ir

        # Stage 2: Edge blending
        edge = self.edge_proj(self.sobel_edge(vis) + self.sobel_edge(ir))
        alpha = torch.sigmoid(self.alpha)
        blended = (1 - alpha) * weighted_fused + alpha * edge

        return self.act(self.norm_out(self.proj_out(blended)))
