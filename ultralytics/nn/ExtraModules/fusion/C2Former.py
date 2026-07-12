"""
C2Former: Cross-modal Cross-former Fusion Module
=================================================
来源: C2Former (旋转目标检测中的双模态融合模块)

核心机制:
    1. 可变形交叉注意力: 使用 offset 机制实现可变形采样
    2. 双向交叉注意力: VIS 查询 IR 特征，IR 查询 VIS 特征
    3. ModalityNorm: 用源模态统计量重缩放参考模态
    4. 残差连接: out_vis = vis_x + out_lwir, out_lwir = lwir_x + out_vis

对 RGBT 的价值:
    - 可变形采样自适应对齐两个模态的特征位置
    - 双向交叉注意力实现真正的跨模态交互
    - ModalityNorm 处理模态间的分布差异

与已有融合模块的区别:
    - CIFusion/AFF/iAFF: 简单加权融合
    - ICAFusion: 标准交叉注意力
    - C2Former: 可变形交叉注意力 + ModalityNorm，更灵活的跨模态对齐

用法:
    C2Former_Fusion: [[vis, ir], 1, C2Former_Fusion, [c2, num_heads, n_groups, offset_range_factor]]

参考:
    C2Former: Cross-modal Cross-former for RGB-T Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModalityNorm(nn.Module):
    """
    ModalityNorm: 模态归一化

    机制: 用源模态的统计量（均值、标准差）重缩放参考模态
    对 RGBT 的价值: 处理 VIS 和 IR 模态间的分布差异

    参数:
        nf: 通道数
        use_residual: 是否使用残差学习 (默认 True)
        learnable: 是否使用可学习的缩放参数 (默认 True)
    """
    def __init__(self, nf, use_residual=True, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True)
            )
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.use_residual = use_residual

            # 初始化为零，确保初始行为接近恒等映射
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        """
        参数:
            lr: 源模态特征 (用于生成 gamma/beta)
            ref: 参考模态特征 (被归一化后重缩放)
        """
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta
        return out


class LayerNormProxy(nn.Module):
    """LayerNorm 的 2D 版本 (将通道维放到最后做 LayerNorm)"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class C2FormerModule(nn.Module):
    """
    C2FormerModule: 跨模态可变形交叉注意力模块

    机制:
        1. Concat VIS+IR 生成 combined query，用于预测 offset
        2. 可变形采样: 根据 offset 对 VIS 特征进行 grid_sample
        3. 双向交叉注意力:
           - q_lwir (来自 VIS 的 ModalityNorm) 查询 k_vis, v_vis (采样后)
           - q_vis (来自 IR 的 ModalityNorm) 查询 k_lwir, v_lwir
        4. 残差连接: vis_x += out_lwir, lwir_x += out_vis

    参数:
        nc: 通道数
        num_heads: 注意力头数
        n_groups: 分组数
        offset_range_factor: offset 范围因子
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout
    """
    def __init__(self, nc, num_heads=8, n_groups=1, offset_range_factor=2,
                 attn_drop=0.0, proj_drop=0.0, kernel_size=5, sr_ratio=1):
        super().__init__()
        self.nc = nc
        self.num_heads = num_heads
        self.n_head_channels = nc // num_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_groups = n_groups
        self.n_group_channels = nc // n_groups
        self.offset_range_factor = offset_range_factor
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.sr = nn.Identity()

        # Offset 预测网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                     kernel_size, 1, kernel_size // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        # Combined query 投影 (VIS+IR concat -> nc)
        self.proj_combinq = nn.Conv2d(nc * 2, nc, 1, 1, 0)

        # QKV 投影
        self.proj_q_vis = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_q_lwir = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_k_vis = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_k_lwir = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_v_vis = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_v_lwir = nn.Conv2d(nc, nc, 1, 1, 0)

        # 输出投影
        self.proj_out_vis = nn.Conv2d(nc, nc, 1, 1, 0)
        self.proj_out_lwir = nn.Conv2d(nc, nc, 1, 1, 0)

        # Dropout
        self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

        # ModalityNorm
        self.vis_MN = ModalityNorm(nc, use_residual=True, learnable=True)
        self.lwir_MN = ModalityNorm(nc, use_residual=True, learnable=True)

    @torch.no_grad()
    def _get_ref_points(self, H, W, B, dtype, device):
        """生成参考网格点"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W).mul_(2).sub_(1)
        ref[..., 0].div_(H).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, vis_x, lwir_x):
        B, C, H, W = vis_x.size()
        dtype, device = vis_x.dtype, vis_x.device

        # Concat 两个模态生成 combined query
        x = torch.cat([vis_x, lwir_x], 1)
        combin_q = self.proj_combinq(x)

        # 分组处理 offset
        q_off = combin_q.reshape(B * self.n_groups, self.n_group_channels, H, W)
        offset = self.conv_offset(q_off)

        # 限制 offset 范围
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / H, 1.0 / W], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 生成参考点和采样位置
        reference = self._get_ref_points(H, W, B, dtype, device)
        vis_pos = reference + offset.permute(0, 2, 3, 1)
        lwir_pos = reference

        # 可变形采样
        vis_x_sampled = F.grid_sample(
            vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            vis_pos[..., (1, 0)], mode='bilinear', align_corners=True
        )
        lwir_x_sampled = F.grid_sample(
            lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            lwir_pos[..., (1, 0)], mode='bilinear', align_corners=True
        )

        # Reshape 为序列
        n_sample = H * W
        vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
        lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)

        # ---- Key/Value 空间下采样（大幅降低注意力矩阵大小） ----
        if self.sr_ratio > 1:
            vis_grid = vis_x_sampled.reshape(B, C, H, W)
            lwir_grid = lwir_x_sampled.reshape(B, C, H, W)
            vis_grid = self.sr(vis_grid)
            lwir_grid = self.sr(lwir_grid)
            n_sample = vis_grid.shape[2] * vis_grid.shape[3]
            vis_x_sampled = vis_grid.reshape(B, C, 1, n_sample)
            lwir_x_sampled = lwir_grid.reshape(B, C, 1, n_sample)

        # 双向交叉注意力
        # VIS 查询 IR: q 来自 ModalityNorm(vis, lwir), k/v 来自采样后的 vis
        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.num_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)

        # IR 查询 VIS: q 来自 ModalityNorm(lwir, vis), k/v 来自采样后的 lwir
        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.num_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)
        v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.num_heads, self.n_head_channels, n_sample)

        # 注意力计算 (VIS -> IR)
        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        attn_vis = self.vis_attn_drop(attn_vis)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
        out_vis = out_vis.reshape(B, C, H, W)
        out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

        # 注意力计算 (IR -> VIS)
        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        attn_lwir = self.lwir_attn_drop(attn_lwir)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
        out_lwir = out_lwir.reshape(B, C, H, W)
        out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

        return out_vis, out_lwir


class C2Former_Fusion(nn.Module):
    """
    C2Former_Fusion: C2Former 融合模块 (YOLO 包装器)

    机制: 双向可变形交叉注意力 + 残差连接
    对 RGBT 的价值: 自适应跨模态对齐和融合
    用法: [[vis, ir], 1, C2Former_Fusion, [c2, num_heads, n_groups, offset_range_factor]]

    参数:
        c1: 输入通道数列表 (由 parse_model 自动注入)
        c2: 输出通道数
        num_heads: 注意力头数 (默认 8)
        n_groups: 分组数 (默认 1)
        offset_range_factor: offset 范围因子 (默认 2)
        attn_drop: 注意力 dropout (默认 0.0)
        proj_drop: 投影 dropout (默认 0.0)
        kernel_size: offset 卷积核大小 (默认 5)
    """
    def __init__(self, c1, c2, num_heads=8, n_groups=1, offset_range_factor=2,
                 attn_drop=0.0, proj_drop=0.0, kernel_size=5, sr_ratio=8):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]
        else:
            c_vis = c_ir = c1

        # 通道对齐
        self.proj_vis = nn.Conv2d(c_vis, c2, 1) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1) if c_ir != c2 else nn.Identity()

        # C2Former 核心模块（sr_ratio=8: 160x160→20x20, 注意力矩阵缩小64x）
        self.c2former = C2FormerModule(
            nc=c2,
            num_heads=num_heads,
            n_groups=n_groups,
            offset_range_factor=offset_range_factor,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            kernel_size=kernel_size,
            sr_ratio=sr_ratio
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis_x, ir_x = x[0], x[1]
        else:
            vis_x = ir_x = x

        # 通道对齐
        vis_x = self.proj_vis(vis_x)
        ir_x = self.proj_ir(ir_x)

        # C2Former 双向交叉注意力
        out_vis, out_ir = self.c2former(vis_x, ir_x)

        # 残差连接
        vis_x = vis_x + out_ir
        ir_x = ir_x + out_vis

        # 融合输出 (相加)
        return vis_x + ir_x


class C2PSA_C2Former(nn.Module):
    """
    C2PSA_C2Former: 集成 C2Former 的 C2PSA 模块

    机制: Split-Concat 结构 + C2Former 处理一个分支
    用法: [-1, 1, C2PSA_C2Former, [c2, num_heads, n_groups, offset_range_factor]]

    参数:
        c1: 输入通道数 (由 parse_model 自动注入)
        c2: 输出通道数
        n: C2Former 重复次数 (默认 1)
        e: 扩展比例 (默认 0.5)
        num_heads: 注意力头数 (默认 8)
        n_groups: 分组数 (默认 1)
        offset_range_factor: offset 范围因子 (默认 2)
    """
    def __init__(self, c1, c2, n=1, e=0.5, num_heads=8, n_groups=1, offset_range_factor=2):
        super().__init__()
        assert c1 == c2, f"C2PSA_C2Former requires c1 == c2, got c1={c1}, c2={c2}"
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()

        # C2Former 需要双输入，这里简化为单输入版本
        self.m = nn.Sequential(*[
            C2FormerSingle(self.c, num_heads, n_groups, offset_range_factor)
            for _ in range(n)
        ])

        self.cv2 = nn.Conv2d(2 * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        a, b = y.split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.bn2(self.cv2(torch.cat((a, b), 1)))


class C2FormerSingle(nn.Module):
    """
    C2FormerSingle: C2Former 的单输入版本 (用于 C2PSA 包装器)

    将输入分割为两半，分别作为 vis 和 ir 进行交叉注意力
    不使用可变形采样，使用标准注意力
    """
    def __init__(self, nc, num_heads=8, n_groups=1, offset_range_factor=2,
                 kernel_size=5, sr_ratio=8):
        super().__init__()
        self.nc = nc
        self.num_heads = num_heads
        # half_nc 用于 QKV，需要能被 num_heads 整除
        self.half_nc = nc // 2
        self.n_head_channels = self.half_nc // num_heads
        self.scale = self.n_head_channels ** -0.5
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.sr = nn.Identity()

        # QKV 投影 (使用 half_nc)
        self.proj_q_vis = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_q_lwir = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_k_vis = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_k_lwir = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_v_vis = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_v_lwir = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)

        # 输出投影
        self.proj_out_vis = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)
        self.proj_out_lwir = nn.Conv2d(self.half_nc, self.half_nc, 1, 1, 0)

        # ModalityNorm
        self.vis_MN = ModalityNorm(self.half_nc, use_residual=True, learnable=True)
        self.lwir_MN = ModalityNorm(self.half_nc, use_residual=True, learnable=True)

    def forward(self, x):
        B, C, H, W = x.size()

        # 分割通道为两半
        vis_x, lwir_x = x.split(self.half_nc, dim=1)

        # ---- Key/Value 空间下采样 ----
        if self.sr_ratio > 1:
            vis_pooled = self.sr(vis_x)
            lwir_pooled = self.sr(lwir_x)
            n_sampled = vis_pooled.shape[2] * vis_pooled.shape[3]
        else:
            vis_pooled = vis_x
            lwir_pooled = lwir_x
            n_sampled = H * W

        # QKV
        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.num_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_pooled).reshape(B * self.num_heads, self.n_head_channels, n_sampled)
        v_vis = self.proj_v_vis(vis_pooled).reshape(B * self.num_heads, self.n_head_channels, n_sampled)

        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.num_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_pooled).reshape(B * self.num_heads, self.n_head_channels, n_sampled)
        v_lwir = self.proj_v_lwir(lwir_pooled).reshape(B * self.num_heads, self.n_head_channels, n_sampled)

        # 注意力计算 (VIS -> IR)
        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
        out_vis = out_vis.reshape(B, self.half_nc, H, W)
        out_vis = self.proj_out_vis(out_vis)

        # 注意力计算 (IR -> VIS)
        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
        out_lwir = out_lwir.reshape(B, self.half_nc, H, W)
        out_lwir = self.proj_out_lwir(out_lwir)

        # 残差连接 + 拼接
        vis_x = vis_x + out_lwir
        lwir_x = lwir_x + out_vis

        return torch.cat([vis_x, lwir_x], dim=1)
