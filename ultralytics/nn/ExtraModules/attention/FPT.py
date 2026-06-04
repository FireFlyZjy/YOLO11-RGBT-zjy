import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# FPT Internal Implementations (完全自包含, 无mmcv依赖)
# ============================================================

class _SelfTrans(nn.Module):
    """FPT内部: 自注意力层内特征聚合 (Self-attention for intra-layer aggregation)

    对单层特征图通过1x1卷积生成q/k/v投影, 在(HW)空间维做注意力,
    支持gaussian/embedded/dot/concatenate四种模式。
    """
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
        super().__init__()
        assert dimension in (1, 2, 3)
        assert mode in ('gaussian', 'embedded', 'dot', 'concatenate'), \
            f'mode must be gaussian/embedded/dot/concatenate, got {mode}'
        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if dimension == 2:
            conv_nd, bn = nn.Conv2d, nn.BatchNorm2d
        elif dimension == 3:
            conv_nd, bn = nn.Conv3d, nn.BatchNorm3d
        else:
            conv_nd, bn = nn.Conv1d, nn.BatchNorm1d

        self.g = conv_nd(in_channels, self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(self.inter_channels, in_channels, kernel_size=1),
                bn(in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if mode in ('embedded', 'dot', 'concatenate'):
            self.theta = conv_nd(in_channels, self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels, self.inter_channels, kernel_size=1)

        if mode == 'concatenate':
            self.W_f = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == 'gaussian':
            theta_x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode in ('embedded', 'dot'):
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
        else:  # concatenate
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            h, w = theta_x.size(2), phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            f = self.W_f(torch.cat([theta_x, phi_x], dim=1)).view(batch_size, h, w)

        if self.mode in ('gaussian', 'embedded'):
            f_div_C = F.softmax(f, dim=-1)
        else:  # dot, concatenate
            f_div_C = f / f.size(-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        return self.W_z(y) + x


class _GroundTrans(nn.Module):
    """FPT内部: 跨层接地注意力 (Cross-layer grounding attention)

    x_low(浅层,更大分辨率)作为query, x_high(深层,更小分辨率)作为key/value,
    深层语义信息被注入浅层细节特征中,实现"自上而下"的语义指导。
    不同于SelfTrans, 输出不含残差连接(返回纯注意力结果)。
    """
    def __init__(self, in_channels, inter_channels=None, mode='dot', dimension=2, bn_layer=True):
        super().__init__()
        assert dimension in (1, 2, 3)
        assert mode in ('gaussian', 'embedded', 'dot', 'concatenate'), \
            f'mode must be gaussian/embedded/dot/concatenate, got {mode}'
        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if dimension == 2:
            conv_nd, bn = nn.Conv2d, nn.BatchNorm2d
        elif dimension == 3:
            conv_nd, bn = nn.Conv3d, nn.BatchNorm3d
        else:
            conv_nd, bn = nn.Conv1d, nn.BatchNorm1d

        self.g = conv_nd(in_channels, self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(self.inter_channels, in_channels, kernel_size=1),
                bn(in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        if mode in ('embedded', 'dot', 'concatenate'):
            self.theta = conv_nd(in_channels, self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels, self.inter_channels, kernel_size=1)

        if mode == 'concatenate':
            self.W_f = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_low, x_high):
        batch_size = x_low.size(0)
        g_x = self.g(x_high).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == 'gaussian':
            theta_x = x_low.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
            phi_x = x_high.view(batch_size, self.in_channels, -1)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode in ('embedded', 'dot'):
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
        else:  # concatenate
            theta_x = self.theta(x_low).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_high).view(batch_size, self.inter_channels, 1, -1)
            h, w = theta_x.size(2), phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            f = self.W_f(torch.cat([theta_x, phi_x], dim=1)).view(batch_size, h, w)

        if self.mode in ('gaussian', 'embedded'):
            f_div_C = F.softmax(f, dim=-1)
        else:  # dot, concatenate
            f_div_C = f / f.size(-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_low.size()[2:])
        return self.W_z(y)


class _RenderTrans(nn.Module):
    """FPT内部: 跨层渲染融合 (Cross-layer rendering fusion)

    使用x_low(浅层特征)的全局池化作为通道调制上下文,
    x_high(深层特征)经3x3卷积生成空间掩码,
    两者逐元素相乘实现特征调制,最后对x_high上采样/下采样后相加输出。
    """
    def __init__(self, channels_high, channels_low, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_high)
        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_high, kernel_size=1, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_high, x_low):
        """x_high: 深层特征(低分辨率), x_low: 浅层特征(高分辨率)"""
        b, c, h, w = x_low.shape
        x_low_gp = F.avg_pool2d(x_low, (h, w)).view(b, c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)

        x_att = x_high_mask * x_low_gp
        if self.upsample:
            out = self.relu(self.bn_upsample(self.conv_upsample(x_high)) + x_att)
        else:
            out = self.relu(self.bn_reduction(self.conv_reduction(x_high)) + x_att)
        return out


# ============================================================
# YOLO-Compatible Wrappers (c1, c2 签名, 支持通道投影)
# ============================================================

class Att_SelfTrans(nn.Module):
    """[FPT] 自注意力层内特征聚合

    机制:
        对单层特征图通过1x1卷积生成query/key/value三组投影,
        在(HW)空间维度做自注意力计算, 输出经BN加残差连接。
        支持4种注意力模式: gaussian / embedded / dot / concatenate,
        默认dot模式(O(n)线性归一化,省去softmax高计算量)。

    与CBAM/SE等注意力的区别:
        FPT自注意力是全局非局部(non-local)操作, 感受野为全图,
        而CBAM/SE仅基于局部卷积或全局池化, 建模能力弱于非局部。

    RGBT价值:
        - 单模态内的长程空间上下文建模, 弥补卷积有限感受野
        - IR小目标因视野局限易丢失, 自注意力捕获全局关系
        - 替换融合后的C3k2/Bottleneck, 强化P5/P4全局依赖性

    YAML用法 (单输入, from=-1):
        - [-1, 1, Att_SelfTrans, [c2]]
        - [-1, 1, Att_SelfTrans, [c2, 'dot']]
        - [-1, 1, Att_SelfTrans, [c2, 'dot', None, True]]

    注册需求: 需在 tasks.py 的 attention 模块列表中添加 Att_SelfTrans
        参考 CBAM, CoordAtt 的注册方式:
        elif m in {..., Att_SelfTrans, ...}:
            c2 = ch[f]
            args = [c2, *args]
    """
    def __init__(self, c1, c2, inter_channels=None, mode='dot', bn_layer=True):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        inter_channels = inter_channels if inter_channels is not None else c2 // 2
        self.self_trans = _SelfTrans(in_channels=c2, inter_channels=inter_channels,
                                     mode=mode, dimension=2, bn_layer=bn_layer)

    def forward(self, x):
        x = self.proj(x)
        return self.self_trans(x)


class Att_GroundTrans(nn.Module):
    """[FPT] 跨层接地注意力 (Cross-layer Grounding Attention)

    机制:
        双输入交叉注意力: x_low(查询) 关注 x_high(键/值)。
        将浅层特征作为query, 深层特征作为key/value,
        通过query-key相似度矩阵计算深层对浅层的注意力加权,
        实现"从上到下"的信息注入(深层语义指导浅层细节)。

    与Att_RenderTrans的区别:
        GroundTrans 使用点积注意力(softmax/normalize)做全局相关性,
        RenderTrans 使用全局池化调制+上采样做轻量空间对齐。

    RGBT价值:
        - 天然支持双输入: RGB↔IR跨模态语义传递
        - P5高层语义→P3/P4细节引导, 跨尺度信息融合
        - IR大目标语义指导RGB小目标定位(反之亦然)

    YAML用法 (双输入, from为列表):
        - [[layer_low, layer_high], 1, Att_GroundTrans, [c2]]
        - [[layer_low, layer_high], 1, Att_GroundTrans, [c2, 'dot']]
        其中 layer_low 提供x_low(query), layer_high 提供x_high(key/value)

    注册需求: 需在 tasks.py 的多输入模块列表中添加 Att_GroundTrans
        参考 Att_AFF, Att_iAFF 的注册方式:
        if m in (..., Att_GroundTrans, ...):
            c1 = [ch[x] for x in f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
    """
    def __init__(self, c1, c2, inter_channels=None, mode='dot', bn_layer=True):
        super().__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) == 2, \
            f'Att_GroundTrans expects c1 as [ch_low, ch_high], got {c1}'
        self.proj_low = nn.Conv2d(c1[0], c2, 1) if c1[0] != c2 else nn.Identity()
        self.proj_high = nn.Conv2d(c1[1], c2, 1) if c1[1] != c2 else nn.Identity()
        inter_channels = inter_channels if inter_channels is not None else c2 // 2
        self.ground_trans = _GroundTrans(in_channels=c2, inter_channels=inter_channels,
                                          mode=mode, dimension=2, bn_layer=bn_layer)

    def forward(self, x_low, x_high):
        x_low = self.proj_low(x_low)
        x_high = self.proj_high(x_high)
        return self.ground_trans(x_low, x_high)


class Att_RenderTrans(nn.Module):
    """[FPT] 跨层渲染融合 (Cross-layer Rendering Fusion)

    机制:
        双输入融合: x_high(深层特征)提供空间掩码,
        x_low(浅层特征)经全局池化提供通道调制上下文,
        两者逐元素相乘实现调制, 最后对x_high上采样/下采样后相加。

        upsample=True:  将x_high上采样到x_low尺寸(用于P5→P4/P3)
        upsample=False: 将x_high降采样到匹配尺寸(用于P5/P4间)

    与Att_GroundTrans的区别:
        RenderTrans使用全局上下文调制(轻量,参数少),
        GroundTrans使用全注意力(计算量大但全局建模更强)。

    RGBT价值:
        - 低计算量的跨模态特征融合, 适合替代高分辨率层的Concat
        - 深层语义掩码突出IR温度异常区域
        - P5→P3上采样路径增强小目标细节恢复

    YAML用法 (双输入, from为列表):
        - [[layer_high, layer_low], 1, Att_RenderTrans, [c2, True]]   # upsample
        - [[layer_high, layer_low], 1, Att_RenderTrans, [c2, False]]  # downsample
        其中 layer_high 提供x_high(深层语义), layer_low 提供x_low(浅层细节)

    注册需求: 需在 tasks.py 的多输入模块列表中添加 Att_RenderTrans
        参考 Att_AFF, Att_iAFF 的注册方式:
        if m in (..., Att_RenderTrans, ...):
            c1 = [ch[x] for x in f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
    """
    def __init__(self, c1, c2, upsample=True):
        super().__init__()
        assert isinstance(c1, (list, tuple)) and len(c1) == 2, \
            f'Att_RenderTrans expects c1 as [ch_high, ch_low], got {c1}'
        c_high, c_low = c1[0], c1[1]
        self.proj_high = nn.Conv2d(c_high, c2, 1) if c_high != c2 else nn.Identity()
        self.proj_low = nn.Conv2d(c_low, c2, 1) if c_low != c2 else nn.Identity()
        self.render_trans = _RenderTrans(channels_high=c2, channels_low=c2, upsample=upsample)

    def forward(self, x_high, x_low):
        x_high = self.proj_high(x_high)
        x_low = self.proj_low(x_low)
        return self.render_trans(x_high, x_low)
