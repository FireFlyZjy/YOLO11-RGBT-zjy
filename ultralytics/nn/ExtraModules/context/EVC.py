"""
EVCBlock — 高效视觉编解码块

原理（Efficient Visual Coding, EVC）：
    结合可学习视觉码本（LVC）与轻量 MLP 的双分支设计，
    在通道维度拼接后融合，捕获全局上下文与局部细节。

模块组成：
    LVCBlock (Learnable Visual Codebook):
        - ConvBlock (1x1 → 3x3 → 1x1) 提取特征
        - Encoding: 可学习码本 (num_codes × C) 与特征进行 scaled L2 距离匹配
        - 通过 softmax 分配权重，聚合得到编码特征
        - Sigmoid 门控: gam * x + x 的残差形式
    LightMLPBlock:
        - DWConv (depthwise) + 线性层（可学习位置编码）
        - GroupNorm → MLP (1x1 → 1x1)
        - Layer Scale + DropPath 双残差连接
    EVCBlock:
        - Conv1 (7x7) + MaxPool 下采样
        - 并行 LVC + LightMLP
        - concat 融合 → 1x1 投影

与 RGBT 关联：
    LVC 的码本可对 RGB 和 Thermal 模态分别学习原型特征，
    LightMLP 提供轻量局部建模，二者互补，适合多模态特征增强。

用法（在 yaml 中配置）：
    - [-1, 1, EVCBlock, [out_channels, num_codes]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# ---------------------------------------------------------------------------
# 工具组件
# ---------------------------------------------------------------------------

class GroupNorm(nn.GroupNorm):
    """分组归一化（单组），等价于 LayerNorm 的 2D 版本。

    Args:
        num_channels (int): 通道数
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    """1x1 卷积实现的 MLP。

    Args:
        in_features (int):          输入通道
        hidden_features (int):      隐藏层通道，默认同输入
        out_features (int):         输出通道，默认同输入
        act_layer (nn.Module):      激活函数类，默认 nn.GELU
        drop (float):               Dropout 率，默认 0.0
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# LVC (Learnable Visual Codebook)
# ---------------------------------------------------------------------------

class Encoding(nn.Module):
    """可学习视觉码本编码层。

    维护一组可学习的 codewords，将输入特征与码本进行 scaled L2 距离匹配，
    并通过 softmax 分配权重，聚合得到紧凑的编码特征。

    Args:
        in_channels (int):  输入通道数
        num_codes (int):    码本数量，默认 64

    Shape:
        Input:  (B, C, H, W)
        Output: (B, num_codes) — 编码后的全局特征
    """

    def __init__(self, in_channels: int, num_codes: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_codes = num_codes

        std = 1.0 / ((num_codes * in_channels) ** 0.5)
        # 可学习码本: (num_codes, in_channels)
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels).uniform_(-std, std),
            requires_grad=True
        )
        # 可学习缩放因子: (num_codes,)
        self.scale = nn.Parameter(
            torch.empty(num_codes).uniform_(-1, 0), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_channels, f"输入通道 {C} != 期望 {self.in_channels}"

        # (B, C, H, W) → (B, C, N) → (B, N, C)  where N = H*W
        x_flat = x.view(B, C, -1).transpose(1, 2).contiguous()  # (B, N, C)

        # scaled L2 距离: (B, N, num_codes)
        expanded_x = x_flat.unsqueeze(2)  # (B, N, 1, C)
        reshaped_codes = self.codewords.view(1, 1, self.num_codes, C)  # (1, 1, K, C)
        diff = (expanded_x - reshaped_codes).pow(2).sum(dim=3)  # (B, N, K)
        scaled_diff = self.scale.view(1, 1, self.num_codes) * diff  # (B, N, K)

        # softmax 分配权重: (B, N, num_codes)
        assignment = torch.softmax(scaled_diff, dim=2)

        # 聚合: weight * (x - code) → sum over N → (B, num_codes, C)
        residual = (x_flat.unsqueeze(2) - self.codewords.view(1, 1, self.num_codes, C))  # (B, N, K, C)
        encoded = (assignment.unsqueeze(3) * residual).sum(dim=1)  # (B, K, C)
        # (B, K, C) → (B, K * C)
        return encoded.view(B, -1)


class LVCBlock(nn.Module):
    """Learnable Visual Codebook Block — 可学习码本全局编码。

    结构:
        x → ConvBlock → Encoding → BN → ReLU → Mean → FC (Sigmoid) → scale + residual

    Args:
        in_channels (int):     输入通道数
        out_channels (int):    输出通道数
        num_codes (int):       码本数量，默认 64

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, num_codes: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes

        # 特征提取
        self.conv_block = nn.Sequential(
            Conv(in_channels, in_channels, k=1, act=nn.ReLU()),
            Conv(in_channels, in_channels, k=3, s=1, act=nn.ReLU()),
            Conv(in_channels, out_channels, k=1, act=False),
        )

        # 编码 + 门控
        self.encoding = Encoding(in_channels=out_channels, num_codes=num_codes)
        self.bn = nn.LayerNorm(num_codes * out_channels)  # LayerNorm替代BN, 兼容batch=1
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sequential(
            nn.Linear(num_codes * out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        B, C, H, W = x.shape

        en = self.encoding(x)  # (B, K*C)
        en = self.act(self.bn(en))
        gam = self.gate(en).view(B, C, 1, 1)  # (B, C, 1, 1)

        return F.relu_(x + x * gam)


# ---------------------------------------------------------------------------
# LightMLPBlock
# ---------------------------------------------------------------------------

class LightMLPBlock(nn.Module):
    """轻量 MLP 块 — Depthwise Conv + MLP + Layer Scale。

    结构:
        x → GN → DWConv → LS → + x → GN → MLP → LS → + x

    Args:
        in_channels (int):              输入通道数
        out_channels (int):             输出通道数
        mlp_ratio (float):              MLP 隐藏层通道倍数，默认 4.0
        drop (float):                   Dropout 率，默认 0.0
        act_layer (nn.Module):          激活函数类，默认 nn.GELU
        use_layer_scale (bool):         是否使用 Layer Scale，默认 True
        layer_scale_init_value (float): 初始缩放值，默认 1e-5
        drop_path (float):              DropPath 率，默认 0.0
        norm_layer (nn.Module):         归一化层类，默认 GroupNorm

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0,
                 act_layer=nn.GELU,
                 use_layer_scale: bool = True,
                 layer_scale_init_value: float = 1e-5,
                 drop_path: float = 0.0,
                 norm_layer=GroupNorm):
        super().__init__()
        self.out_channels = out_channels

        # Depthwise Conv 分支
        self.dw = nn.Sequential(
            Conv(in_channels, in_channels, k=3, s=1, g=in_channels, act=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

        # MLP 分支
        mlp_hidden = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=mlp_hidden,
            out_features=out_channels,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(out_channels), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(out_channels), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# EVCBlock
# ---------------------------------------------------------------------------

class EVCBlock(nn.Module):
    """Efficient Visual Coding Block — LVC + LightMLP 双分支融合。

    结构:
        x → Conv (7x7) → MaxPool
          ├→ LVCBlock ─┐
          └→ LightMLP  → ─→ Concat → 1x1 Conv → out

    Args:
        in_channels (int):      输入通道数
        out_channels (int):     输出通道数
        num_codes (int):        LVC 码本数量，默认 64
        mlp_ratio (float):      LightMLP 中间通道倍数，默认 4.0

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_codes: int = 64, mlp_ratio: float = 4.0):
        super().__init__()
        # 双分支: LVC 改变通道, LightMLP 保持通道
        # LVC: in → out, LightMLP: in → in
        # 拼接后: out + in → 1x1 conv → out
        ch_concat = in_channels + out_channels

        # Stem: 7x7 conv + maxpool, 保持 in_channels
        self.conv1 = Conv(in_channels, in_channels, k=7, act=nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.lvc = LVCBlock(
            in_channels=in_channels, out_channels=out_channels, num_codes=num_codes
        )
        self.l_mlp = LightMLPBlock(
            in_channels, in_channels, mlp_ratio=mlp_ratio,
            act_layer=nn.GELU, drop=0.0,
            use_layer_scale=True, layer_scale_init_value=1e-5,
            drop_path=0.0, norm_layer=GroupNorm
        )

        # 融合: LVC输出(out_c) + LightMLP输出(in_c) → 1x1 → out_c
        self.fusion = nn.Conv2d(ch_concat, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.maxpool(self.conv1(x))
        x_lvc = self.lvc(x1)
        x_lmlp = self.l_mlp(x1)
        x_cat = torch.cat([x_lvc, x_lmlp], dim=1)
        return self.fusion(x_cat)


class DropPath(nn.Module):
    """DropPath / Stochastic Depth — 按样本随机丢弃路径。

    参考: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return mask * x / keep_prob
