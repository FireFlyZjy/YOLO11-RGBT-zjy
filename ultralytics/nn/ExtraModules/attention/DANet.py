import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DANet Internal Implementations (完全自包含, 无mmcv依赖)
# 来源: https://github.com/junfu1115/DANet
# ============================================================

class PAM_Module(nn.Module):
    """DANet内部: 位置注意力模块 (Position Attention Module)

    计算(H*W)x(H*W)空间位置相似度矩阵。
    每个位置的响应是所有位置特征的加权和, 权重由特征相似度决定。
    使用1x1卷积降维(in_dim->in_dim/8)减小计算量,
    gamma参数可学习, 初始为0(从恒等映射开始训练)。
    """
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W),  output = gamma * attention_out + x
        """
        B, C, H, W = x.size()
        # query: (B, HW, C//8)
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        # key: (B, C//8, HW)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        # energy: (B, HW, HW)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # value: (B, C, HW)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        # out: (B, C, HW) -> (B, C, H, W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """DANet内部: 通道注意力模块 (Channel Attention Module)

    计算CxC通道相似度矩阵。
    每个通道的响应是所有通道特征的加权和,
    权重由通道间特征相关性决定(注意不使用1x1卷积降维)。
    使用max(energy)-energy的数值稳定处理后再做softmax。
    """
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W),  output = gamma * attention_out + x
        """
        B, C, H, W = x.size()
        # query: (B, C, HW)
        proj_query = x.view(B, C, -1)
        # key: (B, HW, C)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        # energy: (B, C, C)
        energy = torch.bmm(proj_query, proj_key)
        # numerical stability: max subtract
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # value: (B, C, HW)
        proj_value = x.view(B, C, -1)
        # out: (B, C, HW) -> (B, C, H, W)
        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


# ============================================================
# YOLO-Compatible Wrappers (c1, c2 签名, 支持通道投影)
# ============================================================

class Att_PAM(nn.Module):
    """[DANet] 位置注意力模块 (Position Attention Module)

    机制:
        计算(H*W)x(H*W)空间位置相似度矩阵。每个位置的响应是所有位置
        特征的加权和, 权重由特征相似度决定。使用1x1卷积降维
        (c2->c2/8)减小计算量, gamma参数初始为0(从恒等映射开始训练)。

    复杂度: O((H*W)^2), 建议仅用于P5(最小空间尺寸, 8x8~20x20)
        P4及更大尺寸将导致显存爆炸!

    RGBT价值:
        - 全局空间上下文: 任意两位置的远距离依赖直接建模
        - IR小目标: 微弱信号可通过全局上下文被关联和增强
        - 跨模态: 放在Concat融合之后, 强化模态间空间位置对应

    YAML用法 (单输入, from=-1):
        - [-1, 1, Att_PAM, [c2]]

    注册需求: 需在 tasks.py 的 attention 模块列表中添加 Att_PAM
        参考 CBAM, CoordAtt 的注册方式:
        elif m in {..., Att_PAM, ...}:
            c2 = ch[f]
            args = [c2, *args]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.pam = PAM_Module(c2)

    def forward(self, x):
        x = self.proj(x)
        return self.pam(x)


class Att_CAM(nn.Module):
    """[DANet] 通道注意力模块 (Channel Attention Module)

    机制:
        计算CxC通道相似度矩阵。每个通道的响应是所有通道特征的加权和,
        权重由通道间特征相关性决定(不使用1x1卷积降维)。
        使用max(energy)-energy数值稳定处理后再做softmax。

    复杂度: O(C^2), C为通道数(512/1024), 远小于PAM的O(H^2*W^2)

    RGBT价值:
        - 通道选择: 强化对检测任务有益的通道(如IR温度通道)
        - 模态互补: Concat融合后突出RGB颜色+IR热辐射通道组合
        - 低计算量: 适合在所有融合层使用

    YAML用法 (单输入, from=-1):
        - [-1, 1, Att_CAM, [c2]]

    注册需求: 需在 tasks.py 的 attention 模块列表中添加 Att_CAM
        参考 CBAM, CoordAtt 的注册方式:
        elif m in {..., Att_CAM, ...}:
            c2 = ch[f]
            args = [c2, *args]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.cam = CAM_Module(c2)

    def forward(self, x):
        x = self.proj(x)
        return self.cam(x)


class Att_DANet(nn.Module):
    """[DANet] 双注意力网络 (Dual Attention Network)

    机制:
        串行组合 PAM(位置注意力) + CAM(通道注意力)。
        PAM: (H*W)x(H*W)空间相似度 -> 全局空间上下文
        CAM: CxC通道相似度 -> 通道间依赖建模
        先空间后通道: 先聚合全局空间信息, 再重标定通道重要性。

    复杂度: O((H*W)^2 + C^2), 建议仅在P5层使用。
        在P3/P4层PAM的空间注意力计算量过大, 可选仅用CAM。

    RGBT价值:
        - 位置+通道双重注意力: 全面建模空间和通道维度全局依赖
        - 空间: 建立不同位置的远距离关联(小目标热力图扩散)
        - 通道: 选择对RGBT检测最关键的语义通道
        - 适合作为检测头前的最终特征精炼模块

    YAML用法 (单输入, from=-1):
        - [-1, 1, Att_DANet, [c2]]

    注册需求: 需在 tasks.py 的 attention 模块列表中添加 Att_DANet
        参考 CBAM, CoordAtt 的注册方式:
        elif m in {..., Att_DANet, ...}:
            c2 = ch[f]
            args = [c2, *args]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.pam = PAM_Module(c2)
        self.cam = CAM_Module(c2)

    def forward(self, x):
        x = self.proj(x)
        x = self.pam(x)
        x = self.cam(x)
        return x
