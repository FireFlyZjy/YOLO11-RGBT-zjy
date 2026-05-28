"""
CoordConv — 坐标感知卷积

原理（Coordinate Convolution）：
    在输入特征图上叠加归一化的 xy 坐标通道，使卷积核能够感知像素的空间位置，
    从而打破标准卷积的平移不变性限制，提升空间定位能力。

机制：
    AddCoords 在输入通道末尾拼接两个（或三个）坐标通道：
        - x_channel: 宽度方向归一化坐标，映射到 [-1, 1]
        - y_channel: 高度方向归一化坐标，映射到 [-1, 1]
        - r_channel (可选): 距离中心点的归一化半径
    Conv_Coord 封装 AddCoords + Conv，可直接替换 YOLO 中的标准 Conv。

与 RGBT 关联：
    多模态融合场景中，不同模态的特征图可能空间对齐有细微偏差。
    CoordConv 通过显式的坐标信息帮助模型更好地学习跨模态的空间对应关系。

用法（在 yaml 中配置）：
    - [-1, 1, Conv_Coord, [out_channels, kernel_size, stride]]
    - [-1, 1, Conv_Coord, [out_channels, kernel_size, stride, with_r]]
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class AddCoords(nn.Module):
    """为输入特征图添加归一化的 xy 坐标通道。

    Args:
        with_r (bool): 是否额外添加距离中心点的半径通道 r。默认 False。

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_in + 2 (或 +3 if with_r), H, W)
    """

    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (B, C, H, W)

        Returns:
            拼接坐标通道后的张量
        """
        B, _, H, W = x.shape

        # 生成归一化坐标网格，范围 [-1, 1]
        xx = torch.arange(W, device=x.device, dtype=x.dtype) / (W - 1) * 2 - 1  # [W]
        yy = torch.arange(H, device=x.device, dtype=x.dtype) / (H - 1) * 2 - 1  # [H]

        # 扩展为 (1, 1, H, W)
        xx_grid = xx.view(1, 1, 1, W).expand(B, 1, H, W)
        yy_grid = yy.view(1, 1, H, 1).expand(B, 1, H, W)

        ret = [x, xx_grid, yy_grid]

        if self.with_r:
            # 半径通道：到中心点 (0,0) 的欧氏距离，再归一化到 [0, 1]
            rr = torch.sqrt(xx_grid ** 2 + yy_grid ** 2) / 2 ** 0.5
            ret.append(rr)

        return torch.cat(ret, dim=1)


class Conv_Coord(nn.Module):
    """CoordConv：先添加坐标通道，再执行标准 Conv。

    可直接替换 YOLO 中的 Conv 模块，用于需要空间感知的任务（如检测头、融合层）。

    Args:
        c1 (int):   输入通道数
        c2 (int):   输出通道数
        k (int):    卷积核大小，默认 1
        s (int):    步长，默认 1
        p (int):    填充，默认 None（自动计算 same padding）
        d (int):    空洞率，默认 1
        g (int):    分组数，默认 1
        act (bool | nn.Module): 激活函数，默认 True 使用 SiLU
        with_r (bool): 是否使用半径通道，默认 False

    Shape:
        Input:  (B, C_in, H, W)
        Output: (B, C_out, H_out, W_out)
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: int = None, g: int = 1, d: int = 1, act: bool = True,
                 with_r: bool = False):
        super().__init__()
        self.add_coords = AddCoords(with_r=with_r)

        # 坐标通道占用 2 或 3 个额外通道
        extra = 3 if with_r else 2
        self.conv = Conv(c1 + extra, c2, k=k, s=s, p=p, g=g, d=d, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先添加坐标再卷积。"""
        x = self.add_coords(x)
        return self.conv(x)
