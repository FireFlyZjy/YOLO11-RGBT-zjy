import torch
import torch.nn as nn
from ..modules.conv import Conv, autopad
from ultralytics.nn.ExtraModules.SE import SEAttention


# -------------------------------------------------- SEAttention begin------------------------------------------------------
class Conv_SE(nn.Module):
    # 标准卷积层 + SE 注意力机制
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道, 输出通道, 卷积核, 步长, 填充, 分组, 激活函数
        super(Conv_SE, self).__init__()
        # 自动计算 padding 保证尺寸符合预期
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2) # 批归一化
        # 如果 act 为 True 使用 SiLU，否则使用指定的激活函数或恒等映射
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.att = SEAttention(c2) # 在激活后进行通道注意力加权

    def forward(self, x):
        # 前向传播：卷积 -> 归一化 -> 激活 -> SE 注意力
        return self.att(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        # 融合模式下的前向传播（通常用于推理时加速，省去 BN 层）
        return self.att(self.act(self.conv(x)))


class CSP_ATT(nn.Module):
    # CSP 跨阶段局部网络 https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=False, g=1, e=0.5):
        super(CSP_ATT, self).__init__()
        c_ = int(2 * c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个分支的起始 1x1 卷积
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二个分支（捷径分支）的 1x1 卷积
        self.cv3 = Conv(c_, c_, 3, 1)  # 3x3 卷积，用于进一步特征提取
        self.cv4 = Conv(c_, c_, 1, 1)  # 1x1 卷积，用于调整通道
        # 空间金字塔池化：使用不同核大小进行最大池化，padding 保持特征图尺寸不变
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)  # 处理池化拼接后结果的卷积（4倍通道是因为 1个原始+3个池化）
        self.cv6 = Conv(c_, c_, 3, 1)  # 进一步加强空间特征提取
        self.cv7 = Conv(2 * c_, c2, 1, 1)  # 融合两个主分支并调整到输出通道 c2
        self.att = SEAttention(c2)  # SE 注意力机制层

    def forward(self, x):
        # 第一路径：卷积 -> SPP 多尺度池化 -> 卷积
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))

        # 第二路径：简单的 1x1 卷积
        y2 = self.cv2(x)

        # 拼接两个路径，融合后通过注意力机制
        return self.att(self.cv7(torch.cat((y1, y2), dim=1)))

# -------------------------------------------------- SEAttention end------------------------------------------------------

