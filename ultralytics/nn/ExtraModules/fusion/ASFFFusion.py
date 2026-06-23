import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASFFFusion(nn.Module):
    """ASFFFusion: 自适应空间特征融合(RGB-T双模态版)
    机制: 将ASFF的多尺度单模态融合改造为双模态融合:
          vis和ir分别经过compress→计算通道权重→softmax加权融合→expand输出
    对RGBT的价值: 学习RGB和IR的自适应融合权重, 空间和通道双重自适应
    来源: ASFF (arXiv:1911.09516), 改造为双模态融合
    用法: [[vis_layer, ir_layer], 1, ASFFFusion, [c2]]
    """
    def __init__(self, c1, c2):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]
        else:
            c_vis = c_ir = c1
        self.inter_dim = c2
        self.compress_vis = Conv(c_vis, self.inter_dim, 1, 1)
        self.compress_ir = Conv(c_ir, self.inter_dim, 1, 1)
        self.weight_vis = Conv(self.inter_dim, 8, 1, 1)
        self.weight_ir = Conv(self.inter_dim, 8, 1, 1)
        self.weight_levels = Conv(16, 2, 1, 1)
        self.expand = Conv(self.inter_dim, c2, 3, 1)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            vis, ir = x[0], x[1]
        else:
            vis = ir = x
        vis = self.compress_vis(vis)
        ir = self.compress_ir(ir)
        w_vis = self.weight_vis(vis)
        w_ir = self.weight_ir(ir)
        w = torch.cat((w_vis, w_ir), 1)
        w = self.weight_levels(w)
        w = F.softmax(w, dim=1)
        out = vis * w[:, 0:1, :, :] + ir * w[:, 1:2, :, :]
        out = self.expand(out)
        return out
