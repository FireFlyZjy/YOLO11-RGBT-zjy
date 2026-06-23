import torch
import torch.nn as nn
import torch.nn.functional as F


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention: 十字交叉注意力, 高效近似non-local
    机制: 对每个像素沿行和列两个方向计算注意力, O(H*W*(H+W))复杂度
    对RGBT的价值: 建模跨模态长距离空间依赖, 轻量级全局上下文
    来源: CCNet (arXiv:1811.08838), yoloair-main
    用法: [-1, 1, CrissCrossAttention, [c2]]
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        c = c2
        self.q_conv = nn.Conv2d(c, c // 8, 1, bias=False)
        self.k_conv = nn.Conv2d(c, c // 8, 1, bias=False)
        self.v_conv = nn.Conv2d(c, c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        q = self.q_conv(x)  # b, c//8, h, w
        k = self.k_conv(x)
        v = self.v_conv(x)

        # horizontal attention
        q_h = q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # b*w, h, c//8
        k_h = k.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # b*w, c//8, h
        v_h = v.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # b*w, c, h

        attn_h = torch.bmm(q_h, k_h)  # b*w, h, h
        attn_h = self.softmax(attn_h)
        out_h = torch.bmm(v_h, attn_h.permute(0, 2, 1))  # b*w, c, h
        out_h = out_h.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()  # b, c, h, w

        # vertical attention
        q_v = q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # b*h, w, c//8
        k_v = k.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # b*h, c//8, w
        v_v = v.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # b*h, c, w

        attn_v = torch.bmm(q_v, k_v)  # b*h, w, w
        attn_v = self.softmax(attn_v)
        out_v = torch.bmm(v_v, attn_v.permute(0, 2, 1))  # b*h, c, w
        out_v = out_v.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()  # b, c, h, w

        return x + self.gamma * (out_h + out_v)
