import torch
import torch.nn as nn

class TripletAttention(nn.Module):
    """Triplet Attention: (H,W)/(C,H)/(C,W)三维度旋转空间注意力"""
    def __init__(self):
        super().__init__()
        def make_gate():
            return nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3, bias=False), nn.Sigmoid())
        self.gate_hw = make_gate()
        self.gate_cw = make_gate()
        self.gate_hc = make_gate()

    def forward(self, x):
        hw_pool = torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], dim=1)
        hw_out = x * self.gate_hw(hw_pool)
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        cw_pool = torch.cat([x_perm1.max(1, keepdim=True)[0], x_perm1.mean(1, keepdim=True)], dim=1)
        cw_out = x_perm1 * self.gate_cw(cw_pool)
        cw_out = cw_out.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        hc_pool = torch.cat([x_perm2.max(1, keepdim=True)[0], x_perm2.mean(1, keepdim=True)], dim=1)
        hc_out = x_perm2 * self.gate_hc(hc_pool)
        hc_out = hc_out.permute(0, 3, 2, 1).contiguous()
        return (hw_out + cw_out + hc_out) / 3
