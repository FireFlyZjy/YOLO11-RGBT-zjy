import torch
import torch.nn as nn

class AIFI(nn.Module):
    """AIFI: RT-DETR的Transformer自注意力+2D sin-cos位置编码, 全局上下文建模"""
    def __init__(self, c1, num_heads=8, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(c1, c1 * 4)
        self.fc2 = nn.Linear(c1 * 4, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

    def forward(self, x):
        b, c, h, w = x.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)
        src2 = self.norm1(src)
        q = k = src2 + pos_embed
        src2 = self.ma(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()
