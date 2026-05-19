import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class AdaptivePool2d(nn.Module):
    """自适应池化: 当输入尺寸大于目标尺寸时下采样, 小于时上采样, 确保输出尺寸精确匹配"""
    def __init__(self, output_h, output_w, pool_type='avg'):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape
        if input_h > self.output_h or input_w > self.output_w:
            stride_h = input_h // self.output_h
            stride_w = input_w // self.output_w
            kernel_size = (input_h - (self.output_h - 1) * stride_h,
                           input_w - (self.output_w - 1) * stride_w)
            if self.pool_type == 'avg':
                y = F.avg_pool2d(x, kernel_size, stride=(stride_h, stride_w))
            else:
                y = F.max_pool2d(x, kernel_size, stride=(stride_h, stride_w))
        elif input_h < self.output_h or input_w < self.output_w:
            y = F.interpolate(x, size=(self.output_h, self.output_w), mode='nearest')
        else:
            y = x
        return y


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class LearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2


class CrossAttention(nn.Module):
    """交叉注意力: RGB↔IR 双向交叉注意力, 各自独立的QKV投影"""
    def __init__(self, d_model, h, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.que_proj_vis = nn.Linear(d_model, d_model)
        self.key_proj_vis = nn.Linear(d_model, d_model)
        self.val_proj_vis = nn.Linear(d_model, d_model)

        self.que_proj_ir = nn.Linear(d_model, d_model)
        self.key_proj_ir = nn.Linear(d_model, d_model)
        self.val_proj_ir = nn.Linear(d_model, d_model)

        self.out_proj_vis = nn.Linear(d_model, d_model)
        self.out_proj_ir = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

    def forward(self, x):
        rgb_fea_flat, ir_fea_flat = x[0], x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_vis = self.key_proj_vis(rgb_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_vis = self.val_proj_vis(rgb_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_ir = self.key_proj_ir(ir_fea_flat).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_ir = self.val_proj_ir(ir_fea_flat).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        # 交叉注意力: IR query × VIS key → VIS attention; VIS query × IR key → IR attention
        att_vis = torch.matmul(q_ir, k_vis) / math.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / math.sqrt(self.d_k)
        att_vis = self.attn_drop(F.softmax(att_vis, -1))
        att_ir = self.attn_drop(F.softmax(att_ir, -1))

        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis))
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir))

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    """跨模态Transformer块: 交叉注意力 + MLP + 可学习系数残差"""
    def __init__(self, d_model, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        super().__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_ir = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # 可学习系数
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat, ir_fea_flat = x[0], x[1]
        bs, nx, c = rgb_fea_flat.shape

        for _ in range(self.loops):
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

        return [rgb_fea_flat, ir_fea_flat]


class TransformerFusionBlock(nn.Module):
    """ICAFusion核心: Transformer跨模态特征融合
    流程: 自适应池化 → 位置编码 → 交叉Transformer → 上采样+跳跃连接 → 拼接+1x1卷积
    """
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))

        self.avgpool = AdaptivePool2d(vert_anchors, horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(vert_anchors, horz_anchors, 'max')

        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        self.apply(self._init_weights)

        self.crosstransformer = nn.Sequential(*[
            CrossTransformerBlock(d_model, h, block_exp, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])

        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea, ir_fea = x[0], x[1]
        bs, c, h, w = rgb_fea.shape

        # 下采样 + 位置编码
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_h, new_w = new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.view(bs, c, -1).permute(0, 2, 1) + self.pos_emb_vis

        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.view(bs, c, -1).permute(0, 2, 1) + self.pos_emb_ir

        # 交叉Transformer
        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        # 上采样 + 跳跃连接
        rgb_fea_CFE = rgb_fea_flat.view(bs, new_h, new_w, c).permute(0, 3, 1, 2)
        rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=(h, w), mode='nearest')
        new_rgb_fea = rgb_fea_CFE + rgb_fea

        ir_fea_CFE = ir_fea_flat.view(bs, new_h, new_w, c).permute(0, 3, 1, 2)
        ir_fea_CFE = F.interpolate(ir_fea_CFE, size=(h, w), mode='nearest')
        new_ir_fea = ir_fea_CFE + ir_fea

        # 拼接 + 1x1卷积投影
        new_fea = torch.cat([new_rgb_fea, new_ir_fea], dim=1)
        new_fea = self.conv1x1_out(new_fea)

        return new_fea
