"""
MemoryAttention: 记忆增强注意力

论文灵感: UCMNet (IG_MSA_memory — 记忆增强自注意力)
核心机制:
  1. Memory Bank: 可学习码本 {(K, C)}, 存储典型特征原型
  2. 输入特征通过余弦相似度匹配最近的 memory slots
  3. Soft label 加权聚合 memory vectors → 增强特征
  4. 残差连接: 输出 = 输入 + 增强特征

对 RGBT 价值:
  Memory bank 学习 RGB-T 模态共享特征原型,
  通过 memory 匹配增强对模态不变特征的响应,
  抑制单一模态的噪声响应

用法 (YAML — 单输入, 替换 Conv/SPPF):
  - [-1, 1, MemoryAttention, [c2, num_mem, dim_head]]

参数:
  c1: 输入通道
  c2: 输出通道
  num_mem: memory slots 数量 (默认 64)
  dim_head: 注意力头维度 (默认 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAttention(nn.Module):
    """MemoryAttention — 记忆增强注意力"""

    def __init__(self, c1, c2, num_mem=64, dim_head=64):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

        self.c = c2
        self.k = num_mem
        self.dim_head = dim_head

        # Memory Bank: (K, C) 可学习码本
        self.memory = nn.Parameter(torch.randn(num_mem, c2), requires_grad=True)
        # Prompt 向量: (K, C) 
        self.prompt = nn.Parameter(torch.randn(num_mem, c2), requires_grad=True)

        # QKV 投影
        self.to_q = nn.Linear(c2, c2, bias=False)
        self.to_k = nn.Linear(c2, c2, bias=False)
        self.to_v = nn.Linear(c2, c2, bias=False)

        # 输出投影
        self.proj_out = nn.Linear(c2, c2)

        # 可学习缩放
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.memory, std=0.02)
        nn.init.normal_(self.prompt, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: (B, C, H, W)
        """
        x = self.proj(x)
        B, C, H, W = x.shape

        # 1. Memory 匹配
        # x_flat: (B*H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        N = x_flat.size(0)

        # 余弦相似度: (N, K)
        xn = F.normalize(x_flat, dim=1)  # (N, C)
        mn = F.normalize(self.memory, dim=1)  # (K, C)
        score = torch.matmul(xn, mn.t())  # (N, K)

        # Soft label: (N, K)
        soft_label = F.softmax(score * 10.0, dim=1)  # temperature=10

        # Memory 读取: (N, C)
        mem_out = torch.matmul(soft_label, self.prompt)
        mem_out = mem_out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 2. 自注意力增强
        # 降采样到 1/2 分辨率
        x_pool = F.avg_pool2d(x, kernel_size=2, stride=2)  # (B, C, H/2, W/2)
        B2, C2, H2, W2 = x_pool.shape

        x_pool_flat = x_pool.flatten(2).permute(0, 2, 1)  # (B, N2, C)
        mem_flat = mem_out.flatten(2).permute(0, 2, 1)    # (B, N, C)

        # 降采样 memory 到匹配分辨率
        mem_pool = F.avg_pool2d(mem_out, kernel_size=2, stride=2)
        mem_pool_flat = mem_pool.flatten(2).permute(0, 2, 1)  # (B, N2, C)

        # Q: 输入, K/V: memory
        q = self.to_q(x_pool_flat)  # (B, N2, C)
        k = self.to_k(mem_pool_flat)  # (B, N2, C)
        v = self.to_v(mem_pool_flat)  # (B, N2, C)

        # 多头拆分
        num_heads = C // self.dim_head
        q = q.reshape(B, H2 * W2, num_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, H, N2, D)
        k = k.reshape(B, H2 * W2, num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.reshape(B, H2 * W2, num_heads, self.dim_head).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)  # (B, H, N2, N2)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H2 * W2, C)  # (B, N2, C)
        out = self.proj_out(out)

        # 恢复空间分辨率
        out = out.permute(0, 2, 1).reshape(B, C, H2, W2)  # (B, C, H/2, W/2)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # 3. 残差输出
        return x + self.gamma * out + mem_out


__all__ = ['MemoryAttention']
