"""
DAIFusion: Depth Adaptive Injection Fusion (RGB-T adapted)

Source: VCP-DCN (ECCV 2026) — DAI module for RGB-D camouflaged object detection.
Adaptation: Depth -> IR, learnable prototypes instead of mask-pooled,
content-adaptive weights, lightweight conv instead of deformable conv.

Core mechanism:
  1. Learnable prototypes: modality-consistent (shared) + modality-specific (per-modality)
  2. Compute content-adaptive similarity weights from prototype distances
  3. Decompose features into consistency (shared) and specificity (unique) parts
  4. Fuse via concat + 1x1 conv + depthwise refine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAIFusion(nn.Module):
    """Depth Adaptive Injection Fusion for RGB-T dual-modal detection.

    Replaces Concat at P3/P4/P5 fusion points. Uses prototype-based
    adaptive weighting to decompose RGB and IR features into consistency
    and specificity components, preventing modal homogenization.
    """

    def __init__(self, c1, c2):
        """
        Args:
            c1: input channels — list [c_rgb, c_ir] or int (halved into two equal halves).
            c2: output channel count.
        """
        super().__init__()

        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = int(c1[0]), int(c1[1])
        else:
            c_vis = c_ir = int(c1) // 2

        # Projection layers (1x1, no bias)
        self.proj_rgb = nn.Conv2d(c_vis, c2, 1, bias=False) if c_vis != c2 else nn.Identity()
        self.proj_ir = nn.Conv2d(c_ir, c2, 1, bias=False) if c_ir != c2 else nn.Identity()

        # Learnable prototypes (SPE simplified)
        # Modality-consistent prototype (shared representation)
        self.P_con = nn.Parameter(torch.randn(1, c2, 1, 1) * 0.02)
        # RGB-specific prototype
        self.P_rgb_s = nn.Parameter(torch.randn(1, c2, 1, 1) * 0.02)
        # IR-specific prototype
        self.P_ir_s = nn.Parameter(torch.randn(1, c2, 1, 1) * 0.02)

        # Content-adaptive weight prediction
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Output fusion: concat(F_c, F_s_rgb, F_s_ir) = 3*c2 -> c2
        self.reduce = nn.Conv2d(c2 * 3, c2, 1, bias=False)
        self.norm = nn.GroupNorm(1, c2)  # GroupNorm for P5 1x1 + bs=1 compatibility
        self.act = nn.SiLU(inplace=True)

        # Spatial refinement (depthwise 3x3)
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.GroupNorm(1, c2),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _cosine_sim(a, b):
        """Cosine similarity along channel dimension.

        Args:
            a: (B, C, 1, 1) or (1, C, 1, 1)
            b: (1, C, 1, 1)
        Returns:
            (B, 1, 1, 1) similarity in [-1, 1]
        """
        return F.cosine_similarity(a, b, dim=1, eps=1e-6).view(-1, 1, 1, 1)

    def forward(self, x):
        """Fuse RGB and IR features.

        Args:
            x: list/tuple [rgb_feat, ir_feat] or single tensor (split by channel).
        Returns:
            Fused tensor (B, c2, H, W).
        """
        if isinstance(x, (list, tuple)):
            rgb, ir = x[0], x[1]
        else:
            C_half = x.shape[1] // 2
            rgb, ir = x[:, :C_half], x[:, C_half:]

        # Step 1: Project to c2 channels
        rgb = self.proj_rgb(rgb)  # (B, c2, H, W)
        ir = self.proj_ir(ir)    # (B, c2, H, W)

        # Step 2: Content-adaptive weight computation (DAI core)
        dtype = rgb.dtype
        rgb_f = rgb.float()
        ir_f = ir.float()
        p_con = self.P_con.float()
        p_rgb_s = self.P_rgb_s.float()
        p_ir_s = self.P_ir_s.float()

        g_rgb = self.gap(rgb_f)  # (B, c2, 1, 1)
        g_ir = self.gap(ir_f)    # (B, c2, 1, 1)

        # Content-prototype similarity (content-adaptive)
        sim_rgb = self._cosine_sim(g_rgb, p_con)  # (B, 1, 1, 1)
        sim_ir = self._cosine_sim(g_ir, p_con)   # (B, 1, 1, 1)

        # Prototype-prototype similarity (learnable bias)
        proto_sim_rgb = self._cosine_sim(p_rgb_s, p_con)  # (1, 1, 1, 1)
        proto_sim_ir = self._cosine_sim(p_ir_s, p_con)    # (1, 1, 1, 1)

        # Combined adaptive weights (float32 for numerical stability)
        W_r = torch.sigmoid(sim_rgb + proto_sim_rgb)  # (B, 1, 1, 1)
        W_d = torch.sigmoid(sim_ir + proto_sim_ir)   # (B, 1, 1, 1)

        # Cast weights back to original dtype for conv compatibility
        W_r = W_r.to(dtype)
        W_d = W_d.to(dtype)

        # Step 3: Feature decomposition (in original dtype)
        F_c = W_r * rgb + W_d * ir                # consistency feature
        F_s_rgb = (1.0 - W_r) * rgb                # RGB specificity
        F_s_ir = (1.0 - W_d) * ir                  # IR specificity

        # Step 4: Concatenate and fuse
        out = torch.cat([F_c, F_s_rgb, F_s_ir], dim=1)  # (B, 3*c2, H, W)
        out = self.reduce(out)                           # (B, c2, H, W)
        out = self.norm(out)
        out = self.act(out)
        out = self.refine(out)

        return out

    def __repr__(self):
        c = self.P_con.shape[1]
        return f"DAIFusion(c={c})"
