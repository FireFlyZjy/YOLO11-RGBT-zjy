# 原始模块
from .attention.SE import *
from .conv.ACBlock import *
from .conv.BlazeBlock import *
from .attention.GatingContext import *

# 2026-05-13 注意力模块 (来自 cv-attention)
from .attention.CBAM import *
from .attention.CoordAtt import *
from .attention.SimAM import *
from .attention.CPCA import *
from .attention.EMA import *
from .attention.ECA import *
from .attention.ShuffleAtt import *
from .attention.LSKA import *
from .attention.TripletAtt import *
from .attention.GAM import *
from .attention.ELA import *

# 2026-05-14 模块 (yolo-improve + Plug-and-play module)
from .attention.AIFI import *
from .attention.ULSAM import *
from .attention.StripPool import *

# 2026-05-28 scSE 模块 (Spatial + Channel SE)
from .attention.scSE import *
from .fusion.AFF import *
from .fusion.vHeat_Fusion import *
from .fusion.FCMMFusion import *
from .fusion.DMAF import *
from .fusion.EI2Fusion import *

# 2026-08-06 模块 (原创融合: DCAF - 差异补偿自适应融合)
from .fusion.DCAF import *

# 2026-08-04 模块 (VCP-DCN - DAI融合)
from .fusion.DAIFusion import *

# 2026-05-19 模块 (CMFADet)
from .attention.SFEM import *
from .attention.IR_AFAB import *
from .fusion.CIFusion import *
from .conv.DEConv import *

# 2026-05-26 模块 (CMFADet - ATAH检测头)
from .head import *

# 2026-05-27 模块 (yolo-improve 新增)
from .conv.CoordConv import *
from .conv.PConv import *
from .conv.TridentBlock import *
from .conv.StarBlock import *
from .dynamic.DynamicConv import *
from .dynamic.DGC import *
from .dynamic.RepMLP import *
from .context.ContextAgg import *
from .context.EVC import *

# 2026-05-27 颈部模块 (neck)
from .neck import *

# 2026-05-27 Mamba/SSM模块 (EfficientViM, MobileMamba)
from .mamba import *

# 2026-05-27 频域模块 (FDConv, vHeat, TOST, FADC, DarkIR, SFSConv)
from .frequency.FDConv import *
from .frequency.vHeat import *
from .frequency.TOST import *
from .frequency.FADC import *
from .frequency.DarkIR import *
from .frequency.SFSConv import *
from .frequency.vHeat_Block import *

# 2026-05-28 FPT: Feature Pyramid Transformer — 自注意力/跨层接地/渲染融合
from .attention.FPT import *

# 2026-05-28 DANet: Dual Attention Network — 位置注意力+通道注意力
from .attention.DANet import *

# 2026-05-28 BAM: Bottleneck Attention Module — 通道+空洞空间注意力
from .attention.BAM import *

# 2026-05-28 Conv增强模块 (DOConv, RFB)
from .conv.DOConv import *
from .conv.RFB import *

# 2026-05-27 损失工具 (NWD, Soft-NMS, 独立使用不修改 ultralytics 源码)
from .loss import *

# 2026-06-23 新增注意力 (来自 Fracture_Detection/timm, Gather-Excite)
from .attention.GatherExcite import *   # GatherExcite: 局部/全局聚合+MLP通道门控

# 2026-06-23 新增模块 (来自 LCAFNet, 跨模态交叉注意力融合)
from .fusion.HAFFormer import *         # HAFFormer: 层级注意力融合Transformer (DW-Conv QKV + 门控)
from .fusion.CrossAttention_M import *  # CrossAttention_M: 双向DW-Conv交叉注意力

# 2026-06-27 新增模块 (来自 CVPR2026 BinaryAttention, 1-bit QK注意力)
from .attention.BinaryAttention import *  # BinaryAttention: 1-bit量化注意力, C3k2_BinaryAttention

# 2026-06-27 新增模块 (AFFN频域自相关融合)
from .frequency.AFFN import *  # AFFN: 频域自相关融合网络, C2PSA_AFFN

# 2026-06-27 新增模块 (Flickerformer频域/相位注意力)
from .frequency.Flickerformer import *  # PAM, PhaseGuidedFilter, FSAS, SCAM

# 2026-06-27 新增模块 (PolyNeXt多项式神经网络)
from .attention.PolyNeXt import *  # PolyAttention, PolyConv, C2PSA_PolyAttention, C3k2_PolyConv

# 2026-06-30 新增模块 (WDAM小波双注意力)
from .frequency.WDAM import *  # WDAM, C2PSA_WDAM

# 2026-07-03 新增模块 (C2Former跨模态可变形交叉注意力)
from .fusion.C2Former import *  # C2Former_Fusion, C2PSA_C2Former

# 2026-07-03 新增模块 (MARSS: CVPR2026 雷达语义分割模块)
from .fusion.MARSS import *  # REM, RADE, RFAF_Fusion, RADM, C2PSA_RADM

# 2026-07-05 新增模块 (CKConv多尺度十字形卷积)
from .conv.CKConv import *  # CKConv, C3k2_CKConv

# 2026-07-05 新增模块 (CROWn微观多相共注意力)
from .frequency.CROWn import *  # muPCAD_2D, C2PSA_muPCAD, CrossSourceMHA

# 2026-06-21 新增模块 (来自 yoloair-main)
# --- attention/ 新增注意力 ---
from .attention.CrissCross import *   # CrissCrossAttention: 十字交叉注意力
from .attention.SOCA import *         # SOCA: 二阶通道注意力(协方差池化)
from .attention.SKAttention import *  # SKAttention: 选择性核注意力
from .attention.NAM import *          # NAMAttention: 极轻量BN统计通道注意力
from .attention.S2Attention import *  # S2Attention: 空间移位注意力
from .attention.ACmix import *        # ACmix: 混合注意力+卷积
from .attention.BoT3 import *         # BoT3: 瓶颈Transformer
# --- conv/ 新增卷积 ---
from .conv.GSConv import *            # GSConv/VoVGSCSP: Ghost Shuffle轻量卷积
from .conv.Involution import *        # Involution: 逆卷积算子
# --- context/ 新增上下文 ---
from .context.SPPCSPC import *        # SPPCSPC/SPPFCSPC: CSP增强SPP
# --- fusion/ 新增融合 ---
from .fusion.ASFFFusion import *      # ASFFFusion: 自适应空间特征融合(双模态)

# 2026-06-30 新增模块 (来自 GeoFuse-YOLO, SOEP小目标增强金字塔)
from .conv.SPDConv import *            # SPDConv: 空间到深度卷积, 保持分辨率的小目标增强
from .conv.CSPOmniKernel import *      # CSPOmniKernel: CSP多尺度全方向卷积+频域门控

# 2026-07-07 新增模块 (来自 exps/HVPNet — RGB-T 显著目标检测)
from .attention.SRA import *          # SRA: Strip Recurrent Attention, 条带循环注意力
from .fusion.SCA import *             # SCA: Spatial-Channel RGB-T Fusion
from .fusion.GFM import *             # GFM: Global Fusion Module, 全局融合模块(SAttention)
from .fusion.EDS import *             # EDS: Edge-aware Dynamic Sampling, 边缘感知融合
from .attention.HA import *           # HA: Holistic Attention, 全局注意力精炼
from .conv.RFBBranch import *         # RFBBranch: 4分支非对称RFB(与BasicRFB不同)
from .conv.COI import *               # COI: Conv-One-Identity 三重残差

# 2026-07-07 新增模块 (来自 exps/rmae-progress — ProGRess 渐进式融合)
from .neck.ProgressiveAgg import *    # ProgressiveAgg: 渐进式特征聚合
from .neck.Aggregation import *       # Aggregation: 3尺度密集注意力聚合

# 2026-07-07 新增模块 (来自 exps/UCMNet — 记忆增强注意力)
from .attention.MemoryAttention import *  # MemoryAttention: 记忆增强注意力(可学习码本)

# 2026-07-13 新增模块 (来自 exps HTML教程 — CSDN YOLO11魔术师专栏)
from .conv.PConv_Windmill import *    # PConv: 风车形卷积 AAAI2025
from .attention.CAMixing import *     # CAMixing: 卷积-注意融合模块
from .attention.MDCR import *         # MDCR: 多膨胀通道精炼 红外小目标
from .attention.PPA import *          # PPA: 并行化注意力设计 红外小目标
from .attention.DASI import *         # DASI: 维度感知选择性集成 红外小目标
from .attention.StripBlock import *   # StripBlock: 大型条带卷积 StripR-CNN 2025
from .attention.SHViT import *        # SHViT: 单头自注意力 CVPR2024
from .attention.DWR import *          # DWR: 可扩张残差注意力
from .attention.MSDA import *         # MSDA: 多尺度空洞注意力
from .attention.HSFPN import *        # HS-FPN: 多级特征融合金字塔
from .attention.DAB import *          # DAB: 双注意力块 遥感去雾
from .conv.CMUNeXt import *          # CMUNeXt: 大核倒瓶颈设计
from .neck.DySample import *          # DySample: 动态上采样

# YOLO 包装器 (必须在独立模块之后导入)
from .common import *
