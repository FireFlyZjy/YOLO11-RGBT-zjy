"""
Mamba/SSM 模块初始化
====================
基于状态空间模型(SSM)的模块集合，适用于 RGB-Infrared 多模态检测任务。

包含模块:
    - EfficientViM_Block: 隐状态混合 SSD 变体 (CVPR2025)
    - WTE_Mamba:          小波变换增强 Mamba (CVPR2025)
    - 纯 PyTorch 实现的 SSM 工具函数 (无 mamba-ssm 依赖)
"""

from .EfficientViM import EfficientViM_Block
from .MobileMamba import WTE_Mamba

__all__ = [
    "EfficientViM_Block",
    "WTE_Mamba",
]
