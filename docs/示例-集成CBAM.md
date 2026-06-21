# 端到端完整示例：集成 CBAM

## 三、端到端完整示例：集成 CBAM

> 以下以 CBAM (Convolutional Block Attention Module) 为例，展示完整流程。

### 示例步骤1: 扫描筛选

```
读: objectdetection_script/cv-attention/CBAM.py
判断: 纯PyTorch, ~50K参数, 通道+空间双重注意力
与已有对照: SE仅通道, CoordAtt仅坐标 → CBAM互补
决定: ✅ 纳入
```

### 示例步骤2: 实现模块

创建 `ultralytics/nn/ExtraModules/attention/CBAM.py`:

```python
import torch
import torch.nn as nn

class Att_CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module (通道+空间双重注意力)

    机制: 通道注意力(MLP+双池化) 串行 空间注意力(7×7卷积池化)
    对RGBT的价值: 通道注意力筛选模态重要特征, 空间注意力抑制背景噪声
    用法: [-1, 1, Att_CBAM, [c2, reduction_ratio]]

    参数:
        c1: 输入通道数
        c2: 输出通道数
        reduction_ratio: 通道压缩比 (default 16)
    """
    def __init__(self, c1, c2, reduction_ratio=16):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(c2, c2//reduction_ratio, 1), nn.ReLU(),
            nn.Conv2d(c2//reduction_ratio, c2, 1), nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.proj(x)
        # 通道注意力
        c_att = self.channel_att(x)
        x = x * c_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * s_att
```

### 示例步骤3: 注册

**3a. `__init__.py`**: 追加一行
```python
from .attention.CBAM import *
```

**3b. `tasks.py` base_modules**: 在 `}` 之前追加
```python
            Att_CBAM,    # CBAM: 通道注意力(MLP双池化)+空间注意力(7×7卷积)，串行抑制背景噪声
```

### 示例步骤4: 创建 YAML

创建 `ultralytics/cfg/models/26-RGBT/2026-05-13/yolo26-RGBT-midfusion-Att_CBAM.yaml`:

```yaml
# CBAM: 通道+空间双重注意力
# 改动: P3/P4/P5 Concat融合之后各插入一个Att_CBAM
# 层号位移: 新增层22,24,26; Backbone+3; Head全部+3

nc: 3
scales:
  s: [0.50, 0.50, 1024]
ch: 4
backbone:
  # ... (前21层同baseline)
  - [[6, 16], 1, Concat, [1]]                                         # 21-P3 融合
  - [-1, 1, Att_CBAM, [512]]                                          # 22 ← NEW
  - [[8, 18], 1, Concat, [1]]                                         # 23-P4 融合 (原22)
  - [-1, 1, Att_CBAM, [512]]                                          # 24 ← NEW
  - [[10, 20], 1, Concat, [1]]                                        # 25-P5 融合 (原23)
  - [-1, 1, Att_CBAM, [1024]]                                         # 26 ← NEW
  - [-1, 1, SPPF, [1024, 5, 3, True]]                                # 27 (原24)
  - [-1, 2, C2PSA, [1024]]                                           # 28 (原25)
head:
  # 所有层号+3
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]                       # 29 (原26)
  - [[-1, 24], 1, Concat, [1]]                                        # 30 (原27, ref 22→24)
  - [-1, 2, C3k2, [512, False]]                                       # 31 (原28)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]                       # 32 (原29)
  - [[-1, 22], 1, Concat, [1]]                                        # 33 (原30, ref 21→22)
  - [-1, 2, C3k2, [256, False]]                                       # 34 (原31)
  - [-1, 1, Conv, [256, 3, 2]]                                       # 35 (原32)
  - [[-1, 31], 1, Concat, [1]]                                        # 36 (原33, ref 28→31)
  - [-1, 2, C3k2, [512, False]]                                       # 37 (原34)
  - [-1, 1, Conv, [512, 3, 2]]                                       # 38 (原35)
  - [[-1, 28], 1, Concat, [1]]                                        # 39 (原36, ref 25→28)
  - [-1, 2, C3k2, [1024, True, 0.5, True]]                          # 40 (原37)
  - [[34, 37, 40], 1, Detect, [nc]]                                   # 41 (原38, ref 31,34,37→34,37,40)

# 层号位移对照:
# Backbone: 22,24,26 新增 → 23(22→24),24→27,25→28
# Head: 全部+3, Detect [31,34,37]→[34,37,40]
```

### 示例步骤5: 验证

```bash
python -c "from ultralytics.nn.ExtraModules.attention.CBAM import Att_CBAM; print('import OK')"
python -c "import torch; m=__import__('...CBAM', fromlist=['Att_CBAM']).Att_CBAM(64,128); print(m(torch.randn(2,64,32,32)).shape)"
python -c "from ultralytics import YOLO; YOLO('ultralytics/cfg/models/26-RGBT/2026-05-13/yolo26-RGBT-midfusion-Att_CBAM.yaml').info()"
```

### 示例步骤6: 更新文档

```markdown
## 2026-05-13

### 新增 CBAM 注意力模块
| 模块 | 论文 | 类型 | 参数量 |
|------|------|------|--------|
| Att_CBAM | CBAM (ECCV 2018) | 通道+空间 | +0.05M |
插入策略: P3/P4/P5 Concat融合之后
验证结果: 通过实例化和forward pass
```

---
