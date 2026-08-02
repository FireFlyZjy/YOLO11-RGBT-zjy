# 代码模板与注册插入位置

> 本文件从 `指令.md` 拆分，包含模块类定义模板和 tasks.py 注册代码片段。
> 实施步骤 2/3 时按需查阅。

---

## 一、模块类定义模板

### 单输入模块 (替换 Conv/C3k2)

```python
class YourModule(nn.Module):
    """模块名: 一句话描述

    机制: (2-3句核心原理)
    对RGBT的价值: (1句)
    用法: [-1, 1, YourModule, [c2, ...args]]

    参数:
        c1: 输入通道数 (由parse_model自动注入)
        c2: 输出通道数
        ...: 其他参数
    """
    def __init__(self, c1, c2, ...):  # 必须含 c1, c2
        super().__init__()
        # 1x1 通道对齐 (必须)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        # 核心逻辑
        ...
    def forward(self, x):
        x = self.proj(x)
        ...
        return x
```

### 多输入模块 (替换 Concat, from 为列表)

```python
class YourFusion(nn.Module):
    """融合模块: 一句话描述

    用法: [[vis_layer, ir_layer], 1, YourFusion, [c2, ...args]]
    注意: from是列表时, c1会被parse_model自动转为list
    """
    def __init__(self, c1, c2, ...):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            c_vis, c_ir = c1[0], c1[1]  # 分别取两个输入通道
        else:
            c_vis = c_ir = c1
        self.proj_vis = nn.Conv2d(c_vis, c2, 1) if c_vis != c2 else nn.Identity()
        self.proj_ir  = nn.Conv2d(c_ir, c2, 1)  if c_ir != c2  else nn.Identity()
        ...
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            a, b = x[0], x[1]
        else:
            a = b = x
        a, b = self.proj_vis(a), self.proj_ir(b)
        ...
        return fused
```

### 检测头模块 (替换 Detect)

```python
class YourHead(nn.Module):
    """检测头: 一句话描述
    用法: [[P3, P4, P5], 1, YourHead, [nc, hidc]]
    """
    # 必需的类属性 (YOLO框架要求)
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        # 1x1投影对齐P3/P4/P5不同通道
        self.proj = nn.ModuleList([nn.Conv2d(c, hidc, 1, bias=False) for c in ch])
        ...
    def forward(self, x):
        x = [self.proj[i](x[i]) for i in range(self.nl)]
        # ... 检测逻辑
        # 训练路径返回 list; 推理路径返回 (y, x)
        if self.training: return x
        ...
        return y if self.export else (y, x)
    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[:self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)
    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
```

---

## 二、tasks.py 精确插入位置

### base_modules 追加位置

```
文件: ultralytics/nn/tasks.py
行号: 约1080行 (Att_ICAFusion 之后, } 之前)
操作: 追加新行

# 格式:
            ModuleName,  # 中文功能说明
```

### repeat_modules 追加位置

```
文件: ultralytics/nn/tasks.py
行号: 约1104行 (Att_IRAFAB 之后, } 之前)
操作: 仅当模块需要 n(重复次数) 参数时追加

# 格式:
            ModuleName,  # 中文说明
```

### parse_model 多输入分支追加位置

```
文件: ultralytics/nn/tasks.py
行号: 约1121行
操作: 在 if m in (...) 的括号内追加模块名

# 原行:
if m in (Att_ScalSeq, Att_AFF, Att_iAFF, Att_CIFusion, Att_ICAFusion):
# 追加后:
if m in (Att_ScalSeq, Att_AFF, Att_iAFF, Att_CIFusion, Att_ICAFusion,
         NewFusionModule):
```

### parse_model 检测头分支追加位置

```
文件: ultralytics/nn/tasks.py
行号: 约1125行
操作: 在 elif m in (...) 的括号内追加

# 原行:
elif m in (Detect_ATAH, DecoupledHead):
# 追加后:
elif m in (Detect_ATAH, DecoupledHead, NewHeadModule):
```

### isinstance 检查追加位置

```
文件: ultralytics/nn/tasks.py
行号: 约296行  (self._apply 方法内)
      约362行  (stride构建处)
操作: 追加 or isinstance(m, NewHeadModule)

# 格式:
if isinstance(m, DETECT_CLASS) or isinstance(m, (Detect_ATAH, DecoupledHead, NewHead)):
```
