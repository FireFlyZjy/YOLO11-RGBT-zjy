# yolo26 end2end 移植记录

> 将 yolo26 的 end2end（端到端检测）特性移植到 YOLO11to26-Deepseek
> 日期：2026-07-17

---

## 移植内容

### 1. E2ELoss 损失函数
**文件**：`ultralytics/utils/loss.py`

新增 `E2ELoss` 类，实现了动态权重衰减的双头损失：

- `one2many`：标准检测损失（TAL topk=10）
- `one2one`：一对一匹配损失（TAL topk=7, topk2=1）
- 训练过程中 `o2m` 权重从 0.8 线性衰减到 0.1，`o2o` 权重从 0.2 增长到 0.9

### 2. Detect 双头输出
**文件**：`ultralytics/nn/modules/head.py`

新增两个 property：

- `one2many`：返回 `dict(box_head=self.cv2, cls_head=self.cv3)`
- `one2one`：返回 `dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)`

配合已有的 `forward_end2end` 方法，`end2end=True` 时输出 `{"one2many": ..., "one2one": ...}` 字典。

### 3. tasks.py end2end 集成
**文件**：`ultralytics/nn/tasks.py`

- `parse_model` 从 YAML 解析 `end2end` 标志
- Detect 头部传入 `end2end` 参数
- `base_modules` 分支改为 `elif`，避免多输入模块冲突
- 融合模块（GFM、SCA、EDS）加入多输入处理

### 4. unwrap_model 工具函数
**文件**：`ultralytics/utils/torch_utils.py`

新增 `unwrap_model` 函数，用于解包 DataParallel/DDP 包装的模型。

### 5. 训练器改进（已完成）
**文件**：`ultralytics/engine/trainer.py`

- NaN/Inf 损失自动恢复
- 断点续训完整状态加载
- BN 冻结支持

---

## 使用方式

### 普通 YAML（不启用 end2end，默认行为）
```yaml
nc: 3
scales:
  s: [0.50, 0.50, 1024]
ch: 4
# 不加 end2end 字段，或设为 false
```

### End2End YAML
```yaml
nc: 3
scales:
  s: [0.50, 0.50, 1024]
ch: 4
end2end: True
```

训练命令：
```bash
python train_AddModules.py --model_yaml path/to/end2end/model.yaml --batch 4
```

推理时 end2end 模型自动使用 one2one 头输出，无需 NMS 后处理。

---

## 验证

```bash
python -c "
from ultralytics import YOLO
model = YOLO('path/to/model.yaml')
model.info()
"
```
