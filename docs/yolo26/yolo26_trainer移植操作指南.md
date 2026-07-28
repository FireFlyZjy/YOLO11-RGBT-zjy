# yolo26 trainer.py 移植操作指南

> 将 yolo26 的训练稳定性改进移植到 YOLOv11-RGBT
> 目标文件：`ultralytics/engine/trainer.py`

---

## 改动清单（共 4 处）

### 改动 1：`__init__` 末尾 — 新增两个属性

**位置**：`__init__` 方法末尾

**添加**：
```python
self.nan_recovery_attempts = 0  # NaN 恢复尝试次数
self.best_fitness = None         # 最佳 fitness 值
```

**说明**：用于 `_handle_nan_recovery` 追踪恢复尝试次数和最佳性能。

---

### 改动 2：`_setup_train` 末尾 — 新增 freeze_layer_names

**位置**：`_setup_train` 方法末尾，`self.run_callbacks("on_pretrain_routine_end")` 之前

**添加**：
```python
# freeze layer names
freeze_list = (
    self.args.freeze
    if isinstance(self.args.freeze, list)
    else [self.args.freeze]
    if isinstance(self.args.freeze, str)
    else []
)
always_freeze_names = [  # always freeze these layers
    "model.38.dfl.conv.weight",
]
freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
self.freeze_layer_names = freeze_layer_names
```

**说明**：解析 `args.freeze` 参数，生成需要冻结的层名列表，`_model_train` 据此冻结 BN。

---

### 改动 3：新增三个方法

在 `save_metrics` 方法之后添加以下三个方法：

#### 3.1 `_model_train`

```python
def _model_train(self):
    """Set model in training mode."""
    self.model.train()
    # Freeze BN stat
    for n, m in self.model.named_modules():
        if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
            m.eval()
```

#### 3.2 `_handle_nan_recovery`

```python
def _handle_nan_recovery(self, epoch):
    """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
    loss_nan = self.loss is not None and not self.loss.isfinite()
    fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
    fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
    corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
    reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
    if RANK != -1:  # DDP: broadcast to all ranks
        broadcast_list = [corrupted if RANK == 0 else None]
        dist.broadcast_object_list(broadcast_list, 0)
        corrupted = broadcast_list[0]
    if not corrupted:
        return False
    if epoch == self.start_epoch or not self.last.exists():
        LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
        return False
    self.nan_recovery_attempts += 1
    if self.nan_recovery_attempts > 3:
        raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
    LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
    self._model_train()
    _, ckpt = load_checkpoint(self.last)
    ema_state = ckpt["ema"].float().state_dict()
    if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
        raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
    unwrap_model(self.model).load_state_dict(ema_state)
    self._load_checkpoint_state(ckpt)
    del ckpt, ema_state
    self.scheduler.last_epoch = epoch - 1
    return True
```

#### 3.3 `_load_checkpoint_state`

```python
def _load_checkpoint_state(self, ckpt):
    """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
    if ckpt.get("optimizer") is not None:
        self.optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scaler") is not None:
        self.scaler.load_state_dict(ckpt["scaler"])
    if self.ema and ckpt.get("ema"):
        self.ema = ModelEMA(self.model)
        self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
        self.ema.updates = ckpt["updates"]
    self.best_fitness = ckpt.get("best_fitness", 0.0)
```

---

### 改动 4：`_do_train` 中插入两个调用点

#### 4.1 每个 epoch 开始前调用 `_model_train`

**位置**：`self.scheduler.step()` 之后，在 warmup 之前

```python
self._model_train()
```

#### 4.2 每个 epoch 验证后调用 `_handle_nan_recovery`

**位置**：`self.save_metrics(...)` 之后，`self.stop` 判断之前

```python
# NaN recovery
if self._handle_nan_recovery(epoch):
    continue
```

---

## 所需新增 import

在文件头部 imports 中添加：

```python
from ultralytics.utils.torch_utils import unwrap_model
```

（如已有则跳过）

---

## 验证

```bash
python -c "compile(open('ultralytics/engine/trainer.py').read(), 'trainer.py', 'exec'); print('Syntax OK')"
```
