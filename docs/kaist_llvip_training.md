# KAIST / LLVIP 训练命令 — 论文达标实验复现

> 本文档给出论文达标实验（FLIR 上 mAP50 > 0.865）在 **KAIST** 和 **LLVIP** 数据集上的训练命令。
> 基线模型：`yolo26-RGBT-midfusion.yaml`
> 达标标准：mAP50 > 0.865（见 `docs/指令.md` §12.4 与 `docs/paper_candidates/README.md`）

## 一、模型 YAML 目录（已整理好，训练无需任何准备步骤）

`ultralytics/cfg/models/26-RGBT/` 下新增两个目录，结构一一对应：

- **`flir-nc3/`**：FLIR 用（`nc: 3`），7 个达标实验原始 YAML 已移入
- **`kaist-llvip-nc1/`**：KAIST/LLVIP 用（`nc: 1`，person 单类），由 flir-nc3 复制并改 nc 生成

```
26-RGBT/
├── flir-nc3/                       # FLIR (3类) 原始配置
│   ├── origin/yolo26-RGBT-midfusion.yaml
│   ├── combo/yolo26-RGBT-midfusion-DASI-ProgAgg.yaml
│   ├── 2026-07-16/attention/yolo26-RGBT-midfusion-DASI.yaml
│   ├── 2026-07-16/attention/yolo26-RGBT-midfusion-DWR.yaml
│   ├── 2026-07-13/conv/yolo26-RGBT-midfusion-COI.yaml
│   ├── 2026-07-13/neck/yolo26-RGBT-midfusion-Aggregation.yaml
│   └── 2026-07-13/neck/yolo26-RGBT-midfusion-ProgressiveAgg.yaml
└── kaist-llvip-nc1/                # KAIST/LLVIP (1类) 同结构副本, nc=1
    └── （与 flir-nc3 相同子目录/文件名）
```

> 原位置（`origin/`、`combo/`、`2026-07-16/`、`2026-07-13/`）下的这 7 个文件已移入 `flir-nc3/`，其余 YAML 未动。

## 二、候选实验清单（FLIR 实测 mAP50）

| 配置 | mAP50 | mAP50-95 | 模型 YAML（`kaist-llvip-nc1/` 下相对路径） |
|------|-------|----------|--------------------------------------------|
| **基线** | 0.8495* | 0.4900 | `origin/yolo26-RGBT-midfusion.yaml` |
| **DASI-ProgAgg** | **0.8891** | 0.5271 | `combo/yolo26-RGBT-midfusion-DASI-ProgAgg.yaml` |
| **DASI** | 0.8860 | 0.5232 | `2026-07-16/attention/yolo26-RGBT-midfusion-DASI.yaml` |
| **DWR** | 0.8830 | 0.5262 | `2026-07-16/attention/yolo26-RGBT-midfusion-DWR.yaml` |
| **COI-150** | 0.8799 | 0.5216 | `2026-07-13/conv/yolo26-RGBT-midfusion-COI.yaml`（150 epochs） |
| **COI** | 0.8693 | 0.5088 | `2026-07-13/conv/yolo26-RGBT-midfusion-COI.yaml`（100 epochs） |
| **Aggregation** | 0.8674 | 0.5058 | `2026-07-13/neck/yolo26-RGBT-midfusion-Aggregation.yaml` |
| **ProgressiveAgg** | 0.8654 | 0.4998 | `2026-07-13/neck/yolo26-RGBT-midfusion-ProgressiveAgg.yaml` |

\* paper_candidates 记录值（FLIR 3 类 person/car/bicycle）。KAIST/LLVIP 只有 person 1 类，`kaist-llvip-nc1/` 下的 YAML 已把 `nc` 改为 1，可直接使用。

## 三、公共训练配置（与 FLIR 达标实验完全一致）

| 项目 | 配置 |
|------|------|
| 预训练权重 | `weights/yolo26s.pt`（已验证存在） |
| 模态 | RGBT 4 通道（RGB 3ch + IR 1ch），`--use_simotm RGBT --channels 4` |
| 数据配对 | `--pairs_rgb_ir "visible,infrared"`（KAIST 已重组为 images/visible + images/infrared 结构） |
| 超参 | `--epochs 100 --batch 4 --imgsz 640 --workers 0 --device 0 --optimizer SGD --close_mosaic 10` |
| 数据集 | `ultralytics/cfg/datasets/kaist.yaml`（7601/2252）· `ultralytics/cfg/datasets/llvip.yaml`（12025/3463） |
| 结果保存 | `runs/KAIST/` 与 `runs/LLVIP/`（与 `runs/FLIR/` 并列） |

> 显存提醒：FLIR 实验在 4GB 显卡上 COI 用了 batch=2（其余 batch=4）。KAIST/LLVIP 图像同为 640×512，建议沿用：
> COI / COI-150 用 `--batch 2`，其余 `--batch 4`；显存不足时全部降到 2。

## 四、训练命令

### 4.1 一键脚本（8 配置 × 单数据集）

> 该脚本已保存为项目根目录 **`train_kaist_llvip.sh`**（已加执行权限），直接用参数选择数据集，无需改文件：
>
> ```bash
> bash train_kaist_llvip.sh kaist    # 训练 KAIST → runs/KAIST/
> bash train_kaist_llvip.sh llvip    # 训练 LLVIP → runs/LLVIP/
> ```
>
> 脚本内置前置校验（数据集 yaml / 模型 yaml 存在性）。以下是脚本内容（如需自定义可复制修改）：

```bash
#!/bin/bash
# ============================================================================
# 论文达标实验一键训练脚本 — KAIST / LLVIP
# 用法:
#   bash train_kaist_llvip.sh kaist    # 训练 KAIST (默认)
#   bash train_kaist_llvip.sh llvip    # 训练 LLVIP
# 在项目根目录 /home/lyh/zjy/code/YOLOv11to26-RGBT 下运行
# 模型 YAML 取自 kaist-llvip-nc1/ (nc=1, 已就绪), 无需任何准备步骤
# 详细说明见 docs/kaist_llvip_training.md 与 docs/指令.md 第十三章
# ============================================================================
set -e

# ---- 数据集参数 ----
DATASET="${1:-kaist}"
case "$DATASET" in
  kaist)
    DATA="ultralytics/cfg/datasets/kaist.yaml"
    PROJECT="runs/KAIST"
    ;;
  llvip)
    DATA="ultralytics/cfg/datasets/llvip.yaml"
    PROJECT="runs/LLVIP"
    ;;
  *)
    echo "用法: bash train_kaist_llvip.sh [kaist|llvip]"
    exit 1
    ;;
esac

NC1_DIR="ultralytics/cfg/models/26-RGBT/kaist-llvip-nc1"

# ---- 实验配置: 名称|模型yaml(相对NC1_DIR)|epochs|batch ----
CONFIGS=(
  "baseline|origin/yolo26-RGBT-midfusion.yaml|100|4"
  "DASI-ProgAgg|combo/yolo26-RGBT-midfusion-DASI-ProgAgg.yaml|100|4"
  "DASI|2026-07-16/attention/yolo26-RGBT-midfusion-DASI.yaml|100|4"
  "DWR|2026-07-16/attention/yolo26-RGBT-midfusion-DWR.yaml|100|4"
  "COI|2026-07-13/conv/yolo26-RGBT-midfusion-COI.yaml|100|2"
  "COI-150|2026-07-13/conv/yolo26-RGBT-midfusion-COI.yaml|150|2"
  "Aggregation|2026-07-13/neck/yolo26-RGBT-midfusion-Aggregation.yaml|100|4"
  "ProgressiveAgg|2026-07-13/neck/yolo26-RGBT-midfusion-ProgressiveAgg.yaml|100|4"
)

# ---- 前置校验: 数据集 yaml 与所有模型 yaml 必须存在 ----
[ -f "$DATA" ] || { echo "[错误] 数据集 yaml 不存在: $DATA"; exit 1; }
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r _ YAML _ _ <<< "$c"
  if [ ! -f "$NC1_DIR/$YAML" ]; then
    echo "[错误] 模型 yaml 不存在: $NC1_DIR/$YAML"
    echo "       若为新模块, 请先按 docs/指令.md 13.4 移入 flir-nc3/ 并生成 nc=1 副本"
    exit 1
  fi
done

echo "===================================================================="
echo "  数据集: $DATA"
echo "  保存到: $PROJECT/"
echo "  模型目录: $NC1_DIR"
echo "===================================================================="

# ---- 逐个训练 ----
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME YAML EPOCHS BATCH <<< "$c"
  echo
  echo "===== [$DATASET] 训练 $NAME ($NC1_DIR/$YAML)  epochs=$EPOCHS batch=$BATCH ====="
  python train_AddModules.py \
    --model_yaml "$NC1_DIR/$YAML" \
    --data "$DATA" \
    --pairs_rgb_ir "visible,infrared" \
    --use_simotm RGBT --channels 4 \
    --pretrained weights/yolo26s.pt \
    --epochs "$EPOCHS" --batch "$BATCH" --imgsz 640 --workers 0 --device 0 \
    --optimizer SGD --close_mosaic 10 \
    --project "$PROJECT" --name "yolo26s-RGBT-midfusion-$NAME"
done

echo
echo "全部完成: $PROJECT/"
```

### 4.2 单条命令模板（单独跑某个实验）

```bash
python train_AddModules.py \
  --model_yaml ultralytics/cfg/models/26-RGBT/kaist-llvip-nc1/2026-07-16/attention/yolo26-RGBT-midfusion-DASI.yaml \
  --data ultralytics/cfg/datasets/kaist.yaml \
  --pairs_rgb_ir "visible,infrared" \
  --use_simotm RGBT --channels 4 \
  --pretrained weights/yolo26s.pt \
  --epochs 100 --batch 4 --imgsz 640 --workers 0 --device 0 \
  --optimizer SGD --close_mosaic 10 \
  --project runs/KAIST --name yolo26s-RGBT-midfusion-DASI
```

按需替换的字段：

| 字段 | 说明 |
|------|------|
| `--model_yaml` | 换成 `kaist-llvip-nc1/` 下对应 YAML（见第二节表格，路径含子目录；基线用 `origin/yolo26-RGBT-midfusion.yaml`） |
| `--data` | `kaist.yaml` 或 `llvip.yaml` |
| `--epochs / --batch` | COI 系列 `100/2` 或 `150/2`，其余 `100/4`（见 4.1 表） |
| `--project / --name` | `runs/KAIST` 或 `runs/LLVIP`；`--name` 与 FLIR 实验保持一致命名 `yolo26s-RGBT-midfusion-<模块>` |

## 五、训练后验证与记录

`train_AddModules.py` 训练完会自动验证并写入 `runs/val_results.csv`。批量验证某个数据集的所有实验用 `batch_val.py`
（扫描 `--search_dir` 下全部 `best.pt`；其验证默认 `use_simotm=RGBT`、`channels=4`、配对规则 `visible→infrared`，对重组后的 KAIST/LLVIP 正好适用）：

```bash
# 批量验证 runs/KAIST 下所有实验
python batch_val.py --search_dir runs/KAIST --data ultralytics/cfg/datasets/kaist.yaml --device 0

# 批量验证 runs/LLVIP 下所有实验
python batch_val.py --search_dir runs/LLVIP --data ultralytics/cfg/datasets/llvip.yaml --device 0
```

训练完成后把 mAP50 记录到 `runs/val_results.csv`，并对照 FLIR 结果检查涨点趋势。

## 六、注意事项

1. **训练 KAIST/LLVIP 直接用 `kaist-llvip-nc1/`**：该目录下 YAML 已全部是 `nc: 1`，无需再手动改 nc。`flir-nc3/` 保留 `nc: 3` 原件供 FLIR 使用，两者互不影响。
2. **KAIST 已重组**：`/home/lyh/zjy/dataset/KAIST` 已改为 `images/visible` + `images/infrared` + `labels/visible`（infrared 为软链接），默认配对 `visible→infrared` 即可，无需自定义 pairs。
3. **LLVIP 路径**：`images/visible` + `images/infrared` 均存在，默认配对即可。
4. 若中途中断，用 `--resume runs/KAIST/xxx/weights/last.pt` 续训（需去掉 `--pretrained`）。
5. 7 个 YAML 已从原位置（`origin/`、`combo/`、`2026-07-16/`、`2026-07-13/`）移入 `flir-nc3/`；若其他脚本/文档仍引用旧路径，请相应更新为 `flir-nc3/` 或 `kaist-llvip-nc1/`。
