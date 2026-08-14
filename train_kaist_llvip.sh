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
