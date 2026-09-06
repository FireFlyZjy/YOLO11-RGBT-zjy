#!/bin/bash
# ============================================================================
# 自适应续训脚本 — KAIST / LLVIP (对应 train_kaist_llvip.sh)
# 自动检测每个实验的状态:
#   0 = 已完成  -> 跳过 (results.csv 最后一行 epoch >= 目标)
#   1 = 中断中  -> 从 weights/last.pt 续训
#   2 = 未开始  -> 从头训练
#
# 用法:
#   bash resume_kaist_llvip.sh kaist    # 默认 KAIST
#   bash resume_kaist_llvip.sh llvip    # 训练 LLVIP
#
# 在项目根目录下运行
# 实验清单从 configs/experiments_kaist_llvip.txt 读取 (与 train_kaist_llvip.sh 共用)
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
    echo "用法: bash resume_kaist_llvip.sh [kaist|llvip]"
    exit 1
    ;;
esac

NC1_DIR="ultralytics/cfg/models/26-RGBT/kaist-llvip-nc1"
PRETRAINED="weights/yolo26s.pt"
CONFIG_FILE="configs/experiments_kaist_llvip.txt"

# ---- 从配置文件加载实验清单 (与 train_kaist_llvip.sh 共用) ----
[ -f "$CONFIG_FILE" ] || { echo "[错误] 实验配置文件不存在: $CONFIG_FILE"; exit 1; }
CONFIGS=()
while IFS= read -r line; do
  # 跳过空行和注释行
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  CONFIGS+=("$line")
done < "$CONFIG_FILE"

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "[错误] 配置文件 $CONFIG_FILE 中没有有效实验条目"
  exit 1
fi

# ============================================================================
# 状态检测
#   0 = 已完成 (results.csv 最后一行 epoch >= 目标)
#   1 = 有 last.pt 可续训
#   2 = 未开始 (无 last.pt)
# ============================================================================
check_status() {
  local name="$1" epochs="$2"
  local dir="$PROJECT/yolo26s-RGBT-midfusion-$name"

  # 目录不存在 -> 未开始
  if [ ! -d "$dir" ]; then
    echo "2"; return
  fi

  # 读 results.csv 最后一行的 epoch 列
  local csv="$dir/results.csv"
  if [ -f "$csv" ]; then
    local last_epoch
    last_epoch=$(tail -1 "$csv" | cut -d',' -f1 | tr -d ' ')
    if [[ "$last_epoch" =~ ^[0-9]+$ ]] && [ "$last_epoch" -ge "$epochs" ]; then
      echo "0"; return
    fi
  fi

  # 有 last.pt 但未完成 -> 续训
  if [ -f "$dir/weights/last.pt" ]; then
    echo "1"; return
  fi

  # 目录存在但既无 results.csv 也无 last.pt (例如上一次启动即崩溃) -> 视为未开始
  echo "2"
}

# ============================================================================
# 执行单个实验
# ============================================================================
run_experiment() {
  local name="$1" yaml="$2" epochs="$3" batch="$4" status="$5"
  local dir="$PROJECT/yolo26s-RGBT-midfusion-$name"
  local last_pt="$dir/weights/last.pt"

  echo
  echo "===================================================================="
  echo "  [$DATASET] $name"
  if [ "$status" = "0" ]; then
    echo "  状态: [跳过] 已完成"
  elif [ "$status" = "1" ]; then
    echo "  状态: [续训] 从 $last_pt 恢复"
  else
    echo "  状态: [新训] 从头开始"
  fi
  echo "  YAML:   $NC1_DIR/$yaml"
  echo "  Epochs: $epochs  Batch: $batch"
  echo "===================================================================="

  # 已完成则直接返回
  if [ "$status" = "0" ]; then
    return 0
  fi

  if [ "$status" = "1" ]; then
    # ---- 续训模式: --resume 加载 checkpoint, 自动恢复 optimizer / scheduler / epoch ----
    # 注意: 续训时不需要 --pretrained 和 --optimizer (由 checkpoint 决定)
    python train_AddModules.py \
      --resume "$last_pt" \
      --data "$DATA" \
      --pairs_rgb_ir "visible,infrared" \
      --use_simotm RGBT --channels 4 \
      --epochs "$epochs" --batch "$batch" --imgsz 640 --workers 0 --device 0 \
      --close_mosaic 10 \
      --project "$PROJECT" --name "yolo26s-RGBT-midfusion-$name"
  else
    # ---- 新训模式: YAML + 预训练权重 ----
    [ -f "$NC1_DIR/$yaml" ] || { echo "[错误] 模型 yaml 不存在: $NC1_DIR/$yaml"; exit 1; }
    python train_AddModules.py \
      --model_yaml "$NC1_DIR/$yaml" \
      --data "$DATA" \
      --pairs_rgb_ir "visible,infrared" \
      --use_simotm RGBT --channels 4 \
      --pretrained "$PRETRAINED" \
      --epochs "$epochs" --batch "$batch" --imgsz 640 --workers 0 --device 0 \
      --optimizer SGD --close_mosaic 10 \
      --project "$PROJECT" --name "yolo26s-RGBT-midfusion-$name"
  fi
}

# ============================================================================
# 前置校验
# ============================================================================
[ -f "$DATA" ]        || { echo "[错误] 数据集 yaml 不存在: $DATA"; exit 1; }
[ -f "$PRETRAINED" ]  || { echo "[错误] 预训练权重不存在: $PRETRAINED"; exit 1; }

echo "===================================================================="
echo "  自适应续训脚本"
echo "===================================================================="
echo "  数据集:   $DATA"
echo "  保存到:   $PROJECT/"
echo "  模型目录: $NC1_DIR"
echo "  预训练:   $PRETRAINED"
echo "  实验清单: $CONFIG_FILE  (${#CONFIGS[@]} 个)"
echo "===================================================================="

# ============================================================================
# 第一轮: 状态扫描 (只做检测, 不训练)
# ============================================================================
echo
echo "---- 状态扫描 ----"
done_count=0
resume_count=0
fresh_count=0
STATUSES=()
NAMES=()
YAMLS=()
EPOCHS=()
BATCHES=()

for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name yaml epochs batch <<< "$c"
  status=$(check_status "$name" "$epochs")
  NAMES+=("$name"); YAMLS+=("$yaml"); EPOCHS+=("$epochs"); BATCHES+=("$batch"); STATUSES+=("$status")

  case "$status" in
    0) printf "  [跳过] %-16s  (目标 %s epochs) - 已完成\n" "$name" "$epochs"; ((done_count++)) ;;
    1) printf "  [续训] %-16s  (目标 %s epochs) - 从 last.pt 恢复\n" "$name" "$epochs"; ((resume_count++)) ;;
    2) printf "  [新训] %-16s  (目标 %s epochs) - 从头开始\n" "$name" "$epochs"; ((fresh_count++)) ;;
  esac
done

echo
echo "  汇总: 已完成 $done_count | 待续训 $resume_count | 待新训 $fresh_count"
echo

# 如果没有任何待执行项, 直接退出
if [ "$resume_count" -eq 0 ] && [ "$fresh_count" -eq 0 ]; then
  echo "所有实验均已完成, 无需执行。"
  exit 0
fi

# ============================================================================
# 第二轮: 执行 (跳过已完成的)
# ============================================================================
n=${#NAMES[@]}
i=0
for (( i=0; i<n; i++ )); do
  run_experiment "${NAMES[$i]}" "${YAMLS[$i]}" "${EPOCHS[$i]}" "${BATCHES[$i]}" "${STATUSES[$i]}"
done

echo
echo "===================================================================="
echo "  本轮完成: 跳过 $done_count | 续训 $resume_count | 新训 $fresh_count"
echo "  全部结果: $PROJECT/"
echo "  如需重跑已完成实验, 删除对应目录后再次运行本脚本"
echo "===================================================================="
