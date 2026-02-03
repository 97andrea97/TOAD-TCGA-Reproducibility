#!/bin/bash
set -euo pipefail

label_frac="$1"   # can be 0.1 or 10
gpu="$2"

num_repetitions=1

# Normalize label fraction to [0,1]
lf_float=$(python3 - <<'PY' "$label_frac"
import sys
v = float(sys.argv[1])
lf = v/100.0 if v > 1.0 else v
print(f"{lf:.6f}")
PY
)
pct_int=$(python3 - <<'PY' "$lf_float"
import sys
lf = float(sys.argv[1])
print(int(round(lf*100)))
PY
)
echo "Normalized label fraction: ${lf_float}  (${pct_int}%)"

# This path is now YOUR repo (bind-mounted)
cd /app/TOAD_repo

echo ">>> Running pipeline for label_frac=${lf_float} (${pct_int}%)"

echo "Start splitting"
CUDA_VISIBLE_DEVICES=$gpu python3 create_splits.py \
  --task TCGA --seed 1 --k 10 --label_frac "$lf_float"
echo "create_splits.py done"

SPLIT_DIR_BASENAME="TCGA_${pct_int}"
EXP_CODE="TCGA_label_${pct_int}"

echo "Start training"
CUDA_VISIBLE_DEVICES=$gpu python3 main_mtl_concat.py --weighted_sample --drop_out --early_stopping \
  --lr 2e-4 --k 10 --k_start 0 --k_end $num_repetitions --seed 1 \
  --exp_code "${EXP_CODE}" \
  --task TCGA \
  --split_dir "${SPLIT_DIR_BASENAME}" \
  --log_data \
  --results_dir /app/results \
  --data_root_dir /app/data/FEATURES
echo "main_mtl_concat.py done"

echo "Start evaluation on test"
CUDA_VISIBLE_DEVICES=$gpu python3 eval_mtl_concat.py --drop_out --k 10 --k_start 0 --k_end $num_repetitions \
  --fold 0 \
  --models_exp_code "${EXP_CODE}_s${num_repetitions}" \
  --save_exp_code "${EXP_CODE}_eval_test" \
  --task TCGA \
  --splits_dir "splits/${SPLIT_DIR_BASENAME}" \
  --results_dir /app/results \
  --split test \
  --data_root_dir /app/data/FEATURES
echo "eval_mtl_concat.py (test) done"

echo "Start evaluation on train"
CUDA_VISIBLE_DEVICES=$gpu python3 eval_mtl_concat.py --drop_out --k 10 --k_start 0 --k_end $num_repetitions \
  --fold 0 \
  --models_exp_code "${EXP_CODE}_s${num_repetitions}" \
  --save_exp_code "${EXP_CODE}_eval_train" \
  --task TCGA \
  --results_dir /app/results \
  --splits_dir "splits/${SPLIT_DIR_BASENAME}" \
  --split train \
  --data_root_dir /app/data/FEATURES
echo "eval_mtl_concat.py (train) done"
