#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_clam_single_input.sh <input_dir> <meta_out_dir> <features_out_dir> <slide.svs> <batch_size> <encoder> <target_patch_size>

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <input_dir> <meta_out_dir> <features_out_dir> <slide.svs> <batch_size> [encoder] [target_patch_size]"
  exit 1
fi

INPUT_DIR="$1"
META_OUT="$2"
FEATURES_OUT="$3"
SLIDE_NAME="$(basename "$4")"
BATCH_SIZE="${5:-800}"
ENCODER="${6:-resnet50_trunc}"
TARGET_PATCH_SIZE="${7:-224}"

SLIDE_STEM="${SLIDE_NAME%.*}"

mkdir -p \
  "$META_OUT/masks" \
  "$META_OUT/patches" \
  "$META_OUT/stitches" \
  "$META_OUT/csv_files" \
  "$META_OUT/logs"

# shared features root (single global folder)
mkdir -p "$FEATURES_OUT"

# Patch creation
echo "Starting patch creation"
python3 create_patches_fp.py \
  --source "$INPUT_DIR" \
  --save_dir "$META_OUT" \
  --preset tcga.csv \
  --patch_level 0 \
  --seg --patch \
  > "$META_OUT/logs/${SLIDE_STEM}_patch_creation.log" 2>&1

cp -f "$META_OUT/process_list_autogen.csv" \
  "$META_OUT/csv_files/${SLIDE_STEM}_process_list_autogen.csv"

echo "Starting feature extraction"
# Feature extraction (write into GLOBAL folder)
python extract_features_fp.py \
  --data_h5_dir "$META_OUT" \
  --data_slide_dir "$INPUT_DIR" \
  --csv_path "$META_OUT/csv_files/${SLIDE_STEM}_process_list_autogen.csv" \
  --feat_dir "$FEATURES_OUT" \
  --slide_ext .svs \
  --batch_size "$BATCH_SIZE" \
  --target_patch_size "$TARGET_PATCH_SIZE" \
  --model_name "$ENCODER" \
  > "$META_OUT/logs/${SLIDE_STEM}_feature_extraction.log" 2>&1

echo "Feature extraction completed"