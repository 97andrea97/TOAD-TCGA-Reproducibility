#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIMG_DEFAULT="$SCRIPT_DIR/assets/containers/singularity_preprocessing.simg"
CODE_DEFAULT="$SCRIPT_DIR/src_preprocessing/CLAM_encoder"

INPUT_DIR=""
OUT_ROOT=""
GPU="0"
BATCH_SIZE="32"
ENCODER=uni_v1 #"resnet50_trunc"
TARGET_PATCH_SIZE="224"
MAXDEPTH="2"
JOBS="1"

usage() {
  cat <<EOF
Usage:
  bash $(basename "$0") --input-dir /abs/path --out-root /abs/path [options]

Required:
  --input-dir   Folder containing .svs files (searched up to depth ${MAXDEPTH})
  --out-root    preprocessing_output folder

Options:
  --gpu               GPU id (default: ${GPU})
  --jobs              Parallel slides on the same GPU (default: ${JOBS})
  --batch-size        Batch size (default: ${BATCH_SIZE})
  --encoder           resnet50_trunc | uni_v1 | conch_v1 (default: ${ENCODER})
  --target-patch-size Target patch size (default: ${TARGET_PATCH_SIZE})
  --maxdepth          Search depth (default: ${MAXDEPTH})
  --image             Singularity image path (default: ${SIMG_DEFAULT})
  --code              CLAM code folder (default: ${CODE_DEFAULT})
EOF
}

SIMG="$SIMG_DEFAULT"
CLAM_CODE_DIR="$CODE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --encoder) ENCODER="$2"; shift 2;;
    --target-patch-size) TARGET_PATCH_SIZE="$2"; shift 2;;
    --maxdepth) MAXDEPTH="$2"; shift 2;;
    --image) SIMG="$2"; shift 2;;
    --code) CLAM_CODE_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -n "$INPUT_DIR" ]] || { echo "ERROR: --input-dir is required"; usage; exit 1; }
[[ -n "$OUT_ROOT" ]]  || { echo "ERROR: --out-root is required"; usage; exit 1; }

INPUT_REAL="$(readlink -f "$INPUT_DIR")"
OUT_REAL="$(readlink -f "$OUT_ROOT")"
CODE_REAL="$(readlink -f "$CLAM_CODE_DIR")"
SIMG_REAL="$(readlink -f "$SIMG")"

META_ROOT="$OUT_REAL/logs_and_metadata"
FEATURES_ROOT="$OUT_REAL/FEATURES"

mkdir -p "$META_ROOT" "$FEATURES_ROOT"

export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU"

mapfile -d '' SLIDES < <(find "$INPUT_REAL" -maxdepth "$MAXDEPTH" -type f -iname "*.svs" -print0)

if (( ${#SLIDES[@]} == 0 )); then
  echo "No .svs found in $INPUT_REAL (maxdepth $MAXDEPTH)"
  exit 0
fi

limit_jobs() {
  local max="$1"
  while (( $(jobs -rp | wc -l) >= max )); do
    sleep 0.2
  done
}

run_one() {
  local svs_real="$1"

  local rel="${svs_real#"$INPUT_REAL"/}"
  local file_id
  file_id="$(cut -d/ -f1 <<< "$rel")"

  local slide_file
  slide_file="$(basename "$svs_real")"
  local slide_stem="${slide_file%.*}"
  local hash
  hash="$(printf "%s" "$rel" | sha1sum | awk '{print substr($1,1,8)}')"

  local meta_dir="$META_ROOT/$file_id/${slide_stem}__${hash}"
  mkdir -p "$meta_dir"

  local tmp_in
  tmp_in="$(mktemp -d)"

  echo "Processing: $svs_real"
  echo "Meta out:   $meta_dir"
  echo "Features:   $FEATURES_ROOT"

  singularity exec --nv --cleanenv --containall \
    --bind "$CODE_REAL:/app/CLAM:ro" \
    --bind "$tmp_in:/app/input_data" \
    --bind "$svs_real:/app/input_data/$slide_file:ro" \
    --bind "$meta_dir:/app/meta_out" \
    --bind "$FEATURES_ROOT:/app/features_out" \
    "$SIMG_REAL" \
    bash -lc 'cd /app/CLAM && bash run_clam_single_input.sh /app/input_data /app/meta_out /app/features_out "$1" "$2" "$3" "$4"' \
    -- "$slide_file" "$BATCH_SIZE" "$ENCODER" "$TARGET_PATCH_SIZE" \
    > "$meta_dir/run.log" 2>&1

  rm -rf "$tmp_in"
}

for SVS in "${SLIDES[@]}"; do
  SVS_REAL="$(readlink -f "$SVS")"
  limit_jobs "$JOBS"
  run_one "$SVS_REAL" &
done

wait
