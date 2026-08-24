#!/usr/bin/env bash
# Resume the fixed P2 checkpoint without changing its config fingerprint.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "usage: $0 MODEL HELIOS_ROOT DATASET_ROOT OUTPUT_DIR CHECKPOINT GPU_LIST" >&2
  exit 2
fi
MODEL=$1; HELIOS_ROOT=$2; DATASET_ROOT=$3; OUTPUT_DIR=$4; CHECKPOINT=$5; GPU_LIST=$6
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT_MANIFEST="$DATASET_ROOT/manifest_train_units_3chunk.json"
P3_MANIFEST="$DATASET_ROOT/manifest_train_p3.json"
[[ -f "$UNIT_MANIFEST" && -f "$P3_MANIFEST" && -f "$CHECKPOINT" ]] || { echo "missing manifest/checkpoint" >&2; exit 2; }
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --standalone --nproc_per_node=4 scripts/train_sightline_rgbd.py \
  --train --config configs/sightline.yaml --model "$MODEL" --helios-root "$HELIOS_ROOT" \
  --manifest "$UNIT_MANIFEST" --p3-manifest "$P3_MANIFEST" --resume "$CHECKPOINT" --output-dir "$OUTPUT_DIR"
