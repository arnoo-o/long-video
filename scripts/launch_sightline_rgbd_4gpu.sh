#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 MODEL HELIOS_ROOT DATASET_ROOT OUTPUT_DIR [LATENT_CACHE_ROOT]" >&2
  exit 2
fi

MODEL=$1
HELIOS_ROOT=$2
DATASET_ROOT=$3
OUTPUT_DIR=$4
LATENT_CACHE_ROOT=${5:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

command -v torchrun >/dev/null
cd "$REPO_ROOT"
COMMAND=(torchrun --standalone --nproc_per_node=4 scripts/train_sightline_rgbd.py
  --train --config configs/sightline.yaml --model "$MODEL" --helios-root "$HELIOS_ROOT"
  --manifest "$DATASET_ROOT/manifest_train.json" --expected-records 400
  --output-dir "$OUTPUT_DIR")
if [[ -n "$LATENT_CACHE_ROOT" ]]; then
  COMMAND+=(--latent-cache-root "$LATENT_CACHE_ROOT")
fi
exec "${COMMAND[@]}"
