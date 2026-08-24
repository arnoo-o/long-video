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
UNIT_MANIFEST="$DATASET_ROOT/manifest_train_units_3chunk.json"
P3_MANIFEST="$DATASET_ROOT/manifest_train_p3.json"

command -v torchrun >/dev/null
[[ -f "$UNIT_MANIFEST" && -f "$P3_MANIFEST" ]] || { echo "missing unified P1/P2 or P3 manifest" >&2; exit 2; }
cd "$REPO_ROOT"
COMMAND=(torchrun --standalone --nproc_per_node=4 scripts/train_sightline_rgbd.py
  --train --config configs/sightline.yaml --model "$MODEL" --helios-root "$HELIOS_ROOT"
  --manifest "$UNIT_MANIFEST" --p3-manifest "$P3_MANIFEST"
  --output-dir "$OUTPUT_DIR")
if [[ -n "$LATENT_CACHE_ROOT" ]]; then
  COMMAND+=(--latent-cache-root "$LATENT_CACHE_ROOT")
fi
exec "${COMMAND[@]}"
