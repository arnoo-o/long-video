#!/usr/bin/env bash
set -euo pipefail

base=/ephemeral/ubuntu_cache/sightline-parallel
dataset_root=$base/rgbd_memory_dataset
archive=$dataset_root/raw/nrgbd/neural_rgbd_data.zip
extract_root=$dataset_root/extracted/nrgbd

unzip -tq "$archive"
mkdir -p "$extract_root"
unzip -q -o "$archive" -d "$extract_root"
touch "$extract_root/.complete"
cd "$base/repo"
PYTHONPATH="$base/repo" "$base/rgbd-env/bin/python" scripts/build_rgbd_memory_dataset.py \
  --dataset nrgbd \
  --source-root "$extract_root" \
  --output-root "$dataset_root/processed" \
  --correspondence-pixel-stride 4
