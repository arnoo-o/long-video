#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${1:?raw 7Scenes archive directory required}
EXTRACT_ROOT=${2:?7Scenes extraction directory required}
OUTPUT_ROOT=${3:?processed dataset directory required}
REPO_ROOT=${4:?repository directory required}

mkdir -p "$EXTRACT_ROOT"

unzip_allow_warning() {
  local code=0
  unzip "$@" || code=$?
  if (( code > 1 )); then
    return "$code"
  fi
}

for archive in "$RAW_ROOT"/*.zip; do
  marker="$EXTRACT_ROOT/.top-$(basename "$archive").complete"
  if [[ ! -f "$marker" ]]; then
    unzip_allow_warning -tq "$archive" >/dev/null
    unzip_allow_warning -q -o "$archive" -d "$EXTRACT_ROOT"
    touch "$marker"
  fi
done

while IFS= read -r sequence_archive; do
  sequence_dir="${sequence_archive%.zip}"
  marker="${sequence_dir}.complete"
  if [[ ! -f "$marker" ]]; then
    mkdir -p "$sequence_dir"
    unzip_allow_warning -tq "$sequence_archive" >/dev/null
    unzip_allow_warning -q -o "$sequence_archive" -d "$sequence_dir"
    touch "$marker"
  fi
done < <(find "$EXTRACT_ROOT" -type f -name 'seq-*.zip' | sort)

cd "$REPO_ROOT"
PYTHONPATH=. /ephemeral/ubuntu_cache/sightline-parallel/rgbd-env/bin/python \
  scripts/build_rgbd_memory_dataset.py \
  --dataset 7scenes \
  --source-root "$EXTRACT_ROOT" \
  --output-root "$OUTPUT_ROOT"
