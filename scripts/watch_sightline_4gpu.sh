#!/usr/bin/env bash
set -u

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR TRAIN_ARGS..." >&2
  exit 2
fi

output_dir=$1
shift
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$output_dir"
cd "$repo_root"

attempt=0
while true; do
  train_args=("$@" --output-dir "$output_dir")
  checkpoint=$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint-*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
  if [[ -n "$checkpoint" ]]; then
    train_args+=(--resume "$checkpoint")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${SIGHTLINE_GPU_LIST:-1,2,3,7}" \
    torchrun --standalone --nproc_per_node=4 scripts/train_sightline_rgbd.py "${train_args[@]}"
  status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "$(date -Is) training exited status=$status; retry=$attempt; checkpoint=${checkpoint:-none}" >&2
  sleep 1
done
