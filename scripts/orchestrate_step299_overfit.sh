#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot operational handoff at checkpoint-000299.
# The main four-GPU run is stopped cleanly, three GPUs resume from 299 while
# the released GPU runs the exact step-299 training record, then the three-GPU
# overlap run is stopped and the main four-GPU watcher resumes from 299.

if [[ $# -ne 7 ]]; then
  echo "usage: $0 OUTPUT_DIR REPO_ROOT MODEL_ROOT HELIOS_ROOT CONFIG MANIFEST P3_MANIFEST" >&2
  exit 2
fi

output_dir=$1
repo_root=$2
model_root=$3
helios_root=$4
config=$5
manifest=$6
p3_manifest=$7

checkpoint="$output_dir/checkpoint-000299.pt"
release_gpu=7
train_gpu_list="1,2,3"
main_gpu_list="1,2,3,7"
record_id="scannet__scene0011_01__000008__frames_000_096"
record_base="/ephemeral/arnoo/sightline-parallel/rgbd_memory_dataset/scannet/processed/records/scannet/scannet__scene0011_01__000008"
inference_chunks=2
overlap_output="$output_dir/step299-3gpu-overlap-${record_id}-chunks${inference_chunks}"
inference_output="$output_dir/step299-overfit-inference-${record_id}-chunks${inference_chunks}"
inference_prefix="$inference_output/${record_id}"
main_log="/ephemeral/arnoo/sightline-parallel/train-v10-watch.log"
overlap_log="$output_dir/step299-3gpu-overlap.log"
inference_log="$inference_output/inference.log"
result_file="$inference_output/result.json"
stop_flag="$overlap_output/STOP"

record_rgb_dir="$record_base/rgb"
record_intrinsics="$record_base/intrinsics.npy"
record_c2w="$record_base/c2w_local.npy"
record_near_depth="1.184000015258789"

export PATH="/ephemeral/JerryHouse/miniconda3/envs/videox-fun/bin:/usr/bin:/bin"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

base_args=(
  --train
  --allow-world-size-migration
  --skip-manifest-validation
  --config "$config"
  --model "$model_root"
  --helios-root "$helios_root"
  --manifest "$manifest"
  --p3-manifest "$p3_manifest"
)

log() { echo "$(date -Is) $*"; }

main_pids() {
  ps -eo pid=,args= | awk -v out="$output_dir" '
    index($0,out) && $0 !~ /orchestrate_step299_overfit\.sh/ && $0 !~ /awk -v out=/ && ($0 ~ /watch_sightline_4gpu\.sh/ || $0 ~ /torchrun/ || $0 ~ /train_sightline_rgbd\.py/) {print $1}
  ' | sort -nu
}

overlap_pids() {
  ps -eo pid=,args= | awk -v out="$overlap_output" '
    index($0,out) && $0 !~ /awk -v out=/ && ($0 ~ /torchrun/ || $0 ~ /train_sightline_rgbd\.py/) {print $1}
  ' | sort -nu
}

stop_main() {
  local pids
  pids="$(main_pids || true)"
  if [[ -n "$pids" ]]; then
    log "stopping existing four-GPU run: $pids"
    kill -TERM $pids 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    pids="$(main_pids || true)"
    [[ -z "$pids" ]] && return 0
    sleep 1
  done
  pids="$(main_pids || true)"
  [[ -z "$pids" ]] || kill -KILL $pids 2>/dev/null || true
}

stop_overlap() {
  local pids
  touch "$stop_flag"
  pids="$(overlap_pids || true)"
  if [[ -n "$pids" ]]; then
    log "stopping three-GPU overlap run: $pids"
    kill -TERM $pids 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    pids="$(overlap_pids || true)"
    [[ -z "$pids" ]] && return 0
    sleep 1
  done
  pids="$(overlap_pids || true)"
  [[ -z "$pids" ]] || kill -KILL $pids 2>/dev/null || true
}

if [[ ! -f "$checkpoint" ]]; then
  echo "missing required checkpoint: $checkpoint" >&2
  exit 2
fi

mkdir -p "$overlap_output" "$inference_output"
rm -f "$stop_flag" "$result_file"

# Prevent the original watcher from immediately respawning four ranks.
stop_main

source_image="$(find "$record_rgb_dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%p\n' | sort | head -n 1)"
if [[ -z "$source_image" ]]; then
  echo "no source image found in $record_rgb_dir" >&2
  exit 2
fi

log "launching three-GPU overlap training on CUDA_VISIBLE_DEVICES=$train_gpu_list from $checkpoint"
(
  while [[ ! -f "$stop_flag" ]]; do
    set +e
    CUDA_VISIBLE_DEVICES="$train_gpu_list" \
      torchrun --standalone --nproc_per_node=3 \
      scripts/train_sightline_rgbd.py "${base_args[@]}" \
      --resume "$checkpoint" \
      --save-every 100 \
      --output-dir "$overlap_output" \
      >> "$overlap_log" 2>&1
    status=$?
    set -e
    [[ -f "$stop_flag" ]] && break
    log "three-GPU overlap exited status=$status; restarting from checkpoint-000299" >> "$overlap_log"
    sleep 1
  done
) </dev/null > /dev/null 2>&1 &
overlap_supervisor=$!

log "launching step-299 overfit inference on physical GPU $release_gpu"
set +e
CUDA_VISIBLE_DEVICES="$release_gpu" \
  python3.11 scripts/infer_sightline.py \
  --source "$source_image" \
  --model "$model_root" \
  --out "$inference_prefix" \
  --helios-root "$helios_root" \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --intrinsics "$record_intrinsics" \
  --c2w "$record_c2w" \
  --near-depth "$record_near_depth" \
  --prompt "A realistic video of the same scene." \
  --chunks "$inference_chunks" \
  --steps 2 \
  > "$inference_log" 2>&1
inference_status=$?
set -e

inference_result="$(tail -n 1 "$inference_log" 2>/dev/null || true)"
stop_overlap
kill -TERM "$overlap_supervisor" 2>/dev/null || true
wait "$overlap_supervisor" 2>/dev/null || true

log "resuming four-GPU training from checkpoint-000299 on CUDA_VISIBLE_DEVICES=$main_gpu_list"
nohup env SIGHTLINE_GPU_LIST="$main_gpu_list" \
  bash scripts/watch_sightline_4gpu.sh "$output_dir" "${base_args[@]}" \
  > "$main_log" 2>&1 < /dev/null &
main_watcher_pid=$!

python3.11 - "$result_file" "$inference_status" "$checkpoint" "$inference_prefix.npy" "$inference_log" "$main_watcher_pid" "$inference_result" <<'PY'
import json,sys
out,status,checkpoint,result,log,pid,line=sys.argv[1:]
payload={'inference_status':int(status),'checkpoint':checkpoint,'result_npy':result,'inference_log':log,'four_gpu_watcher_pid':int(pid),'inference_output':line}
open(out,'w').write(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload,ensure_ascii=False))
PY

log "step-299 handoff complete; four-GPU watcher pid=$main_watcher_pid"
