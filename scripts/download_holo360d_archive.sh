#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 URL TARGET EXPECTED_BYTES [PARTS]" >&2
  exit 2
fi

url=$1
target=$2
expected_bytes=$3
part_count=${4:-8}

if (( expected_bytes <= 0 || part_count <= 0 )); then
  echo "EXPECTED_BYTES and PARTS must be positive" >&2
  exit 2
fi

target_dir=$(dirname -- "$target")
target_name=$(basename -- "$target")
parts_dir="$target_dir/${target_name%.zip}.parts"
chunk_size=$(( (expected_bytes + part_count - 1) / part_count ))
mkdir -p "$target_dir" "$parts_dir"

validate_zip() {
  local archive=$1
  if command -v unzip >/dev/null 2>&1; then
    unzip -tqq "$archive"
  else
    python3 -m zipfile -t "$archive" >/dev/null
  fi
}

if [[ -f "$target" ]]; then
  actual=$(stat -c '%s' "$target")
  if [[ $actual -eq $expected_bytes ]]; then
    validate_zip "$target"
    sha256sum "$target" > "$target.sha256"
    echo "archive already complete: $target"
    exit 0
  fi
  mv -- "$target" "$target.incomplete.$actual"
fi

download_part() {
  local index=$1
  local start=$(( index * chunk_size ))
  local end=$(( start + chunk_size - 1 ))
  if (( end >= expected_bytes )); then end=$(( expected_bytes - 1 )); fi
  local expected=$(( end - start + 1 ))
  local output
  output=$(printf '%s/part-%02d' "$parts_dir" "$index")
  if [[ -f "$output" && $(stat -c '%s' "$output") -eq $expected ]]; then
    return
  fi
  rm -f -- "$output.tmp"
  curl -L --fail --retry 8 --retry-delay 2 \
    --range "$start-$end" -o "$output.tmp" "$url"
  [[ $(stat -c '%s' "$output.tmp") -eq $expected ]]
  mv -- "$output.tmp" "$output"
}

export -f download_part
export url expected_bytes parts_dir chunk_size
seq 0 $((part_count - 1)) | xargs -n1 -P"$part_count" bash -c 'download_part "$0"'

temporary="$target.tmp"
if [[ ! -f "$temporary" || $(stat -c '%s' "$temporary") -ne $expected_bytes ]]; then
  rm -f -- "$temporary"
  for index in $(seq 0 $((part_count - 1))); do
    printf -v part '%s/part-%02d' "$parts_dir" "$index"
    dd if="$part" of="$temporary" bs=16M oflag=append conv=notrunc status=none
  done
fi
[[ $(stat -c '%s' "$temporary") -eq $expected_bytes ]]
validate_zip "$temporary"
mv -- "$temporary" "$target"
sha256sum "$target" > "$target.sha256"
echo "download complete: $target"
