#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WAH_ROOT:?Set WAH_ROOT to the clean Warp-as-History checkout}"
PATCH="${PROJECT_ROOT}/patches/wah_confidence.patch"
TRAINING_PATCH="${PROJECT_ROOT}/patches/wah_stage2_training.patch"

if ! git -C "${WAH_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "WAH repository not found: ${WAH_ROOT}" >&2
  exit 1
fi
if git -C "${WAH_ROOT}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "WAH confidence patch is already applied."
else
  git -C "${WAH_ROOT}" apply --check "${PATCH}"
  git -C "${WAH_ROOT}" apply "${PATCH}"
  echo "Applied WAH confidence patch to ${WAH_ROOT}"
fi
if ! git -C "${WAH_ROOT}" apply --reverse --check "${TRAINING_PATCH}" 2>/dev/null; then
  git -C "${WAH_ROOT}" apply --check "${TRAINING_PATCH}"
  git -C "${WAH_ROOT}" apply "${TRAINING_PATCH}"
fi
echo "Applied WAH Stage2 training patch to ${WAH_ROOT}"
