#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAH_ROOT="${WAH_ROOT:-/ephemeral/mdu/long-video/third_party/Warp-as-History}"
PATCH="${PROJECT_ROOT}/patches/wah_confidence.patch"

if [[ ! -d "${WAH_ROOT}/.git" ]]; then
  echo "WAH repository not found: ${WAH_ROOT}" >&2
  exit 1
fi
if git -C "${WAH_ROOT}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "WAH confidence patch is already applied."
  exit 0
fi
git -C "${WAH_ROOT}" apply --check "${PATCH}"
git -C "${WAH_ROOT}" apply "${PATCH}"
echo "Applied WAH confidence patch to ${WAH_ROOT}"
