#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WAH_ROOT:?Set WAH_ROOT to the clean Warp-as-History checkout}"
PATCH="${PROJECT_ROOT}/patches/wah_confidence.patch"
GEOTOKEN_QK_PATCH="${PROJECT_ROOT}/patches/wah_geotoken_qk_binding.patch"
STAGE2_PATCH="${PROJECT_ROOT}/patches/wah_stage2_training.patch"
WPF_PATCH="${PROJECT_ROOT}/patches/wah_wpf_adaptation_training.patch"

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
if ! grep -q 'GeoToken Q/K binding must not modify attention V' \
    "${WAH_ROOT}/helios/modules/transformer_helios.py"; then
  git -C "${WAH_ROOT}" apply --check "${GEOTOKEN_QK_PATCH}"
  git -C "${WAH_ROOT}" apply "${GEOTOKEN_QK_PATCH}"
fi
echo "Applied WAH GeoToken Q/K binding patch to ${WAH_ROOT}"

# Stage2 completion/pyramid-training and WPF adaptation are intentionally
# disabled for the current GeoToken run.  They are orthogonal experimental
# adapters and must not silently alter the pinned WAH path.  If either patch
# is already present, remove it when the patch applies cleanly in reverse;
# otherwise fail instead of leaving a mixed WAH checkout.
for experimental_patch in "${STAGE2_PATCH}" "${WPF_PATCH}"; do
  if git -C "${WAH_ROOT}" apply --reverse --check "${experimental_patch}" 2>/dev/null; then
    git -C "${WAH_ROOT}" apply --reverse "${experimental_patch}"
    echo "Removed disabled WAH experimental patch: ${experimental_patch##*/}"
  elif git -C "${WAH_ROOT}" apply --check "${experimental_patch}" 2>/dev/null; then
    echo "Disabled WAH experimental patch is absent: ${experimental_patch##*/}"
  elif ! grep -Rqs --exclude='*.pyc' --exclude-dir='__pycache__' -E '_stage2_training_observer|_pyramid_training_observer|_pyramid_adapter_names|pyramid_training' \
      "${WAH_ROOT}/warp_as_history" "${WAH_ROOT}/helios"; then
    # These patches overlap the Q/K patch context, so patch --check can fail
    # even when the disabled experiment is absent. The runtime marker check is
    # authoritative after the checkout has been restored to the pinned commit.
    echo "Disabled WAH experimental patch is absent: ${experimental_patch##*/}"
  else
    echo "WAH checkout has a partial or drifted experimental patch: ${experimental_patch##*/}" >&2
    exit 1
  fi
done
echo "WAH Stage2/WPF experimental training patches are disabled."
