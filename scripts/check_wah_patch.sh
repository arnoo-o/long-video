#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WAH_ROOT:?Set WAH_ROOT to the patched Warp-as-History checkout}"
PATCH="${PROJECT_ROOT}/patches/wah_confidence.patch"
GEOTOKEN_QK_PATCH="${PROJECT_ROOT}/patches/wah_geotoken_qk_binding.patch"
if git -C "${WAH_ROOT}" apply --reverse --check "${PATCH}" 2>/dev/null && \
   git -C "${WAH_ROOT}" apply --reverse --check "${GEOTOKEN_QK_PATCH}" 2>/dev/null && \
   ! grep -RqsE '_stage2_training_observer|_pyramid_training_observer|_pyramid_adapter_names|pyramid_training' \
      "${WAH_ROOT}/warp_as_history" "${WAH_ROOT}/helios"; then
  python_bin="${WAH_PYTHON:-python}"
  "${python_bin}" -m py_compile "${WAH_ROOT}/warp_as_history/pipeline.py" "${WAH_ROOT}/helios/diffusers_version/transformer_helios_diffusers.py" "${WAH_ROOT}/helios/modules/transformer_helios.py" "${WAH_ROOT}/helios/modules/helios_kernels/attention_dispatch.py" "${WAH_ROOT}/warp_as_history/training/core.py" "${WAH_ROOT}/warp_as_history/training/data.py"
  echo "WAH confidence + GeoToken Q/K patches are applied; Stage2/WPF experimental patches are disabled."
  exit 0
fi
echo "WAH confidence patch is not applied cleanly." >&2
exit 1
