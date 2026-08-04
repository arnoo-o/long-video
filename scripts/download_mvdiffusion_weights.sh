#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${MVDIFFUSION_ENV:-/ephemeral/mdu/envs/longvideo-mvdiffusion}"
MODEL_ROOT="${MVDIFFUSION_MODEL_ROOT:-/ephemeral/mdu/long-video-data/models/mvdiffusion}"
CHECKPOINT="${MODEL_ROOT}/pano_outpaint.ckpt"
BASE_MODEL="${MODEL_ROOT}/stable-diffusion-2-inpainting"
mkdir -p "${MODEL_ROOT}"

wget -c -O "${CHECKPOINT}" "https://www.dropbox.com/scl/fi/3mtj06qx6mxt4eme1oz2r/pano_outpaint.ckpt?rlkey=xat6cwt47lzfjawum05xa5ftq&dl=1"

actual_size="$(stat -c %s "${CHECKPOINT}")"
expected_size=6250691956
if [[ "${actual_size}" != "${expected_size}" ]]; then
  echo "Unexpected checkpoint size: ${actual_size}, expected ${expected_size}" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for stabilityai/stable-diffusion-2-inpainting." >&2
  echo "Accept the model license on Hugging Face, then rerun with HF_TOKEN set." >&2
  exit 2
fi

HF_TOKEN="${HF_TOKEN}" BASE_MODEL="${BASE_MODEL}" "${ENV_PREFIX}/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "stabilityai/stable-diffusion-2-inpainting",
    local_dir=os.environ["BASE_MODEL"],
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
    allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "unet/*", "model_index.json"],
)
PY
