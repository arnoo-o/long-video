#!/usr/bin/env bash
set -euo pipefail
: "${HF_TOKEN:?Set a read-only Hugging Face token with FLUX.1-dev access}"
: "${DIT360_ENV:=/ephemeral/mdu/envs/longvideo-dit360}"
: "${MODEL_ROOT:=/ephemeral/mdu/long-video-data/models/dit360}"
: "${HF_HOME:=/ephemeral/mdu/long-video-data/cache/huggingface}"
mkdir -p "$MODEL_ROOT"
HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" "$DIT360_ENV/bin/hf" download \
  black-forest-labs/FLUX.1-dev --local-dir "$MODEL_ROOT/FLUX.1-dev"
HF_HOME="$HF_HOME" "$DIT360_ENV/bin/hf" download \
  Insta360-Research/DiT360-Panorama-Image-Generation \
  --local-dir "$MODEL_ROOT/DiT360-Panorama-Image-Generation"
find "$MODEL_ROOT" -type f -not -path '*/.cache/*' -printf '%p %s\n'
