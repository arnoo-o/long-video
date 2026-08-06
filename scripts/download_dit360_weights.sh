#!/usr/bin/env bash
set -euo pipefail
: "${HF_TOKEN:?Set a read-only Hugging Face token with FLUX.1-dev access}"
: "${DIT360_ENV:?Set DIT360_ENV to the isolated DiT360 environment}"
: "${MODEL_ROOT:?Set MODEL_ROOT outside the repository}"
: "${HF_HOME:?Set HF_HOME outside the repository}"
mkdir -p "$MODEL_ROOT"
HF_HOME="$HF_HOME" HF_TOKEN="$HF_TOKEN" "$DIT360_ENV/bin/hf" download \
  black-forest-labs/FLUX.1-dev --local-dir "$MODEL_ROOT/FLUX.1-dev"
HF_HOME="$HF_HOME" "$DIT360_ENV/bin/hf" download \
  Insta360-Research/DiT360-Panorama-Image-Generation \
  --local-dir "$MODEL_ROOT/DiT360-Panorama-Image-Generation"
find "$MODEL_ROOT" -type f -not -path '*/.cache/*' -printf '%p %s\n'
