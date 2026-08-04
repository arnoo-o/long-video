#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/ephemeral/mdu/miniforge3}"
ENV_PREFIX="${MVDIFFUSION_ENV:-/ephemeral/mdu/envs/longvideo-mvdiffusion}"
REPO="${MVDIFFUSION_REPO:-/ephemeral/mdu/long-video-third-party/MVDiffusion}"

if [[ ! -x "${MINIFORGE_ROOT}/bin/conda" ]]; then
  installer="/ephemeral/mdu/Miniforge3.sh"
  wget -O "${installer}" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "${installer}" -b -p "${MINIFORGE_ROOT}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${MINIFORGE_ROOT}/bin/conda" create -y -p "${ENV_PREFIX}" python=3.10 pip
fi

if [[ ! -d "${REPO}/.git" ]]; then
  mkdir -p "$(dirname "${REPO}")"
  git clone https://github.com/Tangshitao/MVDiffusion.git "${REPO}"
fi

"${ENV_PREFIX}/bin/pip" install -r "${REPO}/requirements.txt"
# The upstream file does not pin these old-API transitive dependencies.
"${ENV_PREFIX}/bin/pip" install "huggingface_hub==0.14.1" "torchmetrics==0.11.4" "setuptools<81"
"${ENV_PREFIX}/bin/python" -c "import torch,diffusers,transformers,pytorch_lightning; print(torch.__version__,torch.version.cuda,diffusers.__version__,transformers.__version__,pytorch_lightning.__version__)"
