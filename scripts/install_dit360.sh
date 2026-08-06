#!/usr/bin/env bash
set -euo pipefail
: "${DIT360_REPO:=/ephemeral/mdu/long-video-third-party/DiT360}"
: "${DIT360_ENV:=/ephemeral/mdu/envs/longvideo-dit360}"
if [[ ! -d "$DIT360_REPO/.git" ]]; then
  git clone https://github.com/Insta360-Research-Team/DiT360.git "$DIT360_REPO"
fi
git -C "$DIT360_REPO" fetch origin
git -C "$DIT360_REPO" checkout 3779fe7965473f6824994c663a0ae7a76bc7aafa
python3 -m venv "$DIT360_ENV"
"$DIT360_ENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$DIT360_ENV/bin/pip" install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$DIT360_ENV/bin/pip" install \
  diffusers==0.36.0 huggingface_hub==0.36.0 transformers==4.52.4 \
  accelerate==1.9.0 peft==0.17.1 protobuf==6.31.1 \
  sentencepiece==0.2.0 opencv-python==4.11.0.86 psutil==7.0.0 \
  PyYAML==6.0.2 scipy==1.16.0 safetensors==0.5.3 einops==0.8.1
"$DIT360_ENV/bin/python" - <<'PY'
import diffusers,torch,transformers
print({"torch":torch.__version__,"diffusers":diffusers.__version__,
       "transformers":transformers.__version__})
PY
