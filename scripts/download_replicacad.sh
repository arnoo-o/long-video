#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${HABITAT_ENV_PREFIX:-/ephemeral/mdu/envs/longvideo-habitat}"
DATA_ROOT="${REPLICACAD_ROOT:-/ephemeral/mdu/long-video-data/raw/replicacad}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
mkdir -p "${DATA_ROOT}"
git lfs install --skip-repo
"${ENV_PREFIX}/bin/python" -m habitat_sim.utils.datasets_download --uids replica_cad_baked_lighting --data-path "${DATA_ROOT}" --no-replace
