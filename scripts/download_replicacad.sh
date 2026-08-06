#!/usr/bin/env bash
set -euo pipefail
: "${HABITAT_ENV_PREFIX:?Set HABITAT_ENV_PREFIX to the isolated Habitat environment}"
ENV_PREFIX="${HABITAT_ENV_PREFIX}"
: "${REPLICACAD_ROOT:?Set REPLICACAD_ROOT outside the repository}"
DATA_ROOT="${REPLICACAD_ROOT}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
mkdir -p "${DATA_ROOT}"
git lfs install --skip-repo
"${ENV_PREFIX}/bin/python" -m habitat_sim.utils.datasets_download --uids replica_cad_baked_lighting --data-path "${DATA_ROOT}" --no-replace
