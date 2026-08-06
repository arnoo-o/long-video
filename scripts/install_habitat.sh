#!/usr/bin/env bash
set -euo pipefail
: "${HABITAT_ENV_PREFIX:?Set HABITAT_ENV_PREFIX to an isolated environment}"
ENV_PREFIX="${HABITAT_ENV_PREFIX}"
: "${CONDA_EXE:?Set CONDA_EXE to a conda executable}"
"${CONDA_EXE}" create -y -p "${ENV_PREFIX}" python=3.9
"${CONDA_EXE}" install -y -p "${ENV_PREFIX}" habitat-sim withbullet headless -c conda-forge -c aihabitat
"${CONDA_EXE}" install -y -p "${ENV_PREFIX}" git-lfs -c conda-forge
"${CONDA_EXE}" run -p "${ENV_PREFIX}" python -c "import habitat_sim; print(habitat_sim.__version__ if hasattr(habitat_sim, '__version__') else 'habitat-sim imported')"
