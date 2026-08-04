#!/usr/bin/env bash
set -euo pipefail
ENV_PREFIX="${HABITAT_ENV_PREFIX:-/ephemeral/mdu/envs/longvideo-habitat}"
CONDA_EXE="${CONDA_EXE:-/ephemeral/mdu/miniforge3/bin/conda}"
"${CONDA_EXE}" create -y -p "${ENV_PREFIX}" python=3.9
"${CONDA_EXE}" install -y -p "${ENV_PREFIX}" habitat-sim withbullet headless -c conda-forge -c aihabitat
"${CONDA_EXE}" install -y -p "${ENV_PREFIX}" git-lfs -c conda-forge
"${CONDA_EXE}" run -p "${ENV_PREFIX}" python -c "import habitat_sim; print(habitat_sim.__version__ if hasattr(habitat_sim, '__version__') else 'habitat-sim imported')"
