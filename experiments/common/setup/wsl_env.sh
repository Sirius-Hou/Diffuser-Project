#!/bin/bash
# Source this inside Linux / WSL to get the Diffuser runtime (Python 3.8 + mujoco-py 2.0 + MuJoCo 2.0).
#
#   source experiments/common/setup/wsl_env.sh
#
# Every path can be overridden by exporting the variable BEFORE sourcing, e.g.
#   export ECE780_ENV=$HOME/miniconda3/envs/diffuser
#   export D4RL_DATASET_DIR=$HOME/.d4rl/datasets
#
# Defaults match Sirius' WSL Ubuntu setup (micromamba env under ~/.local/share/ece780-diffuser).

_this="${BASH_SOURCE[0]}"
export DIFFUSER_REPO="${DIFFUSER_REPO:-$(cd "$(dirname "$_this")/../../.." && pwd)}"

export ECE780_ENV="${ECE780_ENV:-$HOME/.local/share/ece780-diffuser/env}"
export ECE780_OSMESA="${ECE780_OSMESA:-$HOME/.local/share/ece780-diffuser/osmesa/usr}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco200}"
export MUJOCO_PY_MJKEY_PATH="${MUJOCO_PY_MJKEY_PATH:-$HOME/.mujoco/mjkey.txt}"
# D4RL dataset (maze2d-large-sparse-v1.hdf5). Default: <repo>/../artifacts/d4rl (outside git).
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$(cd "$DIFFUSER_REPO/.." && pwd)/artifacts/d4rl}"

export PATH="$ECE780_ENV/bin:$PATH"
# libLLVM / libOSMesa from the local OSMesa bundle must be on the loader path,
# otherwise mujoco_py fails to import and tries (and fails) to recompile itself.
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$ECE780_OSMESA/lib/x86_64-linux-gnu:$ECE780_ENV/lib:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="$ECE780_OSMESA/include:${C_INCLUDE_PATH:-}"
export MUJOCO_PY_FORCE_CPU=1
export MPLBACKEND=Agg
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONPATH="$DIFFUSER_REPO:${PYTHONPATH:-}"
