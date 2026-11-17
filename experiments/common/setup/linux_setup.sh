#!/bin/bash
# One-time setup of the Diffuser runtime on a fresh Ubuntu machine (Colab, or a local Linux VM).
# Same steps and versions as the team notebook (Diffuser_Maze2D_Large_*.ipynb, sections 1-6), laid out
# so that wsl_env.sh / run.sh work with their defaults.
#
#   bash experiments/common/setup/linux_setup.sh                                   (GPU machine, e.g. Colab)
#   TORCH_SPEC=torch==1.9.1+cpu bash experiments/common/setup/linux_setup.sh       (CPU-only machine)
#   CACHE=/content/drive/MyDrive/ece780_diffuser_cache bash experiments/common/setup/linux_setup.sh
#
# CACHE keeps the weights zip and the D4RL dataset (on Colab, put it on Drive to survive a disconnect).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT="$HOME/.local/share/ece780-diffuser"
CACHE="${CACHE:-$ROOT/cache}"
MF="$ROOT/miniforge"
TORCH_SPEC="${TORCH_SPEC:-torch==1.9.1+cu111}"
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
ENV="$ROOT/env"                                   # wsl_env.sh default for ECE780_ENV
PY="$ENV/bin/python"
mkdir -p "$ROOT" "$CACHE/d4rl" "$HOME/.mujoco"

echo "=== [1/7] system libraries (OpenGL / OSMesa headers for mujoco-py)"
$SUDO apt-get -qq update
for p in build-essential git wget libosmesa6-dev libgl1-mesa-dev libglfw3 libglew-dev patchelf unzip; do
  $SUDO apt-get install -y -qq $p > /dev/null 2>&1 && echo "ok $p" || echo "FAIL $p"
done
ls /usr/include/GL/osmesa.h

echo "=== [2/7] MuJoCo 2.0 + key"
cd "$HOME/.mujoco"
if [ ! -d mujoco200 ]; then
  wget -q https://www.roboti.us/download/mujoco200_linux.zip
  unzip -q mujoco200_linux.zip && mv mujoco200_linux mujoco200 && rm mujoco200_linux.zip
fi
[ -f mjkey.txt ] || wget -q https://www.roboti.us/file/mjkey.txt
cp mjkey.txt mujoco200/bin/ 2>/dev/null || true

echo "=== [3/7] Miniforge + Python 3.8 env"
if [ ! -d "$MF" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$MF" > /dev/null
  rm -f /tmp/mf.sh
fi
[ -x "$PY" ] || "$MF/bin/conda" create -y -q -p "$ENV" python=3.8 > /dev/null
"$PY" --version

echo "=== [4/7] Python packages"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:/usr/lib/nvidia:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MJKEY_PATH="$HOME/.mujoco/mjkey.txt" MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"
export MUJOCO_PY_FORCE_CPU=1 MPLBACKEND=Agg
"$PY" -m pip install "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4" "Cython<3" numpy==1.23.5 2>&1 | tail -n 3
"$PY" -m pip install --no-build-isolation gym==0.18.0 2>&1 | tail -n 5
"$PY" -m pip install "glfw>=1.4" "cffi>=1.10" "fasteners~=0.15" "imageio>=2.1.2" 2>&1 | tail -n 2
"$PY" -m pip install -v --no-build-isolation mujoco-py==2.0.2.13 2>&1 | grep -iE "error|fatal|exception|Successfully" | head -n 30 || true
"$PY" -m pip install "$TORCH_SPEC" -f https://download.pytorch.org/whl/torch_stable.html 2>&1 | tail -n 3
"$PY" -m pip install matplotlib==3.3.4 typed-argument-parser einops gitpython imageio scikit-video==1.1.11 h5py termcolor click tqdm pandas pillow 2>&1 | tail -n 3
"$PY" -m pip install "Pillow==7.2.0" 2>&1 | tail -n 1
"$PY" -m pip install --no-deps "git+https://github.com/aravindr93/mjrl@master#egg=mjrl" 2>&1 | tail -n 2
"$PY" -m pip install --no-deps "git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl" 2>&1 | tail -n 2
"$PY" -c "import mujoco_py; print('mujoco_py OK', mujoco_py.__version__)"       # first import compiles, 1-3 min
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== [5/7] weights (Dropbox link from Diffuser GitHub issue #13)"
ZIP="$CACHE/maze2d-logs.zip"
[ -f "$ZIP" ] || wget -q -O "$ZIP" "https://www.dropbox.com/s/za14rwp8to1bosn/maze2d-logs.zip?dl=1"
[ -d "$CACHE/maze2d-logs" ] || unzip -q "$ZIP" -d "$CACHE/maze2d-logs"
CFG="$(find "$CACHE/maze2d-logs" -path '*maze2d-large-v1/diffusion/H384_T256*/model_config.pkl' | head -n 1)"
[ -n "$CFG" ] || { echo "large weights not found under $CACHE/maze2d-logs"; exit 1; }
LOGBASE="$(dirname "$(dirname "$(dirname "$(dirname "$CFG")")")")"
echo "LOGBASE = $LOGBASE"

echo "=== [6/7] link weights and dataset dir to the paths the scripts expect"
ln -sfn "$LOGBASE" "$REPO/logs"                            # <repo>/logs/maze2d-large-v1/diffusion/H384_T256
mkdir -p "$(cd "$REPO/.." && pwd)/artifacts"
ln -sfn "$CACHE/d4rl" "$(cd "$REPO/.." && pwd)/artifacts/d4rl"     # wsl_env.sh default for D4RL_DATASET_DIR

echo "=== [7/7] dataset (D4RL downloads maze2d-large-sparse-v1.hdf5 on first use) and final check"
bash "$REPO/experiments/common/setup/run.sh" python -c "import gym, d4rl; d = gym.make('maze2d-large-v1').get_dataset(); print('observations', d['observations'].shape)"
bash "$REPO/experiments/common/setup/run.sh" python experiments/common/setup/check_env.py
