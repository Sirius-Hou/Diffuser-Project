#!/bin/bash
# Full B0 pipeline: environment check -> geometry check -> 2-episode smoke test ->
# reproducibility re-run -> 30-episode main run -> analysis.
# Run from anywhere inside Linux/WSL:
#   bash experiments/b0_baseline/run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN() { bash "$HERE/../common/env/run.sh" "$@"; }
B0=experiments/b0_baseline

echo "=== [1/6] environment check";        RUN python experiments/common/env/check_env.py
echo "=== [2/6] maze geometry check";      RUN python experiments/common/check_geometry.py
echo "=== [3/6] smoke test (seeds 0,1)";   RUN python $B0/run_b0.py --config $B0/configs/smoke.json
echo "=== [4/6] reproducibility re-run";   RUN python $B0/run_b0.py --config $B0/configs/smoke.json --tag _rep --no_png
RUN python $B0/check_repro.py $B0/results/smoke $B0/results/smoke --tag_b _rep --out $B0/results/smoke/repro_check.json
echo "=== [5/6] main run (30 episodes)";   RUN python $B0/run_b0.py --config $B0/configs/main.json
echo "=== [6/6] analysis";                 RUN python $B0/analyze_b0.py $B0/results/main
