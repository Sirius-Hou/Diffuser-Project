#!/bin/bash
# R3 / R5: smoke run, then the full horizon sweep and batch timing.
#   bash experiments/r3_horizon/run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN() { bash "$HERE/../common/setup/run.sh" "$@"; }
R3=experiments/r3_horizon

echo "=== [1/2] smoke";  RUN python $R3/run_r3.py --config $R3/configs/smoke.json
echo "=== [2/2] main";   RUN python $R3/run_r3.py --config $R3/configs/main.json
