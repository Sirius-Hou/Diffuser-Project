#!/bin/bash
# Blocked-corridor pipeline: scenarios -> every method -> analysis.
#   bash experiments/blocked/run_all.sh smoke            (configs/smoke.json, all methods)
#   bash experiments/blocked/run_all.sh probe B2 ours    (only these methods)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN() { bash "$HERE/../common/setup/run.sh" "$@"; }
NAME="${1:?usage: run_all.sh <config name> [methods...]}"; shift
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(B0 B1 B2 ours oracle)
CFG=experiments/blocked/configs/$NAME.json
OUT=experiments/blocked/results/$NAME

if [ ! -f "$HERE/results/$NAME/scenarios.json" ]; then
  echo "=== scenarios";  RUN python experiments/blocked/make_scenarios.py --config $CFG
fi
for M in "${METHODS[@]}"; do
  echo "=== $M";         RUN python experiments/blocked/run_blocked.py --config $CFG --method $M
done
echo "=== analysis";     RUN python experiments/blocked/analyze_blocked.py $OUT
