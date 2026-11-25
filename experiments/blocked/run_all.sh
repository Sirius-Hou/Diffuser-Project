#!/bin/bash
# Steps 2-3 pipeline: tests -> scenario list -> B2 probe (R2) -> smoke (4 methods) -> main -> analysis.
#   bash experiments/blocked/run_all.sh            (inside Linux/WSL)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN() { bash "$HERE/../common/setup/run.sh" "$@"; }
B=experiments/blocked
echo "=== [1/6] shared-code tests";   RUN python experiments/common/test_common.py
echo "=== [2/6] scenario list";       RUN python $B/make_scenarios.py --config $B/configs/scenarios.json
echo "=== [3/6] B2 probe (R2)";        RUN python $B/probe_b2.py --config $B/configs/probe_b2.json
echo "=== [4/6] smoke, 4 methods";     RUN python $B/run_blocked.py --config $B/configs/smoke.json
RUN python $B/analyze_blocked.py $B/results/smoke
echo "=== [5/6] main run";             RUN python $B/run_blocked.py --config $B/configs/main.json
echo "=== [6/6] analysis";             RUN python $B/analyze_blocked.py $B/results/main
