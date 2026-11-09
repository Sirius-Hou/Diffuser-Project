#!/bin/bash
# Run any command inside the Diffuser environment, from the repository root.
#   bash experiments/common/setup/run.sh python experiments/b0_baseline/run_b0.py --config experiments/b0_baseline/configs/smoke.json
source "$(dirname "${BASH_SOURCE[0]}")/wsl_env.sh"
cd "$DIFFUSER_REPO" || exit 1
exec "$@"
