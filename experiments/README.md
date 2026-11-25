# Experiments — Training-Free Rerouting of a Pretrained Diffuser

Project code for ECE 780 (Ke-Ting Wu, Xin-Yi Lin, Ken Chiang, Sirius Hou).
The upstream Diffuser code (`diffuser/`, `scripts/`, `config/`) is used **frozen**. Everything we add lives under `experiments/`.

## Experiments

| Dir | Step | Status | Result |
|---|---|---|---|
| [`b0_baseline/`](b0_baseline/README.md) | 1: B0, plan once, no obstacle | ✅ done 2026-11-09 | 28/30 success, 2.0 s/plan, reproducible |
| [`blocked/`](blocked/README.md) | 2–3: obstacle, scenario list, B2 probe, B0/B1/B2/Ours/Oracle, partial blockage, 2 ablations | done (Colab T4) | full blockage: Ours 76 % vs B2 20 % |

## Layout conventions (apply to every new experiment)

```
experiments/
  common/                 shared, reusable code. Import it; don't copy it
    setup/                wsl_env.sh (runtime paths), run.sh (run in env), check_env.py
    planner.py            FrozenDiffuserPlanner.plan(obs, goal, n, horizon) + original waypoint controller
    maze_utils.py         geometry, BFS (with blocked cells), wall/contact checks, drawing
    check_geometry.py     verifies the coordinate convention
    mazes.py              maze layout as numpy (no d4rl/mujoco needed)
    obstacles.py          obstacles + collision checks (team interface 2)
    waypoints.py          A* + waypoint selection + segment horizon (team interface 3)
    rollout.py            the shared episode loop; methods.py: B0/B1/B2/Ours
    fakes.py, test_common.py   fake env/planner + tests that run without mujoco
  <experiment_name>/
    README.md             background, objectives, implementation, exact commands, results, analysis, limitations
    configs/*.json        every run is driven by a config (seeds, device, output dir, ...)
    run_*.py              episode loop; writes one JSON (+ npz, png) per episode
    analyze_*.py          aggregates → episodes.csv, summary.json, figures, acceptance checks
    run_all.sh            the whole pipeline in order
    results/<run>/        committed outputs (keep small: per-episode json/npz/png, csv, json, figures)
```

Rules:
1. **Never modify the frozen model or the weights.** Wrap them (see `common/planner.py`).
2. **Every run is seeded and config-driven.** `run_meta.json` records the config, seeds, versions and git commit.
3. **Smoke test → reproducibility check → full run → analysis**, in that order.
4. **Per-episode JSON records share one schema across methods** (B0's columns plus method-specific extras), so results can be concatenated and compared.
5. Weights (`logs/`), datasets (`*.hdf5`) and envs are **not** committed. `results/` under `experiments/` **is** committed.

## Quick start (Linux / WSL)

```bash
bash experiments/common/setup/run.sh python experiments/common/setup/check_env.py
```

```bash
bash experiments/b0_baseline/run_all.sh
```

See [`b0_baseline/README.md` §5](b0_baseline/README.md#5-reproducing) for setting up a new machine.
