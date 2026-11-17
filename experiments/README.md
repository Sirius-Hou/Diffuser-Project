# Experiments — Training-Free Rerouting of a Pretrained Diffuser

Project code for ECE 780 (Ke-Ting Wu, Xin-Yi Lin, Ken Chiang, Sirius Hou).
The upstream Diffuser code (`diffuser/`, `scripts/`, `config/`) is used **frozen**. Everything we add lives under `experiments/`.

## Experiments

| Dir | Step | Status | Result |
|---|---|---|---|
| [`b0_baseline/`](b0_baseline/README.md) | 1: B0, plan once, no obstacle | ✅ done 2026-11-09 | 28/30 success, 2.0 s/plan, reproducible |
| [`r3_horizon/`](r3_horizon/README.md) | 2: how short can a planning horizon be (R3), batch timing (R5) | code written, not run yet | — |
| [`blocked/`](blocked/README.md) | 3: B0 / B1 / B2 / ours / oracle with a mid-episode blockage | code written, not run yet | — |

## Layout conventions (apply to every new experiment)

```
experiments/
  common/                 shared, reusable code. Import it; don't copy it
    setup/                wsl_env.sh (runtime paths), run.sh (run in env), check_env.py
    planner.py            FrozenDiffuserPlanner.plan(obs, goal, n, horizon) + original waypoint controller
    maze_utils.py         geometry, BFS (with blocked cells), wall/contact checks, drawing
    obstacles.py          Obstacle (full / partial cell), collides(trajectory, obstacle), ball_hits
    maze_graph.py         astar(maze, start, goal, blocked), waypoints(maze, path)
    check_geometry.py     verifies the coordinate convention
    test_units.py         numpy-only tests for obstacles.py and maze_graph.py (runs without MuJoCo/torch)
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

## Colab GPU

`experiments/common/setup/linux_setup.sh` builds the same Python 3.8 runtime on a fresh Ubuntu machine, following the team notebook step by step, and downloads the weights. On Colab: set the runtime to T4 GPU, clone the repo, `%cd` into it, then:

```bash
!bash experiments/common/setup/linux_setup.sh
```

On a CPU-only Linux machine or VM, use `TORCH_SPEC=torch==1.9.1+cpu bash experiments/common/setup/linux_setup.sh`.

```bash
!bash experiments/r3_horizon/run_all.sh
```

```bash
!bash experiments/blocked/run_all.sh smoke
```

Configs under `r3_horizon/` and `blocked/` default to `"device": "cuda"`; the planner falls back to CPU when CUDA is not available. GPU sampling is not bit-identical to the CPU B0 numbers, so compare methods only within one runtime.
