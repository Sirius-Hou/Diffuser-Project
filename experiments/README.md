# Experiments — Training-Free Rerouting of a Pretrained Diffuser

Project code for ECE 780 (Ke-Ting Wu, Xin-Yi Lin, Ken Chiang, Sirius Hou).
The upstream Diffuser code (`diffuser/`, `scripts/`, `config/`) is used **frozen**. Everything we add lives under `experiments/`.

## Experiments

| Dir | Step | Status | Result |
|---|---|---|---|
| [`b0_baseline/`](b0_baseline/README.md) | 1: B0, plan once, no obstacle | ✅ done 2026-11-09 | 28/30 success, 2.0 s/plan, reproducible |
| [`r3_horizon/`](r3_horizon/README.md) | 2: how short can a planning horizon be (R3), batch timing (R5) | ✅ done 2026-11-17 (CPU, Lima VM) | 32 steps per cell; straight legs 40/40; `plan(n=16)` 24–33 s on CPU |
| [`blocked/`](blocked/README.md) | 3: B0 / B1 / B2 / ours / oracle with a mid-episode blockage | ✅ `main` done 2026-11-21 (pre-registered, CPU, Lima VM); diagnostics 2026-11-22 to 11-27 (guidance, follow-ups, time budget, corridor preference) | `main`, 29 scenarios with a longer detour: B2 10/29, ours (junctions) 23/29, exact McNemar p = 0.0002; ours (corners) 25/29, astar_line + lag trigger 29/29 |

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

## Local Lima VM (macOS, CPU)

R3 and the blocked probes (2026-11-17) ran in a Lima VM on an Intel Mac (i7-9750H), not on Colab. The VM files live outside this repo, in `final_project/.vm/`, so everything Lima writes stays inside `final_project`. Only `final_project` is mounted into the guest, and port forwarding is off. Apple silicon is untested: a `vz` VM there cannot run `arch: x86_64`.

One-time setup (`<FP>` = the absolute path of `final_project`):

1. Download Lima [v2.3.0](https://github.com/lima-vm/lima/releases/download/v2.3.0/lima-2.3.0-Darwin-x86_64.tar.gz) and unpack it into `<FP>/.vm/lima-dist`.
2. Download the Ubuntu 22.04 [cloud image](https://cloud-images.ubuntu.com/releases/jammy/release-20261020/ubuntu-22.04-server-cloudimg-amd64.img) to `<FP>/.vm/img/jammy.img`.
3. Create `<FP>/.vm/env.sh`:

   ```bash
   export LIMA_HOME=<FP>/.vm/lima                 # VM disk and state stay in final_project
   export PATH=<FP>/.vm/lima-dist/bin:$PATH
   ```

4. Create `<FP>/.vm/vm.yaml`:

   ```yaml
   vmType: vz
   arch: x86_64
   cpus: 8
   memory: 8GiB
   disk: 30GiB
   images:
   - location: <FP>/.vm/img/jammy.img
     arch: x86_64
   mountType: virtiofs
   mounts:
   - location: <FP>
     writable: true
   containerd: {system: false, user: false}
   ssh: {loadDotSSHPubKeys: false, forwardAgent: false}
   portForwards:                                  # Lima forwards the guest's localhost ports by default
   - {guestIP: 0.0.0.0, proto: any, ignore: true}
   ```

5. Start the VM and install the runtime in it (it downloads the weights and the dataset):

   ```bash
   source <FP>/.vm/env.sh
   limactl start --name=d --tty=false <FP>/.vm/vm.yaml
   limactl shell --workdir <FP>/Diffuser-Project d -- bash -c 'TORCH_SPEC=torch==1.9.1+cpu bash experiments/common/setup/linux_setup.sh'
   ```

6. Create `<FP>/.vm/vmrun`, which runs a command in the guest from the repo root:

   ```bash
   #!/bin/bash
   HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   source "$HERE/env.sh"
   exec limactl shell --workdir "$HERE/../Diffuser-Project" d -- "$@"
   ```

Running experiments, from `Diffuser-Project/`:

```bash
../.vm/vmrun bash experiments/common/setup/run.sh python experiments/common/setup/check_env.py
```

```bash
../.vm/vmrun bash experiments/blocked/run_all.sh smoke
```

Things to know:

- **Where the data lives.** The conda env, weights and dataset are on the VM disk, under `~/.local/share/ece780-diffuser` in the guest. `linux_setup.sh` leaves two symlinks in the shared folder: `Diffuser-Project/logs` and `final_project/artifacts/d4rl`. Both point to guest paths, so they dangle on the Mac. Neither is committed.
- **Speed.** `plan(n=16)` takes 24–33 s. One blocked scenario with all 5 methods takes about 7.5 min, so `blocked/main` (50 scenarios) needs 6+ hours.
- **Reproducibility.** Runs are deterministic on this CPU. They are not bit-identical to the committed B0 numbers (a different CPU) or to GPU runs, so compare methods only within one runtime.
- **Stopping the VM.** Run `source <FP>/.vm/env.sh && limactl stop d`; `limactl start d` brings it back. To remove it, run `limactl delete d`, then remove `<FP>/.vm` and the two symlinks. This also deletes the env, weights and dataset.
