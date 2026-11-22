"""
Report figures and animations, drawn from saved results only (no model, no MuJoCo).

    bash experiments/common/setup/run.sh python experiments/blocked/make_figures.py
    bash experiments/common/setup/run.sh python experiments/blocked/make_figures.py --only primary anim_methods
    bash experiments/common/setup/run.sh python experiments/blocked/make_figures.py --no_anim

Reads results/main*, results/probe*/scenarios.json and results/diag; writes figures/<name>.png and .pdf,
and for animations figures/<name>.gif plus a last-frame <name>_final.png. The numbers come from the same
episode files as analyze_main.py (seeds excluded in any method are dropped from all of them).
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from analyze_main import RES, RUNS, PRIMARY, wilson, mcnemar_exact  # noqa: E402
from experiments.common import maze_graph as mg  # noqa: E402
from experiments.common import maze_utils as mu  # noqa: E402
from experiments.common import obstacles as ob  # noqa: E402

OUT = os.path.join(HERE, 'figures')
DT = 0.01                                   # seconds per env step
LARGE_MAZE = ('############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\'
              '#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############')   # d4rl pointmaze LARGE_MAZE

# Reference palette (dataviz skill, light mode). Methods are coloured by family; every bar is also labelled.
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8c8b86', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                                                   '#6250d6', '#e34948')
FAMILY = {'B0': MUTED, 'B1': MUTED, 'B2': MUTED, 'ours (corners)': BLUE, 'ours (junctions)': BLUE,
          'astar_line + lag': ORANGE, 'oracle': INK2}
FAIL_COLOR = {'success': BLUE, 'obstacle_collision': RED, 'timeout_still_moving': YELLOW,
              'stuck_on_wall': VIOLET}
FAIL_LABEL = {'success': 'success', 'obstacle_collision': 'hit obstacle', 'timeout_still_moving': 'timeout (moving)',
              'stuck_on_wall': 'stuck on wall'}
RUN_DIR = {n: (r, m) for n, r, m in RUNS}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'savefig.dpi': 200, 'figure.facecolor': 'white'})


def parse_maze(s):
    rows = s.split('\\')
    return np.array([[mu.WALL if ch == '#' else mu.GOAL if ch == 'G' else mu.EMPTY for ch in r] for r in rows])


MAZE = parse_maze(LARGE_MAZE)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print('wrote', f'figures/{name}.png/.pdf')


# ---------------------------------------------------------------- data

class Main:
    """main's scenarios and episodes, paired the way analyze_main.py pairs them."""

    def __init__(self):
        self.scen = {s['seed']: s for s in json.load(open(os.path.join(RES, 'main', 'scenarios.json')))['scenarios']}
        self.eps = {}
        for name, run, method in RUNS:
            d = os.path.join(RES, run, method, 'episodes')
            self.eps[name] = {s: json.load(open(os.path.join(d, f'seed{s}.json'))) for s in self.scen
                              if os.path.exists(os.path.join(d, f'seed{s}.json'))}
        self.names = [n for n, _, _ in RUNS]
        self.dropped = sorted({s for n in self.names for s, r in self.eps[n].items() if r.get('excluded_reason')})
        self.seeds = sorted(set(self.scen).intersection(*[set(self.eps[n]) for n in self.names]) - set(self.dropped))
        self.longer = [s for s in self.seeds if self.extra(s) > 0]
        self.same = [s for s in self.seeds if self.extra(s) == 0]

    def extra(self, s):
        return self.scen[s]['detour_cells'] - self.scen[s]['base_cells']

    def outcome(self, n, s):
        r = self.eps[n][s]
        return 'success' if r['success'] else r['failure_reason']

    def traj(self, n, s):
        run, method = RUN_DIR[n]
        return np.load(os.path.join(RES, run, method, 'episodes', f'seed{s}.npz'))


def rate_ci(k, n):
    lo, hi = wilson(k, n)
    return k / n, lo, hi


# ---------------------------------------------------------------- maze drawing

def maze_axes(ax):
    mu.draw_maze(ax, MAZE, color='#cfcdc6')
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def xy(p):
    """(x = qpos[1], y = qpos[0]) for draw_maze's orientation."""
    p = np.asarray(p)
    return p[..., 1], p[..., 0]


def goal_marker(ax, target, radius=0.5):
    ax.add_patch(Circle(xy(target), radius, fill=False, ec=VIOLET, lw=1.5, zorder=5))
    ax.plot(*xy(target), marker='*', ms=9, c=VIOLET, zorder=6)


def route_xy(cells):
    return np.array([mu.cell_center(c) for c in cells])


def astar_event(rec):
    return next((e for e in rec['events'] if e['event'] == 'astar' and 'waypoints' in e), None)


def astar_route(rec, rollout, scen):
    """The A* route the method computed at its first A* call (from the ball's cell then)."""
    e = astar_event(rec)
    if e is None:
        return None, None
    start = mu.xy_to_cell(rollout[min(e['t'], len(rollout) - 1)])
    path = mg.astar(MAZE, tuple(start), tuple(scen['goal_cell']), blocked=[tuple(scen['obstacle']['cell'])])
    return path, e['waypoints']


# ---------------------------------------------------------------- static figures

def fig_pipeline(_m):
    """The planning loop, coloured by layer."""
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.9), ax.axis('off')
    lay = {'exec': ('#efeeea', MUTED), 'diff': ('#dce9f9', BLUE), 'route': ('#d6f1e6', AQUA), 'test': ('white', INK2)}

    def box(x, y, w, h, text, kind):
        fc, ec = lay[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.5, color=INK)

    def arrow(x0, y0, x1, y1, label=None, lx=0, ly=0):
        ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.0))
        if label:
            ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=7.5, color=INK2, ha='center')

    box(0.1, 3.3, 2.3, 0.9, 'Follow current plan\n(PD + MuJoCo)', 'exec')
    box(3.0, 3.3, 2.7, 0.9, 'Every k = 10 steps:\ndoes the plan hit the obstacle?', 'test')
    box(6.4, 3.3, 3.5, 0.9, 'Best-of-16 from the Diffuser\n(H = 384, keep a clear sample)', 'diff')
    box(6.4, 1.75, 3.5, 0.9, 'All 16 hit: A* on the maze\nwith the blocked cell removed', 'route')
    box(6.4, 0.2, 3.5, 0.9, 'Waypoints on the A* route\n(junctions, or + corners)', 'route')
    box(2.4, 0.2, 3.4, 0.9, 'One short Diffuser plan per leg\n(best-of-8, H from the leg length)', 'diff')
    box(0.1, 1.75, 2.3, 0.9, 'Next leg when the\nwaypoint is reached', 'exec')
    arrow(2.4, 3.75, 3.0, 3.75)
    arrow(5.7, 3.75, 6.4, 3.75, 'yes', ly=0.08)
    arrow(3.0, 3.5, 2.4, 3.5, 'no', ly=-0.25)
    arrow(8.15, 3.3, 8.15, 2.65, 'none clear', lx=0.55)
    ax.annotate('', (2.4, 4.05), (6.4, 4.05), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.0,
                                                               connectionstyle='arc3,rad=0.25'))
    ax.text(4.4, 4.68, 'clear sample found: adopt it', fontsize=7.5, color=INK2, ha='center')
    arrow(8.15, 1.75, 8.15, 1.1)
    arrow(6.4, 0.65, 5.8, 0.65)
    arrow(2.4, 0.65, 1.25, 1.75)
    arrow(1.25, 2.65, 1.25, 3.3)
    ax.legend(handles=[Patch(fc=lay[k][0], ec=lay[k][1], label=l) for k, l in
                       (('route', 'Route = A*'), ('diff', 'Motion plan = Diffuser'), ('exec', 'Execution = PD + MuJoCo'))],
              loc='lower left', bbox_to_anchor=(0.0, -0.13), ncol=3, frameon=False, fontsize=8)
    save(fig, 'fig_pipeline')


def fig_primary(m):
    """Success on the 29 longer-detour scenarios, every method, and the pre-registered paired test."""
    n = len(m.longer)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[2.3, 1]))
    names = m.names[::-1]
    for i, name in enumerate(names):
        k = sum(m.eps[name][s]['success'] for s in m.longer)
        p, lo, hi = rate_ci(k, n)
        ax.barh(i, p * 100, height=0.62, color=FAMILY[name], zorder=2)
        ax.plot([lo * 100, hi * 100], [i, i], c=INK, lw=1.0, zorder=3)
        ax.text(103, i, f'{k}/{n}', va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(names))), ax.set_yticklabels(names)
    ax.set_xlim(0, 126), ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('success rate (%), 95% Wilson CI')
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.set_title(f'Full blockage with a longer detour (n = {n} paired scenarios)', fontsize=9, loc='left')
    a, b = PRIMARY
    ia, ib = names.index(a), names.index(b)
    sa = [m.eps[a][s]['success'] for s in m.longer]
    sb = [m.eps[b][s]['success'] for s in m.longer]
    only_a = sum(x and not y for x, y in zip(sa, sb))
    only_b = sum(y and not x for x, y in zip(sa, sb))
    p = mcnemar_exact(only_a, only_b)
    xb = 116
    ax.plot([xb - 1.5, xb, xb, xb - 1.5], [ia, ia, ib, ib], c=INK, lw=0.9)
    ax.text(xb + 1, (ia + ib) / 2, f'p = {p:.4f}', rotation=90, va='center', fontsize=7.5)

    both = sum(x and y for x, y in zip(sa, sb))
    neither = n - both - only_a - only_b
    grid = np.array([[both, only_a], [only_b, neither]])
    bx.imshow(grid, cmap='Blues', vmin=0, vmax=n * 0.6)
    for (r, c), v in np.ndenumerate(grid):
        bx.text(c, r, str(v), ha='center', va='center', fontsize=11, color=INK if v < n * 0.35 else 'white')
    bx.set_xticks([0, 1]), bx.set_xticklabels(['success', 'fail'])
    bx.set_yticks([0, 1]), bx.set_yticklabels(['success', 'fail'])
    bx.set_xlabel(f'{a}'), bx.set_ylabel(f'{b}')
    bx.xaxis.set_label_position('top'), bx.xaxis.tick_top()
    bx.set_title(f'Exact McNemar on {only_a} vs {only_b} discordant', fontsize=8, y=-0.25)
    for sp in bx.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, 'fig_primary')


def fig_by_extra(m):
    """Success by how much longer the detour is than the blocked route."""
    bins = [('extra ≤ 6 cells', [s for s in m.longer if m.extra(s) <= 6], BLUE, 'o'),
            ('extra ≥ 8 cells', [s for s in m.longer if m.extra(s) >= 8], ORANGE, 's')]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    names = m.names[::-1]
    for j, (label, ss, col, mk) in enumerate(bins):
        off = 0.17 if j == 0 else -0.17
        for i, name in enumerate(names):
            k = sum(m.eps[name][s]['success'] for s in ss)
            p, lo, hi = rate_ci(k, len(ss))
            ax.plot([lo * 100, hi * 100], [i + off] * 2, c=col, lw=1.2, alpha=0.6)
            ax.plot(p * 100, i + off, marker=mk, ms=6.5, c=col, mec='white', mew=1.2,
                    label=f'{label} (n = {len(ss)})' if i == 0 else None, zorder=3)
            ax.text(103, i + off, f'{k}/{len(ss)}', va='center', fontsize=7, color=INK2)
    ax.set_yticks(range(len(names))), ax.set_yticklabels(names)
    ax.set_xlim(-3, 112), ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('success rate (%), 95% Wilson CI')
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.16), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_by_extra')


def fig_outcomes(m):
    """Every paired scenario x method, sorted by extra detour cells; colour = outcome."""
    order = sorted(m.seeds, key=lambda s: (m.extra(s), s))
    fig, (top, ax) = plt.subplots(2, 1, figsize=(8.6, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 3.2]))
    top.bar(range(len(order)), [m.extra(s) for s in order], color=MUTED, width=0.8)
    top.set_ylabel('extra\ncells', fontsize=8)
    top.set_yticks([0, 6, 12])
    top.grid(axis='y', color=GRID, lw=0.8)
    for i, name in enumerate(m.names):
        for j, s in enumerate(order):
            o = m.outcome(name, s)
            ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, color=FAIL_COLOR.get(o, INK2), lw=0))
    ax.set_xlim(-0.6, len(order) - 0.4), ax.set_ylim(len(m.names) - 0.5, -0.5)
    ax.set_yticks(range(len(m.names))), ax.set_yticklabels(m.names, fontsize=8)
    ax.set_xticks(range(len(order))), ax.set_xticklabels([str(s)[-2:] for s in order], fontsize=5.5, rotation=90)
    ax.set_xlabel('scenario (seed 10xx), sorted by extra detour cells')
    split = len(m.same) - 0.5
    for a in (top, ax):
        a.axvline(split, c=INK, lw=0.8, ls='--')
    top.text(split / 2, 10, 'same-length route', ha='center', fontsize=7.5, color=INK2)
    top.text(split + (len(order) - split) / 2, 10, 'longer detour (primary population)', ha='center', fontsize=7.5,
             color=INK2)
    ax.legend(handles=[Patch(color=FAIL_COLOR[k], label=FAIL_LABEL[k]) for k in FAIL_COLOR],
              loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(h_pad=0.3)
    save(fig, 'fig_outcomes')


def fig_failures(m):
    """Failure reasons on the longer-detour scenarios."""
    kinds = ['obstacle_collision', 'timeout_still_moving', 'stuck_on_wall']
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    names = m.names[::-1]
    for i, name in enumerate(names):
        left = 0
        for k in kinds:
            v = sum(m.outcome(name, s) == k for s in m.longer)
            if v:
                ax.barh(i, v, left=left, height=0.62, color=FAIL_COLOR[k], ec='white', lw=1.5)
                ax.text(left + v / 2, i, str(v), ha='center', va='center', fontsize=7.5, color='white')
            left += v
        if left == 0:
            ax.text(0.2, i, 'no failures', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names))), ax.set_yticklabels(names)
    ax.set_xlabel(f'failed scenarios (of {len(m.longer)} longer detours)')
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.legend(handles=[Patch(color=FAIL_COLOR[k], label=FAIL_LABEL[k]) for k in kinds], loc='lower right',
              frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_failures')


def fig_cost(m):
    """Diffuser planning time per episode (CPU VM), longer-detour scenarios."""
    names = [n for n in m.names if n != 'oracle'][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        t = np.array([m.eps[name][s]['planning_time_s'] for s in m.longer])
        ax.scatter(t, i + rng.uniform(-0.18, 0.18, len(t)), s=9, color=FAMILY[name], alpha=0.55, lw=0)
        med = np.median(t)
        ax.plot([med, med], [i - 0.3, i + 0.3], c=INK, lw=1.6)
        ax.text(med, i + 0.36, f'median {med:.0f} s, mean {t.mean():.0f} s', ha='left', fontsize=7, color=INK)
    ax.set_yticks(range(len(names))), ax.set_yticklabels(names)
    ax.set_xlabel(f'Diffuser planning time per episode (s, CPU VM; {len(m.longer)} longer detours; bar = median)')
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    fig.tight_layout()
    save(fig, 'fig_cost')


def fig_scenario(m, seed=1042):
    """One scenario: the t=0 plan runs into the new obstacle; the A* route and the two waypoint modes."""
    sc = m.scen[seed]
    obst = ob.Obstacle.from_dict(sc['obstacle'])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    tr = m.traj('B0', seed)
    for a in (ax, bx):
        maze_axes(a)
        ob.draw_obstacle(a, obst, color=RED)
        goal_marker(a, sc['target'])
        a.plot(*xy(tr['rollout_xy'][0]), 'o', c=AQUA, ms=7, mec='white', zorder=6)
    p0 = tr['plan0']
    ax.plot([], [], 'o', c=AQUA, ms=7, mec='white', label='start')
    ax.plot(*xy(p0[:, :2]), c=BLUE, lw=1.6, label='Diffuser plan at t = 0')
    k = sc['t_appear']
    ax.plot(*xy(p0[k, :2]), 'v', c=INK, ms=7, zorder=7, label=f'ball when the obstacle appears (t = {k * DT:.1f} s)')
    ax.set_title('(a) a corridor on the planned route is blocked', fontsize=9, loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False)
    for name, mk, col, lab in (('ours (junctions)', 'X', INK, 'junction waypoints'),
                               ('ours (corners)', 'P', ORANGE, 'corner + junction waypoints')):
        rec = m.eps[name][seed]
        path, wps = astar_route(rec, m.traj(name, seed)['rollout_xy'], sc)
        if name == 'ours (junctions)':
            bx.plot(*xy(route_xy(path)), c=AQUA, lw=2.2, alpha=0.8, label='A* route (blocked cell removed)')
            t = astar_event(rec)['t']
            bx.plot(*xy(m.traj(name, seed)['rollout_xy'][t]), 'v', c=INK, ms=7, zorder=8,
                    label=f'ball at the A* call (t = {t * DT:.1f} s)')
        w = route_xy(wps)
        bx.scatter(*xy(w), marker=mk, s=40 if mk == 'X' else 28, c=col, zorder=7, label=lab, lw=0)
    bx.set_title(f'(b) reroute: {sc["detour_cells"] - sc["base_cells"]} cells longer than the blocked route',
                 fontsize=9, loc='left')
    bx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, 'fig_scenario')


def fig_samples(_m):
    """Offline best-of-64 at the moment the obstacle appears: a short detour is sampled, a long one never is."""
    rows = json.load(open(os.path.join(RES, 'diag', 'trigger_sampling', 'rows.json')))
    rest = json.load(open(os.path.join(RES, 'diag', 'trigger_sampling_rest', 'rows.json')))
    fig = plt.figure(figsize=(7.6, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    for col, (name, seed) in enumerate((('probe_multigoal', 14), ('probe', 0))):
        ax = fig.add_subplot(gs[0, col])
        maze_axes(ax)
        sc = next(s for s in json.load(open(os.path.join(RES, name, 'scenarios.json')))['scenarios']
                  if s['seed'] == seed)
        row = next(r for r in rows if r['results'] == name and r['seed'] == seed and r['horizon'] == 384)
        plans = np.load(os.path.join(RES, 'diag', 'trigger_sampling', f'{name}_seed{seed:03d}_H384.npy'))
        obst = ob.Obstacle.from_dict(sc['obstacle'])
        valid = [s['valid'] for s in row['samples']]
        for p, v in sorted(zip(plans, valid), key=lambda x: x[1]):
            ax.plot(*xy(p[:, :2]), c=BLUE if v else MUTED, lw=0.7 if v else 0.5, alpha=0.5 if v else 0.3)
        ob.draw_obstacle(ax, obst, color=RED)
        goal_marker(ax, sc.get('target', [7.0, 9.0]))
        path = mg.astar(MAZE, tuple(sc['cell_at_appear']), tuple(mu.xy_to_cell(sc.get('target', [7.0, 9.0]))),
                        blocked=[tuple(sc['obstacle']['cell'])])
        ax.plot(*xy(route_xy(path)), c=AQUA, lw=2.0, ls='--', alpha=0.9)
        ax.plot(*xy(plans[0, 0, :2]), 'o', c=INK, ms=6, mec='white', zorder=7)
        ex = row['detour_cells'] - row['base_cells']
        ax.set_title(f'({"ab"[col]}) detour {ex} cells longer: {row["n_valid"]}/{row["n"]} samples usable',
                     fontsize=8.5, loc='left')
        top_ax = ax
    handles = ([Line2D([], [], c=BLUE, lw=1.2, label='usable sample (clear, in free space)'),
                        Line2D([], [], c=MUTED, lw=1.2, label='other sample'),
                        Line2D([], [], c=AQUA, lw=2, ls='--', label='A* detour'),
                        Patch(color=RED, alpha=0.6, label='obstacle')])

    cx = fig.add_subplot(gs[1, :])
    styles = {384: (BLUE, 'o', -0.15), 512: (ORANGE, 's', 0.15), 640: (VIOLET, '^', 0.45)}
    for r in rows:
        c, mk, off = styles[r['horizon']]
        cx.plot(r['detour_cells'] - r['base_cells'] + off, 100 * r['n_valid'] / r['n'], marker=mk, ls='', c=c,
                ms=5.5, alpha=0.8, mec='white', mew=0.6)
    for r in rest:
        cx.plot(r['detour_cells'] - r['base_cells'] - 0.45, 100 * r['n_valid'] / r['n'], marker='o', ls='',
                mfc='none', mec=INK2, ms=5.5)
    cx.set_xlabel('extra cells of the A* detour over the blocked route')
    cx.set_ylabel('usable samples (% of 64)')
    cx.set_xticks([2, 4, 6, 8, 10, 12])
    cx.axvspan(7, 13, color=GRID, alpha=0.5, lw=0, zorder=0)
    cx.text(10, 70, 'extra ≥ 8: 0 usable samples\nin every scenario and horizon', ha='center', fontsize=8, color=INK2)
    cx.grid(axis='y', color=GRID, lw=0.8)
    cx.legend(handles=[Line2D([], [], marker=mk, ls='', c=c, label=f'H = {h}' + (' (trained)' if h == 384 else ''))
                       for h, (c, mk, _) in styles.items()] +
              [Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, label='H = 384, ball at rest')],
              loc='center right', bbox_to_anchor=(0.97, 0.45), frameon=False, fontsize=7.5, ncol=2)
    cx.set_title('(c) every probe scenario: the trigger state, 64 samples per horizon', fontsize=8.5, loc='left')
    fig.tight_layout(h_pad=3.5)
    fig.legend(handles=list(handles), loc='upper center', bbox_to_anchor=(0.5, top_ax.get_position().y0 - 0.005),
               ncol=4, frameon=False, fontsize=7.5)
    save(fig, 'fig_samples')


def corridor_legs(maze):
    """Same enumeration as diag/leg_test.py (rows.json has 3 rows per leg, in this order)."""
    nodes = {c for c in mu.free_cells(maze) if mg.degree(maze, c) != 2}
    legs = []
    for a in sorted(nodes):
        for d in mu.NEIGHBORS4:
            path = [a, (a[0] + d[0], a[1] + d[1])]
            if not mu.is_free(maze, path[-1]):
                continue
            while path[-1] not in nodes:
                nxt = [(path[-1][0] + e[0], path[-1][1] + e[1]) for e in mu.NEIGHBORS4]
                path.append(next(c for c in nxt if mu.is_free(maze, c) and c != path[-2]))
            legs.append(path)
    return legs


def fig_legs(_m):
    """diag/leg_test: every directed corridor as one waypoint leg from rest, no obstacle."""
    rows = json.load(open(os.path.join(RES, 'diag', 'leg_test', 'rows.json')))
    legs = corridor_legs(MAZE)
    methods = ('diffuser', 'line', 'line_corners')
    by = {m: [r for r in rows if r['method'] == m] for m in methods}
    for m in methods:
        assert len(by[m]) == len(legs)
        for r, path in zip(by[m], legs):
            assert tuple(r['start']) == path[0] and tuple(r['end']) == path[-1] and r['cells'] == len(path) - 1
    label = {'diffuser': 'Diffuser (best-of-8)', 'line': 'straight line node to node',
             'line_corners': 'line through the corners'}
    mcol = {'diffuser': BLUE, 'line': VIOLET, 'line_corners': ORANGE}
    mmk = {'diffuser': 'o', 'line': 's', 'line_corners': '^'}
    fig = plt.figure(figsize=(7.6, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95])
    for col, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        maze_axes(ax)
        for r, path in zip(by[m], legs):
            c = route_xy(path)
            d = c[-1] - c[0]
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            c = c + 0.13 * n                                   # the two directions side by side
            o = 'fail' if not r['success'] else 'one' if r['plans'] == 1 else 'more'
            colr = {'one': AQUA, 'more': YELLOW, 'fail': RED}[o]
            ax.plot(*xy(c), c=colr, lw=1.6, solid_capstyle='round')
            ax.annotate('', xy(c[-1]), xy(c[-2] + 0.45 * (c[-1] - c[-2])),
                        arrowprops=dict(arrowstyle='-|>', color=colr, lw=1.2, mutation_scale=8))
            if o == 'fail':
                ax.plot(*xy(mu.cell_center(r['final_cell'])), 'X', c=RED, ms=7, mec='white', zorder=7)
        k = sum(r['success'] for r in by[m])
        ax.set_title(f'({"abc"[col]}) {label[m]}: {k}/{len(legs)}', fontsize=8, loc='left')
    fig.legend(handles=[Line2D([], [], c=AQUA, lw=1.8, label='reached with the first plan'),
                        Line2D([], [], c=YELLOW, lw=1.8, label='reached after 2-3 plans'),
                        Line2D([], [], c=RED, lw=1.8, label='not reached (X: where the ball ended)')],
               loc='upper center', ncol=3, frameon=False, fontsize=7.5)

    dx = fig.add_subplot(gs[1, :])
    fail_y = 500
    rng = np.random.default_rng(1)
    for j, m in enumerate(methods):
        cells = np.array([r['cells'] for r in by[m]]) + (j - 1) * 0.22 + rng.uniform(-0.05, 0.05, len(by[m]))
        steps = np.array([r['steps'] if r['success'] else fail_y for r in by[m]], dtype=float)
        dx.plot(cells, steps, mmk[m], ls='', c=mcol[m], ms=6, alpha=0.75, mec='white', mew=0.6,
                label=f'{label[m]} ({sum(r["success"] for r in by[m])}/{len(legs)})')
        failed = [r['cells'] for r in by[m] if not r['success']]
        for c in sorted(set(failed)):
            dx.text(c + (j - 1) * 0.22 + 0.12, fail_y, f'x{failed.count(c)}', va='center', fontsize=7.5, color=INK)
    dx.axhline(fail_y - 35, c=MUTED, lw=0.8, ls='--')
    dx.set_ylim(0, fail_y + 30)
    dx.set_yticks([0, 100, 200, 300, 400, fail_y]), dx.set_yticklabels(['0', '100', '200', '300', '400', 'failed'])
    dx.set_xticks(sorted({r['cells'] for r in rows}))
    dx.set_xlabel('corridor length (cells)')
    dx.set_ylabel('steps to reach the next node')
    dx.grid(axis='y', color=GRID, lw=0.8)
    dx.legend(loc='upper left', bbox_to_anchor=(0.0, 0.92), frameon=False, fontsize=7.5)
    dx.set_title(f'(d) every directed corridor ({len(legs)}), ball starts at rest at a node, no obstacle; '
                 'up to 3 plans per leg', fontsize=8, loc='left')
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.5)
    save(fig, 'fig_legs')
    for m in methods:
        rr = by[m]
        print(f"  {m}: success {sum(r['success'] for r in rr)}/{len(rr)}, first plan "
              f"{sum(r['success'] and r['plans'] == 1 for r in rr)}, 2-3 plans "
              f"{sum(r['success'] and r['plans'] > 1 for r in rr)}, first plan in wall "
              f"{[r['first_plan_in_wall'] for r in rr if r['first_plan_in_wall'] > 1]}, steps when reached "
              f"{min(r['steps'] for r in rr if r['success'])}-{max(r['steps'] for r in rr if r['success'])}, "
              f"failed {[(tuple(r['start']), tuple(r['end']), tuple(r['final_cell'])) for r in rr if not r['success']]}")


# ---------------------------------------------------------------- animations

class Panel:
    """One method's episode on one axis; draw(t) updates it to env step t."""

    def __init__(self, ax, m, name, seed, label=None):
        self.ax, self.name = ax, name
        self.sc = m.scen[seed]
        self.rec = m.eps[name][seed]
        d = m.traj(name, seed)
        self.roll = d['rollout_xy']
        self.starts = list(d['plan_starts'])
        self.plans = [d[f'plan{i}'] for i in range(len(self.starts))]
        self.obst = ob.Obstacle.from_dict(self.sc['obstacle'])
        maze_axes(ax)
        goal_marker(ax, self.sc['target'])
        ax.plot(*xy(self.roll[0]), 'o', c=AQUA, ms=6, mec='white', zorder=6)
        x_lo, x_hi, y_lo, y_hi = self.obst.rect
        self.obst_patch = ax.add_patch(Rectangle((y_lo, x_lo), y_hi - y_lo, x_hi - x_lo, color=RED, alpha=0.0,
                                                 zorder=4))
        path, wps = astar_route(self.rec, self.roll, self.sc)
        self.t_astar = astar_event(self.rec)['t'] if path else None
        self.route = ax.plot(*xy(route_xy(path)), c=AQUA, lw=2.4, alpha=0.0)[0] if path else None
        self.wps = ax.scatter(*xy(route_xy(wps)), marker='X', s=30, c=INK, zorder=7, alpha=0.0, lw=0) if path else None
        self.plan_full = ax.plot([], [], c=BLUE, lw=1.0, alpha=0.25)[0]
        self.plan_line = ax.plot([], [], c=BLUE, lw=1.3, alpha=0.8)[0]
        self.setpoint = ax.plot([], [], 'o', c=BLUE, ms=5, mec='white', mew=0.8, zorder=8)[0]
        self.trail = ax.plot([], [], c=ORANGE, lw=1.8)[0]
        self.ball = ax.plot([], [], 'o', c=ORANGE, ms=7, mec=INK, mew=0.8, zorder=8)[0]
        self.end_mark = ax.plot([], [], 'X', c=RED, ms=11, mec='white', zorder=9)[0]
        self.label = label or name
        self.title = ax.set_title(self.label, fontsize=9, loc='left')
        self.T = len(self.roll) - 1

    def status(self):
        r = self.rec
        if r['success']:
            return f'reached goal at {r["first_success_step"] * DT:.2f} s', BLUE
        return FAIL_LABEL.get(r['failure_reason'], r['failure_reason']), RED

    def draw(self, t):
        k = min(t, self.T)
        if t >= self.obst.t_appear:
            self.obst_patch.set_alpha(0.65)
        if self.route is not None and t >= self.t_astar:
            self.route.set_alpha(0.35)
            self.wps.set_alpha(1.0)
        i = max((j for j, s in enumerate(self.starts) if s <= k), default=None)
        if i is not None:
            # waypoint_action tracks plan point (t - start) + 1, or the last point once the plan is used up
            plan = self.plans[i]
            j = min(k - self.starts[i] + 1, len(plan) - 1)
            self.plan_full.set_data(*xy(plan[:, :2]))
            self.plan_line.set_data(*xy(plan[j:, :2]))
            self.setpoint.set_data(*xy(plan[j, :2]))
        else:
            for line in (self.plan_full, self.plan_line, self.setpoint):
                line.set_data([], [])
        self.trail.set_data(*xy(self.roll[:k + 1]))
        self.ball.set_data(*xy(self.roll[k]))
        if t >= self.T:
            text, col = self.status()
            self.title.set_text(f'{self.label}: {text}')
            self.title.set_color(col)
            if not self.rec['success']:
                self.end_mark.set_data(*xy(self.roll[k]))


def animate(m, name, seed, panels, ncols, fig_w, fig_h, caption, stride=5, fps=20, hold=30, dpi=100):
    fig, axes = plt.subplots(int(np.ceil(len(panels) / ncols)), ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()
    ps = [Panel(ax, m, meth, seed, label) for ax, (meth, label) in zip(axes, panels)]
    for ax in axes[len(ps):]:
        ax.axis('off')
    T = max(p.T for p in ps)
    t_appear = ps[0].obst.t_appear
    clock = fig.text(0.01, 0.01, '', fontsize=9, color=INK)
    fig.text(0.99, 0.01, caption, fontsize=7.5, color=INK2, ha='right')
    handles = [Line2D([], [], c=BLUE, lw=1.3, marker='o', ms=4, markevery=[0], label='current plan (dot: where it wants the ball now)'),
               Line2D([], [], c=ORANGE, lw=1.8, label='executed')]
    if any(p.route is not None for p in ps):
        handles += [Line2D([], [], c=AQUA, lw=2.4, alpha=0.5, label='A* route'),
                    Line2D([], [], marker='X', ls='', c=INK, label='waypoint')]
    handles += [Patch(color=RED, alpha=0.65, label='obstacle'),
                Line2D([], [], marker='*', ls='', c=VIOLET, ms=9, label='goal')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.3 / fig_h, 1, 1 - 0.4 / fig_h))
    frames = list(range(0, T + 1, stride)) + [T] * hold

    def update(t):
        for p in ps:
            p.draw(t)
        note = '   obstacle appeared' if t >= t_appear else f'   obstacle appears at {t_appear * DT:.1f} s'
        clock.set_text(f't = {t * DT:5.2f} s{note}')
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    os.makedirs(OUT, exist_ok=True)
    anim.save(os.path.join(OUT, f'{name}.gif'), writer=PillowWriter(fps=fps), dpi=dpi)
    update(T)
    fig.savefig(os.path.join(OUT, f'{name}_final.png'), dpi=200)
    fig.savefig(os.path.join(OUT, f'{name}_final.pdf'))
    plt.close(fig)
    print('wrote', f'figures/{name}.gif and {name}_final.png/.pdf')


def anim_methods(m):
    animate(m, 'anim_methods', 1042,
            [('B2', 'B2 (best-of-16 on a predicted hit)'), ('ours (junctions)', 'ours (junctions)'),
             ('astar_line + lag', 'astar_line + lag'), ('oracle', 'oracle (A* + PD)')],
            2, 8.4, 6.8, 'main seed 1042: detour 12 cells longer', dpi=80)


def anim_junction(m):
    animate(m, 'anim_junction', 1026,
            [('ours (junctions)', 'ours (junctions)'), ('ours (corners)', 'ours (corners)')],
            2, 8.4, 3.8, 'main seed 1026, (3,3) blocked: the shortest corridor between two junctions is the blocked one')


def anim_lag(m):
    animate(m, 'anim_lag', 1046,
            [('B2', 'B2 (replans only on a predicted hit)'), ('B1', 'B1 (replans every 100 steps)')],
            2, 8.4, 3.8, 'main seed 1046: the ball falls behind a clear plan and stalls against a wall')


FIGURES = dict(pipeline=fig_pipeline, scenario=fig_scenario, primary=fig_primary, by_extra=fig_by_extra,
               outcomes=fig_outcomes, failures=fig_failures, cost=fig_cost, samples=fig_samples, legs=fig_legs,
               anim_methods=anim_methods, anim_junction=anim_junction, anim_lag=anim_lag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', choices=list(FIGURES))
    ap.add_argument('--no_anim', action='store_true')
    args = ap.parse_args()
    m = Main()
    print(f'{len(m.seeds)} paired scenarios (dropped {m.dropped}); {len(m.longer)} longer, {len(m.same)} same length')
    for name, fn in FIGURES.items():
        if (args.only and name not in args.only) or (args.no_anim and name.startswith('anim')):
            continue
        fn(m)


if __name__ == '__main__':
    main()
