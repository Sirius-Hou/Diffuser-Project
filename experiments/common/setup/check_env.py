"""Quick environment check: imports, MuJoCo build, dataset, weights.

    bash experiments/common/setup/run.sh python experiments/common/setup/check_env.py
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
ok = True


def check(name, fn):
    global ok
    try:
        msg = fn()
        print(f'[ OK ] {name}: {msg}')
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f'[FAIL] {name}: {type(e).__name__}: {e}')


check('python', lambda: sys.version.split()[0])
check('torch', lambda: __import__('torch').__version__ + f" (cuda={__import__('torch').cuda.is_available()})")
check('mujoco_py', lambda: __import__('mujoco_py').__version__)
check('gym', lambda: __import__('gym').__version__)


def _env():
    import gym
    import d4rl  # noqa: F401
    env = gym.make('maze2d-large-v1')
    return f"max_episode_steps={env._max_episode_steps}, target={env.unwrapped._target}"


check('maze2d-large-v1', _env)


def _data():
    d = os.environ.get('D4RL_DATASET_DIR', '')
    f = os.path.join(d, 'maze2d-large-sparse-v1.hdf5')
    assert os.path.exists(f), f'missing {f}'
    return f'{f} ({os.path.getsize(f) / 1e6:.0f} MB)'


check('dataset', _data)


def _weights():
    f = os.path.join(REPO, 'logs', 'maze2d-large-v1', 'diffusion', 'H384_T256', 'state_1960000.pt')
    assert os.path.exists(f), f'missing {f}'
    return f'{f} ({os.path.getsize(f) / 1e6:.1f} MB)'


check('weights', _weights)
print('ALL OK' if ok else 'SOME CHECKS FAILED')
sys.exit(0 if ok else 1)
