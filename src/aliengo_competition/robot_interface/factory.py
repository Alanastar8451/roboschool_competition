from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aliengo_competition.common.helpers import get_args
from aliengo_competition.robot_interface.sim import SimAliengoRobot
from scripts.play import load_env


def _clone_args(args, *, task: str, mode: str, headless: bool, load_run=-1, checkpoint=-1):
    clone = SimpleNamespace(**vars(args))
    clone.task = task
    clone.mode = mode
    clone.headless = headless
    clone.num_envs = 1
    clone.load_run = load_run
    clone.checkpoint = checkpoint
    clone.resume = True
    return clone


def _resolve_run_dir(load_run=-1) -> Path:
    runs_root = Path(__file__).resolve().parents[3] / "runs" / "gait-conditioned-agility"

    if load_run in (-1, None, "-1"):
        candidates = [path for path in runs_root.glob("*/*/*") if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {runs_root}")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    candidate = Path(str(load_run)).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    relative_candidate = (Path.cwd() / candidate).resolve()
    if relative_candidate.is_dir():
        return relative_candidate

    load_run_str = str(load_run)
    matches = sorted(
        path for path in runs_root.glob(f"*/*/{load_run_str}") if path.is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"Could not resolve training run '{load_run}' under {runs_root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def make_robot_interface(
    *,
    args=None,
    task: str = "aliengo_flat",
    mode: str = "sim",
    headless: bool = True,
    load_run=-1,
    checkpoint=-1,
):
    if args is None:
        args = get_args()
    _clone_args(args, task=task, mode=mode, headless=headless, load_run=load_run, checkpoint=checkpoint)
    if checkpoint not in (-1, None):
        print("Explicit checkpoints are ignored for controller demo; using the latest exported JIT policy from the selected run.")

    run_dir = _resolve_run_dir(load_run)
    print(f"Loading controller low-level policy from: {run_dir}")
    env, policy = load_env(run_dir, headless=headless)
    return SimAliengoRobot(env=env, policy=policy)
