"""Evaluate an agent on a dataset with the common metrics (``hm3d eval``).

Agent spec:
    "oracle"         shortest-path follower (OracleFollower / GridOracle)
    "random"         uniform random actions
    "module:attr"    a callable ``attr(env) -> policy`` called at the start of every
                     episode; ``policy(obs) -> action``

Episodes: every task of every map in the split (deterministic order), optionally
capped per map with ``per_map``. ``progress=True`` prints one line per map on stderr.

Summary: success rate, SPL, share of episodes with a collision, mean time and path length,
and ``mean_progress``: the share of the shortest path covered by the end of the episode
(1 on success, negative when the agent ended farther away than it started).
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import load_dataset
from .envs.oracle import GridOracle, OracleFollower
from .envs.wrappers import EpisodeRecorder


def _progress(info: dict) -> float:
    """Share of the start's geodesic distance covered at the end of an episode."""
    if info["success"]:
        return 1.0
    if "goal_geodesic" in info:                           # grid: BFS steps
        g0, g1 = info.get("d_bfs"), info["goal_geodesic"]
    else:                                                 # svg (absent with nav_info=False)
        g0, g1 = info.get("geodesic_m"), info.get("goal_geodesic_m")
    if not g0 or g1 is None or not np.isfinite(g1):
        return float("nan")
    return 1.0 - g1 / g0


def _agent_factory(spec: str):
    if spec == "oracle":
        def make(env):
            u = env.unwrapped
            oracle = GridOracle(env) if hasattr(u, "agent_pos") else OracleFollower(env)
            return lambda obs: oracle.act()
        return make
    if spec == "random":
        return lambda env: (lambda obs: env.action_space.sample())
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("agent must be 'oracle', 'random' or 'module:callable'")
    return getattr(importlib.import_module(mod), attr)


def evaluate(dataset, robot=None, agent="oracle", split=None, per_map=None, seed=0,
             out=None, trajectory=False, progress=False, **env_kwargs) -> dict:
    ds = load_dataset(dataset)
    robot = robot or (ds.robots[0] if len(ds.robots) == 1 else None)
    if robot is None:
        raise ValueError(f"choose a robot: {ds.robots}")
    env = ds.make_env(robot=robot, split=split, **env_kwargs)
    if env.action_space is not None:
        env.action_space.seed(seed)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).unlink(missing_ok=True)
        env = EpisodeRecorder(env, out, trajectory=trajectory)
    make_policy = _agent_factory(agent) if isinstance(agent, str) else agent
    recs = []
    maps = ds.maps(robot, split)
    t0 = time.perf_counter()
    for k, m in enumerate(maps, 1):
        n = len(ds.tasks(robot, m))
        for i in range(min(n, per_map or n)):
            obs, _ = env.reset(seed=seed, options={"map_id": m, "task_idx": i})
            policy = make_policy(env)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy(obs))
                done = term or trunc
            recs.append(info)
        if progress:
            ok = sum(r["success"] for r in recs)
            print(f"  map {k}/{len(maps)} {m}: {len(recs)} episodes, {ok} successes "
                  f"({time.perf_counter() - t0:.0f} s)", file=sys.stderr, flush=True)
    if not recs:
        raise ValueError("no episodes (empty split?)")
    succ = np.array([r["success"] for r in recs], float)
    return {"dataset": ds.name, "robot": robot, "agent": agent if isinstance(agent, str) else "custom",
            "split": split or "all", "episodes": len(recs), "success": float(succ.mean()),
            "spl": float(np.mean([r["spl"] for r in recs])),
            "collision_episodes": float(np.mean([r["n_collisions"] > 0 for r in recs])),
            "mean_progress": _mean([_progress(r) for r in recs]),
            "mean_time": float(np.mean([r["time"] for r in recs])),
            "mean_path_m": float(np.mean([r["path_length"] for r in recs]))}


def _mean(xs) -> float | None:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def write_summary(summary: dict, path) -> None:
    Path(path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
