"""Evaluate an agent on a dataset with the common metrics (``hm3d eval``).

Agent spec:
    "oracle"         shortest-path follower (OracleFollower / GridOracle)
    "random"         uniform random actions
    "module:attr"    a callable ``attr(env) -> policy`` called at the start of every
                     episode; ``policy(obs) -> action``

Episodes: every task of every map in the split (deterministic order), optionally
capped per map with ``per_map``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from .dataset import load_dataset
from .envs.oracle import GridOracle, OracleFollower
from .envs.wrappers import EpisodeRecorder


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
             out=None, trajectory=False, **env_kwargs) -> dict:
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
    for m in ds.maps(robot, split):
        n = len(ds.tasks(robot, m))
        for i in range(min(n, per_map or n)):
            obs, _ = env.reset(seed=seed, options={"map_id": m, "task_idx": i})
            policy = make_policy(env)
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(policy(obs))
                done = term or trunc
            recs.append(info)
    if not recs:
        raise ValueError("no episodes (empty split?)")
    succ = np.array([r["success"] for r in recs], float)
    return {"dataset": ds.name, "robot": robot, "agent": agent if isinstance(agent, str) else "custom",
            "split": split or "all", "episodes": len(recs), "success": float(succ.mean()),
            "spl": float(np.mean([r["spl"] for r in recs])),
            "collision_episodes": float(np.mean([r["n_collisions"] > 0 for r in recs])),
            "mean_time": float(np.mean([r["time"] for r in recs])),
            "mean_path_m": float(np.mean([r["path_length"] for r in recs]))}


def write_summary(summary: dict, path) -> None:
    Path(path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
