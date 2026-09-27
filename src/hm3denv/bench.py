"""Throughput of an environment on this machine (``hm3d bench``), for tuning
``num_envs`` and for the throughput table of a paper.

Uniform random actions; episodes are drawn as in normal use (random map, then random
task, from the split). One env: step and reset are timed separately.
Several envs: an AsyncVectorEnv (see vector.make_vec) whose workers reset
automatically, so only the total rate is reported.
"""

from __future__ import annotations

import os
import platform
import time

import gymnasium as gym
import numpy as np

from .core import _accel
from .dataset import load_dataset
from .vector import make_vec

ENV_IDS = {"svg": "HM3D/Svg-v0", "grid": "HM3D/Grid-v0"}


def _env_kwargs(ds, robot, split, map_repeat, env_kwargs):
    robot = robot or (ds.robots[0] if len(ds.robots) == 1 else None)
    if robot is None:
        raise ValueError(f"choose a robot: {ds.robots}")
    return robot, dict(dataset=str(ds.root), robot=robot, split=split, map_repeat=map_repeat,
                       **env_kwargs)


def bench(dataset, robot=None, num_envs=1, seconds=10.0, map_repeat=1, split="train", seed=0,
          envs_per_worker=1, **env_kwargs) -> dict:
    ds = load_dataset(dataset)
    robot, kw = _env_kwargs(ds, robot, split, map_repeat, env_kwargs)
    env_id = ENV_IDS[ds.env_type]
    out = {"dataset": ds.name, "robot": robot, "split": split, "num_envs": num_envs,
           "map_repeat": map_repeat, "envs_per_worker": envs_per_worker,
           "backend": _accel.BACKEND, "cpu_count": os.cpu_count(),
           "python": platform.python_version()}

    env = gym.make(env_id, **kw)                 # also compiles/loads the numba kernels
    env.action_space.seed(seed)
    env.reset(seed=seed)
    if num_envs == 1:
        steps, t_step, resets = 0, 0.0, []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            a = env.action_space.sample()
            t0 = time.perf_counter()
            _, _, term, trunc, _ = env.step(a)
            t1 = time.perf_counter()
            t_step += t1 - t0
            steps += 1
            if term or trunc:
                env.reset()
                resets.append(time.perf_counter() - t1)
        env.close()
        total = t_step + sum(resets)
        out.update(steps_per_s=steps / total, step_ms=1e3 * t_step / steps, resets=len(resets),
                   reset_ms=1e3 * float(np.mean(resets)) if resets else None,
                   reset_p95_ms=1e3 * float(np.percentile(resets, 95)) if resets else None)
        return out

    env.close()
    venv = make_vec(env_id, num_envs, envs_per_worker=envs_per_worker, **kw)
    try:
        venv.action_space.seed(seed)
        venv.reset(seed=seed)
        for _ in range(10):                      # warm-up: first map loads in every worker
            venv.step(venv.action_space.sample())
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            venv.step(venv.action_space.sample())
            steps += num_envs
        rate = steps / (time.perf_counter() - t0)
    finally:
        venv.close()
    out.update(steps_per_s=rate, steps_per_s_per_env=rate / num_envs)
    return out
