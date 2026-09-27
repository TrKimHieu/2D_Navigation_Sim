"""Wrappers.

DiscreteActions   Habitat-style discrete actions for the SVG env (forward/backward 0.25 m,
                  turn 15 deg), executed at the robot's top speed over several dt steps,
                  so slow robots spend more time per action.
EpisodeRecorder   one JSONL line per finished episode (common schema for both envs).
CustomReward      reward = fn(info), built from info["reward_terms"] and the step info.
Coverage          info["coverage"]: share of the map's free area seen by the LiDAR so far.
"""

from __future__ import annotations

import json
import math

import gymnasium as gym
import numpy as np

from .base import END_KEYS


class DiscreteActions(gym.Wrapper):
    """0 stay (one dt), 1 forward, 2 backward, 3 turn left, 4 turn right,
    5 strafe left, 6 strafe right (omnidirectional robots only). A macro stops early
    on collision or when the episode ends."""

    def __init__(self, env, step_m=0.25, turn_deg=15.0):
        super().__init__(env)
        r = env.unwrapped.robot
        self.step_m, self.turn = step_m, math.radians(turn_deg)
        self.action_space = gym.spaces.Discrete(7 if r.omni else 5)

    def _plan(self, a):
        u = self.env.unwrapped
        r, dt, k = u.robot, u.dt, u.robot.action_dim
        vec = np.zeros(k)
        if a == 0:
            return [vec]
        if a in (1, 2):
            speed, amount, axis, sign = (r.v_max if a == 1 else -r.v_min), self.step_m, 0, (1 if a == 1 else -1)
        elif a in (3, 4):
            speed, amount, axis, sign = r.w_max, self.turn, k - 1, (1 if a == 3 else -1)
        else:
            speed, amount, axis, sign = r.vy_max, self.step_m, 1, (1 if a == 5 else -1)
        per = speed * dt
        n_full, rest = int(amount // per), amount % per
        plan = []
        for frac in [1.0] * n_full + ([rest / per] if rest > 1e-9 else []):
            v = vec.copy()
            v[axis] = sign * frac
            plan.append(v)
        return plan

    def step(self, action):
        total, n = 0.0, 0
        for a in self._plan(int(action)):
            obs, r, term, trunc, info = self.env.step(a)
            total += r
            n += 1
            if term or trunc or info["collided"]:
                break
        info["macro_steps"] = n
        return obs, total, term, trunc, info


class EpisodeRecorder(gym.Wrapper):
    """Append each finished episode to a JSONL file: robot, dataset, seed + the common
    end-of-episode keys (see envs.base.END_KEYS); optionally the trajectory."""

    def __init__(self, env, path, trajectory=False):
        super().__init__(env)
        self.path, self.trajectory = path, trajectory
        self._seed = None

    def reset(self, seed=None, options=None):
        self._seed = seed
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        out = self.env.step(action)
        if out[2] or out[3]:
            u, info = self.env.unwrapped, out[4]
            ds = getattr(u, "dataset", None)
            rec = {"robot": getattr(u, "robot_id", None) or u.robot.id,
                   "dataset": ds.name if ds is not None else None, "seed": self._seed,
                   **{k: info[k] for k in END_KEYS if k in info}}
            if self.trajectory and hasattr(u, "traj"):
                rec["trajectory"] = [[round(v, 4) for v in p] for p in u.traj]
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return out


class CustomReward(gym.Wrapper):
    """Replace the reward by ``fn(info)``; the environment's own reward is kept in
    ``info["env_reward"]``. Example (SVG, dense terms)::

        CustomReward(env, lambda i: i["reward_terms"]["progress"] + 5 * i["reward_terms"]["success"]
                     - 0.05 * (i["wall_distance_m"] < 0.3))
    """

    def __init__(self, env, fn):
        super().__init__(env)
        self.fn = fn

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info["env_reward"] = r
        return obs, float(self.fn(info)), term, trunc, info


class Coverage(gym.Wrapper):
    """Share of the free area seen by the LiDAR since the episode start.

    Every step the cells crossed by each beam of the observation, from the robot to the
    measured range, are marked seen (samples every res/2 along the beam; only free
    cells count). info["coverage"] = seen free cells / all free cells of the map (the
    denominator includes parts the robot may not reach); info["coverage_gain"] = the
    increase in this step. SVG: cells of ``res`` m; grid: the grid cells.
    Wrap the environment directly (inside DiscreteActions), or macro-steps only
    count their last observation.
    """

    def __init__(self, env, res=0.1):
        super().__init__(env)
        self.res = res

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = self.env.unwrapped
        if hasattr(u, "agent_pos"):                          # grid
            self._free = u.grid == 0
        else:
            self._free, self._x0, self._y1, _ = u.map.raster(self.res)
        self._seen = np.zeros_like(self._free)
        self._n_free = max(int(self._free.sum()), 1)
        self._mark(obs)
        info["coverage"], info["coverage_gain"] = self._value, self._value
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        before = self._value
        self._mark(out[0])
        out[4]["coverage"], out[4]["coverage_gain"] = self._value, self._value - before
        return out

    def _mark(self, obs):
        u = self.env.unwrapped
        if hasattr(u, "agent_pos"):
            r0, c0 = u.agent_pos
            ranges = np.asarray(obs, float)
            ang = np.linspace(0, 2 * np.pi, len(ranges), endpoint=False)
            t = np.arange(0, ranges.max() + 0.5, 0.5)
            ok = t[None, :] < ranges[:, None]                  # cells before the hit
            rr = (r0 + t[None, :] * np.cos(ang)[:, None]).astype(np.int64)[ok]
            cc = (c0 + t[None, :] * np.sin(ang)[:, None]).astype(np.int64)[ok]
        else:
            x, y, th = u.pose
            ranges = np.asarray(obs["lidar"], float)
            ang = th + u.beam_angles
            t = np.arange(0, ranges.max() + self.res / 2, self.res / 2)
            ok = t[None, :] <= ranges[:, None]
            px = (x + t[None, :] * np.cos(ang)[:, None])[ok]
            py = (y + t[None, :] * np.sin(ang)[:, None])[ok]
            cc = np.floor((px - self._x0) / self.res).astype(np.int64)
            rr = np.floor((self._y1 - py) / self.res).astype(np.int64)
        H, W = self._free.shape
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[inb], cc[inb]
        self._seen[rr, cc] |= self._free[rr, cc]
        self._value = float(self._seen.sum()) / self._n_free
