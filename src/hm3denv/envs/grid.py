"""Grid navigation on robot-sized cells.

    observation  `lidar_rays` LiDAR ranges in CELLS
    action       Discrete(4) 0=UP 1=DOWN 2=LEFT 3=RIGHT; a blocked move leaves the agent in place
    reward       +1 on reaching the goal, 0 otherwise; truncated after `budget` steps

One cell = one robot body; blocked cells are obstacles, floorless and roofless
(outdoor) cells; every task was checked on the 3D mesh.

`difficulty` ('easy' | 'medium' | 'hard' | 'starved') restricts random task draws to one level.

Step info adds goal_geodesic (BFS steps to the goal, inf if unreachable), progress (its
decrease in this step), collided (move blocked) and reward_terms ({"success": 0/1}).
``task_filter`` / ``set_task_filter``: curriculum, see envs.base.

LiDAR readings depend only on (map, cell), so each cell is scanned once and the
result reused (same values). ``map_cache`` defaults to 256 maps: grids are small, so
a whole split stays loaded; ``map_repeat`` as in envs.base.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.gridcore import MOVES, bfs, lidar_scan
from .base import EpisodeSource, spl

LABELS = ("p0", "sa_uniform", "sa_persist", "starved", "level")


@dataclass
class _GridCtx:
    map_id: str
    grid: np.ndarray
    tasks: list
    header: dict
    cell_m: float
    lidar: dict = field(default_factory=dict)      # (rays, range) -> (H, W, rays), NaN = not yet
    goal_bfs: OrderedDict = field(default_factory=OrderedDict)    # task -> BFS steps (LRU 8)


class GridEnv(EpisodeSource, gym.Env):
    """``gym.make("HM3D/Grid-v0", dataset=..., split="train")``. ``grid`` + ``tasks`` run
    a single in-memory map (tests)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, dataset=None, robot=None, split=None, map_id=None, task_idx=None,
                 budget=200, lidar_rays=22, lidar_range=10.0, difficulty=None, render_mode=None,
                 grid=None, tasks=None, cell_m=1.0, map_cache=256, map_repeat=1,
                 task_filter=None):
        super().__init__()
        self.split = split
        if tasks is not None:
            self.dataset = None
            self.robot_id = robot
            mid = tasks.get("map_id", "map")
            ctx = _GridCtx(mid, np.asarray(grid), tasks["tasks"], tasks, cell_m)
            self._init_source([mid], lambda m: ctx, map_id, task_idx, map_cache, map_repeat,
                              task_filter)
        else:
            from ..dataset import load_dataset
            self.dataset = ds = load_dataset(dataset)
            if ds.env_type != "grid":
                raise ValueError(f"dataset {ds.name} is a {ds.env_type} dataset")
            if robot is None:
                if len(ds.robots) != 1:
                    raise ValueError(f"choose a robot: {ds.robots}")
                robot = ds.robots[0]
            self.robot_id = robot
            maps = [map_id] if map_id else ds.maps(robot, split)
            self._init_source(maps, lambda m: _GridCtx(
                m, ds.grid_arrays(m)["grid"], ds.tasks(robot, m), ds.task_file(robot, m),
                ds.grid_meta(m)["cell_m"]), map_id, task_idx, map_cache, map_repeat,
                task_filter)
        self.difficulty = difficulty
        self.budget = budget
        self.lidar_rays, self.lidar_range = lidar_rays, lidar_range
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, lidar_range, (lidar_rays,), np.float32)
        self.agent_pos = None
        self.steps = 0
        self._window = self._clock = None

    def _tasks_of(self, map_id) -> list:
        if self.dataset is None:
            return super()._tasks_of(map_id)
        return self.dataset.tasks(self.robot_id, map_id)

    def _goal_dist(self) -> np.ndarray:
        """BFS steps from the goal to every cell (float, inf = unreachable)."""
        g = self.ctx.goal_bfs
        d = g.get(self.task_idx)
        if d is None:
            steps = bfs(self.grid == 0, self.goal)
            d = g[self.task_idx] = np.where(steps < 0, np.inf, steps).astype(float)
            while len(g) > 8:
                g.popitem(last=False)
        else:
            g.move_to_end(self.task_idx)
        return d

    @property
    def grid(self) -> np.ndarray:
        return self.ctx.grid

    @property
    def tasks(self) -> list:
        return self.ctx.tasks

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        flt = (lambda t: t["labels"].get("level") == self.difficulty) if self.difficulty else None
        self.map_id, self.ctx, self.task_idx = self._choose(options, flt)
        t = self.tasks[self.task_idx]
        self.start, self.goal = list(t["start"]["cell"]), list(t["goal"]["cell"])
        self.d_bfs = t["labels"]["d_bfs"]
        self.agent_pos = list(self.start)
        self.steps = self.moves = self.bumps = 0
        self._gd = self._goal_dist()
        self._geo_prev = float(self._gd[self.agent_pos[0], self.agent_pos[1]])
        info = {"start": self.start, "goal": self.goal, "d_bfs": self.d_bfs,
                **{k: t["labels"].get(k) for k in LABELS},
                "map_id": self.map_id, "task_idx": self.task_idx, "task_id": self.task_idx,
                "split": self.ctx.header.get("split", self.split)}
        return self._get_obs(), info

    def step(self, action):
        dr, dc = MOVES[int(action)]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        R, C = self.grid.shape
        blocked = not (0 <= nr < R and 0 <= nc < C and self.grid[nr, nc] == 0)
        if not blocked:
            self.agent_pos = [nr, nc]
            self.moves += 1
        else:
            self.bumps += 1
        self.steps += 1
        terminated = self.agent_pos == self.goal
        truncated = self.steps >= self.budget
        reward = 1.0 if terminated else 0.0
        geo = float(self._gd[self.agent_pos[0], self.agent_pos[1]])
        info = {"steps": self.steps, "d_bfs": self.d_bfs, "collided": blocked,
                "goal_geodesic": geo, "progress": self._geo_prev - geo,
                "reward_terms": {"success": float(terminated)}}
        self._geo_prev = geo
        if terminated or truncated:
            cm = self.ctx.cell_m
            info.update({"map_id": self.map_id, "task_id": self.task_idx,
                         "split": self.ctx.header.get("split", self.split),
                         "success": bool(terminated), "spl": spl(terminated, self.d_bfs, self.moves),
                         "path_length": self.moves * cm, "time": self.steps,
                         "n_collisions": self.bumps,
                         "termination": "success" if terminated else "time_limit",
                         "geodesic_m": self.d_bfs * cm})
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        key = (self.lidar_rays, self.lidar_range)
        table = self.ctx.lidar.get(key)
        if table is None:
            table = self.ctx.lidar[key] = np.full((*self.grid.shape, self.lidar_rays), np.nan,
                                                  np.float32)
        v = table[self.agent_pos[0], self.agent_pos[1]]
        if np.isnan(v[0]):
            v[:] = lidar_scan(self.grid, self.agent_pos, self.lidar_rays, self.lidar_range)
        return v.copy()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._frame()
        if self.render_mode == "human":
            import pygame
            frame = self._frame()
            if self._window is None:
                pygame.init()
                self._window = pygame.display.set_mode(frame.shape[1::-1])
                self._clock = pygame.time.Clock()
            pygame.surfarray.blit_array(self._window, frame.transpose(1, 0, 2))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

    def _frame(self, cell=8):
        img = np.where(self.grid[..., None] == 0, 255, 0).astype(np.uint8).repeat(3, 2)
        for (r, c), col in ((self.goal, (0, 200, 0)), (self.agent_pos, (0, 0, 255))):
            img[r, c] = col
        return img.repeat(cell, 0).repeat(cell, 1)

    def close(self):
        if self._window is not None:
            import pygame
            pygame.quit()
            self._window = None
