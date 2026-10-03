"""Continuous navigation on SVG maps with the real footprint and speed limits of a robot.

SI units; map frame follows REP-103 (theta counter-clockwise from +x).

Observation ``Dict``:
    lidar     n_beams ranges (m) over the robot's real LiDAR FOV, clipped to [range_min, range_max]
    goal      (distance, sin, cos) of the goal bearing in base_link
    velocity  current (v, [vy], w)
Action ``Box(-1, 1, (k,), float32)``: (v, w) for differential drives (k = 2), (v, vy, w) for
omnidirectional ones (k = 3); each component is multiplied by the robot's limit.

Motion is integrated exactly over dt, split into sub-steps of <= 2 cm and <= 2
degrees so fast robots cannot tunnel through thin walls. A colliding sub-step
stops the robot at its last valid pose. The goal is checked at every sub-step.

Step ``info`` (environment information, not robot senses):
    collided, n_collisions, time, pose, path_length (m)
    reward_terms      raw reward components (see reward below)
    goal_geodesic_m   current geodesic distance to the goal (see Planner.lookup)
    progress_m        decrease of goal_geodesic_m in this step
    goal_distance_m   Euclidean distance to the goal
    wall_distance_m   distance from the robot centre to the nearest edge of ``free``
The last four are skipped with ``nav_info=False``.

Reward: "sparse" = success. "dense" = ((w_progress * progress + w_success * success)
- w_collision * collision) - w_step * step over the raw terms of info["reward_terms"]
(weights: ``reward_weights``, default DENSE_WEIGHTS). Custom rewards: wrappers.CustomReward.

LiDAR noise has its own RNG (seeded from the reset seed), so switching noise on or off
does not change which episodes are drawn.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.geometry import Collider, place
from ..core.kinematics import integrate, wrap
from ..core.planning import TURN_SMOOTH, Planner, simplify, turning
from ..core.svgmap import SvgMap, rings_to_d
from ..robots import Robot, load as load_robot
from .base import EpisodeSource, as_point, custom_endpoints, spl

SUB_DIST = 0.02                 # max sub-step translation (m)
SUB_ANGLE = math.radians(2.0)   # max sub-step rotation (rad)
SUCCESS_BONUS, COLLISION_PEN, STEP_PEN = 10.0, 0.1, 0.01
DENSE_WEIGHTS = {"progress": 1.0, "success": SUCCESS_BONUS, "collision": COLLISION_PEN,
                 "step": STEP_PEN}


def task_turn(planner: Planner, task: dict, pred) -> float:
    """Total turning (rad) along the smoothed shortest path of a task, given the Dijkstra
    predecessors of its goal field (used by the default time limit)."""
    s, g = task["start"], task["goal"]
    pts = [(s["x"], s["y"])] + planner.path(pred, s["x"], s["y"]) + [(g["x"], g["y"])]
    return turning(simplify(pts, TURN_SMOOTH), s["theta"])


@dataclass
class _MapCtx:
    map_id: str
    svgmap: SvgMap
    collider: Collider
    tasks: list
    header: dict
    clearance: float
    r_circ: float
    field_cache: int = 4
    _planner: Planner | None = None
    goal_fields: OrderedDict = field(default_factory=OrderedDict)   # task -> distance field (LRU)
    turns: dict = field(default_factory=dict)                       # task -> turning along path (rad)
    render: dict = field(default_factory=dict)                      # render backgrounds
    fields_path: object = None                                      # disk-cached goal fields

    @property
    def planner(self) -> Planner:
        if self._planner is None:
            self._planner = Planner(self.svgmap, self.r_circ, self.clearance)
        return self._planner


class SvgEnv(EpisodeSource, gym.Env):
    """``gym.make("HM3D/Svg-v0", dataset=..., robot=..., split="train")``.

    dataset/robot/split: which episodes to draw from (see envs.base for the choice
    rules). ``svgmap`` + ``tasks`` (+ ``robot``) run a single in-memory map instead,
    mainly for tests.

    Speed knobs (results unchanged): ``map_cache`` maps kept loaded, ``field_cache``
    goal distance fields kept per map (~1 MB each), ``map_repeat`` consecutive random
    episodes drawn from the same map (fewer map loads; see envs.base), ``disk_cache``
    keep what a map load computes on disk (see hm3denv.cache; dataset envs only).
    """

    metadata = {"render_modes": ["rgb_array", "svg"], "render_fps": 10}

    def __init__(self, dataset=None, robot=None, split=None, map_id=None, task_idx=None,
                 dt=0.1, n_beams=72, lidar_noise=0.0, reward="dense", on_collision="stop",
                 time_limit=None, time_factor=3.0, success_radius=None, render_mode=None,
                 svgmap=None, tasks=None, clearance=None, map_cache=8, field_cache=4,
                 map_repeat=1, disk_cache=True, reward_weights=None, nav_info=True,
                 task_filter=None):
        super().__init__()
        if reward not in ("dense", "sparse"):
            raise ValueError("reward must be 'dense' or 'sparse'")
        if on_collision not in ("stop", "terminate"):
            raise ValueError("on_collision must be 'stop' or 'terminate'")
        unknown = set(reward_weights or {}) - set(DENSE_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown reward_weights {sorted(unknown)}; "
                             f"known: {sorted(DENSE_WEIGHTS)}")
        self.reward_weights = {**DENSE_WEIGHTS, **(reward_weights or {})}
        self.nav_info = nav_info
        self._noise_rng = None
        self.split = split
        self.field_cache = field_cache
        self._clearance_arg = clearance
        self._disk = None
        if tasks is not None:                                   # single in-memory map
            self.dataset = None
            self.robot = robot if isinstance(robot, Robot) else load_robot(robot or "turtlebot4")
            mid = tasks.get("map_id", "map")
            ctx = self._make_ctx(mid, svgmap, tasks, clearance)
            self._init_source([mid], lambda m: ctx, map_id, task_idx, map_cache, map_repeat,
                              task_filter)
        else:
            from ..dataset import load_dataset
            self.dataset = load_dataset(dataset)
            if self.dataset.env_type != "svg":
                raise ValueError(f"dataset {self.dataset.name} is a {self.dataset.env_type} dataset")
            if robot is None:
                if len(self.dataset.robots) != 1:
                    raise ValueError(f"choose a robot: {self.dataset.robots}")
                robot = self.dataset.robots[0]
            rid = robot.id if isinstance(robot, Robot) else robot
            self.robot = robot if isinstance(robot, Robot) else self.dataset.robot(rid)
            self._rid = rid
            if disk_cache:
                from ..cache import DatasetCache
                dc = DatasetCache(self.dataset)
                self._disk = dc if dc.enabled else None
            maps = [map_id] if map_id else self.dataset.maps(rid, split)
            self._init_source(maps, self._load_ctx, map_id, task_idx, map_cache, map_repeat,
                              task_filter)
        self.reward_mode, self.on_collision = reward, on_collision
        self.dt, self.lidar_noise = dt, lidar_noise
        self.time_limit_override, self.time_factor = time_limit, time_factor
        self._success_radius = success_radius
        self.render_mode = render_mode

        L = self.robot.lidar
        self._full = L["fov"] >= 2 * math.pi - 1e-3
        self._a_off = 0.0 if self._full else -L["fov"] / 2
        self._da = 2 * math.pi / n_beams if self._full else L["fov"] / (n_beams - 1)
        self.beam_angles = self._a_off + self._da * np.arange(n_beams)   # in base_link
        k = self.robot.action_dim
        self.action_space = spaces.Box(-1.0, 1.0, (k,), np.float32)
        vmax = np.array([max(self.robot.v_max, -self.robot.v_min)]
                        + ([self.robot.vy_max] if k == 3 else []) + [self.robot.w_max], np.float32)
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(L["range_min"], L["range_max"], (n_beams,), np.float32),
            "goal": spaces.Box(np.array([0, -1, -1], np.float32), np.array([np.inf, 1, 1], np.float32)),
            "velocity": spaces.Box(-vmax, vmax),
        })

    def _make_ctx(self, map_id, svgmap, header, clearance, edge_dist=None):
        return _MapCtx(map_id, svgmap, Collider(svgmap, edge_dist=edge_dist), header["tasks"],
                       header,
                       clearance if clearance is not None else header.get("clearance", 0.05),
                       self.robot.r_circ, max(1, self.field_cache))

    def _tasks_of(self, map_id) -> list:
        if self.dataset is None:
            return super()._tasks_of(map_id)
        return self.dataset.tasks(self._rid, map_id)

    def _load_ctx(self, map_id):
        """Map context from the dataset, through the disk cache when enabled."""
        ds, rid, dc = self.dataset, self._rid, self._disk
        paths = dc.paths(rid, map_id, self.robot.r_circ, self._clearance_arg) if dc else None
        hit = dc.load(paths) if paths else None
        if hit is not None:
            ctx = self._make_ctx(map_id, hit["svgmap"], hit["header"], self._clearance_arg,
                                 hit["edge_dist"])
            ctx._planner = hit["planner"]
            ctx.turns.update(hit["turns"])
            ctx.fields_path = hit["fields"]
            return ctx
        ctx = self._make_ctx(map_id, SvgMap.load(ds.svg_path(rid, map_id)),
                             ds.task_file(rid, map_id), self._clearance_arg)
        if paths:
            dc.save(paths, ctx.svgmap, ctx.collider, ctx.header, ctx.planner)
        return ctx

    # ------------------------------------------------------------ helpers

    @property
    def map(self) -> SvgMap:
        return self.ctx.svgmap

    @property
    def collider(self) -> Collider:
        return self.ctx.collider

    @property
    def planner(self) -> Planner:
        return self.ctx.planner

    @property
    def tasks(self) -> list:
        return self.ctx.tasks

    def _scale(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        r = self.robot
        v = a[0] * (r.v_max if a[0] >= 0 else -r.v_min)
        vy = a[1] * r.vy_max if r.omni else 0.0
        return v, vy, a[-1] * r.w_max

    def _goal_field(self):
        """Geodesic distance field to the current goal (LRU per map). The first time a
        task is seen, one Dijkstra run gives both the field and the path used for the
        turning estimate of the time limit. Dataset tasks are keyed by their index,
        custom episodes by their goal (field) and start + goal (turning)."""
        ctx, i, ti = self.ctx, self._fkey, self._tkey
        f = ctx.goal_fields.get(i)
        need_turn = not self.robot.omni and ti not in ctx.turns
        if f is not None and not need_turn:
            ctx.goal_fields.move_to_end(i)
            return f
        g = self.task["goal"]
        if f is None and ctx.fields_path is not None and not need_turn and isinstance(i, int):
            from ..cache import DatasetCache
            f = DatasetCache.load_field(ctx.fields_path, self.planner, i)
        if need_turn:
            f, pred = self.planner.field(g["x"], g["y"], predecessors=True)
            ctx.turns[ti] = task_turn(self.planner, self.task, pred)
        elif f is None:
            f = self.planner.field(g["x"], g["y"])
        ctx.goal_fields[i] = f
        ctx.goal_fields.move_to_end(i)
        while len(ctx.goal_fields) > ctx.field_cache:
            ctx.goal_fields.popitem(last=False)
        return f

    def _custom_task(self, start, goal) -> dict:
        """A task dict for your own start / goal on the current map, after checking that
        the robot fits at the start and can reach the goal."""
        sx, sy, *th = as_point(start, ("x", "y", "theta"), 2)
        gx, gy = as_point(goal, ("x", "y"), 2)
        th = th[0] if th else math.atan2(gy - sy, gx - sx)        # default: face the goal
        name = f"{self.robot.id} on {self.map_id}"
        if not self.collider.pose_valid(self.robot.footprint, sx, sy, th):
            raise ValueError(f"start ({sx:g}, {sy:g}, {th:.3f}) collides with the map ({name})")
        pl = self.planner
        if not pl.in_cdisk(gx, gy):
            raise ValueError(f"goal ({gx:g}, {gy:g}) is too close to a wall or outside the free "
                             f"area for {name} (needs {pl.r_circ + pl.clearance:.2f} m of clearance)")
        task = {"id": -1, "start": {"x": sx, "y": sy, "theta": th}, "goal": {"x": gx, "y": gy},
                "euclidean_m": math.hypot(gx - sx, gy - sy), "labels": {}}
        self.task, self._fkey, self._tkey = task, ("goal", gx, gy), ("task", sx, sy, th, gx, gy)
        f = self._goal_field()
        r, c = pl.nearest_cell(sx, sy)
        if not np.isfinite(f[r, c]):
            raise ValueError(f"goal ({gx:g}, {gy:g}) cannot be reached from the start ({sx:g}, {sy:g}) "
                             f"by {name}: another room, or a passage too narrow for the robot")
        task["geodesic_m"] = pl.lookup(f, sx, sy)
        return task

    def _default_time_limit(self):
        """time_factor x estimated minimum time: drive the geodesic at v_max and turn
        through every corner of the (5 cm smoothed) path at w_max. Omni robots need no turning."""
        turn = 0.0
        if not self.robot.omni:
            if self._tkey not in self.ctx.turns:
                self._goal_field()
            turn = self.ctx.turns[self._tkey]
        return self.time_factor * (self.geodesic / self.robot.v_max + turn / self.robot.w_max)

    def _geo(self, x, y):
        return self.planner.lookup(self._goal_field(), x, y)

    def _obs(self):
        x, y, th = self.pose
        L = self.robot.lidar
        d = self.collider.raycast_uniform(x, y, th + self._a_off, self._da,
                                          len(self.beam_angles), L["range_max"], self._full)
        if self.lidar_noise > 0:
            d = d + self._noise_rng.normal(0, self.lidar_noise, d.shape)
        g = self.task["goal"]
        dx, dy = g["x"] - x, g["y"] - y
        bearing = wrap(math.atan2(dy, dx) - th)
        return {"lidar": np.clip(d, L["range_min"], L["range_max"]).astype(np.float32),
                "goal": np.array([math.hypot(dx, dy), math.sin(bearing), math.cos(bearing)],
                                 np.float32),
                "velocity": np.array(self.vel, np.float32)}

    # ------------------------------------------------------------ gym api

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self._noise_rng is None:
            # independent stream derived from the episode-selection seed
            self._noise_rng = np.random.default_rng(self.np_random.bit_generator.seed_seq.spawn(1)[0])
        self.map_id, self.ctx, self._cur = self._choose(options)
        custom = custom_endpoints(options)
        if custom:
            self._cur = -1
            self.task = self._custom_task(*custom)
        else:
            self.task = self.tasks[self._cur]
            self._fkey = self._tkey = self._cur
        self.success_radius = self._success_radius or self.ctx.header.get("success_radius", 0.2)
        s = self.task["start"]
        self.pose = (s["x"], s["y"], s["theta"])
        if not self.collider.pose_valid(self.robot.footprint, *self.pose):
            raise RuntimeError(f"{self.map_id} task {self._cur}: invalid start pose")
        self.vel = (0.0,) * self.robot.action_dim
        self.t, self.path_len, self.n_coll, self.steps = 0.0, 0.0, 0, 0
        self.geodesic = float(self.task["geodesic_m"])
        self.time_limit = self.time_limit_override or self._default_time_limit()
        self._track_geo = self.reward_mode == "dense" or self.nav_info
        self._geo_prev = self._geo(s["x"], s["y"]) if self._track_geo else 0.0
        self.traj = [(0.0, *self.pose)]
        return self._obs(), {"map_id": self.map_id, "task_id": self._cur,
                             "split": self.ctx.header.get("split", self.split),
                             "geodesic_m": self.geodesic, "time_limit": self.time_limit}

    def step(self, action):
        v, vy, w = self._scale(action)
        x, y, th = self.pose
        n = max(1, math.ceil(max(math.hypot(v, vy) * self.dt / SUB_DIST,
                                 abs(w) * self.dt / SUB_ANGLE)))
        h = self.dt / n
        g = self.task["goal"]
        collided = success = False
        for _ in range(n):
            nx, ny, nth = integrate(x, y, th, v, vy, w, h)
            if not self.collider.pose_valid(self.robot.footprint, nx, ny, nth):
                collided = True
                break
            self.path_len += math.hypot(nx - x, ny - y)
            x, y, th = nx, ny, nth
            # goal checked at EVERY sub-step: a fast robot (Jackal 2 m/s = 0.2 m per step,
            # equal to the success radius) could otherwise skip over the goal region
            if math.hypot(g["x"] - x, g["y"] - y) <= self.success_radius:
                success = True
                break
        self.pose = (x, y, th)
        self.vel = (0.0,) * self.robot.action_dim if collided else (
            (v, vy, w) if self.robot.omni else (v, w))
        self.t += self.dt
        self.steps += 1
        self.n_coll += collided
        self.traj.append((self.t, x, y, th))

        info = {"collided": collided, "n_collisions": self.n_coll, "time": self.t,
                "pose": self.pose, "path_length": self.path_len}
        if self._track_geo:
            geo = self._geo(x, y)
            progress = self._geo_prev - geo
            if not math.isfinite(progress):     # no finite geodesic (should not happen)
                progress = 0.0
            self._geo_prev = geo
        if self.reward_mode == "dense":
            terms = {"progress": progress, "success": float(success),
                     "collision": float(collided), "step": 1.0}
            w = self.reward_weights
            # same operation order as the original hard-coded reward (bit-identical)
            reward = (w["progress"] * terms["progress"] + w["success"] * terms["success"]) \
                - w["collision"] * terms["collision"] - w["step"] * terms["step"]
        else:
            terms = {"success": float(success)}
            reward = float(success)
        info["reward_terms"] = terms
        if self.nav_info:
            info.update(goal_geodesic_m=geo, progress_m=progress,
                        goal_distance_m=math.hypot(g["x"] - x, g["y"] - y),
                        wall_distance_m=self.collider.wall_distance(x, y))
        terminated = success or (collided and self.on_collision == "terminate")
        truncated = (not terminated) and self.t >= self.time_limit - 1e-9
        if terminated or truncated:
            info.update(self.episode_metrics(success, "success" if success else
                                             "collision" if terminated else "time_limit"))
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def episode_metrics(self, success, reason) -> dict:
        return {"map_id": self.map_id, "task_id": self._cur,
                "split": self.ctx.header.get("split", self.split), "success": bool(success),
                "spl": spl(success, self.geodesic, self.path_len), "path_length": self.path_len,
                "time": self.t, "n_collisions": self.n_coll, "termination": reason,
                "geodesic_m": self.geodesic}

    # ------------------------------------------------------------ render

    def render(self):
        if self.render_mode == "svg":
            return self.render_svg()
        if self.render_mode == "rgb_array":
            return self.render_rgb()

    def render_rgb(self, res=0.04):
        key = round(res, 6)
        if key not in self.ctx.render:
            mask, x0, y1, _ = self.map.raster(res)
            self.ctx.render[key] = (np.where(mask[..., None], 255, 80).astype(np.uint8).repeat(3, 2),
                                    x0, y1, res)
        bg, x0, y1, r = self.ctx.render[key]
        img = bg.copy()

        def px(P):
            return np.stack([(P[:, 0] - x0) / r, (y1 - P[:, 1]) / r], 1).astype(np.int32)

        xs = np.array([[p[1], p[2]] for p in self.traj])
        cv2.polylines(img, [px(xs)], False, (80, 140, 230), 1)
        g = self.task["goal"]
        cv2.circle(img, tuple(px(np.array([[g["x"], g["y"]]]))[0]),
                   max(2, int(self.success_radius / r)), (220, 0, 0), -1)
        cv2.fillPoly(img, [px(place(self.robot.footprint, *self.pose))], (0, 160, 0))
        return img

    def render_svg(self, every=10) -> str:
        """Episode as SVG: map + trajectory + footprint every `every` steps + start/goal."""
        base = self.map.to_svg().rsplit("\n  </g>\n</svg>", 1)[0]
        tr = " ".join(f"{p[1]:.3f},{p[2]:.3f}" for p in self.traj)
        fps = "".join(f'\n    <path class="fp" d="{rings_to_d([place(self.robot.footprint, *p[1:])], 3)}"/>'
                      for p in self.traj[::every] + [self.traj[-1]])
        s, g = self.task["start"], self.task["goal"]
        return (base + "\n    <style>.fp{fill:none;stroke:#2a9d2a;stroke-width:0.01}"
                ".traj{fill:none;stroke:#3c78d8;stroke-width:0.02}</style>"
                f'\n    <polyline class="traj" points="{tr}"/>{fps}'
                f'\n    <circle cx="{s["x"]:.3f}" cy="{s["y"]:.3f}" r="0.06" fill="#2a9d2a"/>'
                f'\n    <circle cx="{g["x"]:.3f}" cy="{g["y"]:.3f}" r="{self.success_radius:.3f}" '
                f'fill="#d62728" fill-opacity="0.5"/>\n  </g>\n</svg>\n')
