"""Oracle agents that follow the shortest path (like Habitat's ShortestPathFollower).

They serve as baselines and as an end-to-end check that tasks are drivable with the
robot's real kinematics and footprint, not only "connected in theory".
"""

from __future__ import annotations

import math

import numpy as np

from ..core.gridcore import MOVES, bfs
from ..core.kinematics import wrap


class OracleFollower:
    """SVG env: regulated pure pursuit along the geodesic path. The aim point must be in
    straight line of sight inside C_disk, so shortcuts never cut wall corners."""

    def __init__(self, env, lookahead=0.3, k_turn=2.0):
        self.u = env.unwrapped
        self.lookahead, self.k_turn = lookahead, k_turn
        g = self.u.task["goal"]
        s = self.u.pose
        _, pred = self.u.planner.field(g["x"], g["y"], predecessors=True)
        self.path = np.array(self.u.planner.path(pred, s[0], s[1]) + [(g["x"], g["y"])])
        self.i = 0

    def _visible(self, x, y, tx, ty, step=0.02):
        """Segment (x, y) -> (tx, ty) inside C_disk (ignoring the first few cm, where the
        robot may have drifted slightly out of C_disk)."""
        n = max(2, int(math.hypot(tx - x, ty - y) / step))
        pl = self.u.planner
        return all(pl.in_cdisk(x + (tx - x) * t, y + (ty - y) * t)
                   for t in np.linspace(0, 1, n)[1:] if math.hypot(tx - x, ty - y) * t > 0.05)

    def _target(self, x, y):
        d = np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)
        self.i = max(self.i, int(np.argmin(d[self.i:self.i + 50])) + self.i)
        j = self.i
        while (j < len(self.path) - 1 and d[j + 1] < self.lookahead
               and self._visible(x, y, *self.path[j + 1])):
            j += 1
        if j == self.i and j < len(self.path) - 1 and d[j] < 0.05:
            j += 1                                           # current point reached
        return self.path[j]

    def act(self) -> np.ndarray:
        """Normalised action, shape (k,), every component in [-1, 1]."""
        x, y, th = self.u.pose
        r = self.u.robot
        tx, ty = self._target(x, y)
        dx, dy = tx - x, ty - y
        err = wrap(math.atan2(dy, dx) - th)
        if r.omni:
            c, s = math.cos(-th), math.sin(-th)
            bx, by = c * dx - s * dy, s * dx + c * dy       # direction to the target in base_link
            n = max(math.hypot(bx, by), 1e-9)
            return np.array([bx / n, by / n, np.clip(self.k_turn * err / r.w_max, -1, 1) * 0.3])
        if abs(err) > math.radians(60):                     # large error: turn in place
            return np.array([0.0, math.copysign(1.0, err)])
        L = max(math.hypot(dx, dy), 1e-3)
        # regulated pure pursuit: the required w = 2 v sin(err) / L must stay <= w_max,
        # so fast-but-slow-turning robots (Jackal 2 m/s, 1 rad/s) slow down in corners
        v = r.v_max * max(0.0, math.cos(err)) ** 2
        v = min(v, r.w_max * L / max(2 * abs(math.sin(err)), 1e-6))
        w = 2 * v * math.sin(err) / L + 0.5 * self.k_turn * err
        return np.array([v / r.v_max, np.clip(w / r.w_max, -1, 1)])


class GridOracle:
    """Grid env: always takes a move that decreases the BFS distance to the goal."""

    def __init__(self, env):
        self.u = env.unwrapped
        self.dist = bfs(self.u.grid == 0, self.u.goal)

    def act(self) -> int:
        r, c = self.u.agent_pos
        R, C = self.dist.shape
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and 0 <= self.dist[nr, nc] < self.dist[r, c]:
                return a
        return 0
