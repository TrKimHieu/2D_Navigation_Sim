"""Grid primitives shared by the grid environment and the grid builder: BFS, exact
goal-reaching probabilities (SA) of random probes, and cell -> metre conversion.

Action convention: 0=UP 1=DOWN 2=LEFT 3=RIGHT; moving into a
blocked cell or off the grid leaves the agent in place.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

STARVED_SA = 1e-4  # success probability below which a task is labelled 'starved'


def level_of(sa: float) -> str:
    if sa < STARVED_SA:
        return "starved"
    if sa < 1e-3:
        return "hard"
    if sa < 0.1:
        return "medium"
    return "easy"


def bfs(free: np.ndarray, start) -> np.ndarray:
    """Shortest step count from `start` to every cell; -1 = unreachable."""
    R, C = free.shape
    dist = np.full((R, C), -1, np.int32)
    r0, c0 = int(start[0]), int(start[1])
    if not free[r0, c0]:
        return dist
    dist[r0, c0] = 0
    q = deque([(r0, c0)])
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and free[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


class Chain:
    """The free-cell graph of a grid, shared by all tasks of a map.

    SA = P(reach the goal within `budget` steps) of a random probe, computed exactly
    as an absorbing Markov chain. The recursion runs backwards in time, so one call
    gives the SA of EVERY start cell for a given goal.
    """

    def __init__(self, free: np.ndarray):
        self.free = np.asarray(free, bool)
        R, C = self.free.shape
        cells = np.argwhere(self.free)
        self.n = len(cells)
        self.index = np.full((R, C), -1, np.int64)
        self.index[cells[:, 0], cells[:, 1]] = np.arange(self.n)
        nbr = np.empty((self.n, 4), np.int64)          # nbr[i, a]: cell after action a
        for a, (dr, dc) in enumerate(MOVES):
            r, c = cells[:, 0] + dr, cells[:, 1] + dc
            ok = (r >= 0) & (r < R) & (c >= 0) & (c < C)
            j = np.full(self.n, -1, np.int64)
            j[ok] = self.index[r[ok], c[ok]]
            nbr[:, a] = np.where(j >= 0, j, np.arange(self.n))
        self.nbr = nbr

    def idx(self, rc) -> int:
        i = int(self.index[int(rc[0]), int(rc[1])])
        if i < 0:
            raise ValueError(f"cell {tuple(rc)} is not free")
        return i

    def sa_uniform(self, goal, budget: int) -> np.ndarray:
        """Uniform random walk (each of the 4 actions w.p. 1/4); SA per start cell."""
        g = self.idx(goal)
        rows = np.repeat(np.arange(self.n), 4)
        cols = self.nbr.reshape(-1)
        keep = cols != g
        P0 = csr_matrix((np.full(keep.sum(), 0.25), (rows[keep], cols[keep])),
                        shape=(self.n, self.n))
        h = 0.25 * (self.nbr == g).sum(axis=1)
        return _accumulate(P0, h, budget)

    def sa_persist(self, goal, budget: int, repeat_p: float = 0.6) -> np.ndarray:
        """Persistent probe: repeats its previous action w.p. `repeat_p`, otherwise picks
        uniformly among the 4. State = (previous heading, cell); the first step has no
        previous heading, so the result is averaged over the 4 headings."""
        g = self.idx(goal)
        n = self.n
        w_other = (1.0 - repeat_p) / 4.0
        w_same = repeat_p + w_other
        rows, cols, vals = [], [], []
        h = np.zeros(4 * n)
        for hd in range(4):
            for a in range(4):
                w = w_same if a == hd else w_other
                j = self.nbr[:, a]
                src = hd * n + np.arange(n)
                hit = j == g
                h[src[hit]] += w
                rows.append(src[~hit])
                cols.append(a * n + j[~hit])
                vals.append(np.full((~hit).sum(), w))
        U0 = csr_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(4 * n, 4 * n))
        acc = _accumulate(U0, h, budget)
        return acc.reshape(4, n).mean(axis=0)


def _accumulate(M, h, budget):
    """sum_{k=0}^{budget-1} M^k h = P(reach the goal within `budget` steps)."""
    v = h.copy()
    acc = h.copy()
    for _ in range(budget - 1):
        v = M @ v
        acc += v
    return acc


def cell_to_world(meta: dict, r: int, c: int) -> list[float]:
    """Centre of cell (r, c) of a cropped grid -> [x, y, z] in metres (GLB frame).

    Pixels of the 2 cm storey map: column i covers [lo_u + i*res, lo_u + (i+1)*res);
    image rows are flipped (row 0 at the top), so row i covers v from
    lo_v + (H-1-i)*res. Cell (r, c) starts at pixel (oy - k + (r0+r)*k, ox - k + (c0+c)*k).
    """
    k, res = meta["k"], meta["res"]
    py = meta["oy"] - k + (meta["r0"] + r) * k + k / 2.0
    px = meta["ox"] - k + (meta["c0"] + c) * k + k / 2.0
    lo = meta["lo"]
    p = [0.0, 0.0, 0.0]
    p[meta["ua"]] = lo[meta["ua"]] + px * res
    p[meta["va"]] = lo[meta["va"]] + (meta["H"] - py) * res
    p[meta["up"]] = meta["floor_z"]
    return p


def lidar_scan(grid: np.ndarray, position, n_rays: int = 22, max_range: float = 10.0) -> np.ndarray:
    """2D grid LiDAR, distances in CELLS.

    Rays sampled every 0.1 cell with int() truncation towards zero; a ray stops at the
    grid edge or at a blocked cell (value 1). A cell is one robot body wide, so
    max_range is in robot bodies, not metres.
    """
    R, C = grid.shape
    r, c = position
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    t = np.arange(0, max_range, 0.1)
    nr = (r + t[None, :] * np.cos(angles)[:, None]).astype(np.int64)
    nc = (c + t[None, :] * np.sin(angles)[:, None]).astype(np.int64)
    out = (nr < 0) | (nr >= R) | (nc < 0) | (nc >= C)
    hit = out.copy()
    hit[~out] = grid[nr[~out], nc[~out]] == 1
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), t[first], max_range).astype(np.float32)
