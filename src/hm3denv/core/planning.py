"""Geodesic distances for a robot on the free region.

C_disk = points at least (footprint circumradius + clearance) away from the free
boundary: with its centre there the robot fits at every heading. Planning on
C_disk is CONSERVATIVE: every path found is drivable at any heading, at the cost
of possibly missing narrow passages a rectangular robot could only pass lengthwise.

16-neighbour graph (4 straight, 4 diagonal, 8 knight moves), no corner cutting:
the anisotropy error against continuous Euclidean distance is at most ~2.8%
(~8.2% for 8-neighbour).
"""

from __future__ import annotations

import json
from collections import OrderedDict

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]
MAX_ANISO_ERR = 0.028
TURN_SMOOTH = 0.05      # simplify() tolerance of the time-limit path (m), see envs.svg


def edt_px(mask: np.ndarray) -> np.ndarray:
    """``ndimage.distance_transform_edt(mask)`` (pixels), ~5x faster via OpenCV's exact L2
    transform. Squared distances are integers: recovering them from OpenCV's float32
    output and taking the float64 sqrt gives scipy's values bit for bit below ~1000 px;
    farther only the last bits may differ (irrelevant for robot-sized thresholds)."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.sqrt(np.rint(d.astype(np.float64) ** 2))


def _passable(grid, dr, dc):
    """Mask of cells (r, c) that can move to (r+dr, c+dc): both ends and the cells in between free."""
    H, W = grid.shape

    def shifted(a, b):
        out = np.zeros_like(grid)
        rs, re = max(0, -a), min(H, H - a)
        cs, ce = max(0, -b), min(W, W - b)
        out[rs:re, cs:ce] = grid[rs + a:re + a, cs + b:ce + b]
        return out

    ok = grid & shifted(dr, dc)
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    if abs(dr) == 1 and abs(dc) == 1:
        ok &= shifted(dr, 0) & shifted(0, dc)
    elif abs(dr) == 1 and abs(dc) == 2:
        ok &= shifted(0, sc) & shifted(dr, sc)
    elif abs(dr) == 2 and abs(dc) == 1:
        ok &= shifted(sr, 0) & shifted(sr, dc)
    return ok


class Planner:
    def __init__(self, svgmap, r_circ, clearance=0.05, res=0.02, plan_res=0.04):
        mask, x0, y1, res = svgmap.raster(res)
        # distance from a pixel centre to the nearest non-free pixel centre; the real
        # boundary may be up to one pixel closer -> subtract res to stay conservative
        edt = edt_px(mask) * res
        fine = edt >= r_circ + clearance + res
        f = max(1, int(round(plan_res / res)))
        H, W = fine.shape[0] // f, fine.shape[1] // f
        self.grid = fine[:H * f, :W * f].reshape(H, f, W, f).all(axis=(1, 3))
        self.x0, self.y1, self.res = x0, y1, res * f
        self.r_circ, self.clearance = r_circ, clearance
        self._index_cells()
        cells = self.cells
        rows, cols, w = [], [], []
        for dr, dc in OFFSETS:
            ok = _passable(self.grid, dr, dc)
            r, c = np.nonzero(ok)
            rows.append(self.node[r, c])
            cols.append(self.node[r + dr, c + dc])
            w.append(np.full(len(r), self.res * np.hypot(dr, dc)))
        n = len(cells)
        self.graph = coo_matrix((np.concatenate(w), (np.concatenate(rows),
                                                     np.concatenate(cols))),
                                shape=(n, n)).tocsr()
        # nearest C_disk cell for every cell (distance lookup when the centre leaves C_disk)
        _, self._near = ndimage.distance_transform_edt(~self.grid, return_indices=True)
        self._comp = None
        self._finite_near: OrderedDict = OrderedDict()

    def _index_cells(self):
        self.cells = np.argwhere(self.grid)
        self.node = np.full(self.grid.shape, -1, np.int64)
        self.node[self.cells[:, 0], self.cells[:, 1]] = np.arange(len(self.cells))

    @property
    def comp(self) -> np.ndarray:
        """8-connected component label of every C_disk cell (0 outside C_disk)."""
        if self._comp is None:
            self._comp = ndimage.label(self.grid, np.ones((3, 3)))
        return self._comp[0]

    @property
    def n_comp(self) -> int:
        self.comp
        return self._comp[1]

    # ------------------------------------------------------------ disk cache

    def to_arrays(self) -> dict:
        """Everything the constructor computes, as arrays (see from_arrays)."""
        near = self._near
        if near.size and near.min() >= 0 and near.max() < 2 ** 16:
            near = near.astype(np.uint16)
        # edge weights take only a few values (one per neighbour offset): store codes
        w_vals, w_code = np.unique(self.graph.data, return_inverse=True)
        # scalars go into one JSON member (floats round-trip exactly through repr)
        info = {"grid_shape": list(self.grid.shape), "near_dtype": str(self._near.dtype),
                "frame": [float(v) for v in (self.x0, self.y1, self.res, self.r_circ,
                                             self.clearance)],
                "w_vals": [float(v) for v in w_vals]}
        return {"grid": np.packbits(self.grid), "g_wcode": w_code.astype(np.uint8),
                "g_indices": self.graph.indices, "g_indptr": self.graph.indptr, "near": near,
                "planner": np.array(json.dumps(info))}

    @classmethod
    def from_arrays(cls, a) -> "Planner":
        """The Planner whose to_arrays() gave ``a``; identical to the original."""
        self = cls.__new__(cls)
        info = json.loads(str(a["planner"]))
        H, W = info["grid_shape"]
        self.grid = np.unpackbits(a["grid"], count=H * W).reshape(H, W).astype(bool)
        self.x0, self.y1, self.res, self.r_circ, self.clearance = info["frame"]
        self._index_cells()
        n = len(self.cells)
        data = np.array(info["w_vals"], float)[a["g_wcode"]]
        self.graph = csr_matrix((data, a["g_indices"], a["g_indptr"]), shape=(n, n))
        self._near = a["near"].astype(info["near_dtype"])
        self._comp = None
        self._finite_near = OrderedDict()
        return self

    def cell(self, x, y):
        return int((self.y1 - y) / self.res), int((x - self.x0) / self.res)

    def center(self, r, c):
        return self.x0 + (c + 0.5) * self.res, self.y1 - (r + 0.5) * self.res

    def in_cdisk(self, x, y) -> bool:
        r, c = self.cell(x, y)
        H, W = self.grid.shape
        return 0 <= r < H and 0 <= c < W and bool(self.grid[r, c])

    def nearest_cell(self, x, y):
        r, c = self.cell(x, y)
        H, W = self.grid.shape
        r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
        return int(self._near[0, r, c]), int(self._near[1, r, c])

    def field(self, x, y, predecessors=False):
        """Geodesic distance field (m) from (x, y) to every C_disk cell, inf if unreachable.
        predecessors=True also returns the Dijkstra predecessor array (for path())."""
        r, c = self.nearest_cell(x, y)
        res = dijkstra(self.graph, directed=False, indices=int(self.node[r, c]),
                       return_predecessors=predecessors)
        d, pred = res if predecessors else (res, None)
        out = np.full(self.grid.shape, np.inf)
        out[self.cells[:, 0], self.cells[:, 1]] = d
        return (out, pred) if predecessors else out

    def lookup(self, field, x, y) -> float:
        """Distance at (x, y): nearest C_disk cell plus the straight hop to its centre.

        C_disk (a disk of radius r_circ + clearance) is more conservative than the real
        footprint, so a robot can stand where the nearest C_disk cell is an isolated
        patch that cannot reach the field's source (field = inf there). The nearest cell
        that can reach it is used instead, so the distance stays finite; everywhere else
        the result is unchanged. inf only if no cell reaches the source at all."""
        r, c = self.nearest_cell(x, y)
        if not np.isfinite(field[r, c]):
            rc = self._nearest_finite(field, x, y)
            if rc is not None:
                r, c = rc
        cx, cy = self.center(r, c)
        return float(field[r, c] + np.hypot(x - cx, y - cy))

    def _nearest_finite(self, field, x, y):
        """Cell with a finite field value nearest to (x, y); None if there is none.
        The index map is built on first need and kept for the last 8 fields."""
        hit = self._finite_near.get(id(field))
        if hit is None or hit[0] is not field:
            fin = np.isfinite(field)
            if not fin.any():
                return None
            _, idx = ndimage.distance_transform_edt(~fin, return_indices=True)
            hit = self._finite_near[id(field)] = (field, idx)
            while len(self._finite_near) > 8:
                self._finite_near.popitem(last=False)
        r, c = self.cell(x, y)
        H, W = self.grid.shape
        r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
        return int(hit[1][0, r, c]), int(hit[1][1, r, c])

    def path(self, pred, x, y):
        """Shortest path from (x, y) to the field's source, following Dijkstra
        predecessors (only valid graph edges), as a list of (x, y)."""
        r, c = self.nearest_cell(x, y)
        i = int(self.node[r, c])
        pts = []
        while i >= 0:
            pts.append(self.center(*self.cells[i]))
            i = int(pred[i])
        return pts


def simplify(pts, tol):
    """Douglas-Peucker: drop points deviating less than tol (m); removes grid staircase."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    d = b - a
    n = np.hypot(*d)
    dist = (np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / n if n > 1e-12
            else np.hypot(*(P - a).T))
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(P[:i + 1], tol)[:-1], simplify(P[i:], tol)])


def turning(pts, theta0=None) -> float:
    """Total |turn| (rad) along a polyline, plus the rotation from theta0 onto the first segment."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return 0.0
    h = np.arctan2(*np.diff(P, axis=0)[:, ::-1].T)
    dh = np.diff(h)
    total = float(np.abs((dh + np.pi) % (2 * np.pi) - np.pi).sum())
    if theta0 is not None:
        total += abs((h[0] - theta0 + np.pi) % (2 * np.pi) - np.pi)
    return total
