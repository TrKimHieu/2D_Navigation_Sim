"""Optional numba kernels for the hot geometry of the SVG environment.

Each kernel is a loop version of a numpy function in core.geometry with the SAME
arithmetic (operation order, tolerances, no fastmath / FMA contraction), so both
backends give bit-identical results; tests/test_perf_equivalence.py checks this.

    ENABLED   numba is installed and HM3D_ACCEL is not "0"

Install with ``pip install hm3denv[fast]``. Kernels are compiled on first use and
cached on disk (``cache=True``), so worker processes after the first start fast.

Segment grid: a uniform
grid lists every segment in each cell its bbox (grown by EPS_REG) touches. The *_grid
kernels visit only nearby cells and give exactly the same results as the full scans:

    raycast_grid       walks the cells along each ray (Amanatides-Woo) and stops once the
                       ray's range is <= the entry distance of the next cell; a (segment,
                       ray) pair is only tested if the full scan would test it (same
                       angular interval), with the same intersection formula
    nearest_grid       rings of cells around the point until best <= ring distance
    seg_hit_poly_grid  only the cells under the footprint's bbox
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    from numba import njit
except ImportError:                                            # pragma: no cover
    njit = None

ENABLED = njit is not None and os.environ.get("HM3D_ACCEL", "1") != "0"
BACKEND = "numba" if ENABLED else "numpy"

EPS = 1e-12
EPS_REG = 1e-6         # segment bboxes are grown by this when registered in grid cells (m)
TOL = 1e-15            # |orientation| at float noise counts as 0 (see segments_hit_polygon)


def _sign(d):
    if abs(d) <= TOL:
        return 0.0
    return 1.0 if d > 0 else -1.0


def _seg_hits(S, i, P, xmin, ymin, xmax, ymax):
    """Whether segment i crosses or touches polygon P (False when the bboxes are apart);
    the per-segment test of segments_hit_polygon."""
    M = P.shape[0]
    ax, ay, bx, by = S[i, 0], S[i, 1], S[i, 2], S[i, 3]
    if max(ax, bx) < xmin or min(ax, bx) > xmax or max(ay, by) < ymin or min(ay, by) > ymax:
        return False
    # endpoint a inside P (even-odd rule, formula of points_in_polygon)
    cnt = 0
    for j in range(M):
        x1, y1 = P[j, 0], P[j, 1]
        k = j + 1 if j + 1 < M else 0
        x2, y2 = P[k, 0], P[k, 1]
        if (y1 > ay) != (y2 > ay):
            xi = x1 + (ay - y1) * (x2 - x1) / (y2 - y1)
            if ax < xi:
                cnt += 1
    if cnt % 2 == 1:
        return True
    rx, ry = bx - ax, by - ay
    for j in range(M):
        q1x, q1y = P[j, 0], P[j, 1]
        k = j + 1 if j + 1 < M else 0
        ex, ey = P[k, 0] - q1x, P[k, 1] - q1y
        q2x, q2y = q1x + ex, q1y + ey
        sx, sy = q2x - q1x, q2y - q1y
        d1 = sx * (ay - q1y) - sy * (ax - q1x)
        d2 = sx * (by - q1y) - sy * (bx - q1x)
        d3 = rx * (q1y - ay) - ry * (q1x - ax)
        d4 = rx * (q2y - ay) - ry * (q2x - ax)
        if _sign(d1) * _sign(d2) <= 0 and _sign(d3) * _sign(d4) <= 0:
            return True
    return False


def _seg_hit_poly(S, P, xmin, ymin, xmax, ymax):
    """segments_hit_polygon(S[bbox overlap], P): any segment crossing or touching P."""
    for i in range(S.shape[0]):
        if _seg_hits(S, i, P, xmin, ymin, xmax, ymax):
            return True
    return False


def _ray_segment(S, i, ox, oy, a0, da, n, full, cosk, sink, out):
    """Intersect segment i with the rays inside the angular interval it subtends."""
    two_pi = 2 * math.pi
    x1, y1 = S[i, 0] - ox, S[i, 1] - oy
    x2, y2 = S[i, 2] - ox, S[i, 3] - oy
    ex, ey = x2 - x1, y2 - y1
    r1 = (math.atan2(y1, x1) - a0) % two_pi
    d = (math.atan2(y2, x2) - a0 - r1 + math.pi) % two_pi - math.pi
    lo = r1 + min(d, 0.0)
    hi = lo + abs(d)
    for band in range(1 if full else 3):
        shift = 0.0 if band == 0 else (-two_pi if band == 1 else two_pi)
        k_lo = math.ceil((lo + shift) / da - 1e-9)
        k_hi = math.floor((hi + shift) / da + 1e-9)
        for kk in range(k_lo, k_hi + 1):
            if full:
                k = kk % n
            elif kk < 0 or kk >= n:
                continue
            else:
                k = kk
            dx, dy = cosk[k], sink[k]
            den = dx * ey - dy * ex
            if abs(den) <= EPS:
                continue
            t = (x1 * ey - y1 * ex) / den
            uu = (x1 * dy - y1 * dx) / den
            if t >= 0 and uu >= -1e-9 and uu <= 1 + 1e-9 and t < out[k]:
                out[k] = t


def _raycast_uniform(S, ox, oy, a0, da, n, r_max, full, cosk, sink, out):
    """Collider.raycast_uniform: rays a0 + k*da (cos/sin given), writes ranges into out."""
    for i in range(S.shape[0]):
        x1, y1 = S[i, 0] - ox, S[i, 1] - oy
        ex, ey = S[i, 2] - ox - x1, S[i, 3] - oy - y1
        L2 = max(ex * ex + ey * ey, EPS)
        u = min(max(-(x1 * ex + y1 * ey) / L2, 0.0), 1.0)
        if math.hypot(x1 + u * ex, y1 + u * ey) <= r_max:     # else every hit is beyond r_max
            _ray_segment(S, i, ox, oy, a0, da, n, full, cosk, sink, out)


def _seg_dist(S, i, x, y):
    """Distance from (x, y) to segment i."""
    ax, ay = S[i, 0] - x, S[i, 1] - y
    ex, ey = S[i, 2] - S[i, 0], S[i, 3] - S[i, 1]
    L2 = max(ex * ex + ey * ey, EPS)
    u = min(max(-(ax * ex + ay * ey) / L2, 0.0), 1.0)
    return math.hypot(ax + u * ex, ay + u * ey)


def _nearest_segment(S, x, y):
    """Distance from (x, y) to the nearest segment (Collider.wall_distance)."""
    best = math.inf
    for i in range(S.shape[0]):
        d = _seg_dist(S, i, x, y)
        if d < best:
            best = d
    return best


# ------------------------------------------------------------------ segment grid

def _build_grid(S, gx0, gy0, cs, nx, ny):
    """CSR lists (start, items) of the segments whose grown bbox touches each cell."""
    N = S.shape[0]
    counts = np.zeros(nx * ny + 1, np.int64)
    rng = np.empty((N, 4), np.int64)
    for i in range(N):
        c0 = max(int(math.floor((min(S[i, 0], S[i, 2]) - EPS_REG - gx0) / cs)), 0)
        c1 = min(int(math.floor((max(S[i, 0], S[i, 2]) + EPS_REG - gx0) / cs)), nx - 1)
        r0 = max(int(math.floor((min(S[i, 1], S[i, 3]) - EPS_REG - gy0) / cs)), 0)
        r1 = min(int(math.floor((max(S[i, 1], S[i, 3]) + EPS_REG - gy0) / cs)), ny - 1)
        rng[i, 0], rng[i, 1], rng[i, 2], rng[i, 3] = c0, c1, r0, r1
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                counts[r * nx + c + 1] += 1
    start = np.cumsum(counts)
    fill = start[:-1].copy()
    items = np.empty(start[-1], np.int64)
    for i in range(N):
        for r in range(rng[i, 2], rng[i, 3] + 1):
            for c in range(rng[i, 0], rng[i, 1] + 1):
                items[fill[r * nx + c]] = i
                fill[r * nx + c] += 1
    return start, items


def _seg_interval(S, i, ox, oy, a0, da, full, ks):
    """Ray index ranges of segment i per band, exactly as _ray_segment computes them."""
    two_pi = 2 * math.pi
    x1, y1 = S[i, 0] - ox, S[i, 1] - oy
    x2, y2 = S[i, 2] - ox, S[i, 3] - oy
    r1 = (math.atan2(y1, x1) - a0) % two_pi
    d = (math.atan2(y2, x2) - a0 - r1 + math.pi) % two_pi - math.pi
    lo = r1 + min(d, 0.0)
    hi = lo + abs(d)
    for band in range(1 if full else 3):
        shift = 0.0 if band == 0 else (-two_pi if band == 1 else two_pi)
        ks[i, 2 * band] = math.ceil((lo + shift) / da - 1e-9)
        ks[i, 2 * band + 1] = math.floor((hi + shift) / da + 1e-9)


def _ray_in_interval(ks, i, k, n, full):
    """Whether the full scan tests ray k against segment i (k ranges in ks[i])."""
    if full:
        lo, hi = ks[i, 0], ks[i, 1]
        if hi < lo:
            return False
        if hi - lo + 1 >= n:
            return True
        return (k - lo) % n <= hi - lo
    for b in range(3):
        if ks[i, 2 * b] <= k <= ks[i, 2 * b + 1]:
            return True
    return False


def _raycast_grid(S, start, items, gx0, gy0, cs, nx, ny, ox, oy, a0, da, n, r_max, full,
                  cosk, sink, out, stamp, ks, call):
    """raycast_uniform through the segment grid (same result, see module docstring).
    stamp / ks cache each segment's ray ranges for the call numbered call[0]."""
    cx = int(math.floor((ox - gx0) / cs))
    cy = int(math.floor((oy - gy0) / cs))
    if cx < 0 or cx >= nx or cy < 0 or cy >= ny:
        _raycast_uniform(S, ox, oy, a0, da, n, r_max, full, cosk, sink, out)
        return
    call[0] += 1
    tag = call[0]
    for k in range(n):
        dx, dy = cosk[k], sink[k]
        ix, iy = cx, cy
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        t_max_x = ((ix + (1 if dx > 0 else 0)) * cs + gx0 - ox) / dx if dx != 0 else math.inf
        t_max_y = ((iy + (1 if dy > 0 else 0)) * cs + gy0 - oy) / dy if dy != 0 else math.inf
        t_dx = cs / abs(dx) if dx != 0 else math.inf
        t_dy = cs / abs(dy) if dy != 0 else math.inf
        while True:
            cell = iy * nx + ix
            for p in range(start[cell], start[cell + 1]):
                i = items[p]
                if stamp[i] != tag:
                    _seg_interval(S, i, ox, oy, a0, da, full, ks)
                    stamp[i] = tag
                if not _ray_in_interval(ks, i, k, n, full):
                    continue
                # the intersection of _ray_segment, operation for operation
                x1, y1 = S[i, 0] - ox, S[i, 1] - oy
                x2, y2 = S[i, 2] - ox, S[i, 3] - oy
                ex, ey = x2 - x1, y2 - y1
                den = dx * ey - dy * ex
                if abs(den) <= EPS:
                    continue
                t = (x1 * ey - y1 * ex) / den
                uu = (x1 * dy - y1 * dx) / den
                if t >= 0 and uu >= -1e-9 and uu <= 1 + 1e-9 and t < out[k]:
                    out[k] = t
            t_next = min(t_max_x, t_max_y)
            if out[k] <= t_next or t_next >= r_max:
                break
            if t_max_x < t_max_y:
                ix += step_x
                t_max_x += t_dx
            else:
                iy += step_y
                t_max_y += t_dy
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                break


def _nearest_grid(S, start, items, gx0, gy0, cs, nx, ny, x, y):
    """nearest_segment through the grid: rings of cells until best <= ring distance."""
    cx = int(math.floor((x - gx0) / cs))
    cy = int(math.floor((y - gy0) / cs))
    if cx < 0 or cx >= nx or cy < 0 or cy >= ny:
        return _nearest_segment(S, x, y)
    best = math.inf
    r_last = max(cx, nx - 1 - cx, cy, ny - 1 - cy)
    for r in range(r_last + 1):
        if r >= 1 and best <= (r - 1) * cs:      # every cell of ring r is >= (r-1) cs away
            break
        for iy in range(max(cy - r, 0), min(cy + r, ny - 1) + 1):
            edge_row = iy == cy - r or iy == cy + r
            for ix in range(max(cx - r, 0), min(cx + r, nx - 1) + 1):
                if not (edge_row or ix == cx - r or ix == cx + r):
                    continue                     # inside the ring: done at a smaller r
                cell = iy * nx + ix
                for p in range(start[cell], start[cell + 1]):
                    d = _seg_dist(S, items[p], x, y)
                    if d < best:
                        best = d
    return best


def _seg_hit_poly_grid(S, start, items, gx0, gy0, cs, nx, ny, P, xmin, ymin, xmax, ymax):
    """seg_hit_poly limited to the grid cells under P's bbox (same answer)."""
    c0 = max(int(math.floor((xmin - gx0) / cs)), 0)
    c1 = min(int(math.floor((xmax - gx0) / cs)), nx - 1)
    r0 = max(int(math.floor((ymin - gy0) / cs)), 0)
    r1 = min(int(math.floor((ymax - gy0) / cs)), ny - 1)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = r * nx + c
            for p in range(start[cell], start[cell + 1]):
                if _seg_hits(S, items[p], P, xmin, ymin, xmax, ymax):
                    return True
    return False


if ENABLED:
    _sign = njit(cache=True)(_sign)
    _seg_hits = njit(cache=True)(_seg_hits)
    seg_hit_poly = njit(cache=True)(_seg_hit_poly)
    _ray_segment = njit(cache=True, inline="always")(_ray_segment)
    _raycast_uniform = raycast_uniform = njit(cache=True)(_raycast_uniform)
    _seg_dist = njit(cache=True, inline="always")(_seg_dist)
    _nearest_segment = nearest_segment = njit(cache=True)(_nearest_segment)
    build_grid = njit(cache=True)(_build_grid)
    _seg_interval = njit(cache=True)(_seg_interval)
    _ray_in_interval = njit(cache=True, inline="always")(_ray_in_interval)
    raycast_grid = njit(cache=True)(_raycast_grid)
    nearest_grid = njit(cache=True)(_nearest_grid)
    seg_hit_poly_grid = njit(cache=True)(_seg_hit_poly_grid)
else:                                                          # pragma: no cover
    seg_hit_poly = raycast_uniform = nearest_segment = None
    build_grid = raycast_grid = nearest_grid = seg_hit_poly_grid = None
