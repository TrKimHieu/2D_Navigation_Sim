"""Computational geometry on the ``free`` region of an SvgMap: footprint collision and LiDAR.

A pose (x, y, theta) is valid iff the footprint polygon placed there crosses no
edge of ``free`` AND its centre lies inside ``free``. If no edge is crossed the
footprint is entirely on one side of the boundary, so testing one point (the
centre) is enough. Touching counts as a collision (conservative).

Speed-ups that never change a result:
a distance field to the map edges answers most pose_valid calls without the exact
test, and core._accel runs the exact tests as numba kernels when available.
"""

from __future__ import annotations

import numpy as np

from . import _accel

EPS = 1e-12
GRID_CELL = 0.25       # segment grid cell (m) of the numba kernels, see core._accel
EDGE_MARGIN_PX = 2.0   # query-point quantisation + line rasterisation, both <= sqrt(2)/2 px


def place(poly: np.ndarray, x: float, y: float, th: float) -> np.ndarray:
    """Polygon in base_link -> map frame at pose (x, y, th)."""
    c, s = np.cos(th), np.sin(th)
    return poly @ np.array([[c, s], [-s, c]]) + (x, y)


class SegmentIndex:
    """Uniform spatial hash of segments: each cell lists the segments whose bbox touches it.
    The hash is built on the first query (the numba kernels never query it)."""

    def __init__(self, segs, cell: float = 0.25):
        self.segs = np.asarray(segs, float)
        self.cell = cell
        self._buckets = None

    @property
    def buckets(self) -> dict:
        if self._buckets is None:
            self._build()
        return self._buckets

    def _build(self):
        cell = self.cell
        lo = np.minimum(self.segs[:, :2], self.segs[:, 2:])
        hi = np.maximum(self.segs[:, :2], self.segs[:, 2:])
        self.origin = lo.min(axis=0) if len(lo) else np.zeros(2)
        i0 = np.floor((lo - self.origin) / cell).astype(int)
        i1 = np.floor((hi - self.origin) / cell).astype(int)
        self.shape = (i1.max(axis=0) + 1) if len(i1) else np.ones(2, int)
        buckets: dict = {}
        for k, (a, b) in enumerate(zip(i0, i1)):
            for ix in range(a[0], b[0] + 1):
                for iy in range(a[1], b[1] + 1):
                    buckets.setdefault((ix, iy), []).append(k)
        self._buckets = {key: np.array(v, np.int64) for key, v in buckets.items()}

    def query(self, xmin, ymin, xmax, ymax) -> np.ndarray:
        buckets = self.buckets
        c = self.cell
        ix0, iy0 = np.floor((np.array([xmin, ymin]) - self.origin) / c).astype(int)
        ix1, iy1 = np.floor((np.array([xmax, ymax]) - self.origin) / c).astype(int)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, self.shape[0] - 1), min(iy1, self.shape[1] - 1)
        parts = [buckets[(ix, iy)] for ix in range(ix0, ix1 + 1)
                 for iy in range(iy0, iy1 + 1) if (ix, iy) in buckets]
        if not parts:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(parts))


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def is_convex_ccw(poly: np.ndarray) -> bool:
    e = np.roll(poly, -1, axis=0) - poly
    cr = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    return bool((cr >= -1e-12).all())


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule for any simple polygon (convex or concave): (N,) bool."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None], poly[:, 1][None]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return ((crosses & (x < xi)).sum(axis=1) % 2).astype(bool)


def triangulate(poly: np.ndarray) -> np.ndarray:
    """Ear-clip a simple CCW polygon into (T, 3, 2) triangles. Collinear vertices are
    dropped first; if clipping still gets stuck (degenerate polygon) the convex hull
    is triangulated instead (conservative: never loses part of the robot)."""
    e_in = poly - np.roll(poly, 1, axis=0)
    e_out = np.roll(poly, -1, axis=0) - poly
    turn = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    poly = poly[np.abs(turn) > 1e-12]
    tris = _ear_clip(poly)
    area = 0.5 * abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    got = sum(0.5 * abs(_cross(*(t[1] - t[0]), *(t[2] - t[0]))) for t in tris)
    if abs(got - area) > 1e-9 * max(area, 1.0):
        import cv2
        hull = cv2.convexHull(poly.astype(np.float32))[:, 0, :].astype(float)
        tris = np.array([[hull[0], hull[i], hull[i + 1]] for i in range(1, len(hull) - 1)])
    return tris


def _ear_clip(poly: np.ndarray) -> np.ndarray:
    P = [tuple(p) for p in poly]
    idx = list(range(len(P)))
    out = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10 * len(P):
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-15:                         # reflex or collinear vertex
                continue
            tri = np.array([a, b, c])
            others = np.array([P[j] for j in idx if j not in (i0, i1, i2)])
            if len(others) and points_in_polygon(others, tri).any():
                continue
            out.append(tri)
            idx.pop(k)
            break
        else:
            break                                               # degenerate polygon
    if len(idx) == 3:
        out.append(np.array([P[j] for j in idx]))
    return np.array(out)


def segments_hit_polygon(segs: np.ndarray, poly: np.ndarray) -> bool:
    """Whether any segment (K, 4) crosses or touches the simple polygon (M, 2)."""
    if not len(segs):
        return False
    a, b = segs[:, :2], segs[:, 2:]
    v = poly
    e = np.roll(poly, -1, axis=0) - poly                       # (M, 2)
    # one endpoint inside is enough: a segment that crosses no edge is either
    # entirely inside or entirely outside
    if points_in_polygon(a, poly).any():
        return True
    q1, q2 = v[None], (v + e)[None]                            # (1, M, 2)
    p1, p2 = a[:, None], b[:, None]                            # (K, 1, 2)
    r = p2 - p1
    s = q2 - q1
    d1 = _cross(s[..., 0], s[..., 1], (p1 - q1)[..., 0], (p1 - q1)[..., 1])
    d2 = _cross(s[..., 0], s[..., 1], (p2 - q1)[..., 0], (p2 - q1)[..., 1])
    d3 = _cross(r[..., 0], r[..., 1], (q1 - p1)[..., 0], (q1 - p1)[..., 1])
    d4 = _cross(r[..., 0], r[..., 1], (q2 - p1)[..., 0], (q2 - p1)[..., 1])
    # Compare SIGNS, not products against an absolute threshold: URDF footprints
    # have millimetre edges, so orientation values are ~1e-7 and the product of two
    # same-sign values (~1e-14) still falls below thresholds like 1e-12, reporting
    # phantom crossings that freeze the robot. Only |d| at float noise counts as 0.
    tol = 1e-15
    sd1, sd2, sd3, sd4 = (np.where(np.abs(d) <= tol, 0, np.sign(d)) for d in (d1, d2, d3, d4))
    return bool(((sd1 * sd2 <= 0) & (sd3 * sd4 <= 0)).any())


class Collider:
    """Collision and LiDAR queries on one map.

    accel: use the numba kernels (default: core._accel.ENABLED). Both backends give
    identical results; False forces the numpy code (tests, debugging). The kernels
    use a segment grid (built on first use) so their cost depends on the segments near
    the query, not on the size of the map.
    """

    def __init__(self, svgmap, cell: float = 0.25, res: float = 0.02, accel: bool | None = None,
                 edge_dist: np.ndarray | None = None):
        self.map = svgmap
        self.index = SegmentIndex(svgmap.segments, cell)
        self.mask, self.x0, self.y1, self.res = svgmap.raster(res)
        self.accel = _accel.ENABLED if accel is None else (accel and _accel.ENABLED)
        self.edge_dist = self._edge_distance() if edge_dist is None else edge_dist
        self._r_poly: dict = {}
        self._grid = None

    def _edge_distance(self) -> np.ndarray:
        """Distance (whole px, uint8, floored, capped at 255) from each pixel centre to the
        nearest pixel touched by an edge of ``free``. The true distance from a point in
        pixel (r, c) to the nearest edge is at least (edge_dist[r, c] - EDGE_MARGIN_PX) * res
        (flooring and capping only lower the bound)."""
        import cv2
        H, W = self.mask.shape
        img = np.full((H, W), 255, np.uint8)
        S = self.index.segs
        if len(S):
            sh = 8
            px = (S[:, [0, 2]] - self.x0) / self.res - 0.5          # pixel-centre coordinates
            py = (self.y1 - S[:, [1, 3]]) / self.res - 0.5
            pts = np.ascontiguousarray(np.round(np.stack([px, py], -1) * (1 << sh)), np.int32)
            cv2.polylines(img, list(pts), False, 0, 1, cv2.LINE_8, sh)
        d = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return np.minimum(np.floor(d), 255).astype(np.uint8)

    @property
    def grid(self):
        """(start, items, gx0, gy0, cs, nx, ny, stamp, ks, call) for the numba kernels."""
        if self._grid is None:
            S = self.index.segs
            if len(S):
                gx0 = float(min(S[:, 0].min(), S[:, 2].min())) - GRID_CELL
                gy0 = float(min(S[:, 1].min(), S[:, 3].min())) - GRID_CELL
                nx = int((max(S[:, 0].max(), S[:, 2].max()) - gx0) // GRID_CELL) + 2
                ny = int((max(S[:, 1].max(), S[:, 3].max()) - gy0) // GRID_CELL) + 2
            else:
                gx0 = gy0 = 0.0
                nx = ny = 1
            start, items = _accel.build_grid(S, gx0, gy0, GRID_CELL, nx, ny)
            self._grid = (start, items, gx0, gy0, GRID_CELL, nx, ny,
                          np.zeros(len(S), np.int64), np.zeros((len(S), 6), np.int64),
                          np.zeros(1, np.int64))
        return self._grid

    def _clear_of_edges(self, x: float, y: float, r: float) -> bool:
        """True when no edge of ``free`` can be within r of (x, y) (conservative)."""
        fc = (x - self.x0) / self.res
        fr = (self.y1 - y) / self.res
        H, W = self.edge_dist.shape
        if not (0.0 <= fr < H and 0.0 <= fc < W):
            return False
        return (float(self.edge_dist[int(fr), int(fc)]) - EDGE_MARGIN_PX) * self.res > r

    def inside(self, x: float, y: float) -> bool:
        """Point inside ``free`` (raster lookup; only used once the footprint crosses no
        edge, so boundary rounding does not matter)."""
        c = int((x - self.x0) / self.res)
        r = int((self.y1 - y) / self.res)
        H, W = self.mask.shape
        return 0 <= r < H and 0 <= c < W and bool(self.mask[r, c])

    def pose_valid(self, poly: np.ndarray, x: float, y: float, th: float) -> bool:
        # Shortcuts with the same answer as the exact test below:
        #  centre outside free -> invalid whether or not an edge is crossed;
        #  every edge farther than the footprint circumradius -> nothing to cross.
        if not self.inside(x, y):
            return False
        r = self._r_poly.get(id(poly))
        if r is None or r[0] is not poly:
            r = self._r_poly[id(poly)] = (poly, float(np.linalg.norm(poly, axis=1).max()))
        if self._clear_of_edges(x, y, r[1]):
            return True
        P = place(poly, x, y, th)
        lo, hi = P.min(axis=0), P.max(axis=0)
        if self.accel:
            g = self.grid
            return not _accel.seg_hit_poly_grid(self.index.segs, *g[:7], P,
                                                lo[0], lo[1], hi[0], hi[1])
        idx = self.index.query(lo[0], lo[1], hi[0], hi[1])
        return not segments_hit_polygon(self.index.segs[idx], P)

    def wall_distance(self, x: float, y: float) -> float:
        """Exact distance (m) from (x, y) to the nearest edge of ``free``."""
        S = self.index.segs
        if not len(S):
            return float("inf")
        if self.accel:
            return float(_accel.nearest_grid(S, *self.grid[:7], float(x), float(y)))
        a = S[:, :2] - (x, y)
        e = S[:, 2:] - S[:, :2]
        L2 = np.maximum((e * e).sum(axis=1), EPS)
        u = np.clip(-(a * e).sum(axis=1) / L2, 0.0, 1.0)
        return float(np.hypot(a[:, 0] + u * e[:, 0], a[:, 1] + u * e[:, 1]).min())

    def pose_valid_exact(self, poly: np.ndarray, x: float, y: float, th: float) -> bool:
        """pose_valid without shortcuts or kernels (reference for tests)."""
        P = place(poly, x, y, th)
        lo, hi = P.min(axis=0), P.max(axis=0)
        idx = self.index.query(lo[0], lo[1], hi[0], hi[1])
        if segments_hit_polygon(self.index.segs[idx], P):
            return False
        return self.inside(x, y)

    def raycast_uniform(self, ox, oy, a0, da, n, r_max, full=False) -> np.ndarray:
        """Like raycast() for n rays at angles a0 + k*da, but each segment is only
        tested against the rays inside the angular interval it subtends from
        (ox, oy): a few thousand pairs instead of n x segments. Intersections are
        still solved exactly per pair.

        full=True: n*da = 2pi (360 degree LiDAR), ray indices wrap modulo n.
        """
        S = self.index.segs
        out = np.full(n, r_max, float)
        if not len(S):
            return out
        if self.accel:
            ang = a0 + np.arange(n) * da              # same values as the numpy path below
            _accel.raycast_grid(S, *self.grid[:7], float(ox), float(oy), float(a0), float(da),
                                int(n), float(r_max), bool(full), np.cos(ang), np.sin(ang),
                                out, *self.grid[7:])
            return out
        x1, y1, x2, y2 = S[:, 0] - ox, S[:, 1] - oy, S[:, 2] - ox, S[:, 3] - oy
        ex, ey = x2 - x1, y2 - y1
        L2 = np.maximum(ex * ex + ey * ey, EPS)
        u = np.clip(-(x1 * ex + y1 * ey) / L2, 0, 1)
        near = np.hypot(x1 + u * ex, y1 + u * ey) <= r_max      # drop segments out of range
        x1, y1, x2, y2, ex, ey = x1[near], y1[near], x2[near], y2[near], ex[near], ey[near]
        two_pi = 2 * np.pi
        r1 = (np.arctan2(y1, x1) - a0) % two_pi
        d = (np.arctan2(y2, x2) - a0 - r1 + np.pi) % two_pi - np.pi   # short arc
        lo = r1 + np.minimum(d, 0)
        hi = lo + np.abs(d)
        if not full:
            # lo lies in (-pi, 2pi): the arc may straddle 0 or 2pi -> add index bands
            # shifted by +-2pi (the full-circle case wraps modulo n instead)
            ids = np.arange(len(lo))
            ids = np.concatenate([ids, ids, ids])
            lo = np.concatenate([lo, lo - two_pi, lo + two_pi])
            hi = np.concatenate([hi, hi - two_pi, hi + two_pi])
        else:
            ids = np.arange(len(lo))
        k_lo = np.ceil(lo / da - 1e-9).astype(np.int64)
        k_hi = np.floor(hi / da + 1e-9).astype(np.int64)
        cnt = np.maximum(k_hi - k_lo + 1, 0)
        if not cnt.sum():
            return out
        seg = np.repeat(ids, cnt)
        k = np.repeat(k_lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        if full:
            k %= n
        else:
            ok = (k >= 0) & (k < n)
            seg, k = seg[ok], k[ok]
        ang = a0 + k * da
        dx, dy = np.cos(ang), np.sin(ang)
        ax, ay, sx, sy = x1[seg], y1[seg], ex[seg], ey[seg]
        den = dx * sy - dy * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * sy - ay * sx) / den
            uu = (ax * dy - ay * dx) / den
        hit = (np.abs(den) > EPS) & (t >= 0) & (uu >= -1e-9) & (uu <= 1 + 1e-9)
        np.minimum.at(out, k[hit], t[hit])
        return out

    def raycast(self, ox, oy, angles, r_max) -> np.ndarray:
        """Distance to the nearest edge per angle (m), r_max when nothing is hit."""
        idx = self.index.query(ox - r_max, oy - r_max, ox + r_max, oy + r_max)
        out = np.full(len(angles), r_max, float)
        if not len(idx):
            return out
        S = self.index.segs[idx]
        dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]    # (B, 1)
        ax, ay = S[None, :, 0] - ox, S[None, :, 1] - oy              # (1, K)
        ex, ey = S[None, :, 2] - S[None, :, 0], S[None, :, 3] - S[None, :, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * ey - ay * ex) / den
            u = (ax * dy - ay * dx) / den
        ok = (np.abs(den) > EPS) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf).min(axis=1)
        return np.minimum(out, t)
