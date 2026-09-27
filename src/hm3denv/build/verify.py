"""Check start/goal poses on the real 3D MESH, independently of the raster images of
the slice and grid stages. Every pose must pass all four tests:

  support  vertical rays cast down from floor+floor_margin on an n x n grid under the
           footprint; >= support_min of them first hit a horizontal face within
           floor_band of floor_z
  clear    the body prism footprint x [floor+floor_margin, floor+height] intersects no
           triangle (exact triangle/box or triangle/prism SAT, no sampling)
  roof     a ray cast straight up from the robot top hits some surface (ceiling, roof,
           the storey above); terraces and balconies fail. No height limit: double-height
           rooms have ceilings above the storey's ceil_z (measured on 00337 storey 2:
           ceiling at 3.5-5.4 m)
  bfs      (grid tasks only) BFS on the grid reproduces d_bfs

`margin_hits` also counts triangles in the safety band around the body, for reference
only: the gridify margin is a pixel disk, the box here has corners sqrt(2) further out.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..core.geometry import is_convex_ccw, points_in_polygon, triangulate
from ..core.gridcore import bfs, cell_to_world
from .slice import scene_mesh

SUPPORT_N = 5         # support rays per side
SUPPORT_MIN = 0.9     # same as gridify floor_cover
HORIZONTAL = 0.7      # min |n_up| of a horizontal face, same as the slice stage


class MeshProbe:
    """Triangles of one storey (filtered by height once) + geometric queries."""

    def __init__(self, V, F, up, z_lo=-np.inf, z_hi=np.inf):
        tri = V[F].astype(np.float64)
        tz = tri[:, :, up]
        keep = (tz.max(axis=1) >= z_lo) & (tz.min(axis=1) <= z_hi)
        tri = tri[keep]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12                       # drop degenerate triangles
        self.tri, n, ln = tri[ok], n[ok], ln[ok]
        self.n_up = np.abs(n[:, up]) / ln
        self.lo, self.hi = self.tri.min(axis=1), self.tri.max(axis=1)
        self.up = up
        self.ua, self.va = [a for a in range(3) if a != up]

    def _near(self, u, v, half):
        ua, va = self.ua, self.va
        return np.where((self.hi[:, ua] >= u - half) & (self.lo[:, ua] <= u + half)
                        & (self.hi[:, va] >= v - half) & (self.lo[:, va] <= v + half))[0]

    def _heights(self, pts, idx):
        """Height of every triangle idx above every point pts (u, v); NaN where the point
        is outside the triangle's projection. Shape (P, T)."""
        t = self.tri[idx]
        a = t[:, 0][None]
        e1 = (t[:, 1] - t[:, 0])[None]
        e2 = (t[:, 2] - t[:, 0])[None]
        ua, va, up = self.ua, self.va, self.up
        du = pts[:, 0][:, None] - a[..., ua]
        dv = pts[:, 1][:, None] - a[..., va]
        det = e1[..., ua] * e2[..., va] - e2[..., ua] * e1[..., va]
        eps = 1e-9
        # vertical triangles have a degenerate projection (det = 0): rejected by `inside`
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (du * e2[..., va] - dv * e2[..., ua]) / det
            w = (e1[..., ua] * dv - e1[..., va] * du) / det
            inside = ((s >= -eps) & (w >= -eps) & (s + w <= 1 + eps) & (np.abs(det) > 1e-14))
            z = a[..., up] + s * e1[..., up] + w * e2[..., up]
        return np.where(inside, z, np.nan)

    def support(self, u, v, half, floor_z, floor_margin, floor_band, n=SUPPORT_N):
        """Fraction of downward rays (from floor+floor_margin) whose first hit is floor."""
        g = np.linspace(-half, half, n) * 0.95
        pts = np.array([(u + a, v + b) for a in g for b in g])
        return self.support_points(pts, floor_z, floor_margin, floor_band)

    def support_points(self, pts, floor_z, floor_margin, floor_band):
        """Like support() on an arbitrary set of (u, v) points."""
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ua, va = self.ua, self.va
        idx = np.where((self.hi[:, ua] >= lo[0]) & (self.lo[:, ua] <= hi[0])
                       & (self.hi[:, va] >= lo[1]) & (self.lo[:, va] <= hi[1]))[0]
        if not len(idx):
            return 0.0
        z = self._heights(pts, idx)
        z = np.where(z <= floor_z + floor_margin, z, np.nan)   # below the ray origin
        hit = 0
        for i in range(len(pts)):
            if np.isnan(z[i]).all():
                continue
            j = np.nanargmax(z[i])                              # first surface hit
            if z[i, j] >= floor_z - floor_band and self.n_up[idx[j]] >= HORIZONTAL:
                hit += 1
        return hit / len(pts)

    def _candidates(self, lo, hi, z0, z1):
        blo, bhi = np.zeros(3), np.zeros(3)
        blo[self.ua], blo[self.va], blo[self.up] = lo[0], lo[1], z0
        bhi[self.ua], bhi[self.va], bhi[self.up] = hi[0], hi[1], z1
        return np.where(((self.hi >= blo) & (self.lo <= bhi)).all(axis=1))[0]

    def prism_hits(self, poly, z0, z1):
        """Number of mesh triangles intersecting the prism poly (M,2) in (u, v) x [z0, z1].
        CONCAVE polygons are ear-clipped; exact SAT on every convex piece."""
        idx = self._candidates(poly.min(axis=0), poly.max(axis=0), z0, z1)
        if not len(idx):
            return 0
        t = self.tri[idx][..., [self.ua, self.va, self.up]]
        if is_convex_ccw(poly):
            return int(prism_overlap(t, poly, z0, z1).sum())
        hit = np.zeros(len(t), bool)
        for piece in triangulate(poly):
            hit |= prism_overlap(t, piece, z0, z1)
        return int(hit.sum())

    def box_hits(self, u, v, half, z0, z1):
        """Number of triangles intersecting the box [u+-half] x [v+-half] x [z0, z1] (SAT)."""
        c = np.zeros(3)
        h = np.zeros(3)
        c[self.ua], c[self.va], c[self.up] = u, v, (z0 + z1) / 2
        h[self.ua] = h[self.va] = half
        h[self.up] = (z1 - z0) / 2
        idx = np.where(((self.hi >= c - h) & (self.lo <= c + h)).all(axis=1))[0]
        if not len(idx):
            return 0
        return int(tri_box_overlap(self.tri[idx] - c, h).sum())

    def roof(self, u, v, z_from):
        """Does a vertical ray from z_from upwards hit any surface?"""
        idx = self._near(u, v, 1e-6)
        if not len(idx):
            return False
        z = self._heights(np.array([[u, v]]), idx)[0]
        return bool((z > z_from).any())


def tri_box_overlap(t, h):
    """Akenine-Moller SAT for many triangles (T,3,3) against a box centred at the origin
    with half extents h. Box axes are already filtered by the bounding boxes."""
    v0, v1, v2 = t[:, 0], t[:, 1], t[:, 2]
    f = (v1 - v0, v2 - v1, v0 - v2)
    sep = np.zeros(len(t), bool)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        for fj in f:
            a = np.cross(e, fj)                               # (T, 3)
            p = np.stack([(a * v).sum(axis=1) for v in (v0, v1, v2)])
            r = (np.abs(a) * h).sum(axis=1)
            sep |= (p.min(axis=0) > r) | (p.max(axis=0) < -r)
    n = np.cross(f[0], v2 - v0)
    d = (n * v0).sum(axis=1)
    sep |= np.abs(d) > (np.abs(n) * h).sum(axis=1)
    return ~sep


def prism_overlap(t, poly, z0, z1, eps=1e-12, chunk=1024):
    """SAT of triangles (T,3,3) (coordinates u, v, z) against the convex CCW prism
    poly (M,2) x [z0, z1]. Separating axes: polygon edge normals, z, triangle normal,
    triangle edge x polygon edge, triangle edge x z.

    Intermediates are T x (4M+5) x 2M, so triangles are processed in chunks and memory
    does not grow with the triangle count (parallel workers on low-RAM machines).
    """
    if len(t) > chunk:
        return np.concatenate([prism_overlap(t[i:i + chunk], poly, z0, z1, eps, chunk)
                               for i in range(0, len(t), chunk)])
    M = len(poly)
    e2 = np.roll(poly, -1, axis=0) - poly                          # (M, 2)
    d = np.column_stack([e2, np.zeros(M)])                        # horizontal edges
    n_poly = np.column_stack([e2[:, 1], -e2[:, 0], np.zeros(M)])
    z = np.array([[0.0, 0.0, 1.0]])
    E = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], 1)  # (T, 3, 3)
    n_tri = np.cross(E[:, 0], E[:, 1])[:, None]                    # (T, 1, 3)
    cross_d = np.cross(E[:, :, None, :], d[None, None]).reshape(len(t), 3 * M, 3)
    cross_z = np.cross(E, z[None])                                 # (T, 3, 3)
    fixed = np.concatenate([n_poly, z])[None].repeat(len(t), 0)    # (T, M+1, 3)
    A = np.concatenate([fixed, n_tri, cross_d, cross_z], axis=1)   # (T, K, 3)
    P = np.concatenate([np.column_stack([poly, np.full(M, z0)]),
                        np.column_stack([poly, np.full(M, z1)])])  # (2M, 3)
    pt = np.einsum("tkd,tvd->tkv", A, t)
    pp = np.einsum("tkd,pd->tkp", A, P)
    sep = (pt.max(-1) < pp.min(-1) - eps) | (pp.max(-1) < pt.min(-1) - eps)
    return ~sep.any(axis=1)


def footprint_points(poly, n=SUPPORT_N):
    """n x n grid of points inside the polygon (convex or not; shrunk 5% towards the bbox
    centre) for the support rays."""
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    c = (lo + hi) / 2
    g = np.array([(a, b) for a in np.linspace(lo[0], hi[0], n) for b in np.linspace(lo[1], hi[1], n)])
    g = c + (g - c) * 0.95
    return g[points_in_polygon(g, poly)]


def check_footprint(probe, poly, floor_z, height, floor_margin=0.05, floor_band=0.08,
                    support_min=SUPPORT_MIN):
    """The four tests for a footprint already placed in the map frame (polygon in (u, v))."""
    z0, z1 = floor_z + floor_margin, floor_z + height
    sup = probe.support_points(footprint_points(poly), floor_z, floor_margin, floor_band)
    hits = probe.prism_hits(poly, z0, z1)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    cu, cv = (lo + hi) / 2
    roof = probe.roof(cu, cv, z1)
    return {"support": round(sup, 3), "clear_hits": hits, "roof": roof,
            "ok": bool(sup >= support_min and hits == 0 and roof)}


def check_point(probe, u, v, floor_z, cell_m, height, margin,
                floor_margin=0.05, floor_band=0.08, support_min=SUPPORT_MIN):
    """The four tests for a square robot of side cell_m centred at (u, v)."""
    half = cell_m / 2
    z0, z1 = floor_z + floor_margin, floor_z + height
    sup = probe.support(u, v, half, floor_z, floor_margin, floor_band)
    hits = probe.box_hits(u, v, half, z0, z1)
    mhits = probe.box_hits(u, v, half + margin, z0, z1) - hits if margin > 0 else 0
    roof = probe.roof(u, v, z1)
    return {"support": round(sup, 3), "clear_hits": hits, "margin_hits": mhits,
            "roof": roof, "ok": bool(sup >= support_min and hits == 0 and roof)}


def storey_probe(ws, scene, up, floor_z, _cache={}):
    """MeshProbe of a storey; the last one is kept because samplers call this in a loop.
    Everything above the floor is kept: the roof test needs the storeys above."""
    key = (str(ws), scene, up, round(floor_z, 4))
    if _cache.get("key") != key:
        V, F = scene_mesh(ws, scene)
        _cache.clear()
        _cache["key"] = key
        _cache["probe"] = MeshProbe(V, F, up, floor_z - 0.3)
    return _cache["probe"]


def check_cell(ws, meta, r, c, height):
    """check_point for cell (r, c) of a grid (meta = the grid's JSON)."""
    probe = storey_probe(ws, meta["scene"], meta["up"], meta["floor_z"])
    p = cell_to_world(meta, r, c)
    res = check_point(probe, p[meta["ua"]], p[meta["va"]], meta["floor_z"], meta["cell_m"],
                      height, meta["margin"], meta.get("floor_margin", 0.05),
                      meta.get("floor_band", 0.08))
    res["world"] = [round(x, 4) for x in p]
    return res


# ------------------------------------------------------------------ re-check a dataset

GRID_COLS = ["map_id", "task", "which", "r", "c", "x", "y", "z", "support",
             "clear_hits", "margin_hits", "roof", "bfs_ok", "ok"]
SVG_COLS = ["robot", "map_id", "task", "which", "x", "y", "theta", "support", "clear_hits", "roof", "ok"]


def write_csv(rows, path, cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def recheck_grid(ws, ds, robot):
    """Re-run the mesh tests + BFS on every task of a grid dataset."""
    height = ds.robot(robot).height
    rows = []
    for map_id in ds.maps(robot):
        meta = ds.grid_meta(map_id)
        grid = ds.grid_arrays(map_id)["grid"]
        for ti, t in enumerate(ds.tasks(robot, map_id)):
            s, g = t["start"]["cell"], t["goal"]["cell"]
            d = bfs(grid == 0, s)[tuple(g)]
            bfs_ok = bool(d == t["labels"]["d_bfs"])
            for which, (r, c) in (("start", s), ("goal", g)):
                chk = check_cell(ws, meta, r, c, height)
                rows.append({"map_id": map_id, "task": ti, "which": which, "r": r, "c": c,
                             **dict(zip("xyz", chk["world"])),
                             **{k: chk[k] for k in ("support", "clear_hits", "margin_hits", "roof")},
                             "bfs_ok": bfs_ok, "ok": bool(chk["ok"] and bfs_ok)})
    return rows


def recheck_svg(ws, ds, robot):
    """Re-run the mesh tests on every start pose (real footprint) and goal (circumscribed
    disk) of an SVG dataset."""
    from ..core.geometry import place
    from ..core.svgmap import SvgMap
    from ..robots import footprint_polygon
    rb = ds.robot(robot)
    disk = footprint_polygon({"type": "circle", "radius": rb.r_circ})
    rows = []
    for map_id in ds.maps(robot):
        meta = SvgMap.load(ds.svg_path(robot, map_id)).meta
        probe = storey_probe(ws, meta["scene"], meta["frame"]["up"], meta["floor_z"])
        for ti, t in enumerate(ds.tasks(robot, map_id)):
            s, g = t["start"], t["goal"]
            for which, poly, th in (("start", place(rb.footprint, s["x"], s["y"], s["theta"]), s["theta"]),
                                    ("goal", place(disk, g["x"], g["y"], 0.0), "")):
                chk = check_footprint(probe, poly, meta["floor_z"], rb.height)
                p = s if which == "start" else g
                rows.append({"robot": robot, "map_id": map_id, "task": ti, "which": which,
                             "x": p["x"], "y": p["y"], "theta": th, **chk})
    return rows


def summarize(rows) -> dict:
    pts = [r for r in rows if r.get("task", -1) != -1]
    if not pts:
        return {"points": 0}
    n = len(pts)
    tasks = {}
    for r in pts:
        tasks.setdefault((r["map_id"], r["task"]), []).append(r["ok"])
    out = {"tasks": len(tasks), "points": n,
           "fail_support": sum(r["support"] < SUPPORT_MIN for r in pts) / n,
           "fail_clear": sum(r["clear_hits"] > 0 for r in pts) / n,
           "fail_roof": sum(not r["roof"] for r in pts) / n,
           "points_ok": sum(r["ok"] for r in pts) / n,
           "tasks_ok": sum(all(v) for v in tasks.values()) / len(tasks)}
    if "bfs_ok" in pts[0] and pts[0]["bfs_ok"] != "":
        out["fail_bfs"] = sum(r["bfs_ok"] is False for r in pts) / n
    return out
