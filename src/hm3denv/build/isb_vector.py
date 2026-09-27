"""SVG maps of Isaac-Scene-Builder scenes built from their vector description (CSV) instead
of re-vectorising the 2.5 cm BEV raster.


Geometry of the scene builder (bev_export.py / utils.py), in editor = USD = map coordinates:

    floor   CELL x CELL square centred on (x, y)
    wall    segment (x, y)-(x2, y2), WALL_M thick, centred, flat ends
    door    a wall whose middle DOOR_GAP fraction is open
    prop    bbox: w x h rectangle; sphere: disc of radius r; mesh: mesh_shapes/<name>.png
            (pixels < 128 = obstacle) spanning CELL x CELL, falling back to bbox when the
            image is missing. All centred on (x, y) and rotated by R(yaw) on (x, y)
            (Image.rotate(-yaw) on the y-down image == R(yaw); see utils.bbox_corners).

free = floors - (walls, doors, props). Mesh footprints are traced on the full-resolution
image, grown by half a pixel + simplify_m and simplified with tolerance simplify_m, so
they still contain every obstacle pixel (conservative) with few vertices; discs use a
circumscribed 64-gon.

The BEV raster is only a cross-check, sampled at pixel centres. It is itself offset by up to
one pixel from its .yaml frame (PIL draws thick lines and rectangles on pixel indices), and
stamps mesh footprints at 2.5 cm after rounding their position, so a band of about one pixel
around walls and props always differs. ``unsafe_tol_m2`` counts vector-free area more than
one pixel inside a BEV obstacle; only that one flags a map (> UNSAFE_TOL_FLAG_M2).
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import shapely
from scipy import ndimage
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from ..core.svgmap import SCHEMA_VERSION, SvgMap
from . import isaac_scene_builder as ISB

log = logging.getLogger(__name__)

DEFAULTS = {"geometry": "vector", "mesh_dir": None, "wall_m": 0.2, "door_gap": [0.35, 0.65],
            "cell_m": 3.0, "simplify_m": 0.01}
MIN_PART_M2 = 0.25              # free parts smaller than this are dropped (as vectorize does)
UNSAFE_TOL_FLAG_M2 = 0.1        # flag maps whose vector free area overlaps BEV obstacles this much

_assets: dict = {}              # (path, mtime, cell, simplify) -> footprint in the asset frame
_scenes: dict = {}              # scene key -> (rings, stats): computed once per build for all classes


# ------------------------------------------------------------------ geometry

def _mask_polygons(mask: np.ndarray, px: float, cx: float, cy: float):
    """Polygons (with holes) traced through the centres of the True pixels; pixel (i, j)
    has its centre at ((j + .5) px - cx, (i + .5) px - cy)."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []

    def pts(c):
        p = c[:, 0, :].astype(float)
        return np.stack([(p[:, 0] + 0.5) * px - cx, (p[:, 1] + 0.5) * px - cy], 1)

    out = []
    for i, c in enumerate(cs):
        if hier[0][i][3] != -1:
            continue                                        # a hole, added with its parent
        outer = pts(c)
        holes = [pts(cs[k]) for k in range(len(cs)) if hier[0][k][3] == i and len(cs[k]) >= 3]
        if len(outer) < 3:                                  # 1-2 pixel blob
            out.append(Point(outer.mean(axis=0)).buffer(px / 2, cap_style=3))
            continue
        out.append(Polygon(outer, holes).buffer(0))
    return out


def asset_footprint(path: Path, cell_m: float, simplify_m: float):
    """Footprint of a mesh asset in its own frame (metres, centred, y as in the image)."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, cell_m, simplify_m)
    if key in _assets:
        return _assets[key]
    img = np.asarray(Image.open(path).convert("L"))
    S = img.shape[0]
    px = cell_m / S
    geom = unary_union(_mask_polygons(img < 128, px, cell_m / 2, cell_m / 2))
    if not geom.is_empty:
        geom = geom.buffer(px / 2 + simplify_m, join_style=2, mitre_limit=2.0)
        if simplify_m > 0:
            geom = geom.simplify(simplify_m, preserve_topology=True)
    _assets[key] = geom
    return geom


def _place(geom, x, y, yaw_deg):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return affinity.affine_transform(geom, [c, -s, s, c, x, y])


def _f(row, key, default=0.0):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else default


def scene_geometry(csv_path: Path, p: dict):
    """(floor, obstacles, free) shapely geometries of one scene."""
    cell, half_wall = float(p["cell_m"]), float(p["wall_m"]) / 2
    g0, g1 = p["door_gap"]
    mesh_dir = Path(p["mesh_dir"])
    floors, obst = [], []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        kind = r["kind"]
        x, y = _f(r, "x"), _f(r, "y")
        if kind == "floor":
            floors.append(box(x - cell / 2, y - cell / 2, x + cell / 2, y + cell / 2))
        elif kind in ("wall", "door"):
            x2, y2 = _f(r, "x2"), _f(r, "y2")
            spans = [(0.0, 1.0)] if kind == "wall" else [(0.0, g0), (g1, 1.0)]
            for t0, t1 in spans:
                seg = LineString([(x + (x2 - x) * t0, y + (y2 - y) * t0),
                                  (x + (x2 - x) * t1, y + (y2 - y) * t1)])
                obst.append(seg.buffer(half_wall, cap_style=2, join_style=2))
        elif kind == "prop":
            shape, yaw = (r.get("shape") or "bbox"), _f(r, "yaw")
            png = mesh_dir / f"{r.get('name', '')}.png"
            if shape == "sphere":
                rr = _f(r, "r") / math.cos(math.pi / 64)          # circumscribed 64-gon
                obst.append(Point(x, y).buffer(rr, quad_segs=16))
            elif shape == "mesh" and r.get("name") and png.exists():
                fp = asset_footprint(png, cell, float(p["simplify_m"]))
                if not fp.is_empty:
                    obst.append(_place(fp, x, y, yaw))
            else:
                w, h = _f(r, "w"), _f(r, "h")
                obst.append(_place(box(-w / 2, -h / 2, w / 2, h / 2), x, y, yaw))
    floor = unary_union(floors)
    obstacles = unary_union(obst) if obst else Polygon()
    free = floor.difference(obstacles)
    parts = [g for g in getattr(free, "geoms", [free]) if g.area >= MIN_PART_M2]
    return floor, obstacles, MultiPolygon(parts) if parts else Polygon()


def geometry_rings(geom) -> list[np.ndarray]:
    rings = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        rings.append(np.asarray(poly.exterior.coords, float)[:-1])
        rings.extend(np.asarray(h.coords, float)[:-1] for h in poly.interiors)
    return rings


def compare_to_bev(free, root: Path, scene_id: str) -> dict:
    """The vector free region against the BEV raster (PGM == free), at PGM pixel centres:
    iou, unsafe (vector free, BEV not), lost (BEV free, vector not), and both beyond a
    one-pixel tolerance (``*_tol_m2``)."""
    img, res, ox, oy = ISB.read_bev(root, scene_id)
    H, W = img.shape
    src = img == ISB.FREE                                 # row r <-> y = -oy - (H - r - .5) res
    X, Y = np.meshgrid(ox + (np.arange(W) + 0.5) * res, -oy - (H - np.arange(H) - 0.5) * res)
    vec = shapely.contains_xy(free, X, Y)
    k = np.ones((3, 3), bool)
    px = res * res
    inter, union = (vec & src).sum(), (vec | src).sum()
    return {"iou": round(float(inter / max(union, 1)), 5),
            "unsafe_m2": round(float((vec & ~src).sum() * px), 4),
            "unsafe_tol_m2": round(float((vec & ndimage.binary_erosion(~src, k)).sum() * px), 4),
            "lost_m2": round(float((~vec & src).sum() * px), 4),
            "lost_tol_m2": round(float((~vec & ndimage.binary_erosion(src, k)).sum() * px), 4),
            "bev_free_m2": round(float(src.sum() * px), 3)}


# ------------------------------------------------------------------ stage

def scene_map(src: dict, scene_id: str, hclass: str, height: float,
              meta_src: dict | None = None) -> tuple[SvgMap, dict]:
    root = Path(src["path"])
    csv_path = root / "bev" / scene_id / f"{scene_id}.csv"
    key = (str(csv_path), csv_path.stat().st_mtime_ns, tuple(sorted(
        (k, str(src[k])) for k in DEFAULTS)))
    if key not in _scenes:
        _, _, free = scene_geometry(csv_path, src)
        rings = geometry_rings(free)
        stats = {"free_m2": round(float(free.area), 3), **compare_to_bev(free, root, scene_id)}
        if len(_scenes) >= 4096:                          # rings are small; bound anyway
            _scenes.clear()
        _scenes[key] = (rings, stats)
    rings, stats = _scenes[key]
    name = ISB.scene_name(src, scene_id)
    # the BEV is the coarser of the two, so only a real overlap with its obstacles flags
    flags = ([f"unsafe_tol {stats['unsafe_tol_m2']:.2f} m2"]
             if stats["unsafe_tol_m2"] > UNSAFE_TOL_FLAG_M2 else [])
    meta = {
        "schema_version": SCHEMA_VERSION, "map_id": f"{name}_s0", "scene": name, "storey": 0,
        "floor_z": 0.0, "ceil_z": None, "height_class": hclass,
        "z_band": [0.05, round(height, 4)],
        "frame": {"up": 2, "u_axis": "X", "v_axis": "Y", "right_handed": True, "units": "m",
                  "note": ISB.FRAME_NOTE},
        "vectorize": {"geometry": "csv", "simplify_m": src["simplify_m"], "n_rings": len(rings),
                      "n_vertices": int(sum(len(r) for r in rings)), **stats},
        "flags": flags,
        "provenance": {"tool": "hm3denv.build.isb_vector", "created": time.strftime("%Y-%m-%d %H:%M"),
                       "sources": [str(csv_path), str(Path(src["mesh_dir"])),
                                   str(root / "bev" / scene_id / f"{scene_id}.pgm")]},
        "source": {"type": ISB.TYPE, "scene_id": scene_id,
                   **{k: (meta_src or {}).get(k) for k in ISB.META_KEYS if k in (meta_src or {})}},
    }
    return SvgMap(rings, meta), stats


def run(src: dict, out_dir: Path, hclass: str, height: float, scene_ids,
        min_free_m2: float = 2.0) -> dict:
    """Write out_dir/<map_id>.svg for every scene; returns {map_id: stats + flags}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = ISB.catalog(Path(src["path"]))
    made = {}
    for sid in scene_ids:
        m, st = scene_map(src, sid, hclass, height, cat.get(sid))
        if st["free_m2"] < min_free_m2:
            continue
        m.save(out_dir / f"{m.meta['map_id']}.svg")
        made[m.meta["map_id"]] = {**m.meta["vectorize"], "flags": m.meta["flags"]}
        log.info("  %s: %d rings, %d vertices, IoU vs BEV %.4f, unsafe >1px %.0f cm2, lost %.2f m2%s",
                 m.meta["map_id"], len(m.rings), m.meta["vectorize"]["n_vertices"], st["iou"],
                 st["unsafe_tol_m2"] * 1e4, st["lost_m2"],
                 f"  FLAG {m.meta['flags']}" if m.meta["flags"] else "")
    prefix = f"{src['prefix']}-"
    for old in out_dir.glob(f"{prefix}*.svg"):
        if old.stem not in made:
            old.unlink()
    return made


def input_files(src: dict, scene_ids) -> list[Path]:
    root = Path(src["path"])
    files = [root / "bev" / i / f"{i}.{ext}" for i in scene_ids for ext in ("csv", "pgm", "yaml")]
    mesh = Path(src["mesh_dir"])
    return files + (sorted(mesh.glob("*.png")) if mesh.exists() else [])
