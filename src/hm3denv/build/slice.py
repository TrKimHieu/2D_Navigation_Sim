"""Slice stage: cut each HM3D scene (GLB) into per-storey 2D maps for a robot height.

  1. Sweep bottom-up in layers of `step` and count occupied pixels per layer. A layer
     cutting a floor slab holds the whole floor surface (40-80% of the footprint)
     while a layer between two floors holds 3-9%, so storeys separate cleanly.
  2. A 20-30 cm slab yields TWO adjacent peaks (ceiling below + floor above); they are
     merged into one slab, otherwise each slab would be counted as two storeys.
  3. Per storey, everything inside [floor + floor_margin, floor + robot_height] is
     rasterised as obstacle (walls, furniture, lowest stair steps). The band is cut at
     the true heights, not the `step` layers, so its edges are exact.
  4. Only real material is marked: map = not obstacle. The floor is NOT used to block
     cells here (scan holes on glass or reflective floors would become walls in the
     middle of rooms); floor and roof masks are written separately for later stages.

Outputs per scene in ``<workspace>/stages/slice/<scene>/``:
    maps/storeyK_hXX.png       obstacle map (white = free) for robot height XX cm
    maps/storeyK_floor.png     raw floor mask (horizontal surfaces at floor height)
    maps/storeyK_roof_hXX.png  roof mask (any surface above the robot top)
    maps/storeyK.json          frame: up axis, origin, resolution, floor/ceiling heights
    auto.json, profile.csv, footprint.png, slices/   (inputs of the review tool)
A ``review.json`` written by ``hm3d review`` overrides the automatic axis and storeys.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import paths
from ..robots import height_tag

log = logging.getLogger(__name__)

CHUNK = 1_000_000                    # sampled points per batch (bounds memory)
FREE, BLOCKED, UNKNOWN = 255, 0, 128


@dataclass
class SliceParams:
    robot_height: float = 0.626
    floor_margin: float = 0.05       # ignore the layer right above the floor surface
    ceil_margin: float = 0.10        # ignore the layer right below the ceiling
    step: float = 0.10               # layer thickness for storey detection (m)
    res: float = 0.02                # map resolution (m/pixel)
    oversample: float = 4.0          # mean surface samples per pixel
    min_ratio: float = 0.30          # min occupied fraction for a layer to be a slab
    min_storey_m: float = 1.8        # min height of a storey
    floor_band: float = 0.08         # half thickness of the floor-surface band (m)
    floor_close_m: float = 0.10      # only for floor_mask (legacy review option)
    floor_mask: bool = False         # legacy: block cells without floor (review tool toggle)
    seed: int = 0
    up: int = -1                     # up axis 0/1/2; -1 = detect
    slices: bool = False             # write per-layer images (needed by the review tool)
    use_review: bool = True          # apply review.json when present


# ------------------------------------------------------------------ mesh

def load_mesh(glb_path: Path, cache: Path | None = None):
    """(vertices, faces), cached as .npz because reading a GLB is slow."""
    cache = cache or glb_path.parent
    npz = cache / (glb_path.stem + ".npz")
    if npz.exists():
        d = np.load(npz)
        return d["V"], d["F"]
    import trimesh  # only when a GLB really has to be read
    # process=False skips vertex merging, skip_materials skips texture decoding:
    # 0.4 s instead of ~140 s per scene
    m = trimesh.load(glb_path, force="mesh", process=False, skip_materials=True)
    V = np.asarray(m.vertices, dtype=np.float32)
    F = np.asarray(m.faces, dtype=np.int32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, V=V, F=F)
    return V, F


def scene_mesh(ws: Path, scene: str):
    return load_mesh(paths.glb_dir(ws) / f"{scene}.glb", paths.cache_dir(ws))


def face_areas_normals(V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    return ln / 2.0, n / np.maximum(ln, 1e-12)[:, None]


def detect_up_axis(V, F, bin_m=0.10) -> int:
    """The up axis is the one whose perpendicular faces are the most CLUSTERED in height.

    Raw Matterport GLBs are Z-up (habitat uses Y-up), so no axis can be assumed.
    "Largest perpendicular area" is a poor criterion: walls are perpendicular to the
    horizontal axes too, so X and Y each get ~30% and Z wins by only ~1.2x. Floors and
    ceilings, unlike walls, pile up at a few discrete heights; the share of area in
    the densest height bin gives a 2-8x margin.
    """
    area, nrm = face_areas_normals(V, F)
    cen = V[F].mean(axis=1)
    score = []
    for a in range(3):
        m = np.abs(nrm[:, a]) > 0.9
        if not m.any():
            score.append(0.0)
            continue
        c, w = cen[m, a], area[m]
        h, _ = np.histogram(c, bins=np.arange(c.min(), c.max() + bin_m + 1e-9, bin_m), weights=w)
        score.append(float(h.max() / w.sum()))
    return int(np.argmax(score))


def sample_faces(tri, area, n_pts, rng):
    """Yield batches of points uniformly distributed by area over the given triangles."""
    if len(tri) == 0 or n_pts <= 0:
        return
    cum = np.cumsum(area)
    cum /= cum[-1]
    done = 0
    while done < n_pts:
        k = min(CHUNK, n_pts - done)
        idx = np.searchsorted(cum, rng.random(k))
        a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
        r1 = np.sqrt(rng.random((k, 1), dtype=np.float32))    # uniform barycentric
        r2 = rng.random((k, 1), dtype=np.float32)
        yield a * (1 - r1) + b * (r1 * (1 - r2)) + c * (r1 * r2)
        done += k


class Frame:
    """Mapping between 3D coordinates and pixel indices of the storey images."""

    def __init__(self, V, up, res):
        self.up = up
        self.ua, self.va = [a for a in range(3) if a != up]
        self.res = res
        self.lo = V.min(axis=0)
        self.hi = V.max(axis=0)
        self.W = int(np.ceil((self.hi[self.ua] - self.lo[self.ua]) / res)) + 1
        self.H = int(np.ceil((self.hi[self.va] - self.lo[self.va]) / res)) + 1

    def pixels(self, P):
        ci = ((P[:, self.ua] - self.lo[self.ua]) / self.res).astype(np.int32)
        ri = ((P[:, self.va] - self.lo[self.va]) / self.res).astype(np.int32)
        np.clip(ci, 0, self.W - 1, out=ci)
        np.clip(ri, 0, self.H - 1, out=ri)
        return self.H - 1 - ri, ci          # flip rows so +v points up in the image


# ------------------------------------------------------------------ layers

def build_layers(V, F, frame, step, pts_per_m2, seed=0):
    """3D occupancy (layer, row, col), True = material. Area-weighted sampling, so
    point density reflects real surface area (floor slabs dark, wall cuts thin)."""
    up = frame.up
    L = int(np.ceil((frame.hi[up] - frame.lo[up]) / step)) + 1
    grids = np.zeros((L, frame.H, frame.W), dtype=bool)
    area, _ = face_areas_normals(V, F)
    rng = np.random.default_rng(seed)
    for P in sample_faces(V[F], area, int(float(area.sum()) * pts_per_m2), rng):
        li = np.clip(((P[:, up] - frame.lo[up]) / step).astype(np.int32), 0, L - 1)
        ri, ci = frame.pixels(P)
        flat = li.astype(np.int64) * (frame.H * frame.W) + ri.astype(np.int64) * frame.W + ci
        grids.reshape(-1)[flat] = True
    return grids, frame.lo[up] + np.arange(L) * step


def band_mask(V, F, frame, z_lo, z_hi, pts_per_m2, horizontal_only=False, seed=0):
    """Project everything inside the height band [z_lo, z_hi) onto the plane. Cut at the
    true heights (not the `step` layers); triangles are pre-filtered by height range so
    samples are not wasted on the rest of the mesh."""
    up = frame.up
    tri_all = V[F]
    tz = tri_all[:, :, up]
    keep = (tz.max(axis=1) >= z_lo) & (tz.min(axis=1) <= z_hi)
    if horizontal_only:
        _, nrm = face_areas_normals(V, F)
        keep &= np.abs(nrm[:, up]) > 0.7
    mask = np.zeros((frame.H, frame.W), dtype=bool)
    if not keep.any():
        return mask
    tri = tri_all[keep]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0
    if area.sum() <= 0:
        return mask
    rng = np.random.default_rng(seed)
    for P in sample_faces(tri, area, int(float(area.sum()) * pts_per_m2), rng):
        P = P[(P[:, up] >= z_lo) & (P[:, up] < z_hi)]
        if not len(P):
            continue
        ri, ci = frame.pixels(P)
        mask[ri, ci] = True
    return mask


def refine_surface_z(V, F, up, z0, z1):
    """True height of the floor/ceiling surface inside layer [z0, z1]: the area-weighted
    median of horizontal faces. A layer edge can be off by tens of cm, which would shift
    the robot band below the floor and turn the floor itself into an obstacle."""
    area, nrm = face_areas_normals(V, F)
    c = V[F][:, :, up].mean(axis=1)
    m = (np.abs(nrm[:, up]) > 0.7) & (c >= z0) & (c <= z1)
    if not m.any():
        return None
    z, w = c[m], area[m]
    o = np.argsort(z)
    cw = np.cumsum(w[o])
    return float(z[o][np.searchsorted(cw, cw[-1] / 2)])


def find_slabs(ratio, heights, min_ratio, max_slab_gap=0.6):
    """Group dense layers into slabs (the two faces of one slab form a single group)."""
    dense = [i for i, r in enumerate(ratio) if r >= min_ratio]
    slabs = []
    for i in dense:
        if slabs and heights[i] - heights[slabs[-1][-1]] <= max_slab_gap:
            slabs[-1].append(i)
        else:
            slabs.append([i])
    return slabs


# ------------------------------------------------------------------ maps

def build_map(V, F, frame, floor_z, ceil_z, p: SliceParams, pts_per_m2):
    """Obstacle map of one storey (FREE / BLOCKED, UNKNOWN only with floor_mask)."""
    z_lo = floor_z + p.floor_margin
    z_hi = min(floor_z + p.robot_height, ceil_z - p.ceil_margin)
    obstacle = band_mask(V, F, frame, z_lo, z_hi, pts_per_m2, seed=p.seed)
    out = np.full((frame.H, frame.W), FREE, np.uint8)
    out[obstacle] = BLOCKED
    if p.floor_mask:
        floor = band_mask(V, F, frame, floor_z - p.floor_band, floor_z + p.floor_band,
                          pts_per_m2, horizontal_only=True, seed=p.seed)
        r = max(1, int(round(p.floor_close_m / p.res)))
        out[~ndimage.binary_closing(floor, np.ones((r, r)))] = UNKNOWN
    px = p.res ** 2
    return out, {"z_lo": z_lo, "z_hi": z_hi, "free_m2": float((out == FREE).sum() * px),
                 "blocked_m2": float((out == BLOCKED).sum() * px)}


def save_map(m, path):
    """Black/white: UNKNOWN is merged into BLOCKED (not drivable)."""
    Image.fromarray(np.where(m == FREE, 255, 0).astype(np.uint8), "L").save(path)


def save_bw(mask, path):
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), "L").save(path)


def save_storey_extras(V, F, frame, k, fz, cz, p: SliceParams, maps_dir: Path, scene: str):
    """Floor mask, roof mask and frame of a storey (inputs of the grid/vector stages).

    The floor mask and frame do not depend on the robot height (no hXX suffix); the roof
    mask does. Neither mask draws obstacles; they decide where a robot may stand:
    real floor below, some surface above (indoors). The roof test has no height limit
    because double-height rooms have ceilings above ceil_z.
    """
    pts_per_m2 = p.oversample / (p.res ** 2)
    floor = band_mask(V, F, frame, fz - p.floor_band, fz + p.floor_band, pts_per_m2,
                      horizontal_only=True, seed=p.seed)
    Image.fromarray(np.where(floor, 255, 0).astype(np.uint8), "L").save(maps_dir / f"storey{k}_floor.png")
    roof = band_mask(V, F, frame, fz + p.robot_height, frame.hi[frame.up] + 1.0, pts_per_m2, seed=p.seed)
    Image.fromarray(np.where(roof, 255, 0).astype(np.uint8), "L").save(
        maps_dir / f"storey{k}_roof_{height_tag(p.robot_height)}.png")
    meta = {"scene": scene, "storey": k, "up": frame.up, "ua": frame.ua, "va": frame.va,
            "lo": [float(x) for x in frame.lo], "hi": [float(x) for x in frame.hi],
            "res": frame.res, "W": frame.W, "H": frame.H, "floor_z": float(fz), "ceil_z": float(cz),
            "floor_band": p.floor_band, "floor_margin": p.floor_margin, "ceil_margin": p.ceil_margin}
    (maps_dir / f"storey{k}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def build_storeys(V, F, frame, items, p: SliceParams, scene, maps_dir: Path, rows: list, t0: float):
    """Write the maps of every decided storey (automatic or reviewed)."""
    tag = height_tag(p.robot_height)
    pts_per_m2 = p.oversample / (p.res ** 2)
    # the storey count may change between runs (review merges/splits storeys): remove the
    # previous files, otherwise later stages would read storeys that no longer exist
    for pat in (f"storey*_{tag}.png", "storey*_floor.png", "storey*.json"):
        for old in maps_dir.glob(pat):
            old.unlink()
    for k, fz, cz in items:
        m, st = build_map(V, F, frame, fz, cz, p, pts_per_m2)
        name = f"storey{k}_{tag}.png"
        save_map(m, maps_dir / name)
        save_storey_extras(V, F, frame, k, fz, cz, p, maps_dir, scene)
        tot = st["free_m2"] + st["blocked_m2"]
        log.info("  storey %d: floor %+.2f ceiling %+.2f (%.2f m) | band %+.2f..%+.2f | free %.0f m2 "
                 "(%.0f%%) -> %s", k, fz, cz, cz - fz, st["z_lo"], st["z_hi"], st["free_m2"],
                 st["free_m2"] / max(tot, 1e-9) * 100, name)
        rows.append({"scene": scene, "storey": k, "floor_z": round(fz, 3), "ceil_z": round(cz, 3),
                     "storey_h": round(cz - fz, 3), "robot_h": p.robot_height,
                     "band_lo": round(st["z_lo"], 3), "band_hi": round(st["z_hi"], 3),
                     "res_m": p.res, "W": frame.W, "H": frame.H,
                     "free_m2": round(st["free_m2"], 1), "blocked_m2": round(st["blocked_m2"], 1),
                     "png": f"{scene}/maps/{name}"})
    if not items:
        log.warning("  %s: no storey found", scene)
    log.info("  %d maps (%.1fs)", len(items), time.time() - t0)


def read_review(scene_dir: Path) -> dict | None:
    rp = scene_dir / "review.json"
    return json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None


def is_excluded(scene_dir: Path) -> bool:
    rv = read_review(scene_dir)
    return bool(rv and rv.get("status") == "bad")


def process_scene(glb_path: Path, out: Path, cache: Path, p: SliceParams, rows: list) -> None:
    t0 = time.time()
    scene = glb_path.stem
    scene_dir = out / scene
    maps_dir = scene_dir / "maps"
    V, F = load_mesh(glb_path, cache)
    review = read_review(scene_dir) if p.use_review else None
    if review and review.get("status") == "bad":
        log.info("  %s: excluded by review", scene)
        return
    up = p.up if p.up >= 0 else detect_up_axis(V, F)
    if review and review.get("up") is not None:
        up = int(review["up"])
    frame = Frame(V, up, p.res)
    pts_per_m2 = p.oversample / (p.res ** 2)

    if review and review.get("storeys"):
        # storeys decided by hand: no density analysis needed
        maps_dir.mkdir(parents=True, exist_ok=True)
        items = [(k, float(s["floor_z"]), float(s["ceil_z"])) for k, s in enumerate(review["storeys"], 1)]
        log.info("  %s: %d faces, up=%s (reviewed), %d storeys", scene, len(F), "XYZ"[up], len(items))
        return build_storeys(V, F, frame, items, p, scene, maps_dir, rows, t0)

    grids, heights = build_layers(V, F, frame, p.step, pts_per_m2, p.seed)
    L = len(grids)
    # the review tool uses the detected axis as default for unreviewed scenes
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "auto.json").write_text(json.dumps({"up": up}), encoding="utf-8")
    log.info("  %s: %d faces, up=%s, %dx%d px", scene, len(F), "XYZ"[up], frame.W, frame.H)

    footprint = grids.any(axis=0)
    ratio = grids.reshape(L, -1).sum(axis=1) / max(int(footprint.sum()), 1)
    slabs = find_slabs(ratio, heights, p.min_ratio)
    maps_dir.mkdir(parents=True, exist_ok=True)
    save_bw(footprint, scene_dir / "footprint.png")
    is_slab = {i for g in slabs for i in g}
    with open(scene_dir / "profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "z_m", "black_px", "black_ratio", "is_slab"])
        for i in range(L):
            w.writerow([i, f"{heights[i]:.3f}", int(grids[i].sum()), f"{ratio[i]:.4f}", int(i in is_slab)])
    if p.slices:
        sdir = scene_dir / "slices"
        sdir.mkdir(parents=True, exist_ok=True)
        # file names carry the height, so a new axis or step would leave stale images
        for old in list(sdir.glob("L*.png")) + list((scene_dir / ".thumbs").glob("*.png")):
            old.unlink()
        for i in range(L):
            save_bw(grids[i], sdir / f"L{i:03d}_z{heights[i]:+.2f}.png")
    del grids                                            # the 3D grid is large

    storeys = [(g[-1], h[0]) for g, h in zip(slabs, slabs[1:])]
    if slabs and heights[-1] - heights[slabs[-1][-1]] >= p.min_storey_m:
        storeys.append((slabs[-1][-1], L - 1))           # top storey; the roof may be sloped
    items = []
    for k, (a, b) in enumerate(storeys, 1):
        fz = refine_surface_z(V, F, up, heights[a], heights[a] + p.step)
        cz = refine_surface_z(V, F, up, heights[b], heights[b] + p.step)
        fz = float(heights[a]) if fz is None else fz
        cz = float(heights[b]) if cz is None else cz
        if cz - fz >= p.min_storey_m:
            items.append((k, fz, cz))
    build_storeys(V, F, frame, items, p, scene, maps_dir, rows, t0)


def up_to_date(scene_dir: Path, tag: str) -> bool:
    """Maps for this height exist and are newer than review.json (if any)."""
    roofs = list((scene_dir / "maps").glob(f"storey*_roof_{tag}.png"))
    if not roofs:
        return False
    rp = scene_dir / "review.json"
    return not rp.exists() or rp.stat().st_mtime <= min(r.stat().st_mtime for r in roofs)


def run(ws: Path, scenes=None, params: SliceParams | None = None, force: bool = False) -> list[dict]:
    """Slice the given scenes (default: every GLB) for one robot height. Scenes already
    sliced for this height (and not reviewed since) are skipped unless `force`."""
    p = params or SliceParams()
    ws = Path(ws)
    out = paths.slice_dir(ws)
    out.mkdir(parents=True, exist_ok=True)
    glbs = sorted(paths.glb_dir(ws).glob("*.glb"))
    if scenes:
        want = set(scenes)
        glbs = [g for g in glbs if g.stem in want]
        missing = want - {g.stem for g in glbs}
        if missing:
            raise FileNotFoundError(f"GLB not found for {sorted(missing)} in {paths.glb_dir(ws)}")
    tag = height_tag(p.robot_height)
    rows = []
    todo = [g for g in glbs if force or not (up_to_date(out / g.stem, tag) or is_excluded(out / g.stem))]
    log.info("slice %s: %d scenes (%d up to date)", tag, len(todo), len(glbs) - len(todo))
    for i, g in enumerate(todo, 1):
        try:
            process_scene(g, out, paths.cache_dir(ws), p, rows)
        except Exception as e:                     # one broken scene must not stop the batch
            log.error("  %s failed: %s: %s", g.stem, type(e).__name__, e)
        if i % 25 == 0:
            log.info("---- %d/%d scenes ----", i, len(todo))
    if rows:
        index = out / "index.csv"
        new = not index.exists()
        with open(index, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
    (out / f"params_{tag}.json").write_text(json.dumps(asdict(p), indent=1), encoding="utf-8")
    return rows
