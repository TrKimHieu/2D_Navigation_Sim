"""Grid stage: turn the 2 cm storey maps of the slice stage into robot grids.

Each square cell is one robot body; the robot moves in 4 directions. Only 2D images
are used (no mesh), so re-running for another robot size is cheap.

Cell states:
    wall     >= 1 obstacle pixel (after dilating by `margin`); conservative: a robot
             centred in the cell never touches an obstacle
    nofloor  floor-covered fraction < floor_cover: stairwells, outside, large holes
    noroof   floor present but roof-covered fraction < roof_cover: terraces, balconies
    free     everything else

The grid origin decides whether narrow doors stay open (a 0.8 m door with 0.34 m
cells can close when out of phase), so 16 origin phases are tried and the one with
the largest 4-connected component wins. That component is "indoors", the only
place where tasks are sampled.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import paths
from ..robots import cell_px, height_tag
from .slice import is_excluded

log = logging.getLogger(__name__)

FREE, WALL, NOFLOOR, NOROOF = 0, 1, 2, 3
CROSS = ndimage.generate_binary_structure(2, 1)     # 4-connectivity


@dataclass
class GridParams:
    margin: float = 0.04             # safety margin around obstacles (m)
    floor_close: float = 0.10        # closing radius of the floor/roof masks (m)
    max_hole_m2: float = 1.0         # fill enclosed floor holes smaller than this
    floor_cover: float = 0.9         # min floor fraction of a non-nofloor cell
    roof_cover: float = 0.9          # min roof fraction of a non-noroof cell
    roof: bool = True                # False: allow outdoor terraces/balconies
    min_area_m2: float = 10.0        # drop storeys whose main region is smaller
    flag_main_frac: float = 0.6      # flag storeys whose main region is a small share


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def clean_mask(mask, res, close_m, max_hole_m2):
    """Small closing to join speckles (surface sampling misses ~2% of pixels), then fill
    ENCLOSED holes smaller than max_hole_m2. Floor: scan holes on glass/occluded floor are
    filled, stairwells are large enough to stay. Roof: ceiling scan holes and lamps are
    filled. Outside regions touch the image border, so they are not enclosed holes."""
    m = mask.copy()
    rc = int(round(close_m / res))
    if rc > 0:
        # pad so the closing does not erode the image border
        m = ndimage.binary_closing(np.pad(m, rc), disk(rc))[rc:-rc, rc:-rc]
    holes = ndimage.binary_fill_holes(m) & ~m
    lab, n = ndimage.label(holes, CROSS)
    if n:
        small = np.bincount(lab.ravel()) * res * res < max_hole_m2
        small[0] = False
        m |= small[lab]
    return m


def prepare(obstacle, floor, res, margin_m, floor_close_m, max_hole_m2):
    """Dilate obstacles by the safety margin; clean the floor mask."""
    r = int(round(margin_m / res))
    obst = ndimage.binary_dilation(obstacle, disk(r)) if r > 0 else obstacle.copy()
    return obst, clean_mask(floor, res, floor_close_m, max_hole_m2)


def cell_states(obst, floor, k, oy, ox, floor_cover, roof=None, roof_cover=0.9):
    """Split the image into k*k px cells with the grid origin shifted by (oy, ox) px.

    The image is padded by k px on every side (no obstacle, no floor, no roof) so every
    phase covers it. Cell (r, c) covers source pixels [oy - k + r*k, oy + r*k).
    `roof` None skips the roof test.
    """
    H, W = obst.shape
    R, C = (H + 2 * k - oy) // k, (W + 2 * k - ox) // k

    def blocks(a):
        p = np.zeros((H + 2 * k, W + 2 * k), bool)
        p[k:k + H, k:k + W] = a
        return p[oy:oy + R * k, ox:ox + C * k].reshape(R, k, C, k)

    wall = blocks(obst).any(axis=(1, 3))
    cover = blocks(floor).mean(axis=(1, 3))
    state = np.full((R, C), FREE, np.uint8)
    if roof is not None:
        state[blocks(roof).mean(axis=(1, 3)) < roof_cover] = NOROOF
    state[cover < floor_cover] = NOFLOOR
    state[wall] = WALL
    return state


def largest_component(free):
    lab, n = ndimage.label(free, CROSS)
    if n == 0:
        return np.zeros_like(free)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def phases(k):
    return sorted({int(round(k * f)) % k for f in (0, 0.25, 0.5, 0.75)})


def choose_offset(obst, floor, k, floor_cover, roof=None, roof_cover=0.9):
    """Origin phase with the largest connected free component; ties keep the smaller phase."""
    best = None
    for oy in phases(k):
        for ox in phases(k):
            st = cell_states(obst, floor, k, oy, ox, floor_cover, roof, roof_cover)
            score = int(largest_component(st == FREE).sum())
            if best is None or score > best[0]:
                best = (score, oy, ox, st)
    return best[1], best[2], best[3]


def gridify(obstacle, floor, res, k, margin_m, floor_close_m=0.10,
            max_hole_m2=1.0, floor_cover=0.9, roof=None, roof_cover=0.9):
    """(cropped state, cropped main region, offset/crop/stats). `roof` None skips the roof test."""
    obst, fl = prepare(obstacle, floor, res, margin_m, floor_close_m, max_hole_m2)
    if roof is not None:
        roof = clean_mask(roof, res, floor_close_m, max_hole_m2)
    oy, ox, state = choose_offset(obst, fl, k, floor_cover, roof, roof_cover)
    free = state == FREE
    main = largest_component(free)

    info = {"k": k, "oy": oy, "ox": ox, "r0": 0, "c0": 0,
            "free_cells": int(free.sum()), "main_cells": int(main.sum())}
    info["main_frac"] = info["main_cells"] / max(info["free_cells"], 1)
    if not main.any():
        return state, main, info

    # crop to the main region + a 1-cell border
    rows, cols = np.where(main.any(axis=1))[0], np.where(main.any(axis=0))[0]
    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, state.shape[0])
    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, state.shape[1])
    info.update(r0=int(r0), c0=int(c0))
    return state[r0:r1, c0:c1], main[r0:r1, c0:c1], info


# ------------------------------------------------------------------ batch

def grid_robot(raw: dict, robot_size: float | None = None) -> dict:
    """Robot preset used for gridding. `robot_size` overrides the footprint with a square
    of that side and yields a derived preset `s<cm>_<height tag>` (e.g. s15_h63)."""
    if robot_size is None:
        return raw
    d = copy.deepcopy(raw)
    d["id"] = f"s{int(round(robot_size * 100))}_{height_tag(raw['height'])}"
    d["name"] = f"{raw['name']} as a {robot_size * 100:.0f} cm square"
    d["footprint"] = {"type": "rectangle", "length": robot_size, "width": robot_size}
    d.pop("footprint_simple", None)
    d["sources"] = {**d["sources"], "footprint": {
        "kind": "estimated", "method": f"grid override robot_size={robot_size} m of preset {raw['id']}"}}
    d["derived_from"] = raw["id"]
    return d


def storeys(slice_root: Path, scenes=None):
    for sj in sorted(Path(slice_root).glob("*/maps/storey*.json")):
        scene = sj.parent.parent.name
        if scenes and scene not in scenes:
            continue
        yield scene, sj


def run(ws: Path, out_dir: Path, robot: dict, scenes=None, params: GridParams | None = None) -> dict:
    """Grid every sliced storey for `robot` (a preset dict, see grid_robot) into
    `out_dir/<map_id>.npz|.json`. Returns {map_id: meta}; stale grids are removed."""
    p = params or GridParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    htag = height_tag(robot["height"])
    scenes = set(scenes) if scenes else None
    made, skipped = {}, 0

    def drop(map_id):
        for f in out_dir.glob(f"{map_id}.*"):
            f.unlink()

    for scene, sj in storeys(paths.slice_dir(ws), scenes):
        smeta = json.loads(sj.read_text(encoding="utf-8"))
        k_st = smeta["storey"]
        map_id = f"{scene}_s{k_st}"
        if is_excluded(sj.parent.parent):
            drop(map_id)
            skipped += 1
            continue
        obst_png = sj.parent / f"storey{k_st}_{htag}.png"
        floor_png = sj.parent / f"storey{k_st}_floor.png"
        roof_png = sj.parent / f"storey{k_st}_roof_{htag}.png"
        need = [obst_png, floor_png] + ([roof_png] if p.roof else [])
        if not all(f.exists() for f in need):
            log.warning("  %s: missing %s; slice with robot_height=%s first", map_id,
                        [f.name for f in need if not f.exists()], robot["height"])
            drop(map_id)
            skipped += 1
            continue

        res = smeta["res"]
        k = cell_px(robot, res)
        obstacle = np.asarray(Image.open(obst_png)) == 0
        floor = np.asarray(Image.open(floor_png)) > 0
        roof = np.asarray(Image.open(roof_png)) > 0 if p.roof else None
        state, main, info = gridify(obstacle, floor, res, k, p.margin, p.floor_close,
                                    p.max_hole_m2, p.floor_cover, roof, p.roof_cover)
        cell_m = k * res
        main_m2 = info["main_cells"] * cell_m ** 2
        if main_m2 < p.min_area_m2:
            log.info("  %s: main region %.1f m2 < %s, dropped", map_id, main_m2, p.min_area_m2)
            drop(map_id)
            skipped += 1
            continue

        flags = [f"main_frac {info['main_frac']:.2f}"] if info["main_frac"] < p.flag_main_frac else []
        meta = {**smeta, **info, "map_id": map_id, "robot": robot["id"], "cell_m": cell_m,
                "margin": p.margin, "floor_close": p.floor_close, "max_hole_m2": p.max_hole_m2,
                "floor_cover": p.floor_cover, "obstacle_png": obst_png.name,
                "roof_check": p.roof, "roof_cover": p.roof_cover,
                "noroof_cells": int((state == NOROOF).sum()),
                "shape": list(state.shape), "main_m2": round(main_m2, 2), "flags": flags}
        np.savez_compressed(out_dir / f"{map_id}.npz", grid=(state != FREE).astype(np.uint8),
                            state=state, main=main)
        (out_dir / f"{map_id}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        made[map_id] = meta
        log.info("  %s: %dx%d cells, offset (%d,%d) px, main %.0f m2 (%.0f%% of free)%s", map_id,
                 *state.shape, info["oy"], info["ox"], main_m2, info["main_frac"] * 100,
                 f"  FLAG {flags}" if flags else "")

    # remove grids of storeys that no longer exist (review merged/dropped storeys);
    # a run filtered by scene only cleans those scenes
    for old in out_dir.glob("*.npz"):
        scene = old.stem.rsplit("_s", 1)[0]
        if old.stem not in made and (not scenes or scene in scenes):
            drop(old.stem)
    log.info("gridify %s: %d grids, %d skipped", robot["id"], len(made), skipped)
    return made


def params_dict(p: GridParams) -> dict:
    return asdict(p)
