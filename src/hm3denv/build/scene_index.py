"""Labelled elements of Isaac-Scene-Builder scenes on the robot grids of a grid dataset.

For every grid ``grids/<map_id>.json`` whose scene comes from Isaac-Scene-Builder, writes
``scene/<robot>/<map_id>.json`` (schema ``isb-nav.scene/1``): the rooms, walls, doors and
objects of ``<stem>_objects.json`` expressed in the cells of that grid, so a prompt can say
"a table at cell (4, 5), 3 x 3 cells, shape .#./.#./###".

A cell belongs to the element owning most of the labelled pixels of ``<stem>_ids.png``
whose centre falls in it (the same rule as Isaac-Scene-Builder's ``scene_objects``). Cells
are (row, col) of the grid: row 0 at the top (largest y), ``up`` = row - 1. The frame block
gives the cell <-> metre conversion (metres = the USD world frame of the scene)::

    x = x0 + col * cell_m        y = y0 - row * cell_m       (cell centres)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.gridcore import cell_to_world
from . import isaac_scene_builder as ISB

log = logging.getLogger(__name__)

SCHEMA = "isb-nav.scene/1"


# ------------------------------------------------------------------ geometry

def frame(meta: dict, shape) -> dict:
    """Cell <-> metre conversion of one grid (cell centres)."""
    cell = meta["k"] * meta["res"]
    p = cell_to_world(meta, 0, 0)
    return {"grid": "(row, col); row 0 at the top (largest y), up = row - 1, right = col + 1",
            "world": "x, y metres = USD world frame of the scene; yaw CCW from +x (deg)",
            "cell_m": round(cell, 6), "shape": [int(shape[0]), int(shape[1])],
            "x0": round(p[meta["ua"]], 6), "y0": round(p[meta["va"]], 6),
            "cell_to_world": "x = x0 + col * cell_m, y = y0 - row * cell_m"}


def world_to_cell(fr: dict, x: float, y: float) -> tuple[int, int]:
    """Cell containing the point (x, y); may lie outside the grid."""
    c = math.floor((x - fr["x0"]) / fr["cell_m"] + 0.5)
    r = math.floor((fr["y0"] - y) / fr["cell_m"] + 0.5)
    return r, c


def pixel_owners(ids: np.ndarray, raster: dict, fr: dict) -> dict[tuple, int]:
    """{(row, col): element id} - per grid cell, the id with most pixels in the cell."""
    res = raster["resolution_m"]
    ox, oy = raster["origin_m"]
    rr, cc = np.nonzero(ids)
    if not rr.size:
        return {}
    x = ox + (cc + 0.5) * res
    y = oy + (rr + 0.5) * res
    col = np.floor((x - fr["x0"]) / fr["cell_m"] + 0.5).astype(np.int64)
    row = np.floor((fr["y0"] - y) / fr["cell_m"] + 0.5).astype(np.int64)
    H, W = fr["shape"]
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    key = (row[ok] * W + col[ok]) * 65536 + ids[rr[ok], cc[ok]].astype(np.int64)
    keys, counts = np.unique(key, return_counts=True)
    best = {}
    for k, n in zip(keys.tolist(), counts.tolist()):
        cell, eid = divmod(k, 65536)
        if n > best.get(cell, (0, 0))[0]:
            best[cell] = (n, eid)
    return {divmod(cell, W): eid for cell, (_, eid) in best.items()}


def ascii_shape(cells) -> str:
    """'#'/'.' rows of the bounding box of `cells`, joined with '/'."""
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    h = max(r for r, _ in cells) - r0 + 1
    w = max(c for _, c in cells) - c0 + 1
    rows = [["."] * w for _ in range(h)]
    for r, c in cells:
        rows[r - r0][c - c0] = "#"
    return "/".join("".join(row) for row in rows)


def _box(cells) -> dict:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return {"cell_rc": [min(rs), min(cs)], "size": [max(rs) - min(rs) + 1, max(cs) - min(cs) + 1],
            "bbox_rc": [[min(rs), min(cs)], [max(rs), max(cs)]]}


def _inside(px, py, ring) -> np.ndarray:
    """Even-odd point-in-polygon of arrays px, py against one ring [[x, y], ...]."""
    inside = np.zeros(px.shape, bool)
    n = len(ring)
    for i in range(n):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
        if y1 == y2:
            continue
        cross = (py >= min(y1, y2)) & (py < max(y1, y2))
        xs = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (px < xs)
    return inside


# ------------------------------------------------------------------ one map

def build_map(meta: dict, grid: np.ndarray, objs: dict, ids: np.ndarray, map_id: str,
              robot: str) -> dict:
    """The scene index of one grid (see the module docstring)."""
    fr = frame(meta, grid.shape)
    owner = pixel_owners(ids, objs["raster"], fr)
    cells_of: dict[int, list] = {}
    for cell, eid in sorted(owner.items()):
        cells_of.setdefault(eid, []).append([int(cell[0]), int(cell[1])])

    # rooms: free cells whose centre lies in the room outline
    H, W = grid.shape
    rr, cc = np.mgrid[0:H, 0:W]
    px, py = fr["x0"] + cc * fr["cell_m"], fr["y0"] - rr * fr["cell_m"]
    room_of = np.full(grid.shape, -1, np.int32)
    rooms = []
    for i, room in enumerate(objs.get("rooms", [])):
        mask = np.zeros(grid.shape, bool)
        for ring in room["outline"]:
            mask ^= _inside(px, py, ring)
        room_of[mask & (room_of < 0)] = i
        free = np.argwhere(mask & (grid == 0))
        rooms.append({"id": room["id"], "type": room["type"], "style": room.get("style"),
                      "area_m2": room["area_m2"], "free_cells": int(len(free)),
                      **(_box(free.tolist()) if len(free) else {}),
                      "doors": [], "neighbors": [], "objects": []})
    by_room = {r["id"]: r for r in rooms}

    def room_at(cell):
        r, c = cell
        return objs["rooms"][room_of[r, c]]["id"] if 0 <= r < H and 0 <= c < W and room_of[r, c] >= 0 else None

    walls = []
    for w in objs.get("walls", []):
        cells = cells_of.get(w["id"], [])
        walls.append({"id": w["id"], "class": w["class"], "cells_rc": cells,
                      **(_box(cells) if cells else {}),
                      "start_m": w["start"], "end_m": w["end"], "thickness_m": w["thickness"],
                      "faces": w.get("faces")})
    doors, graph = [], []
    for d in objs.get("openings", []):
        cells = cells_of.get(d["id"], [])
        centre = world_to_cell(fr, *d["center"])
        doors.append({"id": d["id"], "class": d["class"], "center_rc": list(centre),
                      "center_m": d["center"], "width_m": d["width"], "connects": d["connects"],
                      "open": d.get("open", True), "swing_into": d.get("swing_into"),
                      "cells_rc": cells})
        a, b = d["connects"]
        for x, y in ((a, b), (b, a)):
            if x in by_room:
                by_room[x]["doors"].append(d["id"])
                if y not in by_room[x]["neighbors"]:
                    by_room[x]["neighbors"].append(y)
        graph.append([a, b, d["id"]])
    objects = []
    for o in objs.get("objects", []):
        cells = cells_of.get(o["id"], [])
        p = o["pose"]
        item = {"id": o["id"], "label": o["label"], "name": o["name"], "asset": o["asset"],
                "room": o.get("room"), "cells_rc": cells, "n_cells": len(cells)}
        if cells:
            item.update(_box(cells), shape=ascii_shape(cells))
        item.update(center_rc=list(world_to_cell(fr, p["x"], p["y"])),
                    pose_m={"x": p["x"], "y": p["y"], "yaw_deg": p["yaw_deg"]},
                    size_m=o["size"], height_m=o.get("height"))
        if o.get("room") in by_room:
            by_room[o["room"]]["objects"].append(o["id"])
        objects.append(item)

    labelled = set(owner)
    blocked = {tuple(x) for x in np.argwhere(grid != 0).tolist()}
    return {"schema": SCHEMA, "map_id": map_id, "robot": robot,
            "source": {"type": ISB.TYPE, "scene_id": meta["source"]["scene_id"],
                       "objects_schema": objs.get("schema")},
            "frame": fr, "size_m": objs.get("size_m"),
            **{k: objs.get("generator", {}).get(k) for k in ("theme", "layout", "complexity")},
            "rooms": rooms, "room_graph": graph, "walls": walls, "doors": doors,
            "objects": objects,
            "counts": {"free_cells": int((grid == 0).sum()), "blocked_cells": len(blocked),
                       "labelled_cells": len(labelled),
                       "labelled_free_cells": len(labelled - blocked),
                       "unlabelled_blocked_cells": len(blocked - labelled)}}


# ------------------------------------------------------------------ dataset

def input_files(src: dict, map_ids) -> list[Path]:
    """objects.json + ids.png of the scenes of `map_ids` (those that exist)."""
    files = []
    for m in map_ids:
        sid = m.rsplit("_s", 1)[0][len(src["prefix"]) + 1:]
        d = ISB.bev_dir(src["path"], sid)
        files += [p for p in (d / f"{sid}_objects.json", d / f"{sid}_ids.png") if p.exists()]
    return files


def run(src: dict, root: Path, robot: str, map_ids) -> list[str]:
    """Write scene/<robot>/<map_id>.json for every map of an Isaac-Scene-Builder source;
    returns the map ids written. Maps without objects.json are skipped."""
    out = Path(root) / "scene" / robot
    out.mkdir(parents=True, exist_ok=True)
    keep, done = set(), []
    for m in sorted(map_ids):
        meta = json.loads((Path(root) / "grids" / f"{m}.json").read_text(encoding="utf-8"))
        sid = meta["source"]["scene_id"]
        d = ISB.bev_dir(src["path"], sid)
        op = d / f"{sid}_objects.json"
        if not op.exists():
            log.info("  %s: no %s, no scene index", m, op.name)
            continue
        objs = json.loads(op.read_text(encoding="utf-8"))
        ids = np.asarray(Image.open(d / objs["raster"]["file"])).astype(np.int64)
        with np.load(Path(root) / "grids" / f"{m}.npz") as z:
            grid = z["grid"]
        scene = build_map(meta, grid, objs, ids, m, robot)
        (out / f"{m}.json").write_text(json.dumps(scene, indent=1), encoding="utf-8")
        keep.add(f"{m}.json")
        done.append(m)
        n = scene["counts"]
        log.info("  %s: %d objects, %d walls, %d doors, %d/%d blocked cells labelled",
                 m, len(scene["objects"]), len(scene["walls"]), len(scene["doors"]),
                 n["blocked_cells"] - n["unlabelled_blocked_cells"], n["blocked_cells"])
    for f in out.glob("*.json"):
        if f.name not in keep:
            f.unlink()
    return done
