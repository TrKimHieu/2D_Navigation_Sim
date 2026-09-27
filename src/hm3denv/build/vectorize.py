"""Vector stage: turn the 2 cm storey maps into SVG maps (metres) for the continuous env.

    free = floor (cleaned) & roof (cleaned) & no obstacle (for the height class)

Boundaries come from cv2.findContours, which traces the CENTRES of the free boundary
pixels, so free space shrinks by ~1/2 px around every obstacle (conservative).
approxPolyDP simplifies with tolerance `epsilon_px`. The polygons are rasterised back
into the original pixel frame and compared:

    iou        |vector & source| / |vector | source|
    unsafe_m2  area free in the vector map but obstacle/no floor in the source
               (the NON-conservative part; must be tiny)
    lost_m2    area free in the source but lost by the vector map (conservative part)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .. import paths
from ..core.svgmap import SCHEMA_VERSION, SvgMap, rasterize
from .gridify import clean_mask
from .slice import is_excluded

log = logging.getLogger(__name__)


@dataclass
class VectorParams:
    epsilon_px: float = 0.5      # simplification tolerance (px)
    min_iou: float = 0.98        # flag maps below this IoU (the gap is mostly shrunk free space)
    min_free_m2: float = 2.0     # drop storeys with less free area


def contours_to_rings(mask, meta, eps_px, min_ring_px=0):
    """Polygon rings (m) of the region `mask` (bool, 2 cm slice image)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    res, H = meta["res"], meta["H"]
    lo_u, lo_v = meta["lo"][meta["ua"]], meta["lo"][meta["va"]]
    rings = []
    for c in cs:
        if eps_px > 0:
            c = cv2.approxPolyDP(c, eps_px, True)
        pts = c[:, 0, :].astype(float)
        if len(pts) < 3 or abs(cv2.contourArea(c.astype(np.float32))) < min_ring_px:
            if len(pts) < 3:
                # degenerate ring (1-2 px obstacle): replace with a 1 px square around it
                cx, cy = pts.mean(axis=0)
                pts = np.array([[cx - .5, cy - .5], [cx + .5, cy - .5],
                                [cx + .5, cy + .5], [cx - .5, cy + .5]])
            else:
                continue
        x = lo_u + (pts[:, 0] + 0.5) * res
        y = lo_v + (H - pts[:, 1] - 0.5) * res
        rings.append(np.stack([x, y], 1))
    return rings


def free_mask(maps_dir, k, htag, close_m=0.10, max_hole_m2=1.0, min_free_m2=0.25, res=0.02):
    ob = np.asarray(Image.open(maps_dir / f"storey{k}_{htag}.png")) == 0
    fl = clean_mask(np.asarray(Image.open(maps_dir / f"storey{k}_floor.png")) > 0,
                    res, close_m, max_hole_m2)
    rf = clean_mask(np.asarray(Image.open(maps_dir / f"storey{k}_roof_{htag}.png")) > 0,
                    res, close_m, max_hole_m2)
    free = fl & rf & ~ob
    lab, n = ndimage.label(free)
    if n:
        keep = np.bincount(lab.ravel()) * res * res >= min_free_m2
        keep[0] = False
        free = keep[lab]
    outdoor = fl & ~rf & ~ob
    return free, outdoor


def measure(rings, src, meta):
    res, H, W = meta["res"], meta["H"], meta["W"]
    x0 = meta["lo"][meta["ua"]]
    y1 = meta["lo"][meta["va"]] + H * res
    vec = rasterize(rings, x0, y1, res, H, W)
    inter, union = (vec & src).sum(), (vec | src).sum()
    px = res * res
    return {"iou": round(float(inter / max(union, 1)), 5),
            "unsafe_m2": round(float((vec & ~src).sum() * px), 4),
            "lost_m2": round(float((~vec & src).sum() * px), 4),
            "free_m2": round(float(src.sum() * px), 3)}


def vectorize_storey(sj: Path, hclass: str, height: float, eps_px: float, min_iou: float):
    smeta = json.loads(sj.read_text(encoding="utf-8"))
    scene, k = smeta["scene"], smeta["storey"]
    free, outdoor = free_mask(sj.parent, k, hclass, res=smeta["res"])
    rings = contours_to_rings(free, smeta, eps_px)
    viz = contours_to_rings(outdoor, smeta, max(eps_px, 1.0), min_ring_px=25)
    stats = measure(rings, free, smeta)
    up, ua, va = smeta["up"], smeta["ua"], smeta["va"]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "map_id": f"{scene}_s{k}", "scene": scene, "storey": k,
        "floor_z": smeta["floor_z"], "ceil_z": smeta["ceil_z"],
        "height_class": hclass,
        "z_band": [round(smeta["floor_z"] + smeta.get("floor_margin", 0.05), 4),
                   round(smeta["floor_z"] + height, 4)],
        "frame": {"up": up, "u_axis": "XYZ"[ua], "v_axis": "XYZ"[va],
                  "right_handed": bool((ua, va, up) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))),
                  "units": "m", "note": smeta.get("frame_note",
                                                  "x = GLB u axis, y = GLB v axis; theta CCW from +x")},
        "vectorize": {"res": smeta["res"], "epsilon_px": eps_px,
                      "epsilon_m": round(eps_px * smeta["res"], 4),
                      "n_rings": len(rings), "n_vertices": int(sum(len(r) for r in rings)),
                      **stats},
        "flags": [f"iou {stats['iou']:.3f}"] if stats["iou"] < min_iou else [],
        "provenance": {"tool": "hm3denv.build.vectorize", "created": time.strftime("%Y-%m-%d %H:%M"),
                       "sources": [f"stages/slice/{scene}/maps/storey{k}_{hclass}.png",
                                   f"stages/slice/{scene}/maps/storey{k}_floor.png",
                                   f"stages/slice/{scene}/maps/storey{k}_roof_{hclass}.png",
                                   f"stages/slice/{scene}/maps/storey{k}.json"]},
    }
    if "source" in smeta:                       # storeys imported from another source
        meta["source"] = smeta["source"]
    return SvgMap(rings, meta, {"outdoor": viz} if viz else {}), stats


def run(ws, out_dir: Path, hclass: str, height: float, scenes=None,
        params: VectorParams | None = None) -> dict:
    """Vectorise every sliced storey of height class `hclass` into out_dir/<map_id>.svg.
    Returns {map_id: vectorize stats + flags}; stale SVGs are removed."""
    p = params or VectorParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = set(scenes) if scenes else None
    made = {}
    for sj in sorted(paths.slice_dir(ws).glob("*/maps/storey*.json")):
        scene = sj.parent.parent.name
        if (scenes and scene not in scenes) or is_excluded(sj.parent.parent):
            continue
        k = json.loads(sj.read_text("utf-8"))["storey"]
        if not (sj.parent / f"storey{k}_roof_{hclass}.png").exists():
            log.warning("  %s_s%d: no %s maps; slice with robot_height=%s", scene, k, hclass, height)
            continue
        m, st = vectorize_storey(sj, hclass, height, p.epsilon_px, p.min_iou)
        if st["free_m2"] < p.min_free_m2:
            continue
        m.save(out_dir / f"{m.meta['map_id']}.svg")
        made[m.meta["map_id"]] = {**m.meta["vectorize"], "flags": m.meta["flags"]}
        log.info("  %s: %d rings, %d vertices, IoU %.4f, unsafe %.0f cm2, lost %.2f m2%s",
                 m.meta["map_id"], len(m.rings), m.meta["vectorize"]["n_vertices"], st["iou"],
                 st["unsafe_m2"] * 1e4, st["lost_m2"], f"  FLAG {m.meta['flags']}" if m.meta["flags"] else "")
    for old in out_dir.glob("*.svg"):
        if old.stem not in made and (not scenes or old.stem.rsplit("_s", 1)[0] in scenes):
            old.unlink()
    return made
