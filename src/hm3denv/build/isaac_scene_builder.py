"""Source adapter: scenes built with Isaac-Scene-Builder -> storey maps of the slice stage.


Input (Isaac-Scene-Builder ``output/``)::

    bev/<id>/<id>.pgm     3-level occupancy (ROS map_server): 254 free, 0 occupied, 205 outside
    bev/<id>/<id>.yaml    resolution, origin [x, y, 0] (ROS map frame: y = -editor y)
    manifest.jsonl        one JSON line per scene build (the last line of an id wins)

Output: ``<workspace>/stages/slice/<prefix>-<id>/maps/storey0*`` in the format the slice
stage writes, so vectorize / gridify / tasks / preview work unchanged, plus
``source.json``. The map frame is the USD world frame of the scene (Z up, objects at
editor (x, y)): pixel (r, c) of the PGM has its centre at
``x = origin_x + (c + .5) res``, ``y = -origin_y - (H - r - .5) res``. Images are stored
flipped vertically (row 0 = largest y), as the slice stage stores them.

The BEV has no heights: every piece of furniture is an obstacle for every height class
(conservative), and the whole scene is indoors (roofed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .. import paths

log = logging.getLogger(__name__)

TYPE = "isaac_scene_builder"
FREE, OCC, UNKNOWN = 254, 0, 205
SOURCE_KEYS = {"type", "path", "prefix", "geometry", "mesh_dir", "wall_m", "door_gap", "cell_m",
               "simplify_m"}
META_KEYS = ("source", "level", "occlusion_density_target", "occlusion_density_actual",
             "n_furniture_placed", "traversable", "n_zones_actual", "n_doors", "grid",
             "legacy_id", "usd")
FRAME_NOTE = ("x, y = Isaac Sim USD world frame of the scene (Z up; Isaac-Scene-Builder "
              "editor coordinates); theta CCW from +x. The BEV .yaml uses the ROS map frame "
              "(y flipped).")


def check_source(src: dict) -> dict:
    bad = sorted(set(src) - SOURCE_KEYS)
    if bad:
        raise ValueError(f"unknown key(s) in 'source': {bad}; allowed: {sorted(SOURCE_KEYS)}")
    if not src.get("path"):
        raise ValueError("source.path is required (the Isaac-Scene-Builder output directory)")
    root = Path(src["path"]).expanduser()
    if not (root / "bev").is_dir():
        raise FileNotFoundError(f"{root}: no bev/ directory (Isaac-Scene-Builder output expected)")
    prefix = str(src.get("prefix") or "isb")
    if "_s" in prefix or "/" in prefix or "\\" in prefix:
        raise ValueError(f"source.prefix {prefix!r} may not contain '_s' or path separators")
    from .isb_vector import DEFAULTS
    out = {**DEFAULTS, **{k: v for k, v in src.items() if k in DEFAULTS and v is not None}}
    if out["geometry"] not in ("vector", "raster"):
        raise ValueError("source.geometry must be 'vector' (CSV + mesh footprints) or 'raster' (BEV)")
    gap = out["door_gap"]
    if not (isinstance(gap, (list, tuple)) and len(gap) == 2 and 0 <= gap[0] < gap[1] <= 1):
        raise ValueError("source.door_gap must be [start, end] fractions with 0 <= start < end <= 1")
    for k in ("wall_m", "cell_m", "simplify_m"):
        if not isinstance(out[k], (int, float)) or out[k] < 0:
            raise ValueError(f"source.{k} must be a number >= 0")
    out["door_gap"] = [float(gap[0]), float(gap[1])]
    out["mesh_dir"] = str(Path(out["mesh_dir"] or root / "mesh_shapes").expanduser().resolve())
    return {"type": TYPE, "path": str(root.resolve()), "prefix": prefix, **out}


def catalog(root: Path) -> dict[str, dict]:
    """Scene metadata from manifest.jsonl (last line per scene id wins); {} if missing."""
    out = {}
    mf = Path(root) / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["scene_id"]] = d
    return out


def available(root: Path) -> list[str]:
    """Scene ids with a PGM and a YAML."""
    bev = Path(root) / "bev"
    return sorted(d.name for d in bev.iterdir()
                  if d.is_dir() and (d / f"{d.name}.pgm").exists() and (d / f"{d.name}.yaml").exists())


def resolve(src: dict, scenes) -> list[str]:
    """Source scene ids selected by the config: "all" = those in manifest.jsonl (every
    available scene when there is no manifest.jsonl)."""
    root = Path(src["path"])
    have = available(root)
    if scenes == "all":
        cat = catalog(root)
        return [s for s in have if s in cat] if cat else have
    missing = sorted(set(scenes) - set(have))
    if missing:
        raise FileNotFoundError(f"scenes without a BEV .pgm/.yaml in {root / 'bev'}: {missing}")
    return sorted(scenes)


def scene_name(src: dict, scene_id: str) -> str:
    return f"{src['prefix']}-{scene_id}"


def input_files(src: dict, ids) -> list[Path]:
    root = Path(src["path"])
    files = [root / "bev" / i / f"{i}.{ext}" for i in ids for ext in ("pgm", "yaml")]
    return files + ([root / "manifest.jsonl"] if (root / "manifest.jsonl").exists() else [])


def read_bev(root: Path, scene_id: str):
    """(pgm array, resolution, origin_x, origin_y) of one scene."""
    d = Path(root) / "bev" / scene_id
    info = yaml.safe_load((d / f"{scene_id}.yaml").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(d / info.get("image", f"{scene_id}.pgm")))
    if img.ndim != 2:
        raise ValueError(f"{scene_id}: expected a single-channel PGM, got shape {img.shape}")
    if info.get("negate", 0):
        raise ValueError(f"{scene_id}: negate=1 maps are not supported")
    ox, oy = float(info["origin"][0]), float(info["origin"][1])
    return img, float(info["resolution"]), ox, oy


def import_scene(ws: Path, src: dict, scene_id: str, htags, meta: dict | None = None) -> str:
    """Write the storey maps of one scene for the height tags `htags`; returns its name."""
    img, res, ox, oy = read_bev(Path(src["path"]), scene_id)
    H, W = img.shape
    name = scene_name(src, scene_id)
    sdir = paths.slice_dir(ws) / name
    mdir = sdir / "maps"
    mdir.mkdir(parents=True, exist_ok=True)
    flip = img[::-1]                                       # row 0 = largest y
    floor = np.where(flip != UNKNOWN, 255, 0).astype(np.uint8)
    obstacle = np.where(flip == OCC, 0, 255).astype(np.uint8)
    source = {"type": TYPE, "scene_id": scene_id,
              "bev": str(Path(src["path"]) / "bev" / scene_id / f"{scene_id}.pgm"),
              **{k: (meta or {}).get(k) for k in META_KEYS if k in (meta or {})}}
    Image.fromarray(floor).save(mdir / "storey0_floor.png")
    for tag in htags:
        Image.fromarray(obstacle).save(mdir / f"storey0_{tag}.png")
        Image.fromarray(floor).save(mdir / f"storey0_roof_{tag}.png")
    storey = {"scene": name, "storey": 0, "floor_z": 0.0, "ceil_z": None, "floor_margin": 0.05,
              "up": 2, "ua": 0, "va": 1, "res": res, "H": int(H), "W": int(W),
              "lo": [ox, -oy - H * res, 0.0], "source": source, "frame_note": FRAME_NOTE}
    (mdir / "storey0.json").write_text(json.dumps(storey, indent=1), encoding="utf-8")
    (sdir / "source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")
    return name


def run(ws: Path, src: dict, ids, htags) -> list[str]:
    """Import every selected scene; returns the scene names."""
    cat = catalog(Path(src["path"]))
    names = []
    for i in ids:
        names.append(import_scene(ws, src, i, htags, cat.get(i)))
    log.info("imported %d Isaac-Scene-Builder scenes (%s)", len(names), ", ".join(htags))
    return names


def is_imported(ws: Path, scene: str) -> bool:
    return (paths.slice_dir(ws) / scene / "source.json").exists()
