"""Robot presets: load, validate, build the collision footprint.

A preset is a JSON file (schema 1.0). Every numeric parameter must cite a source:
``official`` (with a URL) or ``estimated`` (with the method used).

The footprint is a simple polygon (possibly concave), counter-clockwise, in the
``base_link`` frame (x forward, y left). Its origin is the rotation centre: for
``polygon`` footprints imported from URDF (``hm3d robots import-urdf``) that is
the true wheel axis; ``rectangle``/``circle`` assume the geometric centre.
Circles are replaced by a circumscribed 16-gon (conservative for collisions).

Lookup order for ``load(id)``: extra directories (e.g. a dataset's robot
snapshot) > the workspace ``robots/`` folder > the packaged presets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths

SCHEMA_VERSION = "1.0"
CIRCLE_SIDES = 16
DRIVES = ("differential", "omnidirectional")
REQUIRED = ("schema_version", "id", "name", "footprint", "height", "height_class",
            "drive", "limits", "lidar", "sources")
PRESETS = Path(__file__).resolve().parent / "presets"
# obstacle height band of each map class (m); a robot uses the class whose height
# covers its own, so one set of maps serves every robot of the class
HEIGHT_CLASSES = {"h36": 0.36, "h63": 0.626, "h145": 1.45}


def footprint_polygon(spec: dict) -> np.ndarray:
    """Counter-clockwise (M, 2) polygon in base_link."""
    t = spec["type"]
    if t == "rectangle":
        hl, hw = spec["length"] / 2, spec["width"] / 2
        return np.array([[hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]], float)
    if t == "circle":
        R = spec["radius"] / math.cos(math.pi / CIRCLE_SIDES)
        a = (np.arange(CIRCLE_SIDES) + 0.5) * 2 * math.pi / CIRCLE_SIDES
        return np.stack([R * np.cos(a), R * np.sin(a)], axis=1)
    if t == "polygon":
        P = np.asarray(spec["points"], float)
        return P[::-1] if polygon_area(P) < 0 else P
    raise ValueError(f"unsupported footprint type: {t}")


def polygon_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _numeric_leaves(d: dict, prefix: str = ""):
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _numeric_leaves(v, p + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield p


def validate(raw: dict) -> None:
    """Raise ValueError on schema problems or on a numeric parameter without a source
    (a `sources` key equal to its dotted path or to a prefix of it)."""
    rid = raw.get("id")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{rid}: missing {missing}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{rid}: schema_version {raw['schema_version']}")
    if raw["drive"] not in DRIVES:
        raise ValueError(f"{rid}: drive must be one of {DRIVES}")
    for k in ("v_max", "v_min", "vy_max", "w_max"):
        if k not in raw["limits"]:
            raise ValueError(f"{rid}: limits.{k} missing")
    for k in ("range_min", "range_max", "fov", "beams"):
        if k not in raw["lidar"]:
            raise ValueError(f"{rid}: lidar.{k} missing")
    for src in raw["sources"].values():
        if src.get("kind") == "official" and not src.get("url"):
            raise ValueError(f"{rid}: official source without url")
        if src.get("kind") == "estimated" and not src.get("method"):
            raise ValueError(f"{rid}: estimated source without method")
    body = {k: raw[k] for k in ("footprint", "height", "limits", "lidar")}
    keys = list(raw["sources"])
    for p in _numeric_leaves(body):
        if not any(p == k or p.startswith(k + ".") for k in keys):
            raise ValueError(f"{rid}: parameter {p} has no source")


@dataclass
class Robot:
    id: str
    name: str
    footprint: np.ndarray
    height: float
    height_class: str
    drive: str
    v_max: float
    v_min: float
    vy_max: float
    w_max: float
    lidar: dict
    raw: dict = field(repr=False)

    @property
    def omni(self) -> bool:
        return self.drive == "omnidirectional"

    @property
    def r_circ(self) -> float:
        """Circumradius of the collision polygon around the rotation centre."""
        return float(np.linalg.norm(self.footprint, axis=1).max())

    @property
    def action_dim(self) -> int:
        return 3 if self.omni else 2

    @classmethod
    def from_dict(cls, raw: dict) -> "Robot":
        validate(raw)
        L = raw["limits"]
        return cls(id=raw["id"], name=raw["name"], footprint=footprint_polygon(raw["footprint"]),
                   height=float(raw["height"]), height_class=raw["height_class"],
                   drive=raw["drive"], v_max=float(L["v_max"]), v_min=float(L["v_min"]),
                   vy_max=float(L["vy_max"]), w_max=float(L["w_max"]),
                   lidar=dict(raw["lidar"]), raw=raw)


def search_dirs(extra=None) -> list[Path]:
    dirs = [Path(d) for d in ([extra] if isinstance(extra, (str, Path)) else (extra or []))]
    ws = paths.user_robots_dir(paths.find_workspace(required=False))
    if ws:
        dirs.append(ws)
    return dirs + [PRESETS]


def preset_path(robot_id: str, extra=None) -> Path:
    for d in search_dirs(extra):
        p = d / f"{robot_id}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"no robot preset '{robot_id}' (searched {[str(d) for d in search_dirs(extra)]}); "
                            f"available: {list_robots(extra)}")


def load(robot_id: str | Path, extra=None) -> Robot:
    """A preset by id (see lookup order above) or from a JSON file path."""
    p = Path(robot_id)
    path = p if p.suffix == ".json" and p.exists() else preset_path(str(robot_id), extra)
    return Robot.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_robots(extra=None) -> list[str]:
    return sorted({p.stem for d in search_dirs(extra) if d.exists() for p in d.glob("*.json")})


def cell_px(robot: Robot | dict, res: float) -> int:
    """Grid cell size in pixels: the robot's longest side rounded UP to the map resolution."""
    raw = robot.raw if isinstance(robot, Robot) else robot
    fp = raw.get("footprint_simple", raw["footprint"])
    if fp["type"] == "rectangle":
        side = max(fp["length"], fp["width"])
    elif fp["type"] == "circle":
        side = 2 * fp["radius"]
    else:
        P = np.asarray(fp["points"])
        side = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])))
    return int(math.ceil(side / res - 1e-9))


def height_tag(height: float) -> str:
    """File suffix for a robot height used by the slice stage: 0.626 -> 'h63'."""
    return f"h{int(round(height * 100)):02d}"
