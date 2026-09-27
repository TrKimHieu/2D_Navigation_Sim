"""On-disk cache of what SvgEnv builds when it loads a map, so a map loads in a few
ms instead of ~100 ms.

Everything stored is the exact array the code computes, so an environment gives
identical results with or without the cache.

    <root>/v3/<dataset>/maps/<height_class>/<map_id>-<key>.npz   rings, raster, edge distance
    <root>/v3/<dataset>/<robot>/<map_id>-<key>.npz               task file, planner, [turns]
    <root>/v3/<dataset>/<robot>/<map_id>-<key>.fields.npy        goal distance per task (optional)

Keys hash the sha256 of the SVG and task files recorded in the dataset manifest (no
file is read to check them) plus every parameter the result depends on. Changed data
gives new keys; old entries are only removed by ``hm3d cache clear``.

    HM3D_CACHE   cache root, or "off" to disable. Default: %LOCALAPPDATA%\\hm3denv\\cache
                 (Windows), $XDG_CACHE_HOME/hm3denv or ~/.cache/hm3denv.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from .core.planning import TURN_SMOOTH, Planner
from .core.svgmap import SvgMap
from .dataset.schema import SCHEMA_VERSION

log = logging.getLogger(__name__)

FORMAT = 3
RASTER = (0.02, 0.2)                          # Collider raster: resolution, pad (m)
PLANNER = (0.02, 0.04)                        # Planner: raster res, plan_res (m)
OFF = ("off", "0", "false", "none", "")


def cache_root() -> Path | None:
    v = os.environ.get("HM3D_CACHE")
    if v is not None:
        return None if v.strip().lower() in OFF else Path(v).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "hm3denv" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "hm3denv"


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _save(path: Path, arrays) -> None:
    """Atomic write (tmp file + rename) of a dict (.npz) or one array (.npy); failures are
    logged, never raised."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            if isinstance(arrays, dict):
                np.savez(fh, **arrays)
            else:
                np.save(fh, arrays)
        os.replace(tmp, path)
    except OSError as e:
        log.debug("hm3denv cache: cannot write %s: %s", path, e)
        tmp.unlink(missing_ok=True)


def _load(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except FileNotFoundError:
        return None
    except Exception as e:                    # corrupt/partial file: rebuild and overwrite
        log.debug("hm3denv cache: unreadable %s: %s", path, e)
        return None


def _pack_rings(layers: dict) -> tuple[dict, dict]:
    """Ring layers -> (arrays, JSON-able info)."""
    names = list(layers)
    rings = [r for n in names for r in layers[n]]
    return ({"ring_lens": np.array([len(r) for r in rings], np.int64),
             "ring_xy": (np.concatenate(rings) if rings else np.zeros((0, 2))).astype(float)},
            {"layer_names": names, "layer_counts": [len(layers[n]) for n in names]})


def _unpack_rings(a: dict, info: dict) -> dict:
    rings = np.split(a["ring_xy"], np.cumsum(a["ring_lens"])[:-1]) if len(a["ring_lens"]) else []
    out, i = {}, 0
    for n, c in zip(info["layer_names"], info["layer_counts"]):
        out[n] = list(rings[i:i + c])
        i += c
    return out


class DatasetCache:
    """Cache entries of one dataset. ``enabled`` is False when there is no cache root or
    the manifest has no file hashes; every method is then a no-op."""

    def __init__(self, dataset, root: Path | None = None):
        self.ds = dataset
        self.root = root if root is not None else cache_root()
        self.files = dataset.manifest.get("files") or {}
        self.enabled = self.root is not None and bool(self.files)

    def paths(self, robot: str, map_id: str, r_circ: float, clearance) -> tuple | None:
        """(map file, robot file, fields file), or None if this entry cannot be cached."""
        if not self.enabled:
            return None
        hc = self.ds.manifest["robots"][robot]["height_class"]
        svg_sha = self.files.get(f"maps/{hc}/{map_id}.svg")
        task_sha = self.files.get(f"tasks/{robot}/{map_id}.json")
        if not svg_sha or not task_sha:
            return None
        base = self.root / f"v{FORMAT}" / self.ds.name
        mk = _key("map", FORMAT, svg_sha, RASTER)
        rk = _key("robot", FORMAT, mk, task_sha, repr(float(r_circ)), repr(clearance),
                  PLANNER, TURN_SMOOTH)
        return (base / "maps" / hc / f"{map_id}-{mk}.npz", base / robot / f"{map_id}-{rk}.npz",
                base / robot / f"{map_id}-{rk}.fields.npy")

    # ------------------------------------------------------------ read

    def load(self, paths) -> dict | None:
        """Everything needed to build a map context, or None on a miss."""
        m, r = _load(paths[0]), _load(paths[1])
        if m is None or r is None:
            return None
        try:
            mi, ri = json.loads(str(m["info"])), json.loads(str(r["info"]))
            layers = _unpack_rings(m, mi)
            svgmap = SvgMap(layers.pop("free"), mi["meta"], layers)
            H, W = mi["mask_shape"]
            mask = np.unpackbits(m["mask"], count=H * W).reshape(H, W).astype(bool)
            x0, y1, res = mi["frame"]
            svgmap._raster[(round(RASTER[0], 6), round(RASTER[1], 6))] = (mask, x0, y1, res)
            header = ri["header"]
            if header.get("schema_version") != SCHEMA_VERSION:
                return None
            planner = Planner.from_arrays(r)
            turns = dict(enumerate(ri["turns"])) if ri.get("turns") is not None else {}
        except (KeyError, ValueError) as e:
            log.debug("hm3denv cache: bad entry %s: %s", paths[1], e)
            return None
        return {"svgmap": svgmap, "edge_dist": m["edge_dist"], "header": header,
                "planner": planner, "turns": turns,
                "fields": paths[2] if paths[2].exists() else None}

    @staticmethod
    def load_field(path: Path, planner, task_idx: int) -> np.ndarray | None:
        """Distance field of one task from a fields file (as Planner.field returns it)."""
        try:
            with open(path, "rb") as fh:                   # read only row task_idx
                version = np.lib.format.read_magic(fh)
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read_header(fh)
                if fortran or len(shape) != 2 or shape[1] != len(planner.cells) \
                        or not 0 <= task_idx < shape[0]:
                    raise ValueError(f"shape {shape}")
                fh.seek(task_idx * shape[1] * dtype.itemsize, os.SEEK_CUR)
                d = np.fromfile(fh, dtype, count=shape[1])
            if len(d) != shape[1]:
                raise ValueError("truncated file")
        except Exception as e:
            log.debug("hm3denv cache: no field %d in %s: %s", task_idx, path, e)
            return None
        out = np.full(planner.grid.shape, np.inf)
        out[planner.cells[:, 0], planner.cells[:, 1]] = d
        return out

    # ------------------------------------------------------------ write

    def save(self, paths, svgmap, collider, header: dict, planner, turns=None) -> None:
        """Write the map and robot files (the map file only if missing). Small values go
        into one JSON member per file; JSON floats round-trip exactly."""
        if not paths[0].exists():
            mask, x0, y1, res = svgmap.raster(*RASTER)
            arrays, info = _pack_rings({"free": svgmap.rings, **svgmap.viz})
            info.update(meta=svgmap.meta, mask_shape=list(mask.shape),
                        frame=[float(x0), float(y1), float(res)])
            _save(paths[0], {**arrays, "mask": np.packbits(mask), "edge_dist": collider.edge_dist,
                             "info": np.array(json.dumps(info, ensure_ascii=False))})
        info = {"header": header, "turns": None if turns is None else [float(t) for t in turns]}
        _save(paths[1], {**planner.to_arrays(),
                         "info": np.array(json.dumps(info, ensure_ascii=False))})

    @staticmethod
    def save_fields(path: Path, node_dists: list) -> None:
        """node_dists[i]: goal field of task i at the C_disk cells (field[cells])."""
        _save(path, np.stack(node_dists) if node_dists else np.zeros((0, 0)))


# ------------------------------------------------------------ hm3d cache build/info/clear

def build_entry(dataset_root: str, robot: str, map_id: str, fields: bool) -> str:
    """Cache one (robot, map) completely: structure, turns of every task, optionally
    the goal fields. Runs in a worker process of ``hm3d cache build``."""
    from .envs.svg import SvgEnv, task_turn
    env = SvgEnv(dataset=dataset_root, robot=robot, map_id=map_id)
    dc = env._disk
    if dc is None:
        return f"{map_id}: not cacheable"
    ctx = env._ctx(map_id)
    paths = dc.paths(robot, map_id, env.robot.r_circ, env._clearance_arg)
    have = _load(paths[1])
    has_turns = (have is not None and "info" in have
                 and json.loads(str(have["info"])).get("turns") is not None)
    if has_turns and (not fields or paths[2].exists()):
        return f"{map_id}: cached"
    pl, turns, dists = ctx.planner, [], []
    for t in ctx.tasks:
        g = t["goal"]
        f, pred = pl.field(g["x"], g["y"], predecessors=True)
        turns.append(task_turn(pl, t, pred))
        if fields:
            dists.append(f[pl.cells[:, 0], pl.cells[:, 1]])
    dc.save(paths, ctx.svgmap, ctx.collider, ctx.header, pl, turns)
    if fields:
        dc.save_fields(paths[2], dists)
    return f"{map_id}: built"


def entries(root: Path | None = None) -> list[dict]:
    root = root if root is not None else cache_root()
    out = []
    base = root / f"v{FORMAT}" if root else None
    if base and base.exists():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            files = [f for f in d.rglob("*.np[yz]")]
            out.append({"dataset": d.name, "files": len(files),
                        "mb": sum(f.stat().st_size for f in files) / 2 ** 20, "path": str(d)})
    return out


def clear(dataset: str | None = None, root: Path | None = None) -> list[Path]:
    """Remove the cache of one dataset, or everything, in every format version."""
    root = root if root is not None else cache_root()
    if root is None or not root.exists():
        return []
    targets = [v / dataset if dataset else v for v in sorted(root.glob("v[0-9]*"))]
    targets = [t for t in targets if t.is_dir()]
    for t in targets:
        shutil.rmtree(t)
    return targets
