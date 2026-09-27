"""``hm3d build <config.yaml>``: one config file -> one dataset.

Config (YAML; unknown keys are errors)::

    name: grid-jetauto-v1          # dataset directory name
    env: grid                      # grid | svg
    source: {type: hm3d}           # or {type: isaac_scene_builder, path: <output dir>, prefix: isb}
    scenes: all                    # "all" GLBs of the workspace (or source scenes), or a list
    exclude: []                    # scenes to leave out (review "bad" scenes always are)
    split: {train: 0.7, val: 0.15, test: 0.15, seed: 0}
    slice: {}                      # SliceParams overrides (res, step, ...)
    # env: grid
    grid: {robot: jetauto_pro, robot_size: null, margin: 0.04, ...}   # + GridParams
    tasks: {k: 20, d_min: 8, budget: 200, ...}                        # GridTaskParams
    # env: svg
    robots: all                    # or a list of preset ids
    height_classes: {}             # override class heights, e.g. {h36: 0.36}
    map: {epsilon_px: 0.5, min_iou: 0.98, min_free_m2: 2.0}           # VectorParams
    tasks: {k: 20, min_geo: 1.0, ...}                                 # SvgTaskParams

With ``source.type: isaac_scene_builder`` the storey maps come from the scene
builder's BEV maps (build.isaac_scene_builder) instead of slicing GLBs: stage
``import_<htag>`` replaces ``slice_<htag>``, ``slice:`` is not allowed and the 3D mesh
checks of the task stages are off (``tasks.verify`` must stay false).

Stages write stamps to ``<dataset>/.stages/<stage>.json`` holding the hash of their
parameters and inputs; a stage whose stamp matches is skipped. ``force`` re-runs the
named stages (and everything after them, because their inputs change).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from .. import paths
from ..dataset.schema import assign_splits, scene_of, write_json
from ..dataset.writer import snapshot_robot, write_manifest
from ..robots import HEIGHT_CLASSES, Robot, height_tag, list_robots, load as load_robot
from . import gridify as G
from . import isaac_scene_builder as ISB
from . import slice as S
from . import tasks_grid as TG
from . import tasks_svg as TS
from . import vectorize as VZ

log = logging.getLogger(__name__)

COMMON = {"name", "env", "source", "scenes", "exclude", "split", "slice", "tasks"}
SOURCES = ("hm3d", ISB.TYPE)
ENV_KEYS = {"grid": COMMON | {"grid"}, "svg": COMMON | {"robots", "height_classes", "map"}}
SPLIT_DEFAULT = {"train": 0.7, "val": 0.15, "test": 0.15, "seed": 0}


class ConfigError(ValueError):
    pass


def _section(cfg: dict, key: str, cls, extra=()) -> dict:
    """Defaults of dataclass `cls` updated with cfg[key]; unknown keys are errors."""
    given = cfg.get(key) or {}
    if not isinstance(given, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    known = {f.name for f in fields(cls)} | set(extra)
    bad = sorted(set(given) - known)
    if bad:
        raise ConfigError(f"unknown key(s) in '{key}': {bad}; allowed: {sorted(known)}")
    base = asdict(cls())
    base.update({k: v for k, v in given.items()})
    return base


def load_config(src) -> dict:
    """Validated config with every default filled in (path, YAML text or dict)."""
    if isinstance(src, dict):
        cfg = copy.deepcopy(src)
    else:
        p = Path(src)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8") if p.exists() else src)
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    env = cfg.get("env")
    if env not in ENV_KEYS:
        raise ConfigError(f"env must be 'grid' or 'svg', got {env!r}")
    if not cfg.get("name"):
        raise ConfigError("name is required")
    bad = sorted(set(cfg) - ENV_KEYS[env])
    if bad:
        raise ConfigError(f"unknown key(s) {bad} for env {env}; allowed: {sorted(ENV_KEYS[env])}")
    src = cfg.get("source") or {"type": "hm3d"}
    if not isinstance(src, dict) or src.get("type") not in SOURCES:
        raise ConfigError(f"source.type must be one of {list(SOURCES)}")
    if src["type"] == "hm3d":
        if set(src) - {"type"}:
            raise ConfigError(f"unknown key(s) in 'source': {sorted(set(src) - {'type'})}")
    else:
        try:
            src = ISB.check_source(src)
        except (ValueError, FileNotFoundError) as e:
            raise ConfigError(str(e)) from e
        if cfg.get("slice"):
            raise ConfigError("'slice' does not apply to an isaac_scene_builder source")
        if (cfg.get("tasks") or {}).get("verify"):
            raise ConfigError("tasks.verify needs 3D meshes; an isaac_scene_builder source has none")
    out = {"name": str(cfg["name"]), "env": env, "source": src,
           "scenes": cfg.get("scenes", "all"), "exclude": list(cfg.get("exclude") or [])}
    if out["scenes"] != "all" and not isinstance(out["scenes"], list):
        raise ConfigError("scenes must be 'all' or a list")
    split = dict(SPLIT_DEFAULT)
    bad = sorted(set(cfg.get("split") or {}) - set(SPLIT_DEFAULT))
    if bad:
        raise ConfigError(f"unknown key(s) in 'split': {bad}")
    split.update(cfg.get("split") or {})
    if abs(split["train"] + split["val"] + split["test"] - 1.0) > 1e-6:
        raise ConfigError("split fractions must sum to 1")
    out["split"] = split
    out["slice"] = _section(cfg, "slice", S.SliceParams)
    for k in ("robot_height", "use_review"):     # decided by the pipeline
        out["slice"].pop(k)
    if (cfg.get("slice") or {}).keys() & {"robot_height", "use_review"}:
        raise ConfigError("slice.robot_height/use_review are set by the pipeline (robot / height_classes)")
    if env == "grid":
        g = _section(cfg, "grid", G.GridParams, extra=("robot", "robot_size"))
        g.setdefault("robot", "jetauto_pro")
        g.setdefault("robot_size", None)
        out["grid"] = g
        out["tasks"] = _section(cfg, "tasks", TG.GridTaskParams)
    else:
        robots = cfg.get("robots", "all")
        if robots != "all" and not isinstance(robots, list):
            raise ConfigError("robots must be 'all' or a list")
        out["robots"] = robots
        hc = dict(HEIGHT_CLASSES)
        hc.update(cfg.get("height_classes") or {})
        out["height_classes"] = hc
        out["map"] = _section(cfg, "map", VZ.VectorParams)
        out["tasks"] = _section(cfg, "tasks", TS.SvgTaskParams)
    if src["type"] == ISB.TYPE:
        out["tasks"]["verify"] = False
    return out


# ------------------------------------------------------------------ stamps

def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _files_hash(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        st = f.stat()
        h.update(f"{f.as_posix()}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


class Stages:
    def __init__(self, root: Path, force=()):
        self.dir = Path(root) / ".stages"
        self.force = set(force)
        self.forcing = False                  # once a stage is forced, later stages re-run too
        self.stamps: dict = {}

    def run(self, name, family, params, inputs, fn, outputs):
        """Run `fn()` unless the stamp matches. `family` is the stage kind used by
        `force` (e.g. "tasks" forces every tasks_<robot> stage). `outputs()` hashes the
        result on disk. Returns the stage result (from fn or from the stamp)."""
        p = self.dir / f"{name}.json"
        ph, ih = _hash(params), _hash(inputs)
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if family in self.force or name in self.force:
            self.forcing = True
        if old and not self.forcing and old["params_hash"] == ph and old["inputs_hash"] == ih:
            log.info("[%s] up to date", name)
            self.stamps[name] = old
            return old.get("result")
        log.info("[%s] running", name)
        t0 = time.time()
        result = fn()
        stamp = {"stage": name, "params_hash": ph, "inputs_hash": ih, "outputs_hash": outputs(),
                 "created": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1),
                 "params": params, "result": result}
        write_json(p, stamp)
        self.stamps[name] = stamp
        return result

    def summary(self) -> dict:
        return {k: {kk: v[kk] for kk in ("params_hash", "inputs_hash", "outputs_hash", "created")}
                for k, v in self.stamps.items()}


# ------------------------------------------------------------------ build

def resolve_scenes(ws: Path, cfg: dict) -> list[str]:
    src = cfg["source"]
    if src["type"] == ISB.TYPE:
        drop = set(cfg["exclude"])
        return [ISB.scene_name(src, i) for i in ISB.resolve(src, cfg["scenes"]) if i not in drop]
    glbs = sorted(p.stem for p in paths.glb_dir(ws).glob("*.glb"))
    if not glbs:
        raise FileNotFoundError(f"no .glb in {paths.glb_dir(ws)}")
    if cfg["scenes"] == "all":
        scenes = glbs
    else:
        missing = sorted(set(cfg["scenes"]) - set(glbs))
        if missing:
            raise FileNotFoundError(f"scenes without GLB in {paths.glb_dir(ws)}: {missing}")
        scenes = sorted(cfg["scenes"])
    drop = set(cfg["exclude"])
    return [s for s in scenes if s not in drop and not S.is_excluded(paths.slice_dir(ws) / s)]


def _slice_stage(st: Stages, ws, cfg, scenes, height):
    if cfg["source"]["type"] == ISB.TYPE:
        return _import_stage(st, ws, cfg, scenes, height)
    tag = height_tag(height)
    params = {**cfg["slice"], "robot_height": height}
    review = sorted(paths.slice_dir(ws).glob("*/review.json"))
    inputs = {"scenes": scenes, "glb": _files_hash([paths.glb_dir(ws) / f"{s}.glb" for s in scenes]),
              "review": {p.parent.name: _hash(json.loads(p.read_text("utf-8"))) for p in review}}

    def outputs():
        return _files_hash([f for s in scenes for f in (paths.slice_dir(ws) / s / "maps").glob("storey*")])

    def fn():
        S.run(ws, scenes, S.SliceParams(robot_height=height, **cfg["slice"]),
              force="slice" in st.force)
        return {"height": height}
    return st.run(f"slice_{tag}", "slice", params, inputs, fn, outputs), outputs()


def _import_stage(st: Stages, ws, cfg, scenes, height):
    """Storey maps from an Isaac-Scene-Builder output (instead of slicing GLBs)."""
    src, tag = cfg["source"], height_tag(height)
    ids = [s[len(src["prefix"]) + 1:] for s in scenes]
    inputs = {"scenes": scenes, "files": _files_hash(ISB.input_files(src, ids))}

    def outputs():
        return _files_hash([f for s in scenes for f in (paths.slice_dir(ws) / s / "maps").glob("storey*")])

    def fn():
        ISB.run(ws, src, ids, [tag])
        return {"height": height}
    return st.run(f"import_{tag}", "slice", {"source": src, "height": height}, inputs, fn,
                  outputs), outputs()


def build(config, ws=None, out=None, force=(), preview=True) -> Path:
    """Build (or update) the dataset described by `config`. Returns its directory."""
    from .. import __version__
    cfg = load_config(config)
    ws = paths.find_workspace(ws)
    root = Path(out) if out else ws / "datasets" / cfg["name"]
    root.mkdir(parents=True, exist_ok=True)
    st = Stages(root, force)
    scenes = resolve_scenes(ws, cfg)
    log.info("build %s (%s): %d scenes -> %s", cfg["name"], cfg["env"], len(scenes), root)
    if cfg["env"] == "grid":
        robots = _build_grid(st, ws, root, cfg, scenes)
    else:
        robots = _build_svg(st, ws, root, cfg, scenes)
    splits = st.stamps["splits"]["result"]
    source = dict(cfg["source"])
    if source["type"] == ISB.TYPE:
        source["verified_3d"] = False
    write_manifest(root, name=cfg["name"], env=cfg["env"], config=cfg, splits=splits,
                   robots=robots, stages=st.summary(), config_hash=_hash(cfg), built_with=__version__,
                   source=source)
    if preview:
        from . import preview as PV
        from ..dataset import Dataset
        ds = Dataset(root)
        for rid in ds.robots:
            if cfg["env"] == "grid":
                PV.grid_sheets(ws, ds, rid)
            else:
                PV.svg_sheets(ds, rid)
        if cfg["env"] == "svg":
            (root / "preview").mkdir(exist_ok=True)
            (root / "preview" / "robots.svg").write_text(
                PV.robots_svg([ds.robot(r) for r in ds.robots]), encoding="utf-8")
    return root


def _splits_stage(st, cfg, map_ids):
    scenes = sorted({scene_of(m) for m in map_ids})
    s = cfg["split"]
    return st.run("splits", "splits", s, scenes,
                  lambda: assign_splits(scenes, {k: s[k] for k in ("train", "val", "test")}, s["seed"]),
                  lambda: _hash(scenes))


def _build_grid(st, ws, root, cfg, scenes):
    gc = cfg["grid"]
    base = load_robot(gc["robot"])
    robot = G.grid_robot(base.raw, gc["robot_size"])
    Robot.from_dict(robot)                                        # validate the derived preset
    _, slice_out = _slice_stage(st, ws, cfg, scenes, robot["height"])
    gp = {k: v for k, v in gc.items() if k not in ("robot", "robot_size")}
    grids = root / "grids"
    made = st.run("gridify", "gridify", {**gp, "robot": robot}, {"slice": slice_out, "scenes": scenes},
                  lambda: sorted(G.run(ws, grids, robot, scenes, G.GridParams(**gp))),
                  lambda: _files_hash(grids.glob("*")))
    splits = _splits_stage(st, cfg, made)
    snapshot_robot(root, robot["id"], robot)
    per_map = st.run(f"tasks_{robot['id']}", "tasks", {**cfg["tasks"], "robot": robot},
                     {"grids": st.stamps["gridify"]["outputs_hash"], "splits": _hash(splits)},
                     lambda: TG.run(ws, root, cfg["name"], robot, splits, TG.GridTaskParams(**cfg["tasks"])),
                     lambda: _files_hash((root / "tasks" / robot["id"]).glob("*.json")))
    for f in (root / "robots").glob("*.json"):                     # snapshots of older builds
        if f.stem != robot["id"]:
            f.unlink()
    return {robot["id"]: {"height_class": height_tag(robot["height"]), "maps": per_map}}


def _build_svg(st, ws, root, cfg, scenes):
    ids = list_robots() if cfg["robots"] == "all" else list(cfg["robots"])
    robots = [load_robot(r) for r in ids]
    hcs = sorted({r.height_class for r in robots})
    for r in robots:
        if r.height_class not in cfg["height_classes"]:
            raise ConfigError(f"{r.id}: unknown height class {r.height_class}")
        if r.height > cfg["height_classes"][r.height_class] + 1e-9:
            raise ConfigError(f"{r.id}: height {r.height} m above its class {r.height_class} "
                              f"({cfg['height_classes'][r.height_class]} m)")
    maps, map_stage = {}, {}
    src = cfg["source"]
    vector = src["type"] == ISB.TYPE and src["geometry"] == "vector"
    for hc in hcs:
        h = cfg["height_classes"][hc]
        if height_tag(h) != hc:
            raise ConfigError(f"height class {hc} has height {h} (tag {height_tag(h)})")
        mdir = root / "maps" / hc
        if vector:                      # SVG straight from the scene builder's CSV geometry
            from . import isb_vector as ISBV            # needs shapely ([build] extra)
            ids = [s[len(src["prefix"]) + 1:] for s in scenes]
            map_stage[hc] = f"polygons_{hc}"
            maps[hc] = st.run(map_stage[hc], "vectorize", {**cfg["map"], "height": h, "source": src},
                              {"scenes": scenes, "files": _files_hash(ISBV.input_files(src, ids))},
                              lambda mdir=mdir, hc=hc, h=h, ids=ids: ISBV.run(
                                  src, mdir, hc, h, ids, cfg["map"]["min_free_m2"]),
                              lambda mdir=mdir: _files_hash(mdir.glob("*.svg")))
            continue
        _, slice_out = _slice_stage(st, ws, cfg, scenes, h)
        map_stage[hc] = f"vectorize_{hc}"
        maps[hc] = st.run(map_stage[hc], "vectorize", {**cfg["map"], "height": h},
                          {"slice": slice_out, "scenes": scenes},
                          lambda mdir=mdir, hc=hc, h=h: VZ.run(ws, mdir, hc, h, scenes, VZ.VectorParams(**cfg["map"])),
                          lambda mdir=mdir: _files_hash(mdir.glob("*.svg")))
    for d in (root / "maps").iterdir():                            # classes no longer used
        if d.is_dir() and d.name not in hcs:
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
    splits = _splits_stage(st, cfg, [m for v in maps.values() for m in v])
    out = {}
    for r in robots:
        snapshot_robot(root, r.id, r.raw)
        per_map = st.run(f"tasks_{r.id}", "tasks", {**cfg["tasks"], "robot": r.raw},
                         {"maps": st.stamps[map_stage[r.height_class]]["outputs_hash"],
                          "splits": _hash(splits)},
                         lambda r=r: TS.run(ws, root, cfg["name"], r, splits, TS.SvgTaskParams(**cfg["tasks"])),
                         lambda r=r: _files_hash((root / "tasks" / r.id).glob("*.json")))
        out[r.id] = {"height_class": r.height_class, "maps": per_map,
                     "map_stats": {m: maps[r.height_class][m] for m in per_map if m in maps[r.height_class]}}
    keep = {r.id for r in robots}
    for d in [root / "robots", root / "verify"]:
        for f in d.glob("*") if d.exists() else []:
            if f.stem not in keep:
                f.unlink()
    for d in (root / "tasks").iterdir() if (root / "tasks").exists() else []:
        if d.is_dir() and d.name not in keep:
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
    return out
