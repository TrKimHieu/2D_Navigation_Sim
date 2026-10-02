"""``hm3d extend``: continue a dataset (a downloaded one or your own) instead of rebuilding it.

    hm3d extend svg-v1 --status                         what is in it, what can be added
    hm3d extend svg-v1 --new-scenes                     every GLB of the workspace not in it yet
    hm3d extend svg-v1 --scenes 00900-abc 00901-def     these scenes
    hm3d extend svg-v1 --add-tasks 10                   10 more tasks on every map
    hm3d extend svg-v1 --robots kobuki                  tasks for another robot (SVG datasets)

What is already in the dataset never changes: its maps and tasks stay byte for byte, its
scenes keep their split (tasks used for training never move to test); new scenes are
split by the dataset's fractions. The result is written to a copy (``<name>-ext`` in the
workspace's datasets/, or ``--out``) unless ``--in-place``; the manifest gets a
``history`` entry and fresh sha256 hashes.

The same configuration as the original build is used (the manifest stores it), so new maps
and tasks are made exactly like the old ones. HM3D datasets need the GLB of every scene
that is sliced or checked on the 3D mesh: new scenes always, the old ones for --add-tasks
and --robots unless --no-verify (tasks are then only checked in 2D, which the manifest
records). Isaac-Scene-Builder datasets need the scene builder's output folder for new
scenes (--source PATH when it moved).
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import shutil
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

from .. import __version__, paths
from ..core.svgmap import SvgMap
from ..dataset import Dataset, load_dataset
from ..dataset.schema import assign_splits_incremental, scene_of
from ..dataset.writer import snapshot_robot, task_file, write_manifest
from ..robots import height_tag, list_robots, load as load_robot
from . import gridify as G
from . import isaac_scene_builder as ISB
from . import slice as S
from . import tasks_grid as TG
from . import tasks_svg as TS
from . import vectorize as VZ
from .verify import GRID_COLS, SVG_COLS, write_csv

log = logging.getLogger(__name__)
MANIFEST_KEYS = {"schema_version", "name", "env", "tool_version", "created", "config", "splits",
                 "robots", "stages", "files"}
HEADER_SKIP = {"schema_version", "dataset", "env", "map_id", "robot", "split", "tasks"}


class ExtendError(ValueError):
    pass


# ------------------------------------------------------------------ inspection

def build_config(ds: Dataset) -> dict:
    """The build config of a dataset with every default filled in. Datasets made by older
    versions store only part of it (no source / slice / height classes): the missing keys
    take the defaults those versions used. Isaac-Scene-Builder configs are complete and
    are not re-validated (that would need the scene builder's output folder)."""
    from .pipeline import load_config
    cfg = copy.deepcopy(ds.manifest["config"])
    if (cfg.get("source") or {}).get("type") == ISB.TYPE:
        return cfg
    cfg.setdefault("env", ds.env_type)
    return load_config(cfg)


def dataset_scenes(ds: Dataset) -> set[str]:
    return {s for v in ds.splits.values() for s in v}


def _glbs(ws: Path | None) -> set[str]:
    return {p.stem for p in paths.glb_dir(ws).glob("*.glb")} if ws else set()


def available_scenes(ds: Dataset, ws: Path | None, source_path=None) -> list[str]:
    """Scenes that could be added: GLBs of the workspace (HM3D) or scenes of the scene
    builder's output (Isaac-Scene-Builder) that are not in the dataset."""
    src = dict(build_config(ds)["source"])
    have = dataset_scenes(ds) | {s for h in ds.manifest.get("history", [])   # tried, no usable map
                                 for s in h.get("tried_scenes", [])}
    if src["type"] == ISB.TYPE:
        if source_path:
            src["path"] = str(source_path)
        if not Path(src["path"]).exists():
            return []
        return sorted({ISB.scene_name(src, i) for i in ISB.resolve(src, "all")} - have)
    return sorted(s for s in _glbs(ws) - have if not S.is_excluded(paths.slice_dir(ws) / s))


def status(name, ws=None, source_path=None) -> dict:
    """What a dataset holds and what `extend` could add to it."""
    ds = load_dataset(name)
    ws = paths.find_workspace(ws, required=False)
    src = build_config(ds)["source"]
    have = dataset_scenes(ds)
    out = {"name": ds.name, "path": str(ds.root), "env": ds.env_type, "source": src["type"],
           "scenes": len(have), "splits": {k: len(v) for k, v in ds.splits.items()},
           "robots": {r: {"maps": v["n_maps"], "tasks": v["n_tasks"]}
                      for r, v in ds.manifest["robots"].items()},
           "workspace": str(ws) if ws else None,
           "new_scenes": available_scenes(ds, ws, source_path),
           "history": ds.manifest.get("history", [])}
    if src["type"] == "hm3d":
        out["scenes_without_glb"] = sorted(have - _glbs(ws))
    if ds.env_type == "svg":
        out["robots_to_add"] = sorted(set(list_robots()) - set(ds.robots))
    return out


# ------------------------------------------------------------------ building blocks

def _need_glbs(ws, scenes, why):
    missing = sorted(set(scenes) - _glbs(ws))
    if missing:
        more = f" ... ({len(missing)} in all)" if len(missing) > 8 else ""
        raise ExtendError(f"{why} needs the GLB files of {missing[:8]}{more} in "
                          f"{paths.glb_dir(ws)} (copy them there, e.g. with build-map), or use "
                          "--no-verify to check the new tasks in 2D only")


def _source(cfg, need: bool):
    src = cfg["source"]
    if src["type"] == ISB.TYPE and need:
        try:
            cfg["source"] = src = ISB.check_source(src)
        except (ValueError, FileNotFoundError) as e:
            raise ExtendError(f"the Isaac-Scene-Builder output of this dataset is needed: {e} "
                              "(give its folder with --source PATH)") from e
    return src


def _isb_ids(src, scenes):
    return [s[len(src["prefix"]) + 1:] for s in scenes]


def _make_maps(ws, cfg, scenes, hc, height, dest: Path) -> dict:
    """SVG maps of `scenes` for height class `hc`, written next to the existing ones
    (built in a staging folder: the map builders remove files they did not make)."""
    src = cfg["source"]
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tmp = Path(tmp)
        if src["type"] == ISB.TYPE and src.get("geometry", "vector") == "vector":
            from . import isb_vector as ISBV                 # needs shapely ([build] extra)
            made = ISBV.run(src, tmp, hc, height, _isb_ids(src, scenes), cfg["map"]["min_free_m2"])
        else:
            if src["type"] == ISB.TYPE:
                ISB.run(ws, src, _isb_ids(src, scenes), [hc])
            else:
                S.run(ws, scenes, S.SliceParams(robot_height=height, **cfg["slice"]))
            made = VZ.run(ws, tmp, hc, height, scenes, VZ.VectorParams(**cfg["map"]))
        for f in tmp.glob("*.svg"):
            shutil.move(str(f), dest / f.name)
    return made


def _make_grids(ws, cfg, scenes, robot: dict, dest: Path) -> dict:
    src = cfg["source"]
    gp = {k: v for k, v in cfg["grid"].items() if k not in ("robot", "robot_size")}
    dest.mkdir(parents=True, exist_ok=True)
    if src["type"] == ISB.TYPE:
        ISB.run(ws, src, _isb_ids(src, scenes), [height_tag(robot["height"])])
    else:
        S.run(ws, scenes, S.SliceParams(robot_height=robot["height"], **cfg["slice"]))
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        made = G.run(ws, Path(tmp), robot, scenes, G.GridParams(**gp))
        for f in Path(tmp).glob("*"):
            shutil.move(str(f), dest / f.name)
    return made


def _append_csv(path: Path, rows: list, cols):
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    write_csv(old + rows, path, cols)


def _rewrite_tasks(root, d: dict, splits, tasks):
    header = {k: v for k, v in d.items() if k not in HEADER_SKIP}
    task_file(root, dataset=d["dataset"], env=d["env"], map_id=d["map_id"], robot=d["robot"],
              splits=splits, tasks=tasks, **header)


def _more_svg_tasks(ws, root, robot, splits, p, map_id, k, gen):
    tf = root / "tasks" / robot.id / f"{map_id}.json"
    d = json.loads(tf.read_text(encoding="utf-8"))
    m = SvgMap.load(root / "maps" / robot.height_class / f"{map_id}.svg")
    rng = np.random.default_rng([p.seed, gen, zlib.crc32(f"{robot.id}/{map_id}".encode())])
    check = TS.checker(ws, m.meta, robot) if p.verify else None
    new, checks, _ = TS.sample_map(m, robot, k, rng, p.min_geo, p.max_geo, p.min_gdr,
                                   p.clearance, check)

    def key(t):
        return (t["start"]["x"], t["start"]["y"], t["goal"]["x"], t["goal"]["y"])
    seen = {key(t) for t in d["tasks"]}
    add = [t for t in new if key(t) not in seen]
    n = len(d["tasks"])
    _rewrite_tasks(root, d, splits, d["tasks"] + [dict(t, id=n + i) for i, t in enumerate(add)])
    return len(add), TS.check_rows(robot.id, map_id, checks)


def _more_grid_tasks(ws, root, robot: dict, splits, p, map_id, k, gen):
    tf = root / "tasks" / robot["id"] / f"{map_id}.json"
    d = json.loads(tf.read_text(encoding="utf-8"))
    meta = json.loads((root / "grids" / f"{map_id}.json").read_text(encoding="utf-8"))
    z = np.load(root / "grids" / f"{map_id}.npz")
    check = TG.checker(ws, meta, robot) if p.verify else None
    rng = np.random.default_rng([p.seed, gen, zlib.crc32(map_id.encode())])
    new, checked = TG.sample_map(z["grid"], z["main"], k, p.d_min, p.budget - p.slack, p.budget,
                                 rng, check, p.repeat_p)
    seen = {(tuple(t["start"]["cell"]), tuple(t["goal"]["cell"])) for t in d["tasks"]}
    add = [t for t in new if (tuple(t["start"]), tuple(t["goal"])) not in seen]
    n = len(d["tasks"])
    _rewrite_tasks(root, d, splits, d["tasks"] + [TG.to_schema(n + i, t, meta) for i, t in enumerate(add)])
    return len(add), TG.check_rows(map_id, checked)


# ------------------------------------------------------------------ extend

def extend(name, *, ws=None, out=None, in_place=False, scenes=None, new_scenes=False,
           add_tasks=0, robots=None, verify=True, source_path=None, from_hf=False,
           token=None) -> Path:
    """Add scenes / tasks / robots to a dataset (see the module docstring). Returns the
    directory of the extended dataset."""
    ws = paths.find_workspace(ws) if ws else paths.ensure_workspace()
    try:
        ds = load_dataset(name)
    except FileNotFoundError:
        if not from_hf:
            raise
        from .. import download as D
        ds = Dataset(D.download([name], token=token)[0])
    man = ds.manifest
    cfg = build_config(ds)
    env = cfg["env"]
    if source_path:
        cfg["source"]["path"] = str(source_path)
    have = dataset_scenes(ds)

    # ---------------------------------------------------------------- what to do
    if scenes:
        dup = sorted(set(scenes) & have)
        if dup:
            raise ExtendError(f"already in {ds.name}: {dup}")
        todo_scenes = sorted(set(scenes))
    elif new_scenes:
        todo_scenes = available_scenes(ds, ws, source_path)
        if not todo_scenes:
            where = (paths.glb_dir(ws) if cfg["source"]["type"] == "hm3d" else cfg["source"].get("path"))
            raise ExtendError(f"no new scene found in {where}")
    else:
        todo_scenes = []
    new_robots = [r for r in (robots or []) if r not in ds.robots]
    if robots and env != "svg":
        raise ExtendError("--robots is for SVG datasets (a grid dataset is built for one robot size)")
    if not (todo_scenes or add_tasks or new_robots):
        raise ExtendError("nothing to do: give --new-scenes, --scenes, --add-tasks or --robots "
                          "(--status shows what can be added)")
    hm3d = cfg["source"]["type"] == "hm3d"
    have_hcs = {v["height_class"] for v in man["robots"].values()}
    new_hcs = {load_robot(r).height_class for r in new_robots} - have_hcs
    _source(cfg, need=bool(todo_scenes or new_hcs))
    if hm3d and todo_scenes:
        _need_glbs(ws, todo_scenes, "slicing the new scenes")
    if cfg["source"]["type"] == "hm3d" and verify and (add_tasks or new_robots):
        _need_glbs(ws, have, "checking the new tasks on the 3D mesh")

    # ---------------------------------------------------------------- target
    if in_place:
        root, out_name = ds.root, ds.name
    else:
        out_name = out or f"{ds.name}-ext"
        root = ws / "datasets" / out_name
        if root.exists():
            raise ExtendError(f"{root} exists: extend it with `hm3d extend {out_name} --in-place ...`, "
                              "or choose another --out")
        log.info("copying %s -> %s", ds.root, root)
        shutil.copytree(ds.root, root)
    splits = {k: list(v) for k, v in man["splits"].items()}
    info = {r: {k: v for k, v in i.items() if k not in ("n_maps", "n_tasks", "flagged")}
            for r, i in man["robots"].items()}
    tasks_cfg = dict(cfg["tasks"], verify=bool(cfg["tasks"].get("verify", True) and verify))
    gen = len(man.get("history", [])) + 1
    added = {"scenes": [], "maps": 0, "tasks": 0, "robots": new_robots}

    def csv_path(rid):
        return root / "verify" / f"{rid}.csv"

    # ---------------------------------------------------------------- new scenes
    if todo_scenes:
        log.info("new scenes: %s", todo_scenes)
        if env == "svg":
            p = TS.SvgTaskParams(**tasks_cfg)
            made_by_hc = {}
            for hc in sorted({v["height_class"] for v in info.values()}):
                made_by_hc[hc] = _make_maps(ws, cfg, todo_scenes, hc, cfg["height_classes"][hc],
                                            root / "maps" / hc)
            new_maps = sorted({m for v in made_by_hc.values() for m in v})
            splits = assign_splits_incremental(splits, {scene_of(m) for m in new_maps},
                                               cfg["split"], cfg["split"]["seed"])
            for rid, v in info.items():
                robot = ds.robot(rid)
                rows = []
                for m in sorted(made_by_hc[v["height_class"]]):
                    st, r = TS.map_tasks(ws, root, out_name, robot, splits, p,
                                         root / "maps" / v["height_class"] / f"{m}.svg")
                    v["maps"][m] = st
                    v.setdefault("map_stats", {})[m] = made_by_hc[v["height_class"]][m]
                    added["tasks"] += st["tasks"]
                    rows += r
                _append_csv(csv_path(rid), rows, SVG_COLS)
        else:
            p = TG.GridTaskParams(**tasks_cfg)
            rid = next(iter(info))
            robot = json.loads((root / "robots" / f"{rid}.json").read_text(encoding="utf-8"))
            made = _make_grids(ws, cfg, todo_scenes, robot, root / "grids")
            new_maps = sorted(made)
            splits = assign_splits_incremental(splits, {scene_of(m) for m in new_maps},
                                               cfg["split"], cfg["split"]["seed"])
            rows = []
            for m in new_maps:
                st, r = TG.map_tasks(ws, root, out_name, robot, splits, p, root / "grids" / f"{m}.json")
                info[rid]["maps"][m] = st
                added["tasks"] += st["tasks"]
                rows += r
            _append_csv(csv_path(rid), rows, GRID_COLS)
        added["maps"] = len(new_maps)
        added["scenes"] = sorted({scene_of(m) for m in new_maps})
        if cfg["scenes"] != "all":
            cfg["scenes"] = sorted(set(cfg["scenes"]) | set(added["scenes"]))
        skipped = sorted(set(todo_scenes) - set(added["scenes"]))
        if skipped:
            log.warning("no usable map in %s (too small, or no free floor)", skipped)

    # ---------------------------------------------------------------- more tasks
    if add_tasks:
        old_maps = {rid: [m for m in v["maps"] if scene_of(m) in have
                          and (root / "tasks" / rid / f"{m}.json").exists()] for rid, v in info.items()}
        for rid, ms in old_maps.items():
            rows = []
            if env == "svg":
                robot, p = ds.robot(rid), TS.SvgTaskParams(**tasks_cfg)
                more = _more_svg_tasks
            else:
                robot = json.loads((root / "robots" / f"{rid}.json").read_text(encoding="utf-8"))
                p, more = TG.GridTaskParams(**tasks_cfg), _more_grid_tasks
            for m in ms:
                n, r = more(ws, root, robot, splits, p, m, add_tasks, gen)
                info[rid]["maps"][m] = dict(info[rid]["maps"][m], tasks=info[rid]["maps"][m].get("tasks", 0) + n)
                added["tasks"] += n
                rows += r
                log.info("  %s %s: +%d tasks", rid, m, n)
            _append_csv(csv_path(rid), rows, SVG_COLS if env == "svg" else GRID_COLS)

    # ---------------------------------------------------------------- new robots
    for rid in new_robots:
        robot = load_robot(rid)
        hc = robot.height_class
        if hc not in cfg["height_classes"]:
            raise ExtendError(f"{rid}: unknown height class {hc}")
        if robot.height > cfg["height_classes"][hc] + 1e-9:
            raise ExtendError(f"{rid}: height {robot.height} m above its class {hc}")
        mdir = root / "maps" / hc
        if not mdir.exists():
            all_scenes = sorted(have | set(added["scenes"]))
            if hm3d:
                _need_glbs(ws, all_scenes, f"a new height class ({hc}, for {rid})")
            made = _make_maps(ws, cfg, all_scenes, hc, cfg["height_classes"][hc], mdir)
        else:
            made = {}
            for v in info.values():
                if v["height_class"] == hc:
                    made.update(v.get("map_stats", {}))
        snapshot_robot(root, rid, robot.raw)
        p = TS.SvgTaskParams(**tasks_cfg)
        per_map, rows = {}, []
        for sp in sorted(mdir.glob("*.svg")):
            st, r = TS.map_tasks(ws, root, out_name, robot, splits, p, sp)
            per_map[sp.stem] = st
            added["tasks"] += st["tasks"]
            rows += r
        write_csv(rows, csv_path(rid), SVG_COLS)
        info[rid] = {"height_class": hc, "maps": per_map,
                     "map_stats": {m: made[m] for m in per_map if m in made}}
        if cfg.get("robots") not in (None, "all"):
            cfg["robots"] = sorted(set(cfg["robots"]) | {rid})
        log.info("robot %s: %d maps, %d tasks", rid, sum(1 for v in per_map.values() if v["tasks"]),
                 sum(v["tasks"] for v in per_map.values()))

    # ---------------------------------------------------------------- manifest
    extra = {k: v for k, v in man.items() if k not in MANIFEST_KEYS}
    src = extra.get("source") or dict(cfg["source"])
    if not tasks_cfg["verify"] and cfg["source"]["type"] == "hm3d" and (add_tasks or new_robots):
        src = dict(src, verified_3d="partial")
    extra["source"] = src
    extra["history"] = list(man.get("history", [])) + [{
        "date": time.strftime("%Y-%m-%d %H:%M"), "from": ds.name, "tool_version": __version__,
        "added_scenes": added["scenes"], "tried_scenes": sorted(set(todo_scenes) - set(added["scenes"])),
        "added_maps": added["maps"], "added_tasks": added["tasks"],
        "added_robots": added["robots"], "tasks_per_map": add_tasks or None,
        "verified_3d": bool(tasks_cfg["verify"])}]
    write_manifest(root, name=out_name, env=env, config=cfg, splits=splits, robots=info,
                   stages={}, **extra)
    return root
