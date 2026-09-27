"""Writing datasets: task files, robot snapshots and the manifest."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import __version__
from ..robots import preset_path
from .schema import SCHEMA_VERSION, file_hashes, split_of, write_json


def task_file(root: Path, *, dataset: str, env: str, map_id: str, robot: str, splits: dict,
              tasks: list, **header) -> Path:
    """Write tasks/<robot>/<map_id>.json (schema 2.0)."""
    p = Path(root) / "tasks" / robot / f"{map_id}.json"
    write_json(p, {"schema_version": SCHEMA_VERSION, "dataset": dataset, "env": env,
                   "map_id": map_id, "robot": robot, "split": split_of(splits, map_id),
                   **header, "tasks": tasks})
    return p


def snapshot_robot(root: Path, robot_id: str, raw: dict | None = None) -> None:
    """Copy the preset used for building into the dataset (robots/<id>.json)."""
    dst = Path(root) / "robots" / f"{robot_id}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        dst.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    else:
        dst.write_text(preset_path(robot_id).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")


def write_manifest(root: Path, *, name: str, env: str, config: dict, splits: dict,
                   robots: dict, stages: dict | None = None, **extra) -> dict:
    """robots: {robot_id: {"height_class": ..., "maps": {map_id: {...stats, "flags": [...]}}}}.
    Task/map counts are recomputed from the task files on disk."""
    root = Path(root)
    rob = {}
    for rid, info in robots.items():
        files = sorted((root / "tasks" / rid).glob("*.json"))
        n_tasks = sum(len(json.loads(f.read_text(encoding="utf-8"))["tasks"]) for f in files)
        maps = info.get("maps", {})
        rob[rid] = {**{k: v for k, v in info.items() if k != "maps"},
                    "n_maps": len(files), "n_tasks": n_tasks,
                    "flagged": sorted(m for m, v in maps.items() if v.get("flags")),
                    "maps": maps}
    man = {"schema_version": SCHEMA_VERSION, "name": name, "env": env,
           "tool_version": __version__, "created": time.strftime("%Y-%m-%d %H:%M"),
           "config": config, "splits": splits, "robots": rob, "stages": stages or {},
           **extra, "files": file_hashes(root)}
    write_json(root / "manifest.json", man)
    return man
