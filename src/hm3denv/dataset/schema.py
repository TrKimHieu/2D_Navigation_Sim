"""On-disk formats of a dataset (schema 2.0).

Task file ``tasks/<robot>/<map_id>.json``::

    {"schema_version": "2.0", "dataset": ..., "env": "svg"|"grid", "map_id": ...,
     "robot": ..., "split": "train"|"val"|"test", "success_radius": 0.2, ...,
     "tasks": [{"id": 0,
                "start": {"x": .., "y": .., "theta": ..}      # svg (grid: "cell": [r, c], no theta)
                "goal":  {"x": .., "y": ..},                  # grid: + "cell": [r, c]
                "geodesic_m": .., "euclidean_m": ..,
                "labels": {...}}]}                            # svg: gdr; grid: d_bfs, p0, sa_*, level

All coordinates are metres in the map frame (GLB horizontal axes).

``manifest.json`` records the tool version, the config and its hash, the split of
every scene, per-robot statistics and flags, build stage stamps, and the sha256 of
every file so ``validate`` can detect edits, missing files or schema drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "2.0"
SPLITS = ("train", "val", "test")


def scene_of(map_id: str) -> str:
    return map_id.rsplit("_s", 1)[0]


def assign_splits(scenes, fractions: dict, seed: int = 0) -> dict[str, list[str]]:
    """Deterministic split BY SCENE (a house never appears in two splits): scenes are
    ordered by sha1(f"{seed}:{scene}") and cut by the given fractions."""
    total = sum(fractions.get(s, 0.0) for s in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    order = sorted(set(scenes), key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
    n = len(order)
    n_train = round(fractions.get("train", 0) * n)
    n_val = round(fractions.get("val", 0) * n)
    return {"train": sorted(order[:n_train]), "val": sorted(order[n_train:n_train + n_val]),
            "test": sorted(order[n_train + n_val:])}


def assign_splits_incremental(old: dict, new_scenes, fractions: dict, seed: int = 0) -> dict:
    """Splits of an extended dataset: every scene of `old` keeps its split (tasks already
    used for training never move to test); the new scenes, in the order of assign_splits,
    each go to the split furthest below its target share of the new total."""
    known = {s for v in old.values() for s in v}
    new = [s for s in sorted(set(new_scenes) - known,
                             key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())]
    out = {k: sorted(old.get(k, [])) for k in SPLITS}
    total = len(known)
    for s in new:
        total += 1
        k = max(SPLITS, key=lambda k: (fractions.get(k, 0.0) * total - len(out[k]),
                                       -SPLITS.index(k)))
        out[k].append(s)
    return {k: sorted(v) for k, v in out.items()}


def split_of(splits: dict, map_id: str) -> str:
    sc = scene_of(map_id)
    for name, scenes in splits.items():
        if sc in scenes:
            return name
    raise KeyError(f"scene {sc} is in no split")


def write_json(path: Path, obj) -> None:
    """LF line endings on every OS: dataset files are hashed byte for byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


def read_tasks(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: task schema {d.get('schema_version')} (expected "
                         f"{SCHEMA_VERSION}; rebuild the dataset with this version of hm3denv)")
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HASHED_DIRS = ("maps", "grids", "tasks", "robots")


def file_hashes(root: Path) -> dict[str, str]:
    """sha256 of every data file the environments read."""
    out = {}
    for d in HASHED_DIRS:
        for p in sorted((root / d).rglob("*")) if (root / d).exists() else []:
            if p.is_file():
                out[p.relative_to(root).as_posix()] = sha256(p)
    return out


def validate(root: Path) -> list[str]:
    """Problems found in a dataset directory (empty list = valid)."""
    root = Path(root)
    problems = []
    mp = root / "manifest.json"
    if not mp.exists():
        return ["manifest.json missing"]
    man = json.loads(mp.read_text(encoding="utf-8"))
    if man.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema {man.get('schema_version')}")
    recorded = man.get("files", {})
    actual = file_hashes(root)
    for f in sorted(set(recorded) - set(actual)):
        problems.append(f"missing file: {f}")
    for f in sorted(set(actual) - set(recorded)):
        problems.append(f"file not in manifest: {f}")
    for f in sorted(set(recorded) & set(actual)):
        if recorded[f] != actual[f]:
            problems.append(f"modified file: {f}")
    for p in sorted((root / "tasks").rglob("*.json")) if (root / "tasks").exists() else []:
        try:
            read_tasks(p)
        except ValueError as e:
            problems.append(str(e))
    return problems
