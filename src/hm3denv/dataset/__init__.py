"""Loading built datasets.

    ds = hm3denv.load_dataset("svg-v1")          # name (searched) or path
    ds.maps("turtlebot4", split="train")
    ds.tasks("turtlebot4", map_id)
    env = ds.make_env(robot="turtlebot4", split="train")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import paths
from ..robots import Robot, load as load_robot
from .schema import SCHEMA_VERSION, SPLITS, read_tasks, scene_of, split_of, validate  # noqa: F401


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.root}: manifest schema {self.manifest.get('schema_version')}")
        self._task_cache: dict = {}
        self._grid_meta: dict = {}

    def __repr__(self):
        return f"Dataset({self.name!r}, env={self.env_type!r}, robots={self.robots})"

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def env_type(self) -> str:
        return self.manifest["env"]

    @property
    def robots(self) -> list[str]:
        return sorted(r for r, v in self.manifest["robots"].items() if v.get("n_tasks"))

    @property
    def splits(self) -> dict[str, list[str]]:
        return self.manifest["splits"]

    def robot(self, robot_id: str) -> Robot:
        """The robot as it was when the dataset was built (snapshot in robots/)."""
        return load_robot(robot_id, extra=self.root / "robots")

    def _check_robot(self, robot_id):
        if robot_id not in self.manifest["robots"]:
            raise KeyError(f"robot '{robot_id}' not in dataset {self.name}; available: {self.robots}")

    def maps(self, robot: str, split: str | None = None) -> list[str]:
        self._check_robot(robot)
        ids = sorted(p.stem for p in (self.root / "tasks" / robot).glob("*.json"))
        if split is None:
            return ids
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return [m for m in ids if scene_of(m) in self.splits[split]]

    def task_file(self, robot: str, map_id: str) -> dict:
        key = (robot, map_id)
        if key not in self._task_cache:
            self._task_cache[key] = read_tasks(self.root / "tasks" / robot / f"{map_id}.json")
        return self._task_cache[key]

    def tasks(self, robot: str, map_id: str) -> list[dict]:
        return self.task_file(robot, map_id)["tasks"]

    # ------------------------------------------------------------ map files

    def svg_path(self, robot: str, map_id: str) -> Path:
        hc = self.manifest["robots"][robot]["height_class"]
        return self.root / "maps" / hc / f"{map_id}.svg"

    def grid_arrays(self, map_id: str) -> dict:
        z = np.load(self.root / "grids" / f"{map_id}.npz")
        return {k: z[k] for k in z.files}

    def grid_meta(self, map_id: str) -> dict:
        if map_id not in self._grid_meta:
            p = self.root / "grids" / f"{map_id}.json"
            self._grid_meta[map_id] = json.loads(p.read_text(encoding="utf-8"))
        return self._grid_meta[map_id]

    def make_env(self, robot: str | None = None, **kwargs):
        """Environment of the right type for this dataset (same as gymnasium.make)."""
        if self.env_type == "svg":
            from ..envs.svg import SvgEnv
            return SvgEnv(dataset=self, robot=robot, **kwargs)
        from ..envs.grid import GridEnv
        return GridEnv(dataset=self, robot=robot, **kwargs)

    def validate(self) -> list[str]:
        return validate(self.root)


def load_dataset(name_or_path, search=None) -> Dataset:
    if isinstance(name_or_path, Dataset):
        return name_or_path
    return Dataset(paths.resolve_dataset(name_or_path, search))


def list_datasets(search=None) -> list[Dataset]:
    seen, out = set(), []
    for d in paths.dataset_dirs(search):
        for mp in sorted(d.glob("*/manifest.json")) if d.exists() else []:
            if mp.parent.name not in seen:
                seen.add(mp.parent.name)
                try:
                    out.append(Dataset(mp.parent))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    return out
