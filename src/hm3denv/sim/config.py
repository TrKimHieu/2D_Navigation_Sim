"""Simulation session configuration (``hm3d sim CONFIG.yaml``, ``sim.bat CONFIG.yaml``).

Every key is optional; unknown keys are errors, so a typo never silently falls back to a
default::

    dataset: isb-svg-v1          # name (data/datasets, the demos, ...) or a path
    robot: turtlebot4            # default: the dataset's only robot, else turtlebot4 / the first
    split: train                 # train | val | test | null (every split)
    num_envs: 4                  # environments run side by side
    vectorization: async         # async (one process per env) | sync (same process)
    envs_per_worker: 1           # async: environments per worker process
    seed: 0
    maps: [isb-0001_s0]          # random episodes only on these maps (default: the split)
    episodes:                    # fixed episodes, played in turn (default: random tasks)
      - {map: isb-0001_s0, task_idx: 3}                      # a task of the dataset
      - {map: isb-0001_s0, start: [1.2, 3.4, 0.0], goal: [5.0, 2.0]}   # your own (metres;
                                                             # grid: [row, col] cells)
    task_filter: {geodesic_m: {min: 1, max: 8}}    # random tasks only (see envs.base)
    env: {dt: 0.1, n_beams: 72, reward: dense}      # any argument of the environment
    auto_reset: true             # start the next episode as soon as one ends
    server: {bind: "tcp://127.0.0.1:5555"}         # hm3d sim --serve
    viewer: {host: 127.0.0.1, port: 8770}          # hm3d sim --view
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

KEYS = {"dataset", "robot", "split", "num_envs", "vectorization", "envs_per_worker", "seed",
        "maps", "episodes", "task_filter", "env", "auto_reset", "server", "viewer"}
EPISODE_KEYS = {"map", "task_idx", "start", "goal"}
DEFAULTS = {"dataset": "demo-svg", "robot": None, "split": None, "num_envs": 1,
            "vectorization": "async", "envs_per_worker": 1, "seed": 0, "maps": None,
            "episodes": None, "task_filter": None, "env": {}, "auto_reset": True,
            "server": {"bind": "tcp://127.0.0.1:5555"},
            "viewer": {"host": "127.0.0.1", "port": 8770}}


class SimConfigError(ValueError):
    pass


def load_sim_config(src=None, **overrides) -> dict:
    """Validated session config from a path, YAML text, a dict or nothing, with
    `overrides` (keys of the config, None = not given) applied on top."""
    if src is None:
        cfg = {}
    elif isinstance(src, dict):
        cfg = copy.deepcopy(src)
    else:
        p = resolve_config(src)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8") if p else str(src)) or {}
    if not isinstance(cfg, dict):
        raise SimConfigError("a session config must be a mapping")
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    bad = sorted(set(cfg) - KEYS)
    if bad:
        raise SimConfigError(f"unknown key(s) {bad}; allowed: {sorted(KEYS)}")
    out = copy.deepcopy(DEFAULTS)
    for k in ("server", "viewer"):
        given = cfg.pop(k, None) or {}
        if not isinstance(given, dict) or set(given) - set(DEFAULTS[k]):
            raise SimConfigError(f"'{k}' takes {sorted(DEFAULTS[k])}")
        out[k].update(given)
    out.update(cfg)

    if out["split"] not in (None, "train", "val", "test"):
        raise SimConfigError(f"split must be train, val, test or null, got {out['split']!r}")
    for k in ("num_envs", "envs_per_worker"):
        if not isinstance(out[k], int) or out[k] < 1:
            raise SimConfigError(f"{k} must be an integer >= 1")
    if out["vectorization"] not in ("async", "sync"):
        raise SimConfigError("vectorization must be 'async' or 'sync'")
    if not isinstance(out["env"], dict):
        raise SimConfigError("'env' must be a mapping of environment arguments")
    reserved = sorted(set(out["env"]) & {"dataset", "robot", "split", "map_id", "task_idx", "task_filter"})
    if reserved:
        raise SimConfigError(f"set {reserved} at the top level of the config, not under 'env'")
    if out["maps"] is not None:
        if isinstance(out["maps"], str):
            out["maps"] = [out["maps"]]
        if not isinstance(out["maps"], list) or not out["maps"]:
            raise SimConfigError("maps must be a non-empty list of map ids")
    if out["task_filter"] is not None and not isinstance(out["task_filter"], dict):
        raise SimConfigError("task_filter must be a mapping")
    if out["episodes"] is not None:
        if not isinstance(out["episodes"], list) or not out["episodes"]:
            raise SimConfigError("episodes must be a non-empty list")
        out["episodes"] = [_episode(e, i) for i, e in enumerate(out["episodes"])]
    return out


PACKAGED = Path(__file__).resolve().parents[1] / "configs" / "sim"


def packaged_configs() -> list[str]:
    """Names of the session configs shipped with hm3denv (`hm3d sim demo`)."""
    return sorted(p.stem for p in PACKAGED.glob("*.yaml"))


def resolve_config(src) -> Path | None:
    """A config file: a path, or the name of a packaged config ('demo'). None when `src`
    is YAML text; SimConfigError when it looks like a file name that does not exist."""
    text = str(src)
    if "\n" in text or ": " in text or text.startswith("{"):        # YAML text
        return None
    p = Path(text)
    if p.is_file():
        return p
    q = PACKAGED / (p.name if p.suffix in (".yaml", ".yml") else f"{p.name}.yaml")
    if len(p.parts) == 1 and q.is_file():
        return q
    raise SimConfigError(f"config not found: {src} (a YAML file, or one of {packaged_configs()})")


def _episode(e, i) -> dict:
    if not isinstance(e, dict):
        raise SimConfigError(f"episodes[{i}] must be a mapping like {{map: ..., task_idx: 0}}")
    e = dict(e)
    if "task" in e and "task_idx" not in e:
        e["task_idx"] = e.pop("task")
    bad = sorted(set(e) - EPISODE_KEYS)
    if bad:
        raise SimConfigError(f"episodes[{i}]: unknown key(s) {bad}; allowed: {sorted(EPISODE_KEYS)}")
    custom = ("start" in e) + ("goal" in e)
    if custom == 1:
        raise SimConfigError(f"episodes[{i}]: give both start and goal")
    if custom and "task_idx" in e:
        raise SimConfigError(f"episodes[{i}]: give either task_idx or start + goal, not both")
    if "task_idx" in e and not isinstance(e["task_idx"], int):
        raise SimConfigError(f"episodes[{i}]: task_idx must be an integer")
    return dict(e)


def episode_options(e: dict) -> dict:
    """Reset options of an episode entry."""
    opt = {}
    if e.get("map") is not None:
        opt["map_id"] = e["map"]
    for k in ("task_idx", "start", "goal"):
        if k in e:
            opt[k] = e[k]
    return opt


def parse_point(text: str | None, n_min: int, n_max: int) -> list[float] | None:
    """'1.2,3.4[,0.5]' -> [1.2, 3.4, 0.5] (command-line start / goal)."""
    if text is None:
        return None
    try:
        v = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    except ValueError:
        raise SimConfigError(f"expected numbers separated by commas, got {text!r}") from None
    if not n_min <= len(v) <= n_max:
        raise SimConfigError(f"expected {n_min}-{n_max} numbers, got {text!r}")
    return v
