"""Grid task stage: physically valid (start, goal) pairs on the grids of the grid stage.

  * only cells of the main region (real floor, indoors, the body fits the cell)
  * every candidate cell is checked on the 3D mesh (verify.check_cell); failures are
    discarded and resampled, every check is logged to verify/<robot>.csv
  * start -> goal in the same component, d_bfs in [d_min, budget - slack]

Distance stratification: after drawing a start, the reachable distances are split into
4 quantile bins and the target bin rotates, so every map gets easy to hard tasks
instead of piling up around the mean distance.

Difficulty labels: p0 = sa_uniform (exact absorbing Markov chain),
sa_persist (direction-keeping probe, repeat_p), starved/level from the thresholds in
core.gridcore (level_of, STARVED_SA).
"""

from __future__ import annotations

import json
import logging
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.gridcore import Chain, bfs, cell_to_world, level_of
from ..dataset.writer import task_file
from .verify import GRID_COLS, check_cell, write_csv

log = logging.getLogger(__name__)
N_BINS = 4


@dataclass
class GridTaskParams:
    k: int = 20                  # tasks per map
    d_min: int = 8               # min d_bfs (steps)
    budget: int = 200            # episode step budget
    slack: int = 20              # d_max = budget - slack
    repeat_p: float = 0.6        # sa_persist probe
    seed: int = 0
    flag_fail: float = 0.10      # flag maps whose mesh-check failure rate exceeds this
    verify: bool = True          # False: skip the mesh checks (fast, NOT physically checked)


def map_rng(seed, map_id):
    return np.random.default_rng([seed, zlib.crc32(map_id.encode())])


def sample_map(grid, main, k, d_min, d_max, budget, rng, check=None, repeat_p=0.6, max_tries=None):
    """(tasks, checks). `check(r, c)` -> dict with key `ok`; None skips the mesh checks
    (tests only)."""
    free = grid == 0
    cand = np.argwhere(main)
    chain = Chain(free)
    checked = {}

    def ok(cell):
        key = (int(cell[0]), int(cell[1]))
        if check is None:
            return True
        if key not in checked:
            checked[key] = check(*key)
        return checked[key]["ok"]

    tasks, seen = [], set()
    max_tries = max_tries or 50 * k
    for _ in range(max_tries):
        if len(tasks) >= k or not len(cand):
            break
        s = cand[rng.integers(len(cand))]
        if not ok(s):
            continue
        dist = bfs(free, s)
        valid = main & (dist >= d_min) & (dist <= d_max)
        cells = np.argwhere(valid)
        if not len(cells):
            continue
        dv = dist[cells[:, 0], cells[:, 1]]
        b = len(tasks) % N_BINS
        lo, hi = np.quantile(dv, [b / N_BINS, (b + 1) / N_BINS])
        pool = cells[(dv >= lo) & (dv <= hi)]
        if not len(pool):
            # interpolated quantiles can fall between two real values (few candidates):
            # empty bin, take the candidate nearest the bin centre
            pool = cells[[np.argmin(np.abs(dv - (lo + hi) / 2))]]
        g = pool[rng.integers(len(pool))]
        key = (tuple(int(x) for x in s), tuple(int(x) for x in g))
        if key in seen or not ok(g):
            continue
        seen.add(key)
        si = chain.idx(s)
        sa_u = float(chain.sa_uniform(g, budget)[si])
        sa_p = float(chain.sa_persist(g, budget, repeat_p)[si])
        tasks.append({"start": list(key[0]), "goal": list(key[1]),
                      "d_bfs": int(dist[g[0], g[1]]), "p0": sa_u,
                      "sa_uniform": sa_u, "sa_persist": sa_p,
                      "starved": bool(level_of(sa_u) == "starved"), "level": level_of(sa_u)})
    return tasks, checked


LABEL_KEYS = ("d_bfs", "p0", "sa_uniform", "sa_persist", "starved", "level")


def to_schema(i, t, meta) -> dict:
    """A sampled task in schema 2.0 (metres in the map frame + grid cells)."""
    def pt(cell):
        p = cell_to_world(meta, *cell)
        return {"x": round(p[meta["ua"]], 4), "y": round(p[meta["va"]], 4), "cell": list(cell)}
    s, g = pt(t["start"]), pt(t["goal"])
    return {"id": i, "start": s, "goal": g,
            "geodesic_m": round(t["d_bfs"] * meta["cell_m"], 4),
            "euclidean_m": round(math.hypot(g["x"] - s["x"], g["y"] - s["y"]), 4),
            "labels": {k: t[k] for k in LABEL_KEYS}}


def run(ws, root: Path, dataset: str, robot: dict, splits: dict,
        params: GridTaskParams | None = None) -> dict:
    """Sample tasks on every grid of the dataset for `robot` (preset dict). Writes
    tasks/<robot>/<map_id>.json and verify/<robot>.csv; returns per-map statistics."""
    p = params or GridTaskParams()
    root = Path(root)
    rid = robot["id"]
    tdir = root / "tasks" / rid
    tdir.mkdir(parents=True, exist_ok=True)
    d_max = p.budget - p.slack
    rows, per_map = [], {}
    for mj in sorted((root / "grids").glob("*.json")):
        map_id = mj.stem
        meta = json.loads(mj.read_text(encoding="utf-8"))
        z = np.load(root / "grids" / f"{map_id}.npz")
        check = (lambda r, c, meta=meta: check_cell(ws, meta, r, c, robot["height"])) if p.verify else None
        tasks, checked = sample_map(z["grid"], z["main"], p.k, p.d_min, d_max, p.budget,
                                    map_rng(p.seed, map_id), check, p.repeat_p)
        for (r, c), chk in checked.items():
            rows.append({"map_id": map_id, "task": "", "which": "candidate", "r": r, "c": c,
                         **dict(zip("xyz", chk["world"])),
                         **{k: chk[k] for k in ("support", "clear_hits", "margin_hits", "roof", "ok")},
                         "bfs_ok": ""})
        n_chk = len(checked)
        fail = sum(not v["ok"] for v in checked.values()) / n_chk if n_chk else 0.0
        flags = list(meta.get("flags", []))
        if fail > p.flag_fail:
            flags.append(f"verify_fail {fail:.0%}")
        if len(tasks) < p.k:
            flags.append(f"only {len(tasks)}/{p.k} tasks")
        levels = {lv: sum(t["level"] == lv for t in tasks) for lv in ("easy", "medium", "hard", "starved")}
        per_map[map_id] = {"tasks": len(tasks), "checked": n_chk, "verify_fail": round(fail, 3),
                           "levels": levels, "shape": meta["shape"], "main_m2": meta["main_m2"],
                           "flags": flags}
        out = tdir / f"{map_id}.json"
        if not tasks:
            # e.g. a roof terrace: every cell fails the roof test -> not in the dataset
            out.unlink(missing_ok=True)
            log.info("  %s: 0 tasks, left out  FLAG %s", map_id, flags)
            continue
        task_file(root, dataset=dataset, env="grid", map_id=map_id, robot=rid, splits=splits,
                  budget=p.budget, cell_m=meta["cell_m"],
                  tasks=[to_schema(i, t, meta) for i, t in enumerate(tasks)])
        log.info("  %s: %d tasks, %d cells checked, %.0f%% failed, %s%s", map_id, len(tasks),
                 n_chk, fail * 100, levels, f"  FLAG {flags}" if flags else "")
    for old in tdir.glob("*.json"):                   # grids removed by gridify
        if not per_map.get(old.stem, {}).get("tasks"):
            old.unlink()
    write_csv(rows, root / "verify" / f"{rid}.csv", GRID_COLS)
    return per_map
