"""SVG task stage: PointGoal tasks (start = SE(2) pose, goal = point) for the continuous
env, using each robot's real footprint and height class.

  * start: a point of C_disk (centred there, the footprint fits at any heading), uniform
    theta, checked exactly with the footprint geometry against the map edges
  * goal: same connected component; geodesic in [min_geo, max_geo] m and
    GDR = geodesic / euclidean >= min_gdr (drops near-straight episodes)
  * stratified: reachable geodesic distances split into 4 quantile bins, rotating
  * every pose is checked on the 3D mesh (verify.check_footprint): start with the real
    footprint at theta, goal with the polygon circumscribing the circumscribed circle
    (any heading); failures are resampled, every check goes to verify/<robot>.csv
"""

from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.geometry import Collider, place
from ..core.planning import MAX_ANISO_ERR, Planner
from ..core.svgmap import SvgMap
from ..dataset.writer import task_file
from ..robots import Robot, footprint_polygon
from .verify import SVG_COLS, check_footprint, storey_probe, write_csv

log = logging.getLogger(__name__)
N_BINS = 4


@dataclass
class SvgTaskParams:
    k: int = 20                  # tasks per map
    min_geo: float = 1.0         # geodesic range (m)
    max_geo: float = 30.0
    min_gdr: float = 1.1         # min geodesic / euclidean
    clearance: float = 0.05      # planning clearance added to the circumradius (m)
    success_radius: float = 0.2
    seed: int = 0
    flag_fail: float = 0.10
    verify: bool = True


def sample_map(svgmap, robot, k, rng, min_geo, max_geo, min_gdr, clearance, check=None,
               max_tries=None):
    """(tasks, 3D checks, planner). check(poly_world, which) -> dict with 'ok'."""
    col = Collider(svgmap)
    pl = Planner(svgmap, robot.r_circ, clearance)
    if not len(pl.cells):
        return [], [], pl
    disk = footprint_polygon({"type": "circle", "radius": robot.r_circ})
    centers = np.array([pl.center(r, c) for r, c in pl.cells])
    tasks, checks, goal_ok = [], [], {}
    for _ in range(max_tries or 40 * k):
        if len(tasks) >= k:
            break
        i = int(rng.integers(len(pl.cells)))
        sx, sy = centers[i]
        th = float(rng.uniform(-math.pi, math.pi))
        if not col.pose_valid(robot.footprint, sx, sy, th):
            continue
        if check is not None:
            c = check(place(robot.footprint, sx, sy, th), "start")
            checks.append(("start", sx, sy, th, c))
            if not c["ok"]:
                continue
        f = pl.field(sx, sy)
        geo = f[pl.cells[:, 0], pl.cells[:, 1]]
        euc = np.hypot(centers[:, 0] - sx, centers[:, 1] - sy)
        with np.errstate(divide="ignore", invalid="ignore"):
            gdr = geo / euc
        cand = np.where(np.isfinite(geo) & (geo >= min_geo) & (geo <= max_geo) & (gdr >= min_gdr))[0]
        if not len(cand):
            continue
        b = len(tasks) % N_BINS
        lo, hi = np.quantile(geo[cand], [b / N_BINS, (b + 1) / N_BINS])
        pool = cand[(geo[cand] >= lo) & (geo[cand] <= hi)]
        if not len(pool):
            # interpolated quantiles can fall between two real values: empty bin, take the
            # candidate nearest the bin centre
            pool = cand[[np.argmin(np.abs(geo[cand] - (lo + hi) / 2))]]
        j = int(pool[rng.integers(len(pool))])
        gx, gy = centers[j]
        if check is not None:
            if j not in goal_ok:
                c = check(place(disk, gx, gy, 0.0), "goal")
                checks.append(("goal", gx, gy, None, c))
                goal_ok[j] = c["ok"]
            if not goal_ok[j]:
                continue
        tasks.append({"id": len(tasks),
                      "start": {"x": round(float(sx), 4), "y": round(float(sy), 4), "theta": round(th, 4)},
                      "goal": {"x": round(float(gx), 4), "y": round(float(gy), 4)},
                      "geodesic_m": round(float(geo[j]), 4),
                      "euclidean_m": round(float(euc[j]), 4),
                      "labels": {"gdr": round(float(gdr[j]), 4)}})
    return tasks, checks, pl


def checker(ws, meta: dict, robot: Robot):
    """check(poly, which) on the 3D mesh of the map's storey (verify.check_footprint)."""
    probe = storey_probe(ws, meta["scene"], meta["frame"]["up"], meta["floor_z"])

    def check(poly, which, probe=probe, fz=meta["floor_z"]):
        return check_footprint(probe, poly, fz, robot.height)
    return check


def check_rows(robot_id, map_id, checks) -> list[dict]:
    """verify/<robot>.csv rows of the 3D checks of one map."""
    return [{"robot": robot_id, "map_id": map_id, "task": "", "which": which,
             "x": round(x, 4), "y": round(y, 4), "theta": "" if th is None else round(th, 4),
             **{k: c[k] for k in ("support", "clear_hits", "roof", "ok")}}
            for which, x, y, th, c in checks]


def task_header(p: SvgTaskParams, robot: Robot, plan_res: float) -> dict:
    """Header fields of an SVG task file (besides dataset, map, robot, split, tasks)."""
    return {"height_class": robot.height_class, "success_radius": p.success_radius,
            "clearance": p.clearance,
            "planning": {"plan_res": plan_res, "neighbourhood": 16,
                         "max_anisotropy_error": MAX_ANISO_ERR,
                         "note": "geodesic on C_disk (circumradius + clearance)"}}


def map_tasks(ws, root: Path, dataset: str, robot: Robot, splits: dict, p: SvgTaskParams,
              svg_path: Path) -> tuple[dict, list]:
    """Sample and write the tasks of one map (tasks/<robot>/<map_id>.json; removed when
    no task fits). Returns (per-map stats, verify rows)."""
    m = SvgMap.load(svg_path)
    map_id, meta = svg_path.stem, m.meta
    rng = np.random.default_rng([p.seed, zlib.crc32(f"{robot.id}/{map_id}".encode())])
    check = checker(ws, meta, robot) if p.verify else None
    tasks, checks, pl = sample_map(m, robot, p.k, rng, p.min_geo, p.max_geo, p.min_gdr,
                                   p.clearance, check)
    rows = check_rows(robot.id, map_id, checks)
    fail = (sum(not c[-1]["ok"] for c in checks) / len(checks)) if checks else 0.0
    flags = list(meta.get("flags", []))
    if fail > p.flag_fail:
        flags.append(f"verify_fail {fail:.0%}")
    if len(tasks) < p.k:
        flags.append(f"only {len(tasks)}/{p.k} tasks")
    stats = {"tasks": len(tasks), "checked": len(checks), "verify_fail": round(fail, 3),
             "flags": flags, "cdisk_m2": round(len(pl.cells) * pl.res ** 2, 2)}
    out = Path(root) / "tasks" / robot.id / f"{map_id}.json"
    if not tasks:
        out.unlink(missing_ok=True)
        log.info("  %s: 0 tasks  FLAG %s", map_id, flags)
        return stats, rows
    task_file(root, dataset=dataset, env="svg", map_id=map_id, robot=robot.id, splits=splits,
              **task_header(p, robot, pl.res), tasks=tasks)
    geo = [t["geodesic_m"] for t in tasks]
    log.info("  %s: %d tasks, geodesic %.1f-%.1f m, %d checks, %.0f%% failed%s", map_id, len(tasks),
             min(geo), max(geo), len(checks), fail * 100, f"  FLAG {flags}" if flags else "")
    return stats, rows


def run(ws, root: Path, dataset: str, robot: Robot, splits: dict,
        params: SvgTaskParams | None = None) -> dict:
    """Sample tasks for `robot` on every map of its height class (maps/<hc>/*.svg).
    Writes tasks/<robot>/<map_id>.json and verify/<robot>.csv; returns per-map stats."""
    p = params or SvgTaskParams()
    root = Path(root)
    tdir = root / "tasks" / robot.id
    tdir.mkdir(parents=True, exist_ok=True)
    rows, per_map = [], {}
    for sp in sorted((root / "maps" / robot.height_class).glob("*.svg")):
        per_map[sp.stem], r = map_tasks(ws, root, dataset, robot, splits, p, sp)
        rows += r
    for old in tdir.glob("*.json"):
        if not per_map.get(old.stem, {}).get("tasks"):
            old.unlink()
    write_csv(rows, root / "verify" / f"{robot.id}.csv", SVG_COLS)
    return per_map
