"""Shared fixtures: a tiny SVG dataset and a tiny grid dataset built from synthetic maps,
so environment, dataset and CLI tests run anywhere (no GLB, no workspace)."""

import os

import numpy as np
import pytest

from hm3denv.core.svgmap import SvgMap
from hm3denv.dataset.schema import assign_splits
from hm3denv.dataset.writer import snapshot_robot, task_file, write_manifest

def require_build_extra():
    """Skip the calling test module when the `build` extra (pillow, trimesh, shapely) is
    not installed: the dataset builders need it, the environments do not."""
    for mod in ("PIL", "trimesh", "shapely"):
        pytest.importorskip(mod, reason='needs the build extra: pip install ".[build]"')


@pytest.fixture(autouse=True, scope="session")
def _isolated_disk_cache(tmp_path_factory):
    """Never touch the user's map cache (see hm3denv.cache) nor the data home of the
    checkout (<repo>/data: its datasets, robot presets and workspace)."""
    old = {k: os.environ.get(k) for k in ("HM3D_CACHE", "HM3D_HOME")}
    os.environ["HM3D_CACHE"] = str(tmp_path_factory.mktemp("hm3d-cache"))
    os.environ["HM3D_HOME"] = str(tmp_path_factory.mktemp("hm3d-home"))
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# 3 "scenes" x 1 storey: 6 x 3 m rooms, the second with a U wall
ROOM = [np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float)]
U_ROOM = ROOM + [np.array([[2.9, 0], [2.9, 2.0], [3.1, 2.0], [3.1, 0]], float)]
SCENES = ["00001-aaa", "00002-bbb", "00003-ccc"]


def svg_task(i, sx, sy, sth, gx, gy, geo):
    return {"id": i, "start": {"x": sx, "y": sy, "theta": sth}, "goal": {"x": gx, "y": gy},
            "geodesic_m": geo, "euclidean_m": float(np.hypot(gx - sx, gy - sy)),
            "labels": {"gdr": 1.0}}


@pytest.fixture(scope="session")
def svg_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "svg-tiny"
    splits = assign_splits(SCENES, {"train": 0.34, "val": 0.33, "test": 0.33}, seed=0)
    robots = {}
    for rid in ("turtlebot4", "jetauto_pro"):
        from hm3denv.robots import load
        hc = load(rid).height_class
        snapshot_robot(root, rid)
        maps = {}
        for k, sc in enumerate(SCENES):
            mid = f"{sc}_s1"
            rings = U_ROOM if k == 1 else ROOM
            (root / "maps" / hc).mkdir(parents=True, exist_ok=True)
            SvgMap(rings, {"schema_version": "2.0", "map_id": mid}).save(root / "maps" / hc / f"{mid}.svg")
            tasks = ([svg_task(0, 1.0, 0.6, 0.0, 5.0, 0.6, 6.4)] if k == 1 else
                     [svg_task(0, 1.0, 1.5, 0.0, 5.0, 1.5, 4.0), svg_task(1, 5.0, 1.5, 3.14, 1.0, 1.5, 4.0)])
            task_file(root, dataset="svg-tiny", env="svg", map_id=mid, robot=rid, splits=splits,
                      tasks=tasks, success_radius=0.2, clearance=0.05, height_class=hc)
            maps[mid] = {"flags": []}
        robots[rid] = {"height_class": hc, "maps": maps}
    write_manifest(root, name="svg-tiny", env="svg", config={"name": "svg-tiny"},
                   splits=splits, robots=robots)
    return root


#   0 1 2 3 4
# 0 . . . # .
# 1 . # . # .
# 2 . # . . .
GRID = np.array([[0, 0, 0, 1, 0],
                 [0, 1, 0, 1, 0],
                 [0, 1, 0, 0, 0]], np.uint8)
LABELS = {"d_bfs": 8, "p0": 0.01, "sa_uniform": 0.01, "sa_persist": 0.02, "starved": False,
          "level": "medium"}


def grid_task(i, s, g, labels):
    return {"id": i, "start": {"x": float(s[1]), "y": float(-s[0]), "cell": list(s)},
            "goal": {"x": float(g[1]), "y": float(-g[0]), "cell": list(g)},
            "geodesic_m": labels["d_bfs"] * 0.5, "euclidean_m": 1.0, "labels": labels}


@pytest.fixture(scope="session")
def grid_dataset(tmp_path_factory):
    import json
    root = tmp_path_factory.mktemp("ds") / "grid-tiny"
    splits = assign_splits(SCENES, {"train": 1.0, "val": 0.0, "test": 0.0}, seed=0)
    (root / "grids").mkdir(parents=True)
    snapshot_robot(root, "jetauto_pro")
    maps = {}
    for sc in SCENES:
        mid = f"{sc}_s1"
        np.savez(root / "grids" / f"{mid}.npz", grid=GRID, state=GRID, main=GRID == 0)
        (root / "grids" / f"{mid}.json").write_text(json.dumps({"map_id": mid, "cell_m": 0.5}))
        task_file(root, dataset="grid-tiny", env="grid", map_id=mid, robot="jetauto_pro",
                  splits=splits, budget=200,
                  tasks=[grid_task(0, (2, 0), (2, 4), LABELS),
                         grid_task(1, (0, 0), (0, 1), {**LABELS, "d_bfs": 1, "level": "easy"})])
        maps[mid] = {"flags": []}
    write_manifest(root, name="grid-tiny", env="grid", config={"name": "grid-tiny"},
                   splits=splits, robots={"jetauto_pro": {"height_class": "h63", "maps": maps}})
    return root
