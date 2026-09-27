"""Disk cache of loaded maps:
an environment must behave identically with no cache, a cold cache and a warm one."""

import json
import shutil

import numpy as np
import pytest

from hm3denv import cache as C
from hm3denv.cli import main
from hm3denv.core.planning import Planner
from hm3denv.core.svgmap import SvgMap
from hm3denv.dataset import load_dataset
from hm3denv.envs import SvgEnv
from hm3denv.envs.svg import task_turn

ROBOT = "turtlebot4"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("HM3D_CACHE", str(d))
    return d


def rollout(env):
    """Every task of every map, fixed actions: the full observable record."""
    rec = []
    rng = np.random.default_rng(0)
    for m in env._map_ids:
        for i in range(len(env.dataset.tasks(ROBOT, m))):
            obs, info = env.reset(seed=1, options={"map_id": m, "task_idx": i})
            rec.append((obs, info))
            for _ in range(40):
                out = env.step(rng.uniform(-1, 1, 2))
                rec.append(out[:4] + (out[4]["pose"],))
                if out[2] or out[3]:
                    break
    return rec


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_no_cold_warm_identical(svg_dataset, cache_dir):
    ref = rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT, disk_cache=False))
    cold = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    assert cold._disk is not None
    assert same(rollout(cold), ref)
    assert len(list(cache_dir.rglob("*.npz"))) == 3 + 3          # 3 map files + 3 robot files
    assert same(rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT)), ref)


def test_warm_cache_reads_no_dataset_file(svg_dataset, cache_dir, monkeypatch):
    rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT))                  # fill the cache

    def boom(*a, **k):
        raise AssertionError("dataset file read on a cache hit")
    monkeypatch.setattr(SvgMap, "load", classmethod(boom))
    monkeypatch.setattr("hm3denv.dataset.Dataset.task_file", boom)
    env = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    for m in env._map_ids:
        env.reset(options={"map_id": m, "task_idx": 0})
        env.step([1.0, 0.0])


def test_planner_arrays_round_trip():
    m = SvgMap([np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float),
                np.array([[2.9, 0], [2.9, 2], [3.1, 2], [3.1, 0]], float)], {})
    p = Planner(m, 0.21, 0.05)
    q = Planner.from_arrays(p.to_arrays())
    assert np.array_equal(p.grid, q.grid) and np.array_equal(p._near, q._near)
    for a in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(p.graph, a), getattr(q.graph, a))
    assert (p.x0, p.y1, p.res, p.r_circ, p.clearance) == (q.x0, q.y1, q.res, q.r_circ, q.clearance)
    assert np.array_equal(p.field(1.0, 0.5), q.field(1.0, 0.5))
    assert np.array_equal(p.comp, q.comp) and p.n_comp == q.n_comp


def test_build_entry_turns_and_fields(svg_dataset, cache_dir):
    ds = load_dataset(svg_dataset)
    for m in ds.maps(ROBOT):
        assert C.build_entry(str(svg_dataset), ROBOT, m, True).endswith("built")
    env = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    for m in env._map_ids:
        ctx = env._ctx(m)
        assert ctx.fields_path is not None and len(ctx.turns) == len(ctx.tasks)
        pl = Planner(ctx.svgmap, env.robot.r_circ, ctx.clearance)
        for i, t in enumerate(ctx.tasks):
            f, pred = pl.field(t["goal"]["x"], t["goal"]["y"], predecessors=True)
            assert ctx.turns[i] == task_turn(pl, t, pred)
            assert np.array_equal(C.DatasetCache.load_field(ctx.fields_path, ctx.planner, i), f)
    assert C.build_entry(str(svg_dataset), ROBOT, env._map_ids[0], True).endswith("cached")
    assert same(rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT)),
                rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT, disk_cache=False)))


def test_changed_file_hash_misses(svg_dataset, cache_dir, tmp_path):
    rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT))
    copy = tmp_path / "svg-tiny"
    shutil.copytree(svg_dataset, copy)
    man = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    svg = next(k for k in man["files"] if k.endswith(".svg"))
    man["files"][svg] = "0" * 64
    (copy / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    ds = load_dataset(copy)
    dc = C.DatasetCache(ds)
    r = ds.robot(ROBOT)
    mid = svg.rsplit("/", 1)[1][:-4]
    changed = dc.paths(ROBOT, mid, r.r_circ, None)
    assert dc.load(changed) is None                                   # new key: not cached
    other = next(m for m in ds.maps(ROBOT) if m != mid)
    assert dc.load(dc.paths(ROBOT, other, r.r_circ, None)) is not None


def test_corrupt_entry_is_rebuilt(svg_dataset, cache_dir):
    ref = rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT, disk_cache=False))
    rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT))
    robot_files = [f for f in cache_dir.rglob("*.npz") if f.parent.name == ROBOT]
    for f in robot_files:
        f.write_bytes(b"not a zip file")
    assert same(rollout(SvgEnv(dataset=svg_dataset, robot=ROBOT)), ref)
    for f in robot_files:
        assert C._load(f) is not None                                 # rewritten


def test_cache_off(svg_dataset, monkeypatch):
    monkeypatch.setenv("HM3D_CACHE", "off")
    assert C.cache_root() is None
    assert SvgEnv(dataset=svg_dataset, robot=ROBOT)._disk is None
    assert main(["cache", "build", str(svg_dataset)]) == 2            # error: cache disabled


def test_cli_build_info_clear(svg_dataset, cache_dir, capsys):
    assert main(["cache", "build", str(svg_dataset), "--robot", ROBOT, "--jobs", "1"]) == 0
    assert (cache_dir / f"v{C.FORMAT}" / "svg-tiny" / ROBOT).exists()
    assert main(["cache", "info"]) == 0
    assert "svg-tiny" in capsys.readouterr().out
    assert main(["cache", "clear", "svg-tiny"]) == 0
    assert not (cache_dir / f"v{C.FORMAT}" / "svg-tiny").exists()
