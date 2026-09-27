"""Trainer API: step info,
reward terms and weights, curriculum filters, the LiDAR-noise RNG, CustomReward, Coverage."""

import math

import gymnasium as gym
import numpy as np
import pytest

from hm3denv.core import _accel
from hm3denv.core.geometry import Collider
from hm3denv.core.gridcore import bfs
from hm3denv.core.svgmap import SvgMap
from hm3denv.envs import Coverage, CustomReward, GridEnv, OracleFollower, SvgEnv
from hm3denv.vector import make_vec

ROBOT = "turtlebot4"
U_MAP = "00002-bbb_s1"            # the U-shaped room of the fixture: one task, geodesic 6.4 m


def drive(env, n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [env.step(rng.uniform(-1, 1, 2)) for _ in range(n)]


# ------------------------------------------------------------ step info and reward

def test_svg_step_info_and_dense_reward(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    env.reset(seed=0, options={"map_id": U_MAP, "task_idx": 0})
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
        t = info["reward_terms"]
        assert set(t) == {"progress", "success", "collision", "step"}
        assert r == ((1.0 * t["progress"] + 10.0 * t["success"]) - 0.1 * t["collision"]) - 0.01
        assert info["progress_m"] == t["progress"]
        assert t["collision"] == float(info["collided"])
        x, y, _ = info["pose"]
        g = env.task["goal"]
        assert info["goal_distance_m"] == math.hypot(g["x"] - x, g["y"] - y)
        assert info["wall_distance_m"] == env.collider.wall_distance(x, y)
        assert info["path_length"] == env.path_len
        if term or trunc:
            break


def test_reward_weights(svg_dataset):
    a = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    b = SvgEnv(dataset=svg_dataset, robot=ROBOT, reward_weights={"step": 0.0, "progress": 2.0})
    for env in (a, b):
        env.reset(seed=0, options={"map_id": U_MAP, "task_idx": 0})
    for ra, rb in zip(drive(a), drive(b)):
        t = ra[4]["reward_terms"]
        assert rb[1] == ((2.0 * t["progress"] + 10.0 * t["success"]) - 0.1 * t["collision"]) - 0.0
    with pytest.raises(ValueError, match="unknown reward_weights"):
        SvgEnv(dataset=svg_dataset, robot=ROBOT, reward_weights={"speed": 1.0})


def test_sparse_terms_and_nav_info_off(svg_dataset):
    on = SvgEnv(dataset=svg_dataset, robot=ROBOT, reward="sparse")
    off = SvgEnv(dataset=svg_dataset, robot=ROBOT, reward="sparse", nav_info=False)
    for env in (on, off):
        env.reset(seed=0, options={"map_id": U_MAP, "task_idx": 0})
    for x, y in zip(drive(on), drive(off)):
        assert x[1] == y[1] and x[4]["pose"] == y[4]["pose"]
        assert x[4]["reward_terms"] == {"success": x[1]}
        assert "goal_geodesic_m" in x[4] and "goal_geodesic_m" not in y[4]
        assert "wall_distance_m" not in y[4]


def test_goal_geodesic_falls_along_the_oracle_path(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    env.reset(seed=0, options={"map_id": U_MAP, "task_idx": 0})
    oracle = OracleFollower(env)
    geo = [env._geo_prev]
    while True:
        _, _, term, trunc, info = env.step(oracle.act())
        geo.append(info["goal_geodesic_m"])
        if term or trunc:
            break
    assert info["success"]
    assert geo[0] > 3.0                                   # the U wall forces a detour
    # lookup on the 4 cm planning grid wobbles by a few cm, e.g. while turning in place
    assert geo[-1] < 0.3 and np.diff(geo).max() < 0.05


@pytest.mark.parametrize("accel", [False, True])
def test_wall_distance(accel):
    if accel and not _accel.ENABLED:
        pytest.skip("numba not installed")
    m = SvgMap([np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
                np.array([[2, 1], [2, 2], [3, 2], [3, 1]], float)], {})
    col = Collider(m, accel=accel)
    assert col.wall_distance(1.0, 1.5) == pytest.approx(1.0)
    assert col.wall_distance(0.5, 2.8) == pytest.approx(0.2)
    assert col.wall_distance(2.5, 0.4) == pytest.approx(0.4)
    ref = Collider(m, accel=False)
    rng = np.random.default_rng(0)
    for x, y in zip(rng.uniform(0, 4, 200), rng.uniform(0, 3, 200)):
        assert col.wall_distance(x, y) == ref.wall_distance(x, y)


# ------------------------------------------------------------ curriculum

def episodes(env, n=12, seed=5):
    _, info = env.reset(seed=seed)
    out = [(info["map_id"], info["task_id"])]
    for _ in range(n - 1):
        _, info = env.reset()
        out.append((info["map_id"], info["task_id"]))
    return out


def test_svg_task_filter(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    assert env.set_task_filter(geodesic_m=(5.0, None)) == {"maps": 1, "tasks": 1}
    assert set(episodes(env)) == {(U_MAP, 0)}
    assert env.set_task_filter(geodesic_m=(None, 4.5), gdr=[1.0]) == {"maps": 2, "tasks": 4}
    assert U_MAP not in {m for m, _ in episodes(env)}
    with pytest.raises(ValueError, match="no task matches"):
        env.set_task_filter(geodesic_m=(100.0, None))
    with pytest.raises(ValueError, match="a range is"):
        env.set_task_filter(geodesic_m=(1.0, 2.0, 3.0))
    assert env.set_task_filter() == {"maps": 3, "tasks": 5}
    assert episodes(env) == episodes(SvgEnv(dataset=svg_dataset, robot=ROBOT))
    fixed = SvgEnv(dataset=svg_dataset, robot=ROBOT, task_filter={"geodesic_m": (5.0, None)})
    assert set(episodes(fixed)) == {(U_MAP, 0)}
    # explicit options still win over the filter
    _, info = fixed.reset(options={"map_id": "00001-aaa_s1", "task_idx": 1})
    assert (info["map_id"], info["task_id"]) == ("00001-aaa_s1", 1)


def test_grid_task_filter_and_difficulty(grid_dataset):
    env = GridEnv(dataset=grid_dataset, split="train")
    env.set_task_filter(level=["easy"])
    assert {t for _, t in episodes(env)} == {1}
    env.set_task_filter(d_bfs=(2, None))
    assert {t for _, t in episodes(env)} == {0}
    both = GridEnv(dataset=grid_dataset, split="train", difficulty="medium",
                   task_filter={"level": "easy"})
    assert {t for _, t in episodes(both)} == {1}          # difficulty falls back within the filter


@pytest.mark.parametrize("vectorization", ["sync", "async"])
def test_task_filter_through_vector_env(svg_dataset, vectorization):
    venv = make_vec("HM3D/Svg-v0", 2, vectorization=vectorization, dataset=str(svg_dataset),
                    robot=ROBOT)
    try:
        assert venv.call("set_task_filter", geodesic_m=(5.0, None)) == \
            ({"maps": 1, "tasks": 1},) * 2
        _, info = venv.reset(seed=0)
        assert list(info["map_id"]) == [U_MAP, U_MAP]
    finally:
        venv.close()


# ------------------------------------------------------------ noise RNG

def test_lidar_noise_does_not_change_episode_order(svg_dataset):
    quiet = SvgEnv(dataset=svg_dataset, robot=ROBOT)
    noisy = SvgEnv(dataset=svg_dataset, robot=ROBOT, lidar_noise=0.05)
    assert episodes(quiet, 20) == episodes(noisy, 20)


def test_lidar_noise_reproducible(svg_dataset):
    a = SvgEnv(dataset=svg_dataset, robot=ROBOT, lidar_noise=0.05)
    b = SvgEnv(dataset=svg_dataset, robot=ROBOT, lidar_noise=0.05)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.array_equal(oa["lidar"], ob["lidar"])
    for x, y in zip(drive(a, 10), drive(b, 10)):
        assert np.array_equal(x[0]["lidar"], y[0]["lidar"])
    oc, _ = a.reset(seed=4)
    assert not np.array_equal(oc["lidar"], oa["lidar"])


# ------------------------------------------------------------ grid info

def test_grid_step_info(grid_dataset):
    env = GridEnv(dataset=grid_dataset, split="train")
    env.reset(seed=0, options={"task_idx": 0})
    d = bfs(env.grid == 0, env.goal)
    rng = np.random.default_rng(1)
    bumps = 0
    for _ in range(200):
        prev = d[env.agent_pos[0], env.agent_pos[1]]
        _, r, term, trunc, info = env.step(int(rng.integers(4)))
        assert info["goal_geodesic"] == d[env.agent_pos[0], env.agent_pos[1]]
        assert info["progress"] == prev - info["goal_geodesic"]
        assert info["reward_terms"] == {"success": r}
        bumps += info["collided"]
        if term or trunc:
            assert info["n_collisions"] == bumps
            break


# ------------------------------------------------------------ wrappers

def test_custom_reward(svg_dataset):
    env = CustomReward(SvgEnv(dataset=svg_dataset, robot=ROBOT),
                       lambda i: 2 * i["reward_terms"]["progress"] - (i["wall_distance_m"] < 0.3))
    env.reset(seed=0, options={"map_id": U_MAP, "task_idx": 0})
    for _, r, _, _, info in drive(env, 20):
        assert r == 2 * info["reward_terms"]["progress"] - (info["wall_distance_m"] < 0.3)
        assert "env_reward" in info


def test_coverage_svg_and_grid(svg_dataset, grid_dataset):
    for env, act in ((Coverage(SvgEnv(dataset=svg_dataset, robot=ROBOT)), None),
                     (Coverage(GridEnv(dataset=grid_dataset, split="train")), "grid")):
        rng = np.random.default_rng(0)
        _, info = env.reset(seed=0)
        cov = [info["coverage"]]
        for _ in range(60):
            a = int(rng.integers(4)) if act else rng.uniform(-1, 1, 2)
            _, _, term, trunc, info = env.step(a)
            assert info["coverage_gain"] == pytest.approx(info["coverage"] - cov[-1])
            cov.append(info["coverage"])
            if term or trunc:
                break
        assert 0 < cov[0] <= cov[-1] <= 1 and all(np.diff(cov) >= 0)
        _, info = env.reset()                              # starts over: first scan only
        assert 0 < info["coverage"] == info["coverage_gain"] <= 1


# ------------------------------------------------------------ C_disk islands (inf/nan reward bug)

def closet_map():
    """4 x 3 m room plus a 0.8 x 1.0 m closet joined through a 0.3 m neck: the disk of a
    turtlebot fits in the closet but not through the neck, so C_disk has an island."""
    return SvgMap([np.array([[0, 0], [4, 0], [4, 1.35], [4.2, 1.35], [4.2, 1.0], [5, 1.0],
                             [5, 2.0], [4.2, 2.0], [4.2, 1.65], [4, 1.65], [4, 3], [0, 3]],
                            float)], {})


def test_lookup_is_finite_on_a_cdisk_island():
    from hm3denv.core.planning import Planner
    pl = Planner(closet_map(), 0.18, 0.05)
    assert pl.n_comp == 2                                        # room + closet island
    f = pl.field(1.0, 1.5)
    island = pl.nearest_cell(4.6, 1.5)
    assert not np.isfinite(f[island])                            # the old lookup gave inf
    d = pl.lookup(f, 4.6, 1.5)
    assert np.isfinite(d) and 3.0 < d < 4.5                      # ~ 3.6 m straight line
    rng = np.random.default_rng(0)                               # elsewhere: unchanged
    for x, y in zip(rng.uniform(0.3, 3.7, 100), rng.uniform(0.3, 2.7, 100)):
        r, c = pl.nearest_cell(x, y)
        cx, cy = pl.center(r, c)
        assert pl.lookup(f, x, y) == float(f[r, c] + np.hypot(x - cx, y - cy))


def test_dense_reward_stays_finite_on_a_cdisk_island():
    ts = {"map_id": "closet", "success_radius": 0.2, "clearance": 0.05,
          "tasks": [{"start": {"x": 4.6, "y": 1.5, "theta": math.pi}, "goal": {"x": 1.0, "y": 1.5},
                     "geodesic_m": 3.6}]}
    env = SvgEnv(robot=ROBOT, svgmap=closet_map(), tasks=ts)
    env.reset(seed=0)
    for obs, r, term, trunc, info in drive(env, 30):
        assert math.isfinite(r) and math.isfinite(info["progress_m"])
        assert math.isfinite(info["goal_geodesic_m"])
