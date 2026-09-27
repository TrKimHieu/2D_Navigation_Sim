"""Speed-ups must not change any result: every fast path is compared with the reference code."""

import math

import numpy as np
import pytest
from scipy import ndimage

from hm3denv.bench import bench
from hm3denv.cli import main
from hm3denv.core import _accel
from hm3denv.core.geometry import Collider, place, segments_hit_polygon
from hm3denv.core.gridcore import lidar_scan
from hm3denv.core.planning import edt_px
from hm3denv.core.svgmap import SvgMap
from hm3denv.envs import GridEnv, SvgEnv
from hm3denv.robots import list_robots, load as load_robot
from hm3denv.vector import make_vec

needs_numba = pytest.mark.skipif(not _accel.ENABLED, reason="numba not installed or HM3D_ACCEL=0")


def cluttered(seed=0, n_holes=30):
    """4 x 3 m room, a 1 cm wall and small square obstacles."""
    rng = np.random.default_rng(seed)
    rings = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
             np.array([[1.995, 0], [1.995, 1.8], [2.005, 1.8], [2.005, 0]], float)]
    for _ in range(n_holes):
        cx, cy, h = rng.uniform(0.3, 3.7), rng.uniform(0.3, 2.7), rng.uniform(0.01, 0.15)
        rings.append(np.array([[cx - h, cy - h], [cx - h, cy + h], [cx + h, cy + h],
                               [cx + h, cy - h]], float))
    return SvgMap(rings, {"map_id": "cluttered"})


def random_poses(rng, n):
    return zip(rng.uniform(-0.1, 4.1, n), rng.uniform(-0.1, 3.1, n), rng.uniform(-4, 4, n))


# ------------------------------------------------------------ collision

@pytest.mark.parametrize("robot", list_robots())
def test_pose_valid_matches_exact_test(robot):
    """Broad phase + numba kernel == the original exact test, for every robot footprint
    (several are concave URDF sections) on poses that often graze obstacles."""
    fp = load_robot(robot).footprint
    col = Collider(cluttered())
    rng = np.random.default_rng(len(robot))
    n_valid = 0
    for x, y, th in random_poses(rng, 1500):
        want = col.pose_valid_exact(fp, x, y, th)
        assert col.pose_valid(fp, x, y, th) == want, (x, y, th)
        n_valid += want
    assert 50 < n_valid < 1450                  # both outcomes are exercised


def test_pose_valid_numpy_backend_matches_exact_test():
    fp = load_robot("pal_tiago").footprint
    col = Collider(cluttered(1), accel=False)
    for x, y, th in random_poses(np.random.default_rng(3), 800):
        assert col.pose_valid(fp, x, y, th) == col.pose_valid_exact(fp, x, y, th)


def test_edge_distance_is_a_lower_bound():
    """Every edge point lies at least (edge_dist - margin) * res from the pixel centre."""
    col = Collider(cluttered(2))
    rng = np.random.default_rng(0)
    S = col.index.segs
    for x, y in zip(rng.uniform(0, 4, 400), rng.uniform(0, 3, 400)):
        d_true = min(_point_segment(x, y, s) for s in S)
        if col._clear_of_edges(x, y, 0.0):
            fc, fr = (x - col.x0) / col.res, (col.y1 - y) / col.res
            bound = (col.edge_dist[int(fr), int(fc)] - 2.0) * col.res
            assert bound <= d_true + 1e-12


def _point_segment(x, y, s):
    a, b = np.array(s[:2]), np.array(s[2:])
    t = np.clip(np.dot((x, y) - a, b - a) / max(np.dot(b - a, b - a), 1e-18), 0, 1)
    return float(np.hypot(*((x, y) - (a + t * (b - a)))))


@needs_numba
def test_seg_hit_poly_kernel_matches_numpy():
    col = Collider(cluttered(4))
    S = col.index.segs
    rng = np.random.default_rng(5)
    fps = [load_robot(r).footprint for r in list_robots()]
    for k, (x, y, th) in enumerate(random_poses(rng, 2000)):
        P = place(fps[k % len(fps)], x, y, th)
        lo, hi = P.min(axis=0), P.max(axis=0)
        idx = col.index.query(lo[0], lo[1], hi[0], hi[1])
        assert _accel.seg_hit_poly(S, P, lo[0], lo[1], hi[0], hi[1]) == \
            segments_hit_polygon(S[idx], P)


# ------------------------------------------------------------ lidar

@needs_numba
@pytest.mark.parametrize("full", [True, False])
def test_raycast_kernel_bit_identical_to_numpy(full):
    fast, ref = Collider(cluttered(6)), Collider(cluttered(6), accel=False)
    rng = np.random.default_rng(7)
    n = 72 if full else 61
    da = 2 * math.pi / n if full else math.radians(270) / (n - 1)
    for x, y, th in random_poses(rng, 300):
        a0 = th if full else th - math.radians(135)
        a = fast.raycast_uniform(x, y, a0, da, n, 8.0, full)
        b = ref.raycast_uniform(x, y, a0, da, n, 8.0, full)
        assert np.array_equal(a, b)


# ------------------------------------------------------------ planner

def test_edt_px_equals_scipy():
    mask = cluttered(8).raster(0.02)[0]
    assert np.array_equal(edt_px(mask), ndimage.distance_transform_edt(mask))


# ------------------------------------------------------------ episode source

def test_map_repeat_keeps_map_for_k_episodes(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot="turtlebot4", map_repeat=3)
    env.reset(seed=0)
    ids = [env.map_id] + [env.reset()[1]["map_id"] for _ in range(11)]
    assert all(len(set(ids[i:i + 3])) == 1 for i in range(0, 12, 3))
    assert len(set(ids)) > 1


def test_map_repeat_one_draws_as_before(svg_dataset):
    """map_repeat=1 uses the RNG exactly like the original code: one map draw, then one
    task draw, per reset."""
    env = SvgEnv(dataset=svg_dataset, robot="turtlebot4")
    env.reset(seed=4)
    got = [(env.map_id, env._cur)] + [(i["map_id"], i["task_id"]) for i in
                                      (env.reset()[1] for _ in range(15))]
    rng = np.random.default_rng(4)
    maps = env._map_ids
    want = []
    for _ in range(16):
        m = maps[int(rng.integers(len(maps)))]
        want.append((m, int(rng.integers(len(env.dataset.tasks("turtlebot4", m))))))
    assert got == want


def test_map_repeat_must_be_positive(svg_dataset):
    with pytest.raises(ValueError):
        SvgEnv(dataset=svg_dataset, robot="turtlebot4", map_repeat=0)


def test_field_cache_is_bounded_and_values_unchanged():
    m = SvgMap([np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float)], {})
    ts = [{"start": {"x": 1, "y": 1.5, "theta": 0}, "goal": {"x": gx, "y": 1.5},
           "geodesic_m": gx - 1} for gx in (2.0, 3.0, 4.0, 5.0, 5.5)]
    hdr = {"map_id": "t", "success_radius": 0.2, "clearance": 0.05, "tasks": ts}
    small = SvgEnv(robot="turtlebot4", svgmap=m, tasks=hdr, field_cache=2)
    big = SvgEnv(robot="turtlebot4", svgmap=m, tasks=hdr, field_cache=100)
    for rep in range(2):
        for i in range(len(ts)):
            a = small.reset(options={"task_idx": i})
            b = big.reset(options={"task_idx": i})
            assert a[1]["time_limit"] == b[1]["time_limit"]
            for _ in range(5):
                ra = small.step([0.5, 0.3])[1]
                assert ra == big.step([0.5, 0.3])[1]
            assert len(small.ctx.goal_fields) <= 2


# ------------------------------------------------------------ grid

def test_grid_lidar_table_equals_scan(grid_dataset):
    env = GridEnv(dataset=grid_dataset, split="train")
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(int(rng.integers(4)))
        assert np.array_equal(obs, lidar_scan(env.grid, env.agent_pos, 22, 10.0))
        obs[:] = -1                                 # callers may modify what they get
        if term or trunc:
            env.reset()
    assert np.array_equal(env._get_obs(), lidar_scan(env.grid, env.agent_pos, 22, 10.0))


# ------------------------------------------------------------ vector env and bench

@pytest.mark.parametrize("vectorization", ["sync", "async"])
def test_make_vec(svg_dataset, vectorization):
    venv = make_vec("HM3D/Svg-v0", 2, vectorization=vectorization, dataset=str(svg_dataset),
                    robot="turtlebot4")
    try:
        obs, _ = venv.reset(seed=0)
        assert obs["lidar"].shape[0] == 2
        obs, rew, term, trunc, _ = venv.step(np.zeros((2, 2), np.float32))
        assert rew.shape == (2,)
    finally:
        venv.close()


def test_bench_and_cli(svg_dataset, capsys):
    r = bench(svg_dataset, robot="turtlebot4", seconds=0.3, split="train")
    assert r["steps_per_s"] > 0 and r["backend"] in ("numba", "numpy")
    assert main(["bench", str(svg_dataset), "--robot", "turtlebot4", "--seconds", "0.2"]) == 0
    assert "step/s" in capsys.readouterr().out


# ------------------------------------------------------------ segment grid (0.6.1)

def dense_map(seed=0, n_holes=1500):
    """20 x 12 m room with many small obstacles: thousands of short segments."""
    rng = np.random.default_rng(seed)
    rings = [np.array([[0, 0], [20, 0], [20, 12], [0, 12]], float)]
    for _ in range(n_holes):
        cx, cy, h = rng.uniform(0.3, 19.7), rng.uniform(0.3, 11.7), rng.uniform(0.005, 0.08)
        k = int(rng.integers(3, 9))
        a = np.sort(rng.uniform(0, 2 * np.pi, k))
        rings.append(np.stack([cx + h * np.cos(a), cy + h * np.sin(a)], 1))
    return SvgMap(rings, {"map_id": "dense"})


def grid_points(col, rng, n):
    """Random points plus points exactly on grid-cell lines and corners."""
    _, _, gx0, gy0, cs, nx, ny = col.grid[:7]
    xs = list(rng.uniform(0.05, 19.95, n))
    ys = list(rng.uniform(0.05, 11.95, n))
    for _ in range(n // 2):
        i, j = int(rng.integers(1, nx - 2)), int(rng.integers(1, ny - 2))
        xs.append(gx0 + i * cs)
        ys.append(gy0 + j * cs if rng.random() < 0.5 else float(rng.uniform(0.05, 11.95)))
    return list(zip(xs, ys))


@needs_numba
@pytest.mark.parametrize("full, r_max", [(True, 12.0), (False, 12.0), (True, 1.0), (False, 3.5)])
def test_grid_raycast_bit_identical_on_a_dense_map(full, r_max):
    m = dense_map()
    fast, ref = Collider(m), Collider(m, accel=False)
    assert len(fast.index.segs) > 5000
    rng = np.random.default_rng(1)
    n = 72 if full else 61
    da = 2 * math.pi / n if full else math.radians(270) / (n - 1)
    for x, y in grid_points(fast, rng, 150):
        for a0 in (0.0, math.pi / 2, float(rng.uniform(-4, 4))):   # 0 and pi/2: rays along cell lines
            a = fast.raycast_uniform(x, y, a0, da, n, r_max, full)
            b = ref.raycast_uniform(x, y, a0, da, n, r_max, full)
            assert np.array_equal(a, b), (x, y, a0)


@needs_numba
def test_grid_wall_distance_and_pose_valid_identical():
    m = dense_map(2)
    fast, ref = Collider(m), Collider(m, accel=False)
    rng = np.random.default_rng(3)
    fps = [load_robot(r).footprint for r in ("turtlebot4", "pal_tiago", "jetauto_pro")]
    for k, (x, y) in enumerate(grid_points(fast, rng, 400)):
        assert fast.wall_distance(x, y) == ref.wall_distance(x, y)
        th = float(rng.uniform(-4, 4))
        assert fast.pose_valid(fps[k % 3], x, y, th) == ref.pose_valid_exact(fps[k % 3], x, y, th)
    assert fast.wall_distance(-5.0, 30.0) == ref.wall_distance(-5.0, 30.0)   # outside the grid


@needs_numba
def test_grid_lists_every_touching_segment():
    col = Collider(dense_map(4, 300))
    start, items, gx0, gy0, cs, nx, ny = col.grid[:7]
    S = col.index.segs
    lo = np.minimum(S[:, :2], S[:, 2:]) - _accel.EPS_REG
    hi = np.maximum(S[:, :2], S[:, 2:]) + _accel.EPS_REG
    for cell in range(nx * ny):
        r, c = divmod(cell, nx)
        x0, y0 = gx0 + c * cs, gy0 + r * cs
        touch = np.nonzero((lo[:, 0] < x0 + cs) & (hi[:, 0] >= x0) &
                           (lo[:, 1] < y0 + cs) & (hi[:, 1] >= y0))[0]
        assert set(items[start[cell]:start[cell + 1]]) == set(touch)
