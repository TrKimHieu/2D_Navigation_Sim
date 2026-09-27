import gymnasium as gym
import numpy as np
import pytest

import hm3denv  # noqa: F401
from hm3denv.core.gridcore import lidar_scan
from hm3denv.envs import GridEnv

MAP = "00001-aaa_s1"


def make(ds, **kw):
    return GridEnv(dataset=ds, map_id=MAP, task_idx=0, **kw)


def test_reset_info_has_task_labels(grid_dataset):
    obs, info = make(grid_dataset).reset(seed=0)
    assert obs.shape == (22,) and obs.dtype == np.float32
    for k in ("start", "goal", "d_bfs", "p0", "sa_uniform", "sa_persist", "starved", "level"):
        assert k in info
    assert info["start"] == [2, 0] and info["d_bfs"] == 8 and info["split"] == "train"


def test_wall_bump_stays_and_shortest_path_reaches_goal(grid_dataset):
    env = make(grid_dataset)
    env.reset()
    env.step(3)                              # RIGHT into the wall at (2, 1)
    assert env.agent_pos == [2, 0]
    for a in (0, 0, 3, 3, 1, 1, 3, 3):       # U U R R D D R R
        obs, r, term, trunc, info = env.step(a)
    assert term and r == 1.0 and env.steps == 9
    assert info["success"] and info["n_collisions"] == 1 and info["spl"] == 1.0
    assert info["path_length"] == pytest.approx(8 * 0.5) and info["termination"] == "success"


def test_truncated_at_budget(grid_dataset):
    env = make(grid_dataset, budget=3)
    env.reset()
    for _ in range(3):
        *_, term, trunc, info = env.step(0)
    assert trunc and not term and info["termination"] == "time_limit"


def test_out_of_bounds_stays(grid_dataset):
    env = make(grid_dataset)
    env.reset()
    env.step(1)                              # DOWN off the grid
    assert env.agent_pos == [2, 0]


def test_difficulty_and_task_idx_option(grid_dataset):
    env = GridEnv(dataset=grid_dataset, difficulty="easy")
    _, info = env.reset(seed=3)
    assert info["level"] == "easy" and info["start"] == [0, 0]
    _, info = env.reset(options={"map_id": MAP, "task_idx": 0})
    assert info["start"] == [2, 0]


def test_gym_make_and_render(grid_dataset):
    env = gym.make("HM3D/Grid-v0", dataset=grid_dataset, render_mode="rgb_array")
    env.reset(seed=0)
    assert env.render().shape == (3 * 8, 5 * 8, 3)
