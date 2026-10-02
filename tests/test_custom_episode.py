"""Your own start / goal at reset: reset(options={"map_id": ..., "start": ..., "goal": ...})."""

import math

import gymnasium as gym
import numpy as np
import pytest

import hm3denv  # noqa: F401  registers HM3D/*
from hm3denv.core.svgmap import SvgMap
from hm3denv.envs import GridEnv, GridOracle, OracleFollower, SvgEnv

ROOM = np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float)
U_RINGS = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
           np.array([[1.9, 0], [1.9, 2.0], [2.1, 2.0], [2.1, 0]], float)]
TWO_ROOMS = [np.array([[0, 0], [2.9, 0], [2.9, 3], [0, 3]], float),
             np.array([[3.1, 0], [6, 0], [6, 3], [3.1, 3]], float)]


def svg_env(rings, robot="turtlebot4", **kw):
    t = {"map_id": "m", "success_radius": 0.2, "clearance": 0.05,
         "tasks": [{"start": {"x": 1.0, "y": 1.5, "theta": 0.0}, "goal": {"x": 2.0, "y": 1.5},
                    "geodesic_m": 1.0}]}
    return SvgEnv(robot=robot, svgmap=SvgMap(rings, {}), tasks=t, **kw)


def drive(env, agent):
    while True:
        _, _, term, trunc, info = env.step(agent.act())
        if term or trunc:
            return info


def test_svg_custom_episode_is_drivable():
    env = svg_env(U_RINGS, reward="sparse")
    obs, info = env.reset(seed=0, options={"start": [1.0, 0.6], "goal": {"x": 3.0, "y": 0.6}})
    assert info["task_id"] == -1
    assert 4.0 < info["geodesic_m"] < 6.0                     # around the wall, not 2 m
    assert env.pose[2] == pytest.approx(0.0)                    # default heading: to the goal
    assert obs["goal"][0] == pytest.approx(2.0, abs=1e-5)
    end = drive(env, OracleFollower(env))
    assert end["success"] and end["task_id"] == -1


def test_svg_custom_start_heading_and_dict_forms():
    env = svg_env([ROOM])
    env.reset(options={"start": {"x": 1, "y": 1.5, "theta": 1.0}, "goal": [5, 1.5]})
    assert env.pose == (1.0, 1.5, 1.0)


@pytest.mark.parametrize("start, goal, msg", [
    ([0.05, 1.5], [5, 1.5], "collides"),
    ([1, 1.5], [5.95, 1.5], "too close to a wall"),
    ([1, 1.5], [9, 1.5], "outside the free"),
    ([1, 1.5], [5, 1.5, 0, 7], "expected x, y"),
])
def test_svg_custom_invalid(start, goal, msg):
    env = svg_env([ROOM])
    with pytest.raises(ValueError, match=msg):
        env.reset(options={"start": start, "goal": goal})


def test_svg_custom_unreachable_goal():
    env = svg_env(TWO_ROOMS)
    with pytest.raises(ValueError, match="cannot be reached"):
        env.reset(options={"start": [1, 1.5], "goal": [5, 1.5]})


def test_svg_custom_needs_both_endpoints():
    env = svg_env([ROOM])
    with pytest.raises(ValueError, match="both 'start' and 'goal'"):
        env.reset(options={"start": [1, 1.5]})


def test_svg_custom_does_not_mix_caches_with_dataset_tasks(svg_dataset):
    env = gym.make("HM3D/Svg-v0", dataset=str(svg_dataset), robot="turtlebot4").unwrapped
    m = env.dataset.maps("turtlebot4")[0]
    _, a = env.reset(options={"map_id": m, "task_idx": 0})
    _, b = env.reset(options={"map_id": m, "start": [5.0, 1.5, 3.14], "goal": [1.0, 2.5]})
    _, c = env.reset(options={"map_id": m, "start": [3.0, 1.5], "goal": [1.0, 2.5]})
    _, d = env.reset(options={"map_id": m, "task_idx": 0})
    assert a["geodesic_m"] == d["geodesic_m"] == env.tasks[0]["geodesic_m"]
    assert b["geodesic_m"] == pytest.approx(math.hypot(4, 1), abs=0.15)
    assert c["geodesic_m"] == pytest.approx(math.hypot(2, 1), abs=0.15)


def test_custom_episode_needs_map_id_with_several_maps(svg_dataset):
    env = gym.make("HM3D/Svg-v0", dataset=str(svg_dataset), robot="turtlebot4")
    with pytest.raises(ValueError, match="map_id"):
        env.reset(options={"start": [1, 1.5], "goal": [5, 1.5]})


# ------------------------------------------------------------------ grid

#   0 1 2 3 4
# 0 . . . # .
# 1 . # . # #
# 2 . # . # .      (2, 4) is cut off
GRID = np.array([[0, 0, 0, 1, 0],
                 [0, 1, 0, 1, 1],
                 [0, 1, 0, 1, 0]], np.uint8)


def grid_env():
    t = {"map_id": "g", "tasks": [{"start": {"cell": [2, 0]}, "goal": {"cell": [0, 2]},
                                   "labels": {"d_bfs": 4}}]}
    return GridEnv(grid=GRID, tasks=t)


def test_grid_custom_episode():
    env = grid_env()
    _, info = env.reset(options={"start": [2, 0], "goal": {"cell": [2, 2]}})
    assert info["task_id"] == -1 and info["d_bfs"] == 6
    end = drive(env, GridOracle(env))
    assert end["success"] and end["spl"] == 1.0
    _, info = env.reset(options={"task_idx": 0})             # dataset task again: its own BFS
    assert info["d_bfs"] == 4 and env.goal == [0, 2]
    assert drive(env, GridOracle(env))["success"]


@pytest.mark.parametrize("start, goal, msg", [
    ([1, 1], [0, 0], "blocked"),
    ([0, 0], [5, 0], "outside"),
    ([0, 0], [0, 0], "same cell"),
    ([0, 0], [2, 4], "cannot be reached"),
])
def test_grid_custom_invalid(start, goal, msg):
    with pytest.raises(ValueError, match=msg):
        grid_env().reset(options={"start": start, "goal": goal})
