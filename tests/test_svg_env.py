import json
import math

import gymnasium as gym
import numpy as np
import pytest

import hm3denv  # noqa: F401  registers HM3D/Svg-v0
from hm3denv.core.svgmap import SvgMap
from hm3denv.envs import DiscreteActions, EpisodeRecorder, OracleFollower, SvgEnv

U_RINGS = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
           np.array([[1.9, 0], [1.9, 2.0], [2.1, 2.0], [2.1, 0]], float)]


def tasks(*items, clearance=0.05):
    return {"schema_version": "1.0", "map_id": "t", "success_radius": 0.2,
            "clearance": clearance, "tasks": list(items)}


def task(sx, sy, sth, gx, gy, geo):
    return {"start": {"x": sx, "y": sy, "theta": sth}, "goal": {"x": gx, "y": gy},
            "geodesic_m": geo}


def make(robot="turtlebot4", rings=None, t=None, **kw):
    m = SvgMap(rings or [np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float)], {})
    return SvgEnv(robot=robot, svgmap=m, tasks=t or tasks(task(1, 1.5, 0, 5, 1.5, 4.0)),
                      **kw)


def test_spaces_and_obs():
    env = make()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["goal"][0] == pytest.approx(4.0) and obs["goal"][2] == pytest.approx(1.0)
    # the front ray (angle 0) hits the wall x=6 at 5 m; the back ray hits x=0 at 1 m
    assert obs["lidar"][0] == pytest.approx(5.0, abs=1e-4)
    assert obs["lidar"][36] == pytest.approx(1.0, abs=1e-4)
    assert env.action_space.shape == (2,)
    assert make("jetauto_pro").action_space.shape == (3,)


def test_drive_straight_to_goal():
    env = make()
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step([1.0, 0.0])
        total += r
        done = term or trunc
    assert info["success"] and info["termination"] == "success"
    assert info["spl"] > 0.95 and info["path_length"] == pytest.approx(3.8, abs=0.05)
    # dense reward ~ geodesic distance covered + bonus
    assert total == pytest.approx(3.8 + 10 - 0.01 * env.unwrapped.steps, abs=0.1)


def test_wall_stops_robot_without_penetration():
    env = make("clearpath_jackal", t=tasks(task(1, 1.5, 0, 1.0, 2.7, 1.2)))   # goal to the side
    env.reset(seed=0)
    for _ in range(40):
        _, _, term, trunc, info = env.step([1.0, 0.0])      # 2 m/s into the wall
    x, y, th = env.unwrapped.pose
    assert info["n_collisions"] > 0
    assert env.unwrapped.collider.pose_valid(env.unwrapped.robot.footprint, x, y, th)
    assert 6 - x - 0.254 < 0.03                              # stops against the wall (2 cm sub-steps)


def test_no_tunnelling_through_1cm_wall():
    rings = [np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float),
             np.array([[3.0, 0.2], [3.0, 2.8], [3.01, 2.8], [3.01, 0.2]], float)]
    env = make("clearpath_jackal", rings=rings, t=tasks(task(1, 1.5, 0, 5, 1.5, 6.0)))
    env.reset(seed=0)
    for _ in range(30):
        env.step([1.0, 0.0])
    assert env.unwrapped.pose[0] < 3.0


def test_omni_strafes():
    env = make("jetauto_pro", t=tasks(task(3, 1.0, 0, 3, 2.5, 1.5)))
    env.reset(seed=0)
    for _ in range(10):
        env.step([0.0, 1.0, 0.0])                             # 0.6 m/s to the left for 1 s
    x, y, th = env.unwrapped.pose
    assert (x, y, th) == pytest.approx((3.0, 1.6, 0.0), abs=1e-6)


def test_time_limit_truncates():
    env = make(time_limit=1.0)
    env.reset(seed=0)
    for _ in range(10):
        _, _, term, trunc, info = env.step([0.0, 0.0])
    assert trunc and not term and info["termination"] == "time_limit"


def test_discrete_macro_actions():
    env = DiscreteActions(make())
    env.reset(seed=0)
    env.step(1)
    assert env.unwrapped.pose[0] == pytest.approx(1.25, abs=1e-6)
    env.step(3)
    assert env.unwrapped.pose[2] == pytest.approx(math.radians(15), abs=1e-6)
    # TurtleBot4 0.31 m/s: 0.25 m takes 9 steps of dt 0.1 s (8 full + 1 partial)
    assert env.unwrapped.steps == 9 + 2


def test_recorder_writes_jsonl(tmp_path):
    p = tmp_path / "ep.jsonl"
    env = EpisodeRecorder(make(time_limit=0.3), p, trajectory=True)
    env.reset(seed=7)
    for _ in range(3):
        env.step([0.5, 0.0])
    rec = json.loads(p.read_text().strip())
    assert rec["seed"] == 7 and rec["termination"] == "time_limit"
    assert len(rec["trajectory"]) == 4


def test_gym_make_and_render():
    env = gym.make("HM3D/Svg-v0", robot="turtlebot4", svgmap=SvgMap(U_RINGS, {}),
                   tasks=tasks(task(1, 0.5, 0, 3, 0.5, 5.0)), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert env.render().ndim == 3
    svg = env.unwrapped.render_svg()
    assert svg.startswith("<?xml") and "polyline" in svg and svg.rstrip().endswith("</svg>")


@pytest.mark.parametrize("robot", ["turtlebot3_burger", "turtlebot4", "jetauto_pro",
                                   "agilex_limo"])
def test_oracle_reaches_goal_around_wall(robot):
    env = make(robot, rings=U_RINGS, t=tasks(task(1, 0.6, 0, 3, 0.6, 5.2)),
               reward="sparse")
    env.reset(seed=0)
    ora = OracleFollower(env)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(ora.act())
        done = term or trunc
    assert info["success"], info
    assert info["spl"] > 0.6


def test_fast_robot_cannot_skip_goal_between_steps():
    # Jackal 2 m/s = 0.2 m per step: the step ends at x=1.2 and x=1.4 are both 0.215 m from the goal (> 0.2);
    # only the middle point x=1.3 is inside the goal region -> check every sub-step
    env = make("clearpath_jackal", t=tasks(task(1.0, 1.5, 0, 1.3, 1.69, 0.4)))
    env.reset(seed=0)
    env.step([1.0, 0.0])
    _, r, term, _, info = env.step([1.0, 0.0])
    assert term and info["success"]
    # stops at the FIRST 2 cm sub-step inside the goal region: |x - 1.3| <= 0.0624 -> x = 1.24
    assert env.unwrapped.pose[0] == pytest.approx(1.24, abs=1e-9)
