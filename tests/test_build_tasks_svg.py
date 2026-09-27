from conftest import require_build_extra

require_build_extra()

import numpy as np

from hm3denv.core.svgmap import SvgMap
from hm3denv.envs import SvgEnv as HM3DSvgEnv
from hm3denv.robots import load as load_robot
from hm3denv.core.geometry import Collider
from hm3denv.envs.oracle import OracleFollower
from hm3denv.build.tasks_svg import sample_map

# 8 x 5 m room with a middle partition (U corridor) and a square obstacle
RINGS = [np.array([[0, 0], [8, 0], [8, 5], [0, 5]], float),
         np.array([[3.9, 0], [3.9, 3.5], [4.1, 3.5], [4.1, 0]], float),
         np.array([[6, 3], [6, 3.6], [6.6, 3.6], [6.6, 3]], float)]


def run(robot_id="turtlebot3_waffle_pi", check=None, k=12, seed=0):
    m = SvgMap(RINGS, {})
    return m, sample_map(m, load_robot(robot_id), k, np.random.default_rng(seed),
                         1.0, 30.0, 1.1, 0.05, check)


def test_tasks_valid_and_filtered():
    m, (tasks, _, pl) = run()
    robot = load_robot("turtlebot3_waffle_pi")
    col = Collider(m)
    assert len(tasks) == 12
    for t in tasks:
        s, g = t["start"], t["goal"]
        assert col.pose_valid(robot.footprint, s["x"], s["y"], s["theta"])
        assert pl.in_cdisk(g["x"], g["y"])
        assert 1.0 <= t["geodesic_m"] <= 30.0 and t["labels"]["gdr"] >= 1.1
        assert t["geodesic_m"] >= t["euclidean_m"] - 1e-6


def test_deterministic_and_bins_spread():
    _, (a, _, _) = run(seed=3)
    _, (b, _, _) = run(seed=3)
    assert a == b
    geo = sorted(t["geodesic_m"] for t in a)
    assert geo[-1] > 1.5 * geo[0]


def test_rejected_points_never_used():
    calls = []

    def check(poly, which):
        cx = poly[:, 0].mean()
        calls.append(which)
        return {"ok": cx < 4.0 or which == "start"}      # goals right of the partition rejected

    _, (tasks, checks, _) = run(check=check)
    assert tasks and all(t["goal"]["x"] < 4.0 for t in tasks)
    assert "goal" in calls and any(not c[-1]["ok"] for c in checks)


def test_oracle_completes_sampled_tasks():
    m, (tasks, _, _) = run("turtlebot4", k=6)
    env = HM3DSvgEnv(robot="turtlebot4", svgmap=m, reward="sparse",
                     tasks={"schema_version": "2.0", "map_id": "t", "success_radius": 0.2,
                            "clearance": 0.05, "tasks": tasks})
    for i in range(len(tasks)):
        env.reset(options={"task_idx": i})
        ora, done = OracleFollower(env), False
        while not done:
            _, _, term, trunc, info = env.step(ora.act())
            done = term or trunc
        assert info["success"] and env.n_coll == 0, (i, info)


def test_sparse_candidates_do_not_crash():
    # narrow room: few candidates -> a quantile bin can be empty
    m = SvgMap([np.array([[0, 0], [2.2, 0], [2.2, 0.8], [0, 0.8]], float)], {})
    tasks, _, _ = sample_map(m, load_robot("turtlebot3_burger"), 8, np.random.default_rng(0),
                             1.0, 30.0, 1.0, 0.05)
    assert tasks
