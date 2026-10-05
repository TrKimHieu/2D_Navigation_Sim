"""Several environments per worker process: the results must equal gymnasium's
SyncVectorEnv exactly."""

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

import hm3denv
from hm3denv.vector import GroupedVectorEnv, make_vec

N, K = 5, 2                       # groups of 2 + 2 + 1


def same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype == object:
            return all(same(x, y) for x, y in zip(a.ravel(), b.ravel()))
        return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def svg_fns(svg_dataset, **kw):
    # 2 s time limit: plenty of truncations and autoresets in 150 steps
    return [hm3denv.env_fn("HM3D/Svg-v0", dataset=str(svg_dataset), robot="turtlebot4",
                           time_limit=2.0, **kw)] * N


def rollout(venv, steps=150, seed=0):
    rng = np.random.default_rng(1)
    out = [venv.reset(seed=seed)]
    for _ in range(steps):
        if isinstance(venv.single_action_space, gym.spaces.Discrete):
            a = rng.integers(0, venv.single_action_space.n, venv.num_envs)
        else:
            a = rng.uniform(-1, 1, (venv.num_envs, *venv.single_action_space.shape)).astype(np.float32)
        out.append(venv.step(a))
    return out


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP])
def test_grouped_equals_sync(svg_dataset, mode, shared):
    fns = svg_fns(svg_dataset)
    ref = gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
    grp = GroupedVectorEnv(fns, envs_per_worker=K, autoreset_mode=mode, shared_memory=shared)
    try:
        a, b = rollout(ref), rollout(grp)
        assert same(a, b)
        n_done = sum(int(np.sum(s[2] | s[3])) for s in a[1:])
        assert n_done >= 10                                   # autoreset was exercised
        assert grp.metadata["autoreset_mode"] == mode
    finally:
        ref.close()
        grp.close()


def test_grouped_grid_env_and_reset_mask(grid_dataset):
    fns = [hm3denv.env_fn("HM3D/Grid-v0", dataset=str(grid_dataset), budget=15)] * N
    ref, grp = gym.vector.SyncVectorEnv(fns), GroupedVectorEnv(fns, envs_per_worker=K)
    try:
        assert same(rollout(ref, 60), rollout(grp, 60))
        mask = np.array([True, False, False, True, True])
        assert same(ref.reset(seed=[7, 8, 9, 10, 11], options={"reset_mask": mask}),
                    grp.reset(seed=[7, 8, 9, 10, 11], options={"reset_mask": mask}))
        assert same(rollout(ref, 20, seed=3), rollout(grp, 20, seed=3))
    finally:
        ref.close()
        grp.close()


def test_grouped_call_attrs_and_errors(svg_dataset):
    grp = GroupedVectorEnv(svg_fns(svg_dataset), envs_per_worker=K)
    try:
        grp.reset(seed=0)
        assert grp.call("set_task_filter", geodesic_m=(5.0, None)) == ({"maps": 1, "tasks": 1},) * N
        _, info = grp.reset(seed=0)
        assert list(info["map_id"]) == ["00002-bbb_s1"] * N
        grp.set_attr("time_factor", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert grp.get_attr("time_factor") == (1.0, 2.0, 3.0, 4.0, 5.0)
        with pytest.raises(RuntimeError, match="AttributeError"):
            grp.call("no_such_method")
        assert grp.get_attr("time_factor") == (1.0, 2.0, 3.0, 4.0, 5.0)  # still usable
    finally:
        grp.close()


def test_make_vec_envs_per_worker(svg_dataset):
    venv = make_vec("HM3D/Svg-v0", 4, envs_per_worker=2, dataset=str(svg_dataset),
                    robot="turtlebot4")
    try:
        assert isinstance(venv, GroupedVectorEnv) and venv.num_envs == 4
        obs, _ = venv.reset(seed=0)
        assert obs["lidar"].shape[0] == 4
    finally:
        venv.close()
    with pytest.raises(ValueError):
        make_vec("HM3D/Svg-v0", 4, envs_per_worker=0, dataset=str(svg_dataset))


@pytest.mark.parametrize("vectorization,epw", [("sync", 1), ("async", 1), ("async", 2)])
def test_make_vec_autoreset_mode(svg_dataset, vectorization, epw):
    venv = make_vec("HM3D/Svg-v0", 2, vectorization=vectorization, envs_per_worker=epw,
                    autoreset_mode="SameStep", dataset=str(svg_dataset), robot="turtlebot4",
                    time_factor=0.05)
    try:
        assert venv.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
        venv.reset(seed=0)
        for _ in range(500):
            _, _, term, trunc, info = venv.step(venv.action_space.sample())
            if (term | trunc).any():
                break
        done = term | trunc
        assert done.any() and (info["_final_obs"] == done).all()
        assert info["final_info"]["termination"][done.argmax()] in ("success", "collision",
                                                                     "time_limit")
    finally:
        venv.close()


def test_shared_observations_copy_semantics(svg_dataset):
    grp = GroupedVectorEnv(svg_fns(svg_dataset), envs_per_worker=K, copy=False)
    cpy = GroupedVectorEnv(svg_fns(svg_dataset), envs_per_worker=K)
    try:
        a, _ = grp.reset(seed=0)
        b, _ = grp.step(np.zeros((N, 2), np.float32))[:2]
        assert a["lidar"] is b["lidar"]                       # the shared buffer itself
        c, _ = cpy.reset(seed=0)
        c["lidar"][:] = -1.0                                  # a copy: the env is not affected
        d = cpy.step(np.zeros((N, 2), np.float32))[0]
        assert (d["lidar"] >= 0).all() and d["lidar"] is not c["lidar"]
    finally:
        grp.close()
        cpy.close()
