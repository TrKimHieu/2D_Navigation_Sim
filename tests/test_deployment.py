"""Deployment helpers: the picklable env factory, mapping-style (YAML / JSON) task-filter
ranges, and the CI workflow file."""

import pickle
from pathlib import Path

import gymnasium as gym
import pytest
import yaml

import hm3denv
from hm3denv.envs import SvgEnv

PKG = Path(__file__).resolve().parents[1]


def test_env_fn_is_picklable_and_works_in_workers(svg_dataset):
    fn = hm3denv.env_fn("HM3D/Svg-v0", dataset=str(svg_dataset), robot="turtlebot4")
    assert pickle.loads(pickle.dumps(fn))().reset(seed=0)[1]["map_id"]
    venv = gym.vector.AsyncVectorEnv([fn, fn])
    try:
        _, info = venv.reset(seed=0)
        assert len(info["map_id"]) == 2
    finally:
        venv.close()


def test_range_mapping_equals_tuple(svg_dataset):
    a = SvgEnv(dataset=svg_dataset, robot="turtlebot4")
    b = SvgEnv(dataset=svg_dataset, robot="turtlebot4")
    assert a.set_task_filter(geodesic_m={"min": 5.0}) == b.set_task_filter(geodesic_m=(5.0, None))
    assert a.set_task_filter(geodesic_m={"max": 4.5}) == b.set_task_filter(geodesic_m=(None, 4.5))
    with pytest.raises(ValueError, match="'min' and 'max'"):
        a.set_task_filter(geodesic_m={"lo": 1})


def test_ci_workflow_parses():
    ci = PKG / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        pytest.skip("not inside the repository")
    wf = yaml.safe_load(ci.read_text(encoding="utf-8"))
    assert set(wf["jobs"]) == {"tests", "docker", "scripts"}
    assert (PKG / "Dockerfile").exists()
