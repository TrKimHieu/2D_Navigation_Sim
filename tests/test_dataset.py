import json

import numpy as np
import pytest

import hm3denv
from hm3denv.dataset.schema import assign_splits, validate
from hm3denv.envs import SvgEnv
from hm3denv.evaluate import evaluate


def test_load_and_query(svg_dataset):
    ds = hm3denv.load_dataset(svg_dataset)
    assert ds.env_type == "svg" and ds.robots == ["jetauto_pro", "turtlebot4"]
    assert sum(len(ds.maps("turtlebot4", s)) for s in ("train", "val", "test")) == 3
    assert ds.tasks("turtlebot4", "00001-aaa_s1")[0]["start"]["x"] == 1.0
    assert ds.robot("turtlebot4").id == "turtlebot4"
    with pytest.raises(KeyError, match="available"):
        ds.maps("kobuki")


def test_dataset_by_name_via_env_var(svg_dataset, monkeypatch):
    monkeypatch.setenv("HM3D_DATASETS", str(svg_dataset.parent))
    assert hm3denv.load_dataset("svg-tiny").name == "svg-tiny"
    assert "svg-tiny" in [d.name for d in hm3denv.list_datasets()]
    with pytest.raises(FileNotFoundError, match="Available"):
        hm3denv.load_dataset("nope")


def test_bundled_demo_datasets(monkeypatch):
    """The demos ship with the package: found by name with nothing configured, valid, and
    free of paths of the machine that built them."""
    monkeypatch.delenv("HM3D_DATASETS", raising=False)
    monkeypatch.delenv("HM3D_WORKSPACE", raising=False)
    for name, env_type in (("demo-svg", "svg"), ("demo-grid", "grid")):
        ds = hm3denv.load_dataset(name)
        assert ds.env_type == env_type and ds.validate() == []
        for f in ds.root.rglob("*"):
            if f.is_file():
                assert b"IsaacProjects" not in f.read_bytes(), f


def test_splits_are_by_scene_and_deterministic():
    scenes = [f"{i:05d}-x" for i in range(100)]
    a = assign_splits(scenes, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert a == assign_splits(scenes, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (70, 15, 15)
    assert not set(a["train"]) & set(a["test"])
    assert a != assign_splits(scenes, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)


def test_env_samples_across_maps(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot="turtlebot4")
    seen = {env.reset(seed=s)[1]["map_id"] for s in range(30)}
    assert len(seen) == 3                                   # every map of the split(s) drawn
    env = SvgEnv(dataset=svg_dataset, robot="turtlebot4", split="train")
    train = hm3denv.load_dataset(svg_dataset).maps("turtlebot4", "train")
    assert {env.reset(seed=s)[1]["map_id"] for s in range(10)} <= set(train)


def test_reset_options_pick_episode(svg_dataset):
    env = SvgEnv(dataset=svg_dataset, robot="turtlebot4")
    _, info = env.reset(options={"map_id": "00003-ccc_s1", "task_idx": 1})
    assert info["map_id"] == "00003-ccc_s1" and info["task_id"] == 1
    assert env.pose[0] == 5.0


def test_validate_detects_tampering(svg_dataset, tmp_path):
    import shutil
    assert validate(svg_dataset) == []
    copy = tmp_path / "copy"
    shutil.copytree(svg_dataset, copy)
    f = next((copy / "tasks" / "turtlebot4").glob("*.json"))
    d = json.loads(f.read_text())
    d["tasks"][0]["goal"]["x"] += 0.1
    f.write_text(json.dumps(d))
    assert any("modified" in p for p in validate(copy))


@pytest.mark.parametrize("fixture,robot", [("svg_dataset", "turtlebot4"), ("svg_dataset", "jetauto_pro"),
                                           ("grid_dataset", None)])
def test_oracle_evaluation_succeeds(fixture, robot, request, tmp_path):
    root = request.getfixturevalue(fixture)
    out = tmp_path / "ep.jsonl"
    s = evaluate(root, robot=robot, agent="oracle", out=out)
    assert s["success"] == 1.0 and s["collision_episodes"] == 0.0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == s["episodes"] and all("spl" in r and "termination" in r for r in recs)


def test_random_agent_runs(grid_dataset):
    s = evaluate(grid_dataset, agent="random", budget=30)
    assert 0.0 <= s["success"] <= 1.0 and s["episodes"] == 6
    assert np.isfinite(s["mean_time"])
