"""End-to-end build tests on a tiny synthetic workspace (box rooms exported as GLB),
plus the CLI. The `workspace` test compares a real rebuild with the datasets of a workspace."""

from conftest import require_build_extra

require_build_extra()

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hm3denv.build.pipeline import ConfigError, load_config
from hm3denv.cli import main
from hm3denv.dataset import Dataset
from hm3denv.evaluate import evaluate

trimesh = pytest.importorskip("trimesh")

SCENES = ["00001-aaa", "00002-bbb", "00003-ccc"]


def room(obstacle_x):
    """6 x 4 m room (Z up): floor z=0, ceiling z=2.5, four walls, one 0.8 m box."""
    parts = []
    for ext, c in (([6, 4, 0.1], [3, 2, -0.05]), ([6, 4, 0.1], [3, 2, 2.55]),
                   ([0.1, 4, 2.5], [-0.05, 2, 1.25]), ([0.1, 4, 2.5], [6.05, 2, 1.25]),
                   ([6, 0.1, 2.5], [3, -0.05, 1.25]), ([6, 0.1, 2.5], [3, 4.05, 1.25]),
                   ([0.8, 0.8, 0.8], [obstacle_x, 2, 0.4])):
        b = trimesh.creation.box(extents=ext)
        b.apply_translation(c)
        parts.append(b)
    return trimesh.util.concatenate(parts)


@pytest.fixture(scope="session")
def tiny_ws(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ws")
    assert main(["init", str(ws)]) == 0
    for i, sc in enumerate(SCENES):
        room(2.0 + i).export(ws / "raw" / "glb" / f"{sc}.glb")
    return ws


def grid_cfg(**kw):
    return {"name": "g", "env": "grid", "slice": {"up": 2},
            "split": {"train": 0.34, "val": 0.33, "test": 0.33, "seed": 0},
            "tasks": {"k": 4, "d_min": 3}, **kw}


@pytest.fixture(scope="session")
def grid_built(tiny_ws, tmp_path_factory):
    out = tmp_path_factory.mktemp("out") / "g"
    cfg = tmp_path_factory.mktemp("cfg") / "g.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(grid_cfg()))
    assert main(["-w", str(tiny_ws), "build", str(cfg), "--out", str(out)]) == 0
    return out, cfg


def test_grid_build_is_valid_and_solvable(grid_built):
    out, _ = grid_built
    ds = Dataset(out)
    assert ds.validate() == [] and ds.robots == ["jetauto_pro"]
    assert sorted(ds.maps("jetauto_pro")) == [f"{s}_s1" for s in SCENES]
    assert {len(v) for v in ds.splits.values()} == {1}
    t = ds.tasks("jetauto_pro", "00001-aaa_s1")[0]
    assert set(t["start"]) == {"x", "y", "cell"} and "d_bfs" in t["labels"]
    assert 0 <= t["start"]["x"] <= 6 and 0 <= t["start"]["y"] <= 4       # metres, map frame
    s = evaluate(out, agent="oracle")
    assert s["success"] == 1.0 and s["episodes"] == 12
    assert (out / "preview" / "jetauto_pro" / "sheets" / "page_01.png").exists()
    rows = (out / "verify" / "jetauto_pro.csv").read_text().splitlines()
    assert len(rows) > 1 and rows[0].startswith("map_id")


def test_rebuild_skips_and_force_reruns(grid_built, tiny_ws):
    out, cfg = grid_built
    f = out / "tasks" / "jetauto_pro" / "00001-aaa_s1.json"
    stamp = json.loads((out / ".stages" / "tasks_jetauto_pro.json").read_text())
    mtime = f.stat().st_mtime_ns
    assert main(["-w", str(tiny_ws), "build", str(cfg), "--out", str(out), "--no-preview"]) == 0
    assert f.stat().st_mtime_ns == mtime
    assert main(["-w", str(tiny_ws), "build", str(cfg), "--out", str(out), "--no-preview",
                 "--force", "tasks"]) == 0
    new = json.loads((out / ".stages" / "tasks_jetauto_pro.json").read_text())
    assert new["created"] >= stamp["created"] and f.stat().st_mtime_ns != mtime
    assert json.loads(f.read_text())["tasks"] == Dataset(out).tasks("jetauto_pro", "00001-aaa_s1")
    assert Dataset(out).validate() == []


def test_grid_robot_size_override(tiny_ws, tmp_path):
    from hm3denv.build.pipeline import build
    out = build(grid_cfg(name="g15", grid={"robot": "jetauto_pro", "robot_size": 0.15}),
                ws=tiny_ws, out=tmp_path / "g15", preview=False)
    ds = Dataset(out)
    assert ds.robots == ["s15_h63"] and ds.grid_meta("00001-aaa_s1")["cell_m"] == pytest.approx(0.16)
    assert ds.robot("s15_h63").raw["derived_from"] == "jetauto_pro"


def test_svg_build(tiny_ws, tmp_path):
    from hm3denv.build.pipeline import build
    cfg = {"name": "s", "env": "svg", "slice": {"up": 2}, "robots": ["turtlebot4", "jetauto_pro"],
           "split": {"train": 0.34, "val": 0.33, "test": 0.33, "seed": 0}, "tasks": {"k": 3}}
    out = build(cfg, ws=tiny_ws, out=tmp_path / "s")
    ds = Dataset(out)
    assert ds.validate() == [] and ds.robots == ["jetauto_pro", "turtlebot4"]
    assert sorted(p.name for p in (out / "maps").iterdir()) == ["h36", "h63"]
    assert (out / "preview" / "robots.svg").exists()
    for r in ds.robots:
        s = evaluate(out, robot=r)
        assert s["success"] == 1.0 and s["collision_episodes"] == 0.0


def test_config_errors():
    with pytest.raises(ConfigError, match="unknown key"):
        load_config({"name": "x", "env": "grid", "tasks": {"kk": 1}})
    with pytest.raises(ConfigError, match="env"):
        load_config({"name": "x", "env": "voxel"})
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config({"name": "x", "env": "svg", "split": {"train": 0.9}})
    with pytest.raises(ConfigError, match="set by the pipeline"):
        load_config({"name": "x", "env": "grid", "slice": {"robot_height": 0.3}})
    c = load_config({"name": "x", "env": "svg"})
    assert c["tasks"]["k"] == 20 and c["height_classes"]["h36"] == 0.36


def test_example_configs_load():
    root = Path(__file__).resolve().parents[1] / "src" / "hm3denv" / "configs"
    names = {load_config(p)["name"] for p in root.glob("*.yaml")}
    assert names == {"grid-jetauto-v1", "grid-s15-v1", "svg-v1"}


def test_cli_dataset_commands(svg_dataset, capsys):
    search = str(svg_dataset.parent)
    assert main(["--search", search, "datasets"]) == 0
    assert "svg-tiny" in capsys.readouterr().out
    assert main(["--search", search, "info", "svg-tiny"]) == 0
    assert "turtlebot4" in capsys.readouterr().out
    assert main(["--search", search, "datasets", "validate", "svg-tiny"]) == 0
    assert main(["--search", search, "info", "missing"]) == 2
    assert "Available" in capsys.readouterr().err
    assert main(["robots", "check"]) == 0


def test_cli_render_and_eval(svg_dataset, tmp_path, capsys):
    search = str(svg_dataset.parent)
    out = tmp_path / "ep.svg"
    assert main(["--search", search, "render", "svg-tiny", "--robot", "turtlebot4",
                 "--map", "00001-aaa_s1", "--out", str(out)]) == 0
    assert out.read_text().startswith("<svg") or "<svg" in out.read_text()
    summ = tmp_path / "s.json"
    assert main(["--search", search, "eval", "svg-tiny", "--robot", "turtlebot4",
                 "--summary", str(summ)]) == 0
    assert json.loads(summ.read_text())["success"] == 1.0


def test_cli_without_workspace_fails_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HM3D_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["slice"]) == 2
    assert "hm3d init" in capsys.readouterr().err


# ------------------------------------------------------------------ real workspace

REAL_WS = Path(os.environ.get("HM3D_WORKSPACE", Path(__file__).resolve().parents[1] / "workspace"))


@pytest.mark.workspace
@pytest.mark.skipif(not (REAL_WS / "datasets" / "grid-jetauto-v1").exists()
                    or not (REAL_WS / "raw" / "glb" / "00800-TEEsavR23oF.glb").exists(),
                    reason="real HM3D workspace (GLB + built datasets) not present")
def test_rebuild_matches_workspace_datasets(tmp_path):
    """Rebuilding two scenes reproduces the tasks of the built datasets exactly."""
    from hm3denv.build.pipeline import build
    scenes = ["00800-TEEsavR23oF", "00337-CFVBbU9Rsyb"]
    g = Dataset(build({"name": "eq-grid", "env": "grid", "scenes": scenes}, ws=REAL_WS,
                      out=tmp_path / "g", preview=False))
    ref = Dataset(REAL_WS / "datasets" / "grid-jetauto-v1")
    for m in g.maps("jetauto_pro"):
        assert g.tasks("jetauto_pro", m) == [dict(t, id=i) for i, t in enumerate(ref.tasks("jetauto_pro", m))]
        assert np.array_equal(g.grid_arrays(m)["state"], ref.grid_arrays(m)["state"])
    s = Dataset(build({"name": "eq-svg", "env": "svg", "scenes": scenes[:1],
                       "robots": ["turtlebot4", "jetauto_pro"]}, ws=REAL_WS, out=tmp_path / "s",
                      preview=False))
    ref = Dataset(REAL_WS / "datasets" / "svg-v1")
    for r in s.robots:
        for m in s.maps(r):
            assert s.tasks(r, m) == ref.tasks(r, m)
