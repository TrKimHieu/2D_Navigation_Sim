"""hm3d extend: new scenes, more tasks and new robots, without touching what is there."""

from conftest import require_build_extra

require_build_extra()

import json  # noqa: E402

import pytest  # noqa: E402

from hm3denv.build.extend import ExtendError, extend, status  # noqa: E402
from hm3denv.build.pipeline import build  # noqa: E402
from hm3denv.cli import main  # noqa: E402
from hm3denv.dataset import Dataset  # noqa: E402
from hm3denv.evaluate import evaluate  # noqa: E402
from room_glb import room  # noqa: E402

SCENES = ["00001-aaa", "00002-bbb", "00003-ccc", "00004-ddd", "00005-eee"]
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 0}


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    ws = tmp_path_factory.mktemp("extend-ws")
    assert main(["init", str(ws)]) == 0
    for i, sc in enumerate(SCENES):
        room(1.5 + 0.7 * i).export(ws / "raw" / "glb" / f"{sc}.glb")
    return ws


def tasks_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "tasks").rglob("*.json")}


@pytest.fixture(scope="module")
def svg_base(ws):
    cfg = {"name": "base", "env": "svg", "slice": {"up": 2}, "robots": ["turtlebot4"],
           "scenes": SCENES[:3], "split": SPLIT, "tasks": {"k": 3}}
    return build(cfg, ws=ws, out=ws / "datasets" / "base", preview=False)


def test_status_lists_what_can_be_added(ws, svg_base):
    st = status(str(svg_base), ws)
    assert st["scenes"] == 3 and st["new_scenes"] == SCENES[3:]
    assert "kobuki" in st["robots_to_add"] and st["scenes_without_glb"] == []


def test_extend_svg_new_scenes_tasks_and_robot(ws, svg_base):
    before = Dataset(svg_base)
    old_tasks = tasks_bytes(svg_base)
    root = extend(str(svg_base), ws=ws, new_scenes=True, add_tasks=2, robots=["jetauto_pro"])
    ds = Dataset(root)
    assert root == ws / "datasets" / "base-ext" and ds.name == "base-ext"
    assert ds.validate() == []
    assert tasks_bytes(svg_base) == old_tasks                       # the original is untouched
    for k, v in before.splits.items():                              # old scenes keep their split
        assert set(v) <= set(ds.splits[k])
    assert {s for v in ds.splits.values() for s in v} == set(SCENES)
    tb = "turtlebot4"
    for m in before.maps(tb):                                        # old tasks kept, 2 more each
        old, new = before.tasks(tb, m), ds.tasks(tb, m)
        assert new[:len(old)] == old and len(new) == len(old) + 2
        assert [t["id"] for t in new] == list(range(len(new)))
    assert set(ds.maps(tb)) > set(before.maps(tb))
    assert ds.robots == ["jetauto_pro", "turtlebot4"]
    assert (root / "maps" / "h63").is_dir() and ds.maps("jetauto_pro")
    h = ds.manifest["history"][-1]
    assert h["added_scenes"] == SCENES[3:] and h["added_robots"] == ["jetauto_pro"] and h["verified_3d"]
    for r in ds.robots:
        s = evaluate(root, robot=r)
        assert s["success"] == 1.0 and s["collision_episodes"] == 0.0


def test_extend_refuses_existing_output_and_nothing_to_do(ws, svg_base):
    with pytest.raises(ExtendError, match="nothing to do"):
        extend(str(svg_base), ws=ws, out="x")
    with pytest.raises(ExtendError, match="already in"):
        extend(str(svg_base), ws=ws, scenes=[SCENES[0]], out="x")
    (ws / "datasets" / "taken").mkdir(exist_ok=True)
    with pytest.raises(ExtendError, match="exists"):
        extend(str(svg_base), ws=ws, add_tasks=1, out="taken")


def test_extend_without_glbs_needs_no_verify(ws, svg_base, tmp_path):
    lone = tmp_path / "lone"                       # a workspace without the old GLBs
    assert main(["init", str(lone)]) == 0
    with pytest.raises(ExtendError, match="--no-verify"):
        extend(str(svg_base), ws=lone, add_tasks=1, out="nv")
    root = extend(str(svg_base), ws=lone, add_tasks=1, out="nv", verify=False)
    ds = Dataset(root)
    assert ds.validate() == [] and ds.manifest["source"]["verified_3d"] == "partial"
    assert ds.manifest["history"][-1]["verified_3d"] is False


def test_extend_grid_in_place_via_cli(ws, capsys):
    cfg = {"name": "g", "env": "grid", "slice": {"up": 2}, "scenes": SCENES[:3], "split": SPLIT,
           "tasks": {"k": 4, "d_min": 3}}
    root = build(cfg, ws=ws, out=ws / "datasets" / "g", preview=False)
    before = Dataset(root)
    old = {m: before.tasks("jetauto_pro", m) for m in before.maps("jetauto_pro")}
    assert main(["-w", str(ws), "extend", str(root), "--scenes", SCENES[4], "--add-tasks", "2",
                 "--in-place"]) == 0
    assert "+1 scenes" in capsys.readouterr().out
    ds = Dataset(root)
    assert ds.validate() == [] and SCENES[4] in {s for v in ds.splits.values() for s in v}
    for m, t in old.items():
        assert ds.tasks("jetauto_pro", m)[:len(t)] == t
    assert evaluate(root)["success"] == 1.0
    assert main(["-w", str(ws), "extend", str(root), "--status"]) == 0
    out = capsys.readouterr().out
    assert "history" in out and "scenes that can be added: 1" in out
    with pytest.raises(SystemExit):
        main(["extend", str(root), "--robots", "kobuki", "--scenes", "x", "--new-scenes"])
    assert main(["-w", str(ws), "extend", str(root), "--robots", "kobuki"]) == 2
    assert "SVG datasets" in capsys.readouterr().err
    json.loads((root / "manifest.json").read_text(encoding="utf-8"))
