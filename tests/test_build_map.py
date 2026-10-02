"""hm3d build-map: GLB files or folders -> a ready dataset in the workspace."""

from conftest import require_build_extra

require_build_extra()

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from hm3denv.build.importer import find_glbs, import_glbs, scene_name  # noqa: E402
from hm3denv.cli import main  # noqa: E402
from hm3denv.dataset import Dataset  # noqa: E402
from hm3denv.evaluate import evaluate  # noqa: E402
from room_glb import room  # noqa: E402


@pytest.mark.parametrize("path, name", [
    ("x/My Room_s1.glb", "My-Room-s1"),
    ("x/00800-TEEsavR23oF/TEEsavR23oF.basis.glb", "00800-TEEsavR23oF"),
    ("x/house.v2.glb", "house-v2"),
    ("x/__.glb", "scene"),
])
def test_scene_names(path, name):
    assert scene_name(Path(path)) == name


def test_find_glbs_skips_semantic_and_duplicates(tmp_path):
    d = tmp_path / "hm3d" / "00009-abcdefghijk"
    d.mkdir(parents=True)
    for f in ("abcdefghijk.basis.glb", "abcdefghijk.glb", "abcdefghijk.semantic.glb"):
        (d / f).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in find_glbs([tmp_path / "hm3d"])] == ["abcdefghijk.basis.glb"]
    with pytest.raises(ValueError, match="not a .glb"):
        find_glbs([tmp_path / "notes.txt"])
    with pytest.raises(FileNotFoundError):
        find_glbs([tmp_path / "missing.glb"])


def test_import_refuses_a_different_file_of_the_same_name(tmp_path):
    ws = tmp_path / "ws"
    a, b = tmp_path / "a" / "room.glb", tmp_path / "b" / "room.glb"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert import_glbs(ws, [a]) == ["room"]
    assert import_glbs(ws, [a]) == ["room"]                  # same content: fine
    with pytest.raises(FileExistsError, match="--replace"):
        import_glbs(ws, [b])
    import_glbs(ws, [b], replace=True)
    assert (ws / "raw" / "glb" / "room.glb").read_bytes() == b"two"


def test_build_map_then_add_more(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HM3D_HOME", str(tmp_path / "home"))       # the data home is the workspace
    monkeypatch.delenv("HM3D_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "scenes"
    (src / "00010-abcdefghijk").mkdir(parents=True)
    room(2.0).export(src / "Living Room.glb")
    room(3.5).export(src / "00010-abcdefghijk" / "abcdefghijk.glb")
    assert main(["build-map", str(src), "--robots", "turtlebot4", "--up", "z", "--no-preview"]) == 0
    out = capsys.readouterr().out
    root = tmp_path / "home" / "datasets" / "my-maps"
    assert "2 new scene(s)" in out and "sim.bat --dataset my-maps" in out
    ds = Dataset(root)
    assert ds.validate() == [] and ds.robots == ["turtlebot4"]
    assert {s for v in ds.splits.values() for s in v} == {"Living-Room", "00010-abcdefghijk"}
    assert evaluate("my-maps", robot="turtlebot4")["success"] == 1.0   # found by name
    before = {m: ds.tasks("turtlebot4", m) for m in ds.maps("turtlebot4")}

    room(4.0).export(tmp_path / "kitchen.glb")
    assert main(["build-map", str(tmp_path / "kitchen.glb"), str(src), "--no-preview"]) == 0
    assert "1 new scene(s)" in capsys.readouterr().out
    ds = Dataset(root)
    assert ds.validate() == [] and "kitchen" in {s for v in ds.splits.values() for s in v}
    for m, t in before.items():
        assert ds.tasks("turtlebot4", m) == t
