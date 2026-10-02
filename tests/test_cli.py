"""Commands of a plain install (no build extra): `hm3d render`, `hm3d robots render` and
`hm3d eval` must work without Pillow; the evaluation summary."""

import sys

import pytest

from hm3denv.cli import main

MAP = "00001-aaa_s1"


@pytest.fixture
def no_pillow(monkeypatch):
    """Make `import PIL` fail and import the preview module afresh, as on a `[fast]` install;
    afterwards the original module is put back (in sys.modules and on the package)."""
    import hm3denv.build as build
    saved = sys.modules.pop("hm3denv.build.preview", None), build.__dict__.pop("preview", None)
    for name in [m for m in sys.modules if m == "PIL" or m.startswith("PIL.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "PIL", None)
    yield
    sys.modules.pop("hm3denv.build.preview", None)
    build.__dict__.pop("preview", None)
    if saved[0] is not None:
        sys.modules["hm3denv.build.preview"] = saved[0]
    if saved[1] is not None:
        build.preview = saved[1]


def test_render_without_pillow(no_pillow, svg_dataset, grid_dataset, tmp_path):
    svg, png, grid = tmp_path / "ep.svg", tmp_path / "ep.png", tmp_path / "grid.png"
    assert main(["render", str(svg_dataset), "--robot", "turtlebot4", "--map", MAP, "--out", str(svg)]) == 0
    assert "<svg" in svg.read_text(encoding="utf-8")
    assert main(["render", str(svg_dataset), "--robot", "turtlebot4", "--map", MAP, "--out", str(png)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert main(["render", str(grid_dataset), "--map", MAP, "--out", str(grid)]) == 0
    assert grid.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_robots_render_without_pillow(no_pillow, tmp_path):
    out = tmp_path / "robots.svg"
    assert main(["robots", "render", "turtlebot4", "--out", str(out)]) == 0
    assert "<svg" in out.read_text(encoding="utf-8")


def test_eval_reports_progress(svg_dataset, capsys):
    assert main(["eval", str(svg_dataset), "--robot", "turtlebot4", "--per-map", "1"]) == 0
    out = capsys.readouterr()
    assert '"success": 1.0' in out.out
    assert "map 1/3" in out.err and "map 3/3" in out.err
    assert main(["eval", str(svg_dataset), "--robot", "turtlebot4", "--per-map", "1", "--quiet"]) == 0
    assert capsys.readouterr().err == ""


def test_eval_mean_progress(svg_dataset, grid_dataset):
    from hm3denv.evaluate import evaluate
    assert evaluate(svg_dataset, robot="turtlebot4", agent="oracle")["mean_progress"] == 1.0
    assert evaluate(grid_dataset, agent="oracle")["mean_progress"] == 1.0
    s = evaluate(svg_dataset, robot="turtlebot4", agent="random", per_map=1)
    assert s["success"] == 0.0 and -1.0 < s["mean_progress"] < 1.0
    # without the distance keys in info (nav_info=False) there is nothing to measure
    s = evaluate(svg_dataset, robot="turtlebot4", agent="random", per_map=1, nav_info=False)
    assert s["mean_progress"] is None
