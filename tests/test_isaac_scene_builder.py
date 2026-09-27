"""Datasets from Isaac-Scene-Builder output (BEV raster geometry)."""

from conftest import require_build_extra

require_build_extra()

import json

import numpy as np
import pytest
from PIL import Image

from hm3denv.build import isaac_scene_builder as ISB
from hm3denv.build.pipeline import ConfigError, build, load_config
from hm3denv.cli import main
from hm3denv.dataset import load_dataset
from hm3denv.envs import GridEnv, SvgEnv
from hm3denv.evaluate import evaluate

RES = 0.025
LEFT, TOP, RIGHT, BOTTOM = 0.2, 0.3, 0.2, 0.2          # asymmetric margins (m)
ROOM_W, ROOM_H = 6.0, 4.0                              # editor x in [0, 6], y in [0, 4]
BOX = (4.0, 4.6, 0.6, 1.4)                             # obstacle x0, x1, y0, y1 (editor = USD)


def bev_image(box=BOX):
    """PGM as Isaac-Scene-Builder draws it: row r, column c <-> editor
    x = (c + .5) res - LEFT, y = (r + .5) res - TOP (y grows downwards in the image)."""
    W = round((LEFT + ROOM_W + RIGHT) / RES)
    H = round((TOP + ROOM_H + BOTTOM) / RES)
    x = (np.arange(W) + 0.5) * RES - LEFT
    y = (np.arange(H) + 0.5) * RES - TOP
    X, Y = np.meshgrid(x, y)
    img = np.full((H, W), ISB.UNKNOWN, np.uint8)
    inside = (X >= 0) & (X <= ROOM_W) & (Y >= 0) & (Y <= ROOM_H)
    img[inside] = ISB.FREE
    wall = inside & ((X < 0.1) | (X > ROOM_W - 0.1) | (Y < 0.1) | (Y > ROOM_H - 0.1))
    img[wall] = ISB.OCC
    x0, x1, y0, y1 = box
    img[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = ISB.OCC
    return img


def write_scene(root, sid, box=BOX):
    d = root / "bev" / sid
    d.mkdir(parents=True, exist_ok=True)
    img = bev_image(box)
    Image.fromarray(img).save(d / f"{sid}.pgm")
    H = img.shape[0]
    # ROS map frame: lower-left pixel corner at origin, y_ros = -editor y
    origin_y = -(H * RES - TOP)
    (d / f"{sid}.yaml").write_text(
        f"image: {sid}.pgm\nresolution: {RES:.6f}\norigin: [{-LEFT:.4f}, {origin_y:.4f}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n", encoding="utf-8")


@pytest.fixture
def isb(tmp_path):
    """Scene builder output (S000, S001 in manifest.jsonl, probe0 not) + a workspace."""
    out = tmp_path / "isb_output"
    write_scene(out, "S000")
    write_scene(out, "S001", box=(1.0, 1.8, 2.5, 3.2))
    write_scene(out, "probe0")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for sid, dens in (("S000", 0.05), ("S001", 0.08), ("S000", 0.07)):   # last S000 wins
            fh.write(json.dumps({"scene_id": sid, "source": "procedural", "level": "d10",
                                 "occlusion_density_actual": dens,
                                 "usd": f"C:/scenes/{sid}.usd"}) + "\n")
    ws = tmp_path / "ws"
    assert main(["init", str(ws)]) == 0
    return out, ws


def config(out, env="svg", **extra):
    cfg = {"name": f"isb-{env}", "env": env,         # these fixtures have no CSV: BEV raster
           "source": {"type": "isaac_scene_builder", "path": str(out), "geometry": "raster"},
           "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 0}}
    if env == "svg":
        cfg.update(robots=["turtlebot4"], tasks={"k": 6, "min_geo": 1.0})
    else:
        cfg.update(grid={"robot": "jetauto_pro"}, tasks={"k": 6, "d_min": 3, "budget": 200})
    cfg.update(extra)
    return cfg


def test_svg_dataset_frame_and_oracle(isb):
    out, ws = isb
    root = build(config(out), ws=ws, preview=False)
    ds = load_dataset(root)
    assert ds.validate() == []
    assert ds.manifest["source"]["verified_3d"] is False
    assert sorted(ds.maps("turtlebot4")) == ["isb-S000_s0", "isb-S001_s0"]   # probe0 left out
    env = SvgEnv(dataset=root, robot="turtlebot4", map_id="isb-S000_s0")
    env.reset(seed=0)
    col = env.collider
    # frame = USD / editor coordinates: the obstacle is where the editor put it ...
    assert not col.inside(4.3, 1.0)
    # ... and not mirrored across the room (a y-flip bug would put it at y = 3.0)
    assert col.inside(4.3, 3.0) and col.inside(1.0, 1.0)
    assert not col.inside(-0.1, 2.0) and not col.inside(3.0, 4.2)      # outside the room
    meta = env.map.meta
    assert meta["source"]["scene_id"] == "S000" and meta["source"]["occlusion_density_actual"] == 0.07
    assert "USD" in meta["frame"]["note"]
    s = evaluate(root, robot="turtlebot4", agent="oracle")
    assert s["episodes"] == 12 and s["success"] == 1.0


def test_grid_dataset_with_preview(isb):
    out, ws = isb
    root = build(config(out, env="grid"), ws=ws, preview=True)
    ds = load_dataset(root)
    assert ds.validate() == []
    rid = ds.robots[0]
    assert len(ds.maps(rid)) == 2
    assert list((root / "preview").rglob("*.png"))
    assert evaluate(root, agent="oracle")["success"] == 1.0
    env = GridEnv(dataset=root)
    env.reset(seed=0)


def test_import_stage_reruns_only_on_change(isb):
    out, ws = isb
    cfg = config(out)
    root = build(cfg, ws=ws, preview=False)
    stamp = next((root / ".stages").glob("import_*.json"))
    first = json.loads(stamp.read_text(encoding="utf-8"))
    build(cfg, ws=ws, preview=False)
    assert json.loads(stamp.read_text(encoding="utf-8")) == first          # up to date: not re-run
    write_scene(out, "S001", box=(2.0, 2.5, 2.0, 2.5))                  # edited in the builder
    build(cfg, ws=ws, preview=False)
    assert json.loads(stamp.read_text(encoding="utf-8"))["inputs_hash"] != first["inputs_hash"]
    env = SvgEnv(dataset=root, robot="turtlebot4", map_id="isb-S001_s0")
    env.reset(seed=0)
    assert not env.collider.inside(2.25, 2.25) and env.collider.inside(1.4, 2.8)


def test_scene_selection(isb):
    out, _ = isb
    src = ISB.check_source({"type": "isaac_scene_builder", "path": str(out)})
    assert ISB.resolve(src, "all") == ["S000", "S001"]
    assert ISB.resolve(src, ["probe0"]) == ["probe0"]
    with pytest.raises(FileNotFoundError):
        ISB.resolve(src, ["S999"])


@pytest.mark.parametrize("bad, match", [
    ({"slice": {"res": 0.02}}, "slice"),
    ({"tasks": {"verify": True}}, "verify"),
    ({"source": {"type": "isaac_scene_builder", "path": "Z:/nowhere"}}, "bev"),
    ({"source": {"type": "isaac_scene_builder", "path": ".", "colour": 1}}, "unknown"),
    ({"source": {"type": "unity"}}, "source.type"),
])
def test_config_errors(isb, bad, match):
    out, _ = isb
    with pytest.raises(ConfigError, match=match):
        load_config(config(out, **bad))


def test_hm3d_config_unchanged():
    cfg = load_config({"name": "x", "env": "grid"})
    assert cfg["source"] == {"type": "hm3d"} and cfg["tasks"]["verify"] is True


def test_verify_refuses_imported_dataset(isb, capsys):
    out, ws = isb
    root = build(config(out), ws=ws, preview=False)
    assert main(["verify", str(root)]) == 2
    assert "no 3D mesh" in capsys.readouterr().err
