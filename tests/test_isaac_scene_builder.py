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


# ------------------------------------------------------------------ ISB_Nav_Dataset scenes/

M = 0.2                                                 # margin of the newer scene builder (m)


def write_dataset_scene(root, sid, box=BOX):
    """A scene folder of ISB_Nav_Dataset/<version>/scenes/: PGM in the USD frame (first row
    = largest y, origin = real lower-left corner), objects.json and the 16-bit id raster
    (rows grow with +y) with the walls as element 1 and the box as element 10."""
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    W = round((2 * M + ROOM_W) / RES)
    H = round((2 * M + ROOM_H) / RES)
    x = (np.arange(W) + 0.5) * RES - M
    y = (np.arange(H) + 0.5) * RES - M
    X, Y = np.meshgrid(x, y)                            # row r <-> y, growing upwards
    inside = (X >= 0) & (X <= ROOM_W) & (Y >= 0) & (Y <= ROOM_H)
    wall = inside & ((X < 0.1) | (X > ROOM_W - 0.1) | (Y < 0.1) | (Y > ROOM_H - 0.1))
    x0, x1, y0, y1 = box
    obj = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    img = np.full((H, W), ISB.UNKNOWN, np.uint8)
    img[inside] = ISB.FREE
    img[wall | obj] = ISB.OCC
    Image.fromarray(img[::-1]).save(d / f"{sid}.pgm")
    (d / f"{sid}.yaml").write_text(
        f"# BEV map for ROS map_server. {ISB.USD_FRAME_MARK} (x, y).\nimage: {sid}.pgm\n"
        f"resolution: {RES:.6f}\norigin: [{-M:.4f}, {-M:.4f}, 0.0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.196\n", encoding="utf-8")
    ids = np.zeros((H, W), np.uint16)
    ids[wall] = 1
    ids[obj] = 10
    Image.fromarray(ids).save(d / f"{sid}_ids.png")
    room = [[0, 0], [ROOM_W, 0], [ROOM_W, ROOM_H], [0, ROOM_H]]
    objs = {"schema": "isaac-scene-builder.objects/1", "size_m": [ROOM_W, ROOM_H],
            "generator": {"theme": "office", "layout": "open", "complexity": 0},
            "rooms": [{"id": "room0", "type": "office", "area_m2": ROOM_W * ROOM_H,
                       "outline": [room]}],
            "walls": [{"id": 1, "class": "wall", "start": [0, 0], "end": [ROOM_W, 0],
                       "thickness": 0.2, "faces": {"left": ["room0"], "right": []}}],
            "openings": [],
            "objects": [{"id": 10, "label": "table", "name": "Desk", "asset": "SM_Desk",
                         "room": "room0", "pose": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                                                   "yaw_deg": 0.0},
                         "size": [x1 - x0, y1 - y0], "height": 0.75}],
            "raster": {"file": f"{sid}_ids.png", "resolution_m": RES, "origin_m": [-M, -M],
                       "width": W, "height": H}}
    (d / f"{sid}_objects.json").write_text(json.dumps(objs), encoding="utf-8")


def test_dataset_scenes_layout_frame_and_scene_index(tmp_path):
    from hm3denv.build import scene_index as SI
    scenes = tmp_path / "ISB_Nav_Dataset" / "v" / "scenes"
    write_dataset_scene(scenes, "v_s06_000")
    write_dataset_scene(scenes, "v_s06_001", box=(1.0, 1.8, 2.5, 3.2))
    (scenes / "index.json").write_text(json.dumps([{"stem": "v_s06_000"}, {"stem": "v_s06_001"}]),
                                       encoding="utf-8")
    ws = tmp_path / "ws"
    assert main(["init", str(ws)]) == 0
    root = build(config(scenes, env="grid"), ws=ws, preview=False)
    ds = load_dataset(root)
    assert ds.validate() == []
    assert sorted(ds.maps(ds.robots[0])) == ["isb-v_s06_000_s0", "isb-v_s06_001_s0"]
    m = "isb-v_s06_000_s0"
    sc = json.loads((root / "scene" / ds.robots[0] / f"{m}.json").read_text(encoding="utf-8"))
    assert sc["schema"] == SI.SCHEMA and sc["source"]["scene_id"] == "v_s06_000"
    grid = np.load(root / "grids" / f"{m}.npz")["grid"]
    fr = sc["frame"]
    # every task end point: metres == frame(cell), inside the room, in the USD frame
    for t in json.loads((root / "tasks" / ds.robots[0] / f"{m}.json").read_text())["tasks"]:
        for e in (t["start"], t["goal"]):
            r, c = e["cell"]
            assert abs(fr["x0"] + c * fr["cell_m"] - e["x"]) < 1e-6
            assert abs(fr["y0"] - r * fr["cell_m"] - e["y"]) < 1e-6
            assert 0 < e["x"] < ROOM_W and 0 < e["y"] < ROOM_H
    # the desk: its cells are blocked, around its pose, not mirrored (y-flip bug: y = 3.0)
    desk = next(o for o in sc["objects"] if o["id"] == 10)
    assert desk["label"] == "table" and desk["n_cells"] > 0
    assert all(grid[r, c] != 0 for r, c in desk["cells_rc"])
    r, c = SI.world_to_cell(fr, 4.3, 1.0)
    assert [r, c] in desk["cells_rc"] and desk["center_rc"] == [r, c]
    assert SI.world_to_cell(fr, 4.3, 3.0) not in {tuple(x) for x in desk["cells_rc"]}
    assert set(desk["shape"]) <= {"#", ".", "/"} and desk["size"][0] * desk["size"][1] >= desk["n_cells"]
    assert sc["counts"]["labelled_free_cells"] == 0
    assert sc["rooms"][0]["objects"] == [10] and sc["rooms"][0]["free_cells"] > 0
    assert any(w["cells_rc"] for w in sc["walls"])
