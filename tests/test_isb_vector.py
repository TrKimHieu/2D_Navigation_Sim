"""SVG maps built from the Isaac-Scene-Builder CSV geometry.

The fixture draws its BEV with the same PIL recipe as Isaac-Scene-Builder (bev_export.py),
so the mesh rotation convention and the BEV cross-check are tested against it."""

from conftest import require_build_extra

require_build_extra()

import csv
import json
import math

import numpy as np
import pytest
import shapely
from PIL import Image, ImageDraw

pytest.importorskip("shapely")

from hm3denv.build import isaac_scene_builder as ISB          # noqa: E402
from hm3denv.build import isb_vector as V                     # noqa: E402
from hm3denv.build.pipeline import ConfigError, build, load_config   # noqa: E402
from hm3denv.cli import main                                  # noqa: E402
from hm3denv.dataset import load_dataset                      # noqa: E402
from hm3denv.envs import SvgEnv                               # noqa: E402
from hm3denv.evaluate import evaluate                         # noqa: E402

CELL, SCALE, WALL_M, GAP = 3.0, 40, 0.2, (0.35, 0.65)
L_YAW = 30.0
FIELDS = ["kind", "name", "x", "y", "yaw", "shape", "w", "h", "r", "x2", "y2"]


def l_asset(size=1024):
    """Asymmetric L footprint (dark = obstacle) in a CELL x CELL frame."""
    a = np.full((size, size), 255, np.uint8)
    s = size / CELL                                          # px per metre
    c = size // 2
    a[int(c - 0.6 * s):int(c + 0.6 * s), int(c - 0.4 * s):int(c - 0.1 * s)] = 0   # long bar
    a[int(c + 0.3 * s):int(c + 0.6 * s), int(c - 0.4 * s):int(c + 0.5 * s)] = 0   # foot
    return a


def scene_rows():
    rows = [dict(kind="floor", x=1.5, y=1.5, shape="box", w=CELL, h=CELL),
            dict(kind="floor", x=4.5, y=1.5, shape="box", w=CELL, h=CELL)]
    for (x, y, x2, y2) in [(0, 0, 3, 0), (3, 0, 6, 0), (0, 3, 3, 3), (3, 3, 6, 3), (0, 0, 0, 3),
                           (6, 0, 6, 3)]:
        rows.append(dict(kind="wall", x=x, y=y, x2=x2, y2=y2))
    rows.append(dict(kind="door", x=3, y=0, x2=3, y2=3))       # open for y in [1.05, 1.95]
    rows += [dict(kind="prop", name="Lshape", x=1.5, y=1.5, yaw=L_YAW, shape="mesh", w=1, h=1, r=0.6),
             dict(kind="prop", name="Box", x=4.5, y=0.8, yaw=0, shape="bbox", w=0.6, h=0.4, r=0.4),
             dict(kind="prop", name="Ball", x=4.8, y=2.2, yaw=0, shape="sphere", w=0.4, h=0.4, r=0.2),
             dict(kind="prop", name="Ghost", x=5.3, y=1.5, yaw=45, shape="mesh", w=0.5, h=0.5, r=0.3)]
    return rows


def draw_bev(rows, mesh_dir, nx=2, ny=1, margin_m=WALL_M, scale=SCALE):
    """Isaac-Scene-Builder's rasterize_to_image, reduced to what the fixture uses."""
    m = int(round(margin_m * scale))
    W, H = int(nx * CELL * scale) + 2 * m, int(ny * CELL * scale) + 2 * m

    def X(v):
        return m + v * scale

    img = Image.new("L", (W, H), ISB.UNKNOWN)
    d = ImageDraw.Draw(img)
    for r in rows:
        if r["kind"] == "floor":
            d.rectangle([X(r["x"] - CELL / 2), X(r["y"] - CELL / 2),
                         X(r["x"] + CELL / 2), X(r["y"] + CELL / 2)], fill=ISB.FREE)
    for r in rows:
        if r["kind"] in ("wall", "door"):
            x0, y0, x1, y1 = r["x"], r["y"], r["x2"], r["y2"]
            segs = [((x0, y0), (x1, y1))] if r["kind"] == "wall" else [
                ((x0, y0), (x0 + (x1 - x0) * GAP[0], y0 + (y1 - y0) * GAP[0])),
                ((x0 + (x1 - x0) * GAP[1], y0 + (y1 - y0) * GAP[1]), (x1, y1))]
            for (ax, ay), (bx, by) in segs:
                d.line([X(ax), X(ay), X(bx), X(by)], fill=ISB.OCC, width=int(round(WALL_M * scale)))
    for r in rows:
        if r["kind"] != "prop":
            continue
        cx, cy = X(r["x"]), X(r["y"])
        png = mesh_dir / f"{r['name']}.png"
        if r["shape"] == "sphere":
            rr = r["r"] * scale
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=ISB.OCC)
        elif r["shape"] == "mesh" and png.exists():                    # _mesh_stamp
            fp = int(round(CELL * scale))
            st = Image.open(png).convert("L").resize((fp, fp), Image.NEAREST)
            st = st.rotate(-r["yaw"], expand=True, fillcolor=255)
            mask = st.point(lambda v: 255 if v < 128 else 0)
            img.paste(Image.new("L", st.size, ISB.OCC),
                      (int(cx - st.width / 2), int(cy - st.height / 2)), mask)
        else:                                                           # bbox / fallback
            a = math.radians(r["yaw"])
            pts = [(cx + (dx * math.cos(a) - dy * math.sin(a)) * scale,
                    cy + (dx * math.sin(a) + dy * math.cos(a)) * scale)
                   for dx, dy in ((-r["w"] / 2, -r["h"] / 2), (r["w"] / 2, -r["h"] / 2),
                                  (r["w"] / 2, r["h"] / 2), (-r["w"] / 2, r["h"] / 2))]
            d.polygon(pts, fill=ISB.OCC)
    return img, m


@pytest.fixture
def isb_csv(tmp_path):
    out = tmp_path / "isb_output"
    mesh = out / "mesh_shapes"
    mesh.mkdir(parents=True)
    Image.fromarray(l_asset()).save(mesh / "Lshape.png")             # "Ghost" has no image
    for sid in ("S000", "S001"):
        d = out / "bev" / sid
        d.mkdir(parents=True)
        rows = scene_rows()
        with open(d / f"{sid}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        img, m = draw_bev(rows, mesh)
        img.save(d / f"{sid}.pgm")
        (d / f"{sid}.yaml").write_text(
            f"image: {sid}.pgm\nresolution: {1 / SCALE:.6f}\n"
            f"origin: [{-m / SCALE:.4f}, {-(1 * CELL) - m / SCALE:.4f}, 0.0]\nnegate: 0\n",
            encoding="utf-8")
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps({"scene_id": s, "level": "d1"}) + "\n" for s in ("S000", "S001")),
        encoding="utf-8")
    ws = tmp_path / "ws"
    assert main(["init", str(ws)]) == 0
    src = ISB.check_source({"type": "isaac_scene_builder", "path": str(out)})
    return out, ws, src


def placed_l_pixels(src, yaw=L_YAW, step=4):
    """Map coordinates of the obstacle pixel centres of the L asset placed at (1.5, 1.5)."""
    a = l_asset()
    S = a.shape[0]
    px = CELL / S
    i, j = np.nonzero(a[::step, ::step] < 128)
    u, v = (j * step + 0.5) * px - CELL / 2, (i * step + 0.5) * px - CELL / 2
    t = math.radians(yaw)
    return 1.5 + u * math.cos(t) - v * math.sin(t), 1.5 + u * math.sin(t) + v * math.cos(t)


# ------------------------------------------------------------------ geometry

def test_scene_geometry_features(isb_csv):
    out, _, src = isb_csv
    _, obst, free = V.scene_geometry(out / "bev" / "S000" / "S000.csv", src)

    def is_free(x, y):
        return bool(shapely.contains_xy(free, x, y))
    assert is_free(0.5, 2.5) and is_free(5.5, 0.5)
    assert not is_free(3.0, 0.5) and not is_free(3.0, 2.5)          # door jambs
    assert is_free(3.0, 1.5)                                         # door opening
    assert not is_free(0.05, 1.5) and not is_free(3.0, 2.95)        # walls, 0.1 m each side
    assert is_free(0.15, 1.5)
    assert not is_free(4.5, 0.8) and is_free(4.5, 1.05)              # bbox 0.6 x 0.4
    assert not is_free(4.8, 2.2 + 0.199) and not is_free(4.8 + 0.2, 2.2)   # disc r = 0.2
    assert not is_free(5.3, 1.5)                                     # missing image -> bbox
    xs, ys = placed_l_pixels(src)
    assert not shapely.contains_xy(free, xs, ys).any()               # conservative


def test_mesh_rotation_matches_the_scene_builder_stamp(isb_csv):
    out, _, src = isb_csv
    fp = V.asset_footprint(out / "mesh_shapes" / "Lshape.png", CELL, src["simplify_m"])
    xs, ys = placed_l_pixels(src)
    for yaw, want in ((L_YAW, True), (-L_YAW, False)):
        placed = V._place(fp, 1.5, 1.5, yaw)
        inside = shapely.contains_xy(placed, xs, ys)
        assert inside.all() == want                                   # -yaw would be wrong
    # and the BEV the scene builder drew agrees around the L (checked at pixel centres)
    st = V.compare_to_bev(V.scene_geometry(out / "bev" / "S000" / "S000.csv", src)[2],
                          out, "S000")
    assert st["unsafe_tol_m2"] == 0.0 and st["iou"] > 0.95


def test_simplified_footprint_is_small_and_conservative(isb_csv):
    out, _, src = isb_csv
    raw = V.asset_footprint(out / "mesh_shapes" / "Lshape.png", CELL, 0.0)
    simp = V.asset_footprint(out / "mesh_shapes" / "Lshape.png", CELL, 0.01)
    assert simp.contains(raw.buffer(-1e-6))
    assert len(simp.exterior.coords) <= 12


# ------------------------------------------------------------------ pipeline

def config(out, **src):
    return {"name": "isb-vec", "env": "svg",
            "source": {"type": "isaac_scene_builder", "path": str(out), **src},
            "split": {"train": 1.0, "val": 0.0, "test": 0.0, "seed": 0},
            "robots": ["turtlebot4", "jetauto_pro"], "tasks": {"k": 6, "min_geo": 1.0}}


def test_build_vector_dataset(isb_csv):
    out, ws, _ = isb_csv
    root = build(config(out), ws=ws, preview=False)
    ds = load_dataset(root)
    assert ds.validate() == []
    stamps = sorted(p.stem for p in (root / ".stages").glob("*.json"))
    assert "polygons_h36" in stamps and "polygons_h63" in stamps
    assert not any(s.startswith(("import_", "vectorize_")) for s in stamps)
    a = SvgEnv(dataset=root, robot="turtlebot4", map_id="isb-S000_s0")
    b = SvgEnv(dataset=root, robot="jetauto_pro", map_id="isb-S000_s0")
    a.reset(seed=0)
    b.reset(seed=0)
    assert a.map.meta["vectorize"]["geometry"] == "csv" and a.map.meta["flags"] == []
    assert a.map.meta["height_class"] != b.map.meta["height_class"]
    assert all(np.array_equal(p, q) for p, q in zip(a.map.rings, b.map.rings))  # same geometry
    # (-0.25, 0) of the L's bar rotated by 30 deg; the door opening is free
    assert not a.collider.inside(1.2835, 1.375) and a.collider.inside(3.0, 1.5)
    assert evaluate(root, robot="turtlebot4", agent="oracle")["success"] == 1.0


def test_polygons_stage_reruns_when_the_csv_changes(isb_csv):
    out, ws, _ = isb_csv
    root = build(config(out), ws=ws, preview=False)
    stamp = root / ".stages" / "polygons_h36.json"
    first = json.loads(stamp.read_text(encoding="utf-8"))
    build(config(out), ws=ws, preview=False)
    assert json.loads(stamp.read_text(encoding="utf-8")) == first
    p = out / "bev" / "S001" / "S001.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("Box,4.5,0.8", "Box,4.5,2.2"),
                 encoding="utf-8")
    build(config(out), ws=ws, preview=False)
    assert json.loads(stamp.read_text(encoding="utf-8"))["inputs_hash"] != first["inputs_hash"]
    env = SvgEnv(dataset=root, robot="turtlebot4", map_id="isb-S001_s0")
    env.reset(seed=0)
    assert env.collider.inside(4.5, 0.8) and not env.collider.inside(4.5, 2.2)


def test_raster_geometry_still_available(isb_csv):
    out, ws, _ = isb_csv
    root = build(config(out, geometry="raster"), ws=ws, preview=False)
    env = SvgEnv(dataset=root, robot="turtlebot4", map_id="isb-S000_s0")
    env.reset(seed=0)
    assert "geometry" not in env.map.meta["vectorize"]


@pytest.mark.parametrize("bad, match", [
    ({"geometry": "mesh"}, "geometry"),
    ({"door_gap": [0.7, 0.3]}, "door_gap"),
    ({"wall_m": -1}, "wall_m"),
])
def test_config_errors(isb_csv, bad, match):
    out, _, _ = isb_csv
    with pytest.raises(ConfigError, match=match):
        load_config(config(out, **bad))
