from conftest import require_build_extra

require_build_extra()

import numpy as np
import pytest
import trimesh

from hm3denv.build.verify import MeshProbe, check_point, tri_box_overlap

CELL, H, MARGIN = 0.34, 0.626, 0.04


def quad(x0, x1, y0, y1, z):
    V = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return np.asarray(m.vertices), np.asarray(m.faces)


def scene():
    """4x4 m room, floor z=0, 2.5 m ceiling over y<3; corner x>3, y>3 is a double-height
    void with a 6 m ceiling; the rest of y>3 is unroofed floor. Dining table top at
    0.75 m on 4 cm legs at the corners; chair seat at 0.45 m."""
    parts = [quad(0, 4, 0, 4, 0.0), quad(0, 4, 0, 3, 2.5), quad(3, 4, 3, 4, 6.0),
             box(1, 2, 1, 2, 0.72, 0.75),
             box(3, 3.5, 1, 1.5, 0.40, 0.45)]
    for x, y in ((1, 1), (1.96, 1), (1, 1.96), (1.96, 1.96)):
        parts.append(box(x, x + 0.04, y, y + 0.04, 0.0, 0.72))
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return MeshProbe(np.concatenate(Vs), np.concatenate(Fs), up=2)


@pytest.fixture(scope="module")
def probe():
    return scene()


def check(probe, u, v):
    return check_point(probe, u, v, 0.0, CELL, H, MARGIN)


def test_open_floor_passes(probe):
    r = check(probe, 0.5, 2.5)
    assert r == {"support": 1.0, "clear_hits": 0, "margin_hits": 0,
                 "roof": True, "ok": True}


def test_under_dining_table_passes(probe):
    r = check(probe, 1.5, 1.5)          # table top 0.72 m > robot 0.626 m
    assert r["ok"]


def test_under_chair_fails(probe):
    r = check(probe, 3.25, 1.25)        # chair seat 0.45 m < robot
    assert r["clear_hits"] > 0 and not r["ok"]


def test_patio_without_roof_fails(probe):
    r = check(probe, 2.0, 3.5)
    assert r["support"] == 1.0 and r["clear_hits"] == 0
    assert not r["roof"] and not r["ok"]


def test_double_height_ceiling_passes(probe):
    assert check(probe, 3.5, 3.5)["ok"]


def test_outside_floor_fails(probe):
    r = check(probe, -1.0, 1.0)
    assert r["support"] == 0.0 and not r["ok"]


def test_half_on_floor_fails_support(probe):
    r = check(probe, 0.0, 1.0)          # cell centre on the floor edge
    assert 0.3 < r["support"] < 0.7 and not r["ok"]


def test_table_leg_in_margin_only(probe):
    # table leg at x=[1,1.04]; body [0.79,1.13] hits it, 0.2 further only the
    # margin band does
    assert check(probe, 0.96, 1.5)["clear_hits"] == 0
    r = check(probe, 0.80, 1.02)
    assert r["clear_hits"] == 0 and r["margin_hits"] > 0 and r["ok"]


def test_sat_big_triangle_crossing_box_without_vertices_inside():
    t = np.array([[[-5, -5, 0.0], [5, -5, 0.0], [0, 5, 0.0]]])   # plane z=0
    assert tri_box_overlap(t, np.array([1.0, 1.0, 0.1]))[0]
    assert not tri_box_overlap(t - [0, 0, 0.2], np.array([1.0, 1.0, 0.1]))[0]
    # tilted triangle passing the box corner without touching
    t2 = np.array([[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]])
    assert not tri_box_overlap(t2, np.array([0.9, 0.9, 0.9]))[0]
    assert tri_box_overlap(t2, np.array([1.1, 1.1, 1.1]))[0]


# ------------------------------------------------------------ rotated prism (SVG branch)

from hm3denv.build.verify import check_footprint, footprint_points, prism_overlap  # noqa: E402


def test_prism_matches_box_sat_when_axis_aligned():
    rng = np.random.default_rng(0)
    t = rng.uniform(-1.5, 1.5, (2000, 3, 3))
    h = np.array([0.6, 0.4, 0.5])
    sq = np.array([[0.6, -0.4], [0.6, 0.4], [-0.6, 0.4], [-0.6, -0.4]])
    # tri_box_overlap skips the 3 box axes (bbox pre-filtered) -> filter the same way
    bb = ((t.max(1) >= -h) & (t.min(1) <= h)).all(1)
    assert (prism_overlap(t, sq, -0.5, 0.5) == (tri_box_overlap(t, h) & bb)).all()


def test_rotated_rectangle_prism():
    # 1 x 0.2 rectangle rotated 45 deg; small triangle in the bbox corner (outside) does not touch
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s], [s, c]])
    rect = np.array([[0.5, -0.1], [0.5, 0.1], [-0.5, 0.1], [-0.5, -0.1]]) @ R.T
    corner = np.array([[[0.3, -0.3, 0.1], [0.34, -0.3, 0.1], [0.3, -0.34, 0.1]]])
    on_axis = np.array([[[0.3, 0.3, 0.1], [0.32, 0.3, 0.1], [0.3, 0.32, 0.1]]])
    assert not prism_overlap(corner, rect, 0, 1)[0]
    assert prism_overlap(on_axis, rect, 0, 1)[0]


def test_check_footprint_rotated_robot(probe):
    from hm3denv.core.geometry import place
    rect = np.array([[0.2, -0.1], [0.2, 0.1], [-0.2, 0.1], [-0.2, -0.1]])
    ok = check_footprint(probe, place(rect, 1.5, 1.5, 0.7), 0.0, H)     # under the table
    assert ok["ok"] and ok["support"] == 1.0
    bad = check_footprint(probe, place(rect, 3.25, 1.25, 0.3), 0.0, H)  # under the chair
    assert bad["clear_hits"] > 0 and not bad["ok"]
    assert len(footprint_points(rect)) == 25


def test_concave_prism_notch_not_hit():
    # mesh triangle inside the U notch of the footprint: no hit; its convex hull would hit
    U = np.array([[0.2, -0.15], [0.2, -0.05], [0.0, -0.05], [0.0, 0.05], [0.2, 0.05],
                  [0.2, 0.15], [-0.2, 0.15], [-0.2, -0.15]], float)
    tri = np.array([[0.1, -0.02, 0.1], [0.15, -0.02, 0.1], [0.12, 0.02, 0.3]])
    pr = MeshProbe(tri.astype(float), np.array([[0, 1, 2]]), up=2)
    assert pr.prism_hits(U, 0.05, 0.5) == 0
    hull = np.array([[0.2, -0.15], [0.2, 0.15], [-0.2, 0.15], [-0.2, -0.15]])
    assert pr.prism_hits(hull, 0.05, 0.5) == 1
