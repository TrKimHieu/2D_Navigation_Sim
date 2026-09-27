import math

import numpy as np
import pytest

from hm3denv.core.geometry import Collider, place, segments_hit_polygon
from hm3denv.core.kinematics import integrate
from hm3denv.core.planning import MAX_ANISO_ERR, Planner
from hm3denv.robots import footprint_polygon, list_robots, validate
from hm3denv.robots import load as load_robot
from hm3denv.core.svgmap import SvgMap


def room(w=4.0, h=3.0, holes=()):
    """Rectangular room [0,w]x[0,h] (CCW) with square holes (cx, cy, half) as obstacles."""
    rings = [np.array([[0, 0], [w, 0], [w, h], [0, h]], float)]
    for cx, cy, hs in holes:
        rings.append(np.array([[cx - hs, cy - hs], [cx - hs, cy + hs],
                               [cx + hs, cy + hs], [cx + hs, cy - hs]], float))
    return SvgMap(rings, {"schema_version": "1.0", "map_id": "room"})


# ------------------------------------------------------------ robots

def test_all_presets_valid_and_sourced():
    ids = list_robots()
    assert len(ids) >= 9
    for rid in ids:
        r = load_robot(rid)
        assert r.v_max > 0 and r.w_max > 0 and r.footprint.shape[1] == 2


def test_missing_source_rejected():
    raw = load_robot("turtlebot3_burger").raw
    bad = {**raw, "sources": {k: v for k, v in raw["sources"].items() if k != "limits.w_max"}}
    with pytest.raises(ValueError, match="w_max"):
        validate(bad)


def test_circle_footprint_circumscribes():
    P = footprint_polygon({"type": "circle", "radius": 0.2})
    # the circle lies inside the polygon: centre-to-edge distance >= r
    e = np.roll(P, -1, 0) - P
    n = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1)[:, None]
    assert np.allclose((n * P).sum(1), 0.2)           # edges are tangent to the circle


# ------------------------------------------------------------ svg

def test_svg_roundtrip(tmp_path):
    m = room(holes=[(2, 1.5, 0.3)])
    p = tmp_path / "m.svg"
    m.save(p)
    m2 = SvgMap.load(p)
    assert m2.meta["map_id"] == "room" and len(m2.rings) == 2
    assert np.allclose(m2.rings[1], m.rings[1], atol=1e-4)


def test_raster_even_odd():
    mask, x0, y1, res = room(holes=[(2, 1.5, 0.3)]).raster(0.02)
    def at(x, y):
        return mask[int((y1 - y) / res), int((x - x0) / res)]
    assert at(1.0, 1.0) and not at(2.0, 1.5) and not at(-0.1, 1.0)


# ------------------------------------------------------------ geometry

def test_pose_valid_walls_and_hole():
    col = Collider(room(holes=[(2, 1.5, 0.3)]))
    sq = footprint_polygon({"type": "rectangle", "length": 0.4, "width": 0.2})
    assert col.pose_valid(sq, 1.0, 1.0, 0.0)
    assert not col.pose_valid(sq, 0.15, 1.0, 0.0)           # touches the wall x = 0
    assert col.pose_valid(sq, 0.15, 1.0, math.pi / 2)       # fits when turned lengthwise
    assert not col.pose_valid(sq, 2.0, 1.5, 0.0)            # centre inside the obstacle
    assert not col.pose_valid(sq, 2.45, 1.5, 0.0)           # crosses the hole
    assert not col.pose_valid(sq, 10.0, 10.0, 0.0)          # outside the map


def test_footprint_containing_whole_small_obstacle_detected():
    tiny = np.array([[1.0, 1.0, 1.02, 1.0]])                 # a 2 cm edge entirely inside
    big = place(footprint_polygon({"type": "rectangle", "length": 1, "width": 1}), 1, 1, 0.3)
    assert segments_hit_polygon(tiny, big)


def test_lidar_analytic_distances():
    col = Collider(room())
    ang = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2, math.atan2(2, 3)])
    d = col.raycast(1.0, 1.0, ang, 20.0)
    assert np.allclose(d[:4], [3.0, 2.0, 1.0, 1.0])
    assert d[4] == pytest.approx(math.hypot(3, 2))            # into the corner (4, 3)
    assert col.raycast(1.0, 1.0, np.array([0.0]), 2.5)[0] == 2.5   # out of range


# ------------------------------------------------------------ kinematics

def test_arc_integration_matches_closed_form_and_fine_euler():
    x, y, th = integrate(0, 0, 0, 1.0, 0, math.pi / 2, 1.0)   # a quarter circle of radius 2/pi
    R = 2 / math.pi
    assert (x, y, th) == pytest.approx((R, R, math.pi / 2))
    p = (0.0, 0.0, 0.3)
    for _ in range(10000):
        p = integrate(*p, 0.5, 0.2, 0.7, 1e-4)
    assert integrate(0, 0, 0.3, 0.5, 0.2, 0.7, 1.0) == pytest.approx(p, abs=1e-6)


def test_omni_strafe():
    x, y, th = integrate(0, 0, math.pi / 2, 0, 0.5, 0, 2.0)   # strafes left while facing +y
    assert (x, y, th) == pytest.approx((-1.0, 0.0, math.pi / 2))


# ------------------------------------------------------------ planning

def test_geodesic_goes_around_wall_and_is_near_euclid_in_open():
    # U-shaped corridor: a middle wall from y=0 to y=2 at x=2
    rings = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
             np.array([[1.9, 0], [1.9, 2.0], [2.1, 2.0], [2.1, 0]], float)]
    pl = Planner(SvgMap(rings, {}), r_circ=0.1, clearance=0.0)
    f = pl.field(3.0, 0.5)
    d = pl.lookup(f, 1.0, 0.5)
    assert d > 2 * (2.0 + 0.1) - 0.3                          # must go around the end of the wall
    g = Planner(room(), r_circ=0.1, clearance=0.0)
    e = g.lookup(g.field(0.5, 0.5), 3.5, 2.5)
    assert math.hypot(3, 2) <= e <= math.hypot(3, 2) * (1 + MAX_ANISO_ERR) + 0.06


def test_cdisk_respects_radius():
    pl = Planner(room(), r_circ=0.5, clearance=0.0)
    assert pl.in_cdisk(2.0, 1.5) and not pl.in_cdisk(0.3, 1.5)
    col = Collider(room())
    circ = footprint_polygon({"type": "circle", "radius": 0.45})
    for r, c in pl.cells[::50]:
        assert col.pose_valid(circ, *pl.center(r, c), 0.0)


def test_path_follows_graph_and_ends_at_source():
    rings = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
             np.array([[1.9, 0], [1.9, 2.0], [2.1, 2.0], [2.1, 0]], float)]
    pl = Planner(SvgMap(rings, {}), r_circ=0.1, clearance=0.0)
    f, pred = pl.field(3.0, 0.5, predecessors=True)
    pts = pl.path(pred, 1.0, 0.5)
    assert pl.in_cdisk(*pts[0]) and math.dist(pts[-1], (3.0, 0.5)) < 0.1
    assert max(y for _, y in pts) > 2.0                        # goes around the end of the wall


@pytest.mark.parametrize("full", [True, False])
def test_uniform_raycast_equals_brute_force(full):
    rng = np.random.default_rng(1)
    holes = [(rng.uniform(0.5, 3.5), rng.uniform(0.5, 2.5), rng.uniform(0.02, 0.2))
             for _ in range(25)]
    col = Collider(room(holes=holes))
    n = 72 if full else 61
    da = 2 * math.pi / n if full else math.radians(270) / (n - 1)
    for _ in range(200):
        ox, oy, th = rng.uniform(0.05, 3.95), rng.uniform(0.05, 2.95), rng.uniform(-4, 4)
        a0 = th if full else th - math.radians(135)
        fast = col.raycast_uniform(ox, oy, a0, da, n, 8.0, full)
        slow = col.raycast(ox, oy, a0 + da * np.arange(n), 8.0)
        assert np.allclose(fast, slow, atol=1e-9)


# ------------------------------------------------------------ footprint lom (URDF)

from hm3denv.core.geometry import points_in_polygon, triangulate  # noqa: E402

# U shape: 0.4 x 0.3 body with a 0.1 wide, 0.2 deep notch in the front face (+x)
U_FP = np.array([[0.2, -0.15], [0.2, -0.05], [0.0, -0.05], [0.0, 0.05], [0.2, 0.05],
                 [0.2, 0.15], [-0.2, 0.15], [-0.2, -0.15]], float)


def test_points_in_concave_polygon():
    pts = np.array([[0.1, 0.0], [-0.1, 0.0], [0.1, 0.1], [0.3, 0.0]])
    assert points_in_polygon(pts, U_FP).tolist() == [False, True, True, False]


def tri_area(t):
    (ax, ay), (bx, by) = t[1] - t[0], t[2] - t[0]
    return 0.5 * abs(ax * by - ay * bx)


def test_triangulate_covers_area_exactly():
    tris = triangulate(U_FP)
    area = sum(tri_area(t) for t in tris)
    assert area == pytest.approx(0.4 * 0.3 - 0.2 * 0.1)
    for rid in list_robots():                       # every preset, URDF sections included
        fp = load_robot(rid).footprint
        a = 0.5 * abs(np.sum(fp[:, 0] * np.roll(fp[:, 1], -1) - np.roll(fp[:, 0], -1) * fp[:, 1]))
        t = triangulate(fp)
        assert sum(tri_area(x) for x in t) == pytest.approx(a)


def test_obstacle_inside_notch_is_not_a_collision():
    # a 4 cm obstacle inside the U notch: no contact (a convex hull would touch it)
    rings = [np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float),
             np.array([[1.08, 1.48], [1.08, 1.52], [1.12, 1.52], [1.12, 1.48]], float)]
    col = Collider(SvgMap(rings, {}))
    assert col.pose_valid(U_FP, 1.0, 1.5, 0.0)
    hull = np.array([[0.2, -0.15], [0.2, 0.15], [-0.2, 0.15], [-0.2, -0.15]])
    assert not col.pose_valid(hull, 1.0, 1.5, 0.0)
    assert not col.pose_valid(U_FP, 1.0, 1.5, math.pi / 2)    # turned 90 deg: the obstacle enters the body


def test_tiny_edges_no_false_intersection():
    # regression: millimetre footprint edges and centimetre map edges, nearly parallel
    # ~0.18 m apart -- same-sign orientations ~2e-7; the old absolute 1e-12 threshold on
    # their PRODUCT reported phantom crossings that froze robots with URDF footprints
    tri = np.array([[0.0, 0.0], [0.001, -0.001], [0.002, 0.002]])      # CCW, edges of 1-3 mm
    assert not segments_hit_polygon(np.array([[0.13, 0.1301, 0.15, 0.1501]]), tri)
    # a real crossing must still be reported
    assert segments_hit_polygon(np.array([[0.0005, -0.002, 0.0015, 0.004]]), tri)
