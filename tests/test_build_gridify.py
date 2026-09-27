from conftest import require_build_extra

require_build_extra()

import numpy as np

from hm3denv.build.gridify import (FREE, NOFLOOR, WALL, cell_states, clean_mask, gridify,
                     largest_component, prepare)

RES = 0.02
K = 17          # jetauto_pro: 0.324 m -> 17 px


def house(door=(108, 131)):
    """4x6 m house, two rooms split by a wall at column 200, door at rows [door).

    The door is 23 px = 0.46 m; after the 2 px margin on each side 19 px remain, which
    fit a 17 px cell only when the grid origin falls on phase oy = 8 (see below).
    """
    H, W = 300, 400
    floor = np.zeros((H, W), bool)
    floor[50:250, 50:350] = True
    obst = np.zeros((H, W), bool)
    obst[50:52, 50:350] = obst[248:250, 50:350] = True
    obst[50:250, 50:52] = obst[50:250, 348:350] = True
    obst[50:250, 199:201] = True
    obst[door[0]:door[1], 199:201] = False
    return obst, floor


def test_margin_dilates_by_disk():
    obst = np.zeros((20, 20), bool)
    obst[10, 10] = True
    o, _ = prepare(obst, np.ones((20, 20), bool), RES, 0.04, 0.0, 0.0)
    assert o.sum() == 13            # disk of radius 2 px


def test_small_floor_hole_filled_big_hole_kept():
    floor = np.ones((300, 300), bool)
    floor[20:55, 20:55] = False      # 35x35 px = 0.49 m2: scan hole
    floor[150:221, 150:221] = False  # 71x71 px = 2.0 m2: stairwell
    _, fl = prepare(np.zeros_like(floor), floor, RES, 0.0, 0.0, 1.0)
    assert fl[20:55, 20:55].all()
    assert not fl[150:221, 150:221].any()


def test_outside_not_a_hole():
    floor = np.zeros((100, 100), bool)
    floor[30:70, 30:70] = True
    _, fl = prepare(np.zeros_like(floor), floor, RES, 0.0, 0.0, 100.0)
    assert not fl[:30].any()         # outside touches the border -> stays no-floor


def test_fixed_phase_closes_door_but_search_opens_it():
    obst, floor = house()
    o, fl = prepare(obst, floor, RES, 0.04, 0.10, 1.0)
    for oy in (0, 4, 13):            # every other phase closes the door
        st = cell_states(o, fl, K, oy, 0, 0.9)
        main = largest_component(st == FREE)
        assert main.sum() < 0.6 * (st == FREE).sum()

    state, main, info = gridify(obst, floor, RES, K, 0.04)
    assert info["oy"] == 8
    assert info["main_frac"] == 1.0  # both rooms connected


def test_outside_is_nofloor_and_main_inside():
    obst, floor = house()
    state, main, info = gridify(obst, floor, RES, K, 0.04)
    assert (state[main] == FREE).all()
    # every main cell lies fully on the floor
    for r, c in np.argwhere(main):
        y0 = info["oy"] - K + (info["r0"] + r) * K
        x0 = info["ox"] - K + (info["c0"] + c) * K
        assert floor[y0:y0 + K, x0:x0 + K].all()
    # crop keeps exactly 1 border cell, wall or nofloor
    assert not main[0].any() and not main[-1].any()
    assert set(np.unique(state[0])) <= {WALL, NOFLOOR}


def test_patio_without_roof_is_noroof_and_excluded():
    """Outdoor terrace at floor level, connected through a door in the right wall: floor
    but no roof -> noroof, not part of the main region."""
    obst, floor = house()
    floor[50:250, 350:390] = True              # 0.8 m terrace on the right
    obst[108:131, 348:350] = False             # door to the terrace, same phase oy = 8
    roof = np.zeros_like(floor)
    roof[50:250, 50:350] = True                # only the house has a roof
    def main_right_edge(main, info):
        return max(info["ox"] - K + (info["c0"] + c + 1) * K for _, c in np.argwhere(main))

    state, main, info = gridify(obst, floor, RES, K, 0.04, roof=roof)
    assert main_right_edge(main, info) <= 350      # main region does not spill onto the terrace

    _, main2, info2 = gridify(obst, floor, RES, K, 0.04)   # no roof test
    assert main_right_edge(main2, info2) > 350             # terrace leaks into the main region

def test_roof_mask_scan_holes_filled():
    roof = np.ones((200, 200), bool)
    roof[50:60, 50:60] = False                 # ceiling lamp / scan hole 0.04 m2
    assert clean_mask(roof, RES, 0.10, 1.0).all()
