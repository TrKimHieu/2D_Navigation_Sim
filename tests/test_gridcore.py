import numpy as np
import pytest

from hm3denv.core.gridcore import MOVES, Chain, bfs, cell_to_world, level_of

# 6x7 room with a partition so that the chain is not trivially symmetric
FREE = np.ones((6, 7), bool)
FREE[2, 1:6] = False


def mc_hit_rate(free, start, goal, budget, repeat_p, n=40_000, seed=0):
    """Monte-Carlo simulation of the probe, to compare with the exact solution."""
    rng = np.random.default_rng(seed)
    R, C = free.shape
    pos = np.tile(np.array(start), (n, 1))
    head = rng.integers(0, 4, n)
    done = np.zeros(n, bool)
    mv = np.array(MOVES)
    for _ in range(budget):
        rnd = rng.integers(0, 4, n)
        if repeat_p is None:
            a = rnd
        else:
            a = np.where(rng.random(n) < repeat_p, head, rnd)
        nxt = pos + mv[a]
        ok = ((nxt[:, 0] >= 0) & (nxt[:, 0] < R) & (nxt[:, 1] >= 0)
              & (nxt[:, 1] < C))
        ok[ok] = free[nxt[ok, 0], nxt[ok, 1]]
        pos = np.where((ok & ~done)[:, None], nxt, pos)
        head = a
        done |= (pos[:, 0] == goal[0]) & (pos[:, 1] == goal[1])
    return done.mean()


def test_bfs_goes_around_wall():
    d = bfs(FREE, (0, 3))
    assert d[3, 3] == 9            # 1 down + 3 across + 2 through the gap in column 0 + 3 back
    assert d[2, 3] == -1           # a wall cell


def test_bfs_blocked_start():
    assert (bfs(FREE, (2, 3)) == -1).all()


@pytest.mark.parametrize("budget", [6, 20])
def test_sa_uniform_matches_monte_carlo(budget):
    ch = Chain(FREE)
    s, g = (0, 3), (4, 3)
    exact = ch.sa_uniform(g, budget)[ch.idx(s)]
    mc = mc_hit_rate(FREE, s, g, budget, None)
    assert abs(exact - mc) < 0.01


@pytest.mark.parametrize("budget", [6, 20])
def test_sa_persist_matches_monte_carlo(budget):
    ch = Chain(FREE)
    s, g = (0, 3), (4, 3)
    exact = ch.sa_persist(g, budget, 0.6)[ch.idx(s)]
    mc = mc_hit_rate(FREE, s, g, budget, 0.6)
    assert abs(exact - mc) < 0.01


def test_sa_zero_when_goal_farther_than_budget():
    ch = Chain(FREE)
    assert ch.sa_uniform((4, 3), 4)[ch.idx((0, 3))] == 0.0


def test_levels():
    assert level_of(5e-5) == "starved"
    assert level_of(5e-4) == "hard"
    assert level_of(0.05) == "medium"
    assert level_of(0.5) == "easy"


def test_cell_to_world_center_of_first_pixel_block():
    # image H=100 px, res 2 cm, z up; cell (0,0) after cropping starts at pixel (oy-k+r0*k)
    meta = dict(k=10, res=0.02, oy=10, ox=10, r0=0, c0=0, lo=[1.0, 2.0, 0.0],
                up=2, ua=0, va=1, H=100, floor_z=0.5)
    x, y, z = cell_to_world(meta, 0, 0)
    assert x == pytest.approx(1.0 + 5 * 0.02)          # centre of columns 0..9
    assert y == pytest.approx(2.0 + (100 - 5) * 0.02)  # row 0 is the top one
    assert z == 0.5
