from conftest import require_build_extra

require_build_extra()

import numpy as np

from hm3denv.build.gridify import gridify
from hm3denv.core.gridcore import bfs
from hm3denv.build.tasks_grid import map_rng, sample_map
from test_build_gridify import K, RES, house


def grid_and_main():
    state, main, _ = gridify(*house(), RES, K, 0.04)
    return (state != 0).astype(np.uint8), main


def run(check=None, seed=0, k=12):
    grid, main = grid_and_main()
    return grid, main, sample_map(grid, main, k, 4, 180, 200,
                                  map_rng(seed, "m"), check)


def test_tasks_satisfy_constraints():
    grid, main, (tasks, _) = run()
    assert len(tasks) == 12
    for t in tasks:
        assert main[tuple(t["start"])] and main[tuple(t["goal"])]
        d = bfs(grid == 0, t["start"])[tuple(t["goal"])]
        assert d == t["d_bfs"] and 4 <= d <= 180
        assert 0.0 <= t["sa_uniform"] <= 1.0 and t["p0"] == t["sa_uniform"]
        assert t["level"] in {"easy", "medium", "hard", "starved"}


def test_deterministic_per_seed():
    a = run(seed=1)[2][0]
    b = run(seed=1)[2][0]
    c = run(seed=2)[2][0]
    assert a == b and a != c


def test_distance_bins_spread():
    tasks = run()[2][0]
    d = sorted(t["d_bfs"] for t in tasks)
    assert d[-1] >= 2 * d[0]         # not piled on one distance


def test_rejected_cells_never_used():
    bad = lambda r, c: {"ok": (r + c) % 3 != 0}   # noqa: E731
    _, _, (tasks, checked) = run(check=bad)
    assert tasks
    for t in tasks:
        for cell in (t["start"], t["goal"]):
            assert sum(cell) % 3 != 0
    assert any(not v["ok"] for v in checked.values())
