"""Episode selection and metrics shared by the grid and SVG environments.

Episode selection at reset (first match wins):
  0. ``reset(options={"map_id": ..., "start": ..., "goal": ...})``: your own start and
     goal on that map instead of a task of the dataset (checked: the start must be free,
     the goal reachable; ValueError otherwise). Continuous env: start (x, y[, theta]),
     goal (x, y) in metres; grid env: start and goal (row, col) cells. task_id is -1.
  1. ``reset(options={"map_id": ..., "task_idx": ...})``
  2. the fixed ``map_id`` / ``task_idx`` given to the constructor
  3. otherwise a map drawn uniformly from the split, then a task uniformly from the
     map (``difficulty`` restricts tasks to one level, grid only), using the env RNG.

Curriculum: ``set_task_filter(**criteria)`` (or ``task_filter={...}`` at construction)
restricts step 3 to tasks matching every criterion, and to maps having at least one:

    env.set_task_filter(geodesic_m=(1.0, 5.0), gdr=(None, 1.5))   # ranges, inclusive
    env.set_task_filter(level=["easy", "medium"])                  # list = allowed values
    env.set_task_filter(geodesic_m={"min": 1.0, "max": 5.0})      # range as a mapping (YAML/JSON)
    env.set_task_filter()                                          # no filter
    venv.call("set_task_filter", geodesic_m=(1.0, 5.0))           # every env of a vector env

A key is looked up in the task (``geodesic_m``, ``euclidean_m``) then in its ``labels``;
a task without the key does not match. Criteria are plain data so they can be sent to
worker processes. With no filter the RNG is used exactly as before.

Loaded maps are kept in an LRU cache so sampling across a split does not reload files.
With ``map_repeat = K`` a map drawn in step 3 is kept for K consecutive episodes (the
task is still drawn each time), cutting map loads by K. K = 1 draws exactly as before;
K > 1 makes consecutive episodes of one env correlated, so report it with results.
"""

from __future__ import annotations

from collections import OrderedDict

END_KEYS = ("map_id", "task_id", "split", "success", "spl", "path_length", "time",
            "n_collisions", "termination", "geodesic_m")


def spl(success: bool, shortest: float, taken: float) -> float:
    """Success weighted by Path Length (Anderson et al. 2018): S * l / max(p, l)."""
    return float(success) * shortest / max(taken, shortest) if shortest > 0 else float(success)


class EpisodeSource:
    """Mixin: owns the list of maps, the LRU cache and the reset-time choice."""

    def _init_source(self, map_ids, loader, map_id=None, task_idx=None, cache_size=8,
                     map_repeat=1, task_filter=None):
        if not map_ids:
            raise ValueError("no maps with tasks for this robot/split")
        if int(map_repeat) < 1:
            raise ValueError("map_repeat must be >= 1")
        self._map_repeat = int(map_repeat)
        self._sticky_map, self._sticky_left = None, 0
        self._map_ids = list(map_ids)
        self._loader = loader
        self._fixed_map, self._fixed_task = map_id, task_idx
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._criteria, self._eligible = None, None
        if task_filter:
            self.set_task_filter(**task_filter)

    def _tasks_of(self, map_id) -> list:
        """Task list of a map; envs backed by a dataset read it without loading the map."""
        return self._ctx(map_id).tasks

    def set_task_filter(self, **criteria) -> dict:
        """Restrict random episodes to matching tasks (see module docstring). Returns the
        number of maps and tasks left; raises ValueError if nothing matches."""
        self._sticky_map, self._sticky_left = None, 0
        if not criteria:
            self._criteria, self._eligible = None, None
            n = sum(len(self._tasks_of(m)) for m in self._map_ids)
            return {"maps": len(self._map_ids), "tasks": n}
        criteria = {k: _as_range(k, v) if isinstance(v, dict) else v for k, v in criteria.items()}
        for k, v in criteria.items():
            if isinstance(v, tuple) and (len(v) != 2 or not all(
                    b is None or isinstance(b, (int, float)) for b in v)):
                raise ValueError(f"{k}: a range is (lo, hi) with numbers or None, got {v!r}")
        eligible = {}
        for m in self._map_ids:
            idx = [i for i, t in enumerate(self._tasks_of(m)) if task_matches(t, criteria)]
            if idx:
                eligible[m] = idx
        if not eligible:
            raise ValueError(f"no task matches {criteria}")
        self._criteria, self._eligible = dict(criteria), eligible
        return {"maps": len(eligible), "tasks": sum(map(len, eligible.values()))}

    def _ctx(self, map_id):
        if map_id in self._cache:
            self._cache.move_to_end(map_id)
            return self._cache[map_id]
        ctx = self._loader(map_id)
        self._cache[map_id] = ctx
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ctx

    def _choose(self, options, task_filter=None):
        options = options or {}
        map_id = options.get("map_id", self._fixed_map)
        if custom_endpoints(options) and map_id is None:
            if len(self._map_ids) > 1:
                raise ValueError("a custom start/goal needs map_id in the reset options "
                                 f"(maps: {self._map_ids[:5]}{' ...' if len(self._map_ids) > 5 else ''})")
            map_id = self._map_ids[0]
        if map_id is None and self._sticky_left > 0:
            map_id = self._sticky_map
            self._sticky_left -= 1
        elif map_id is None:
            maps = self._map_ids if self._eligible is None else list(self._eligible)
            map_id = maps[int(self.np_random.integers(len(maps)))]
            self._sticky_map, self._sticky_left = map_id, self._map_repeat - 1
        ctx = self._ctx(map_id)
        n = len(ctx.tasks)
        idx = options.get("task_idx", self._fixed_task)
        if idx is None:
            base = list(range(n))
            if self._criteria is not None:
                base = (self._eligible.get(map_id) or
                        [i for i in base if task_matches(ctx.tasks[i], self._criteria)] or base)
            pool = [i for i in base if task_filter is None or task_filter(ctx.tasks[i])]
            pool = pool or base
            idx = pool[int(self.np_random.integers(len(pool)))]
        return map_id, ctx, int(idx) % n


def custom_endpoints(options) -> tuple | None:
    """(start, goal) given in reset options, or None. Both are needed."""
    if not options or ("start" not in options and "goal" not in options):
        return None
    if options.get("start") is None or options.get("goal") is None:
        raise ValueError("a custom episode needs both 'start' and 'goal'")
    return options["start"], options["goal"]


def as_point(v, names, n_min) -> list[float]:
    """[a, b, ...] from a sequence or a mapping with keys `names` (trailing ones optional)."""
    if isinstance(v, dict):
        v = [v[k] for k in names if k in v]
    v = [float(x) for x in v]
    if not n_min <= len(v) <= len(names):
        raise ValueError(f"expected {', '.join(names[:n_min])}"
                         f"{'[, ' + ', '.join(names[n_min:]) + ']' if len(names) > n_min else ''}, got {v}")
    return v


_MISSING = object()


def _as_range(key, v: dict) -> tuple:
    """{"min": a, "max": b} (either optional) -> (a, b)."""
    bad = set(v) - {"min", "max"}
    if bad:
        raise ValueError(f"{key}: a range mapping takes 'min' and 'max', got {sorted(bad)}")
    return (v.get("min"), v.get("max"))


def task_matches(task: dict, criteria: dict) -> bool:
    """Whether a task meets every criterion: (lo, hi) range (None = open), list of
    allowed values, or a single value."""
    labels = task.get("labels") or {}
    for key, want in criteria.items():
        v = task.get(key, labels.get(key, _MISSING))
        if v is _MISSING:
            return False
        if isinstance(want, tuple):
            lo, hi = want
            if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
                return False
        elif isinstance(want, (list, set, frozenset)):
            if v not in want:
                return False
        elif v != want:
            return False
    return True
