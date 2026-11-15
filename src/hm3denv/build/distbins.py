"""Absolute distance bins for the task stages (svg and grid).

Tasks are spread evenly over fixed start-goal distance ranges in metres (default 1-3, 3-6,
6-10, 10-15, 15-30 m), so the difficulty by distance does not depend on the size of the
room around each start. Every map gets the same quota per bin; a bin the map cannot fill
(a small flat has no 15 m path) is filled from the bins it can.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DIST_BINS = (1.0, 3.0, 6.0, 10.0, 15.0, 30.0)


def normalize(bins) -> tuple[float, ...]:
    """Validated bin edges (metres): at least two, strictly increasing, positive."""
    edges = tuple(float(b) for b in (bins or DEFAULT_DIST_BINS))
    if len(edges) < 2 or edges[0] < 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"dist_bins must be increasing metres, at least two values: {edges}")
    return edges


def quotas(k: int, n_bins: int) -> list[int]:
    """k tasks split over n_bins as evenly as possible (the first bins take the remainder)."""
    return [k // n_bins + (i < k % n_bins) for i in range(n_bins)]


def label(edges, b: int) -> str:
    lo, hi = edges[b], edges[b + 1]
    return f"{lo:g}-{hi:g}"


def bin_index(dist, edges) -> np.ndarray:
    """Bin of every distance in metres; -1 outside [edges[0], edges[-1]]."""
    d = np.asarray(dist, float)
    b = np.searchsorted(edges, d, side="right") - 1
    b = np.where(d == edges[-1], len(edges) - 2, b)
    return np.where((d < edges[0]) | (d > edges[-1]), -1, b)


def pick_bin(avail, counts, quota, rng) -> int | None:
    """The open bin (reachable from this start, below its quota) with the largest deficit;
    ties at random. None when no bin is open."""
    open_ = [b for b in range(len(quota)) if avail[b] and counts[b] < quota[b]]
    if not open_:
        return None
    deficit = [quota[b] - counts[b] for b in open_]
    best = [b for b, d in zip(open_, deficit) if d == max(deficit)]
    return int(best[rng.integers(len(best))])
