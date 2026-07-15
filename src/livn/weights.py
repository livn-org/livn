from __future__ import annotations

import numpy as np

__all__ = ["normalize_weights"]


def normalize_weights(
    weight: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    group_id: np.ndarray,
    target: float | None = None,
    max_iter: int = 20,
) -> np.ndarray:
    """Return rescaled weights so each group sums to ``target``.

    Parameters
    ----------
    weight, w_min, w_max :
        Parallel float arrays of length ``N`` (one entry per plastic synapse).
    group_id :
        Integer array of length ``N`` assigning each synapse to a postsynaptic
        group. Groups are normalized independently.
    target :
        Desired per-group weight sum. When ``None``, each group targets the
        number of synapses it contains (i.e. the sum if every weight were 1.0).
    max_iter :
        Maximum redistribution iterations per group (converges quickly).

    Returns
    -------
    np.ndarray
        A new weight array (input arrays are not modified).
    """
    weight = np.array(weight, dtype=np.float64, copy=True)
    w_min = np.asarray(w_min, dtype=np.float64)
    w_max = np.asarray(w_max, dtype=np.float64)
    group_id = np.asarray(group_id)

    if weight.size == 0:
        return weight

    order = np.argsort(group_id, kind="stable")
    sorted_groups = group_id[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    slices = np.split(order, boundaries)

    for idx in slices:
        _normalize_group(weight, w_min, w_max, idx, target, max_iter)

    return weight


def _normalize_group(
    weight: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    idx: np.ndarray,
    target: float | None,
    max_iter: int,
) -> None:
    t = float(len(idx)) if target is None else float(target)

    free = idx
    clamped_sum = 0.0
    for _ in range(max_iter):
        free_sum = float(weight[free].sum())
        remaining = t - clamped_sum
        if free_sum <= 0.0 or abs(free_sum - remaining) < 1e-12:
            break
        scale = remaining / free_sum

        new_w = weight[free] * scale
        hi = new_w >= w_max[free]
        lo = new_w <= w_min[free]
        keep = ~(hi | lo)

        weight[free[hi]] = w_max[free[hi]]
        weight[free[lo]] = w_min[free[lo]]
        weight[free[keep]] = new_w[keep]

        clamped_sum += float(w_max[free[hi]].sum()) + float(w_min[free[lo]].sum())

        next_free = free[keep]
        if next_free.size == free.size:
            break
        free = next_free
