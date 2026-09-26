from __future__ import annotations

import numpy as np

__all__ = ["connection_scales", "normalize_weights"]


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


def connection_scales(model, conn, syn, pop_code: dict) -> np.ndarray:
    """Per-connection weight multipliers.

    Ones unless ``model`` exposes ``weight_scales(projection, pre_gid, post_gid,
    syn_id)``. A connection is keyed by (pre gid, post gid, synapse id), so the
    receptors of one contact (AMPA and NMDA) share a multiplier.
    """
    n = 0 if conn is None else int(conn.size)
    scales = np.ones(n, dtype=np.float64)
    hook = getattr(model, "weight_scales", None)
    if n == 0 or not callable(hook):
        return scales
    names = {int(code): name for name, code in pop_code.items()}
    rows = np.asarray(conn.syn_row, dtype=np.int64)
    post_gid = np.asarray(syn.post_gid, dtype=np.int64)[rows]
    syn_id = np.asarray(syn.syn_id, dtype=np.int64)[rows]
    pre_gid = np.asarray(conn.pre_gid, dtype=np.int64)
    pairs = np.stack([conn.pre_pop, conn.post_pop], axis=1)
    for pre_code, post_code in np.unique(pairs, axis=0):
        mask = (conn.pre_pop == pre_code) & (conn.post_pop == post_code)
        projection = f"{names[int(pre_code)]}->{names[int(post_code)]}"
        scales[mask] = hook(projection, pre_gid[mask], post_gid[mask], syn_id[mask])
    return scales
