from __future__ import annotations

import inspect

import numpy as np


def connection_table(env) -> dict[str, np.ndarray]:
    """`Env.connections()` for a backend with `conn` / `syn` tables."""
    conn = env.conn
    if conn is None or not conn.size:
        ints, names = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=object)
        return {
            "pre_gid": ints,
            "post_gid": ints,
            "syn_id": ints,
            "pre_population": names,
            "post_population": names,
            "receptor": names,
            "strength": np.zeros(0),
        }

    def named(codes, table: dict) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int64)
        lut = np.full(max([*table.values(), int(codes.max(initial=0))]) + 2, "")
        lut = lut.astype(object)
        for name, code in table.items():
            lut[int(code)] = name
        return lut[codes]

    rows = np.asarray(conn.syn_row)
    strength = getattr(env, "_wscale", None)
    if strength is None or len(strength) != conn.size:
        strength = np.ones(conn.size)
    return {
        "pre_gid": np.asarray(conn.pre_gid),
        "post_gid": np.asarray(env.syn.post_gid)[rows],
        "syn_id": np.asarray(env.syn.syn_id)[rows],
        "pre_population": named(conn.pre_pop, env._pop_code),
        "post_population": named(conn.post_pop, env._pop_code),
        "receptor": named(conn.receptor, env._receptor_code),
        "strength": np.asarray(strength),
    }


def noise_scale_state(scales, keys) -> tuple[dict[int, float], tuple[str, ...]]:
    """What `Env.set_noise_scale` keeps: `({gid: factor}, keys)`."""
    return {int(g): float(f) for g, f in dict(scales).items()}, tuple(keys)


def scaled_noise(env, gid, params: dict) -> dict:
    """`params` for cell `gid`, with its `set_noise_scale` factor on the scaled keys."""
    scales, keys = getattr(env, "_noise_scale", ({}, ()))
    scale = float(scales.get(int(gid), 1.0))
    if scale == 1.0 or not keys:
        return params
    signature = inspect.signature(env.model.neuron_noise_configure).parameters
    out = dict(params)
    for key in keys:
        if key not in out and key in signature:
            out[key] = signature[key].default
        if key in out:
            out[key] = float(out[key]) * scale
    return out


def checked(values, size: int, what: str) -> np.ndarray:
    """`values` as one float64 per connection, or why not."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(
            f"{array.shape[0] if array.ndim else 0} {what} for {size} connections; "
            "rows as `connections()`"
        )
    return array
