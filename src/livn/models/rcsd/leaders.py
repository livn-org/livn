from __future__ import annotations

import numpy as np

DEFAULTS = {
    "fraction": 0.0,
    "bias": 0.0,
    "rest_noise": 1.0,
    "efferent": 1.0,
    "delay_sd": 0.0,
    "by_input": 0.0,
    "input_spread": 0.0,
    "input_power": 1.0,
}
REST_NOISE_KEYS = ("g_e0", "std_e")


def apply(env, values: dict) -> dict:
    unknown = sorted(set(values) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown leader parameter(s) {unknown}; known: {list(DEFAULTS)}"
        )
    state = {**DEFAULTS, **{k: float(v) for k, v in values.items()}}
    model = env.model

    start, count = env.system.population_ranges["EXC"]
    exc = np.arange(start, start + count)
    table = env.connections()
    if state["by_input"] > 0.5:
        chosen = by_input(env, table, exc, state["fraction"])
    else:
        chosen = exc[np.asarray(model.leader_units(exc)) < state["fraction"]]
    leaders = {int(g) for g in chosen}

    bias = state["bias"] * 1e-3
    env.set_holding_current(dict.fromkeys(leaders, bias) if bias != 0.0 else {})

    rest = state["rest_noise"]
    env.set_noise_scale(
        {int(g): rest for g in exc if int(g) not in leaders} if rest != 1.0 else {},
        keys=REST_NOISE_KEYS,
    )

    # leader efferents and the per-cell input scale
    from_leader = (table["pre_population"] == "EXC") & np.isin(table["pre_gid"], chosen)
    factor = np.where(from_leader, state["efferent"], 1.0)
    if state["input_spread"] != 0.0:
        factor = factor * input_scale(
            env, table, exc, state["input_spread"], state["input_power"]
        )
    env.set_connection_factors(factor)

    # the transmission delay spread
    env.set_delay_offsets(
        model.delay_jitter(
            table["pre_gid"], table["post_gid"], table["syn_id"], state["delay_sd"]
        )
    )
    return {"chosen": sorted(leaders), **state}


def chosen(env) -> set[int]:
    return set((env.group_state.get("leaders") or {}).get("chosen", ()))


def input_scores(env, table: dict, exc: np.ndarray) -> np.ndarray:
    score = np.zeros(len(exc))
    rows = (
        (table["pre_population"] == "EXC")
        & (table["post_population"] == "EXC")
        & (table["receptor"] == "AMPA")
    )
    np.add.at(score, table["post_gid"][rows] - exc[0], table["strength"][rows])
    comm = getattr(env, "comm", None)
    if comm is not None and comm.Get_size() > 1:
        from mpi4py import MPI

        score = comm.allreduce(score, op=MPI.SUM)
    return score


def by_input(env, table: dict, exc: np.ndarray, fraction: float) -> np.ndarray:
    n_lead = round(fraction * len(exc))
    if n_lead <= 0:
        return exc[:0]
    # largest first, ties by gid so every rank and backend agrees
    order = np.lexsort((exc, -input_scores(env, table, exc)))
    return np.sort(exc[order[:n_lead]])


def input_scale(
    env, table: dict, exc: np.ndarray, spread: float, power: float
) -> np.ndarray:
    order = np.lexsort((-exc, input_scores(env, table, exc)))
    rank = np.empty(len(exc))
    rank[order] = np.arange(len(exc)) / max(len(exc) - 1, 1)
    cell = 1.0 - float(spread) * (1.0 - rank) ** float(power)
    rows = (table["pre_population"] == "EXC") & (table["post_population"] == "EXC")
    factor = np.ones(len(table["pre_gid"]))
    factor[rows] = cell[table["post_gid"][rows] - exc[0]]
    return factor
