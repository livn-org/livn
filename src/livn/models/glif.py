from __future__ import annotations

import copy
import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from livn.models.eventloop import (
    BrownianPath,
    SolverConfig,
    event_solve,
    resample,
)
from livn.types import Model

PARAM_NAMES = (
    "tau_m",  # membrane time constant [ms]
    "E_L",  # resting potential [mV]
    "g_L",  # leak conductance [nS]; with I in pA, I/g_L is in mV
    "V_threshold_base",  # Theta_inf, the constant floor of the threshold [mV]
    "theta_decay_rate",  # b_spike [1/ms]
    "theta_jump",  # delta_Theta_s [mV]
    "asc_amp_1",  # after-spike current amplitudes [pA]
    "asc_amp_2",
    "asc_decay_rate_1",  # after-spike current decay rates [1/ms]
    "asc_decay_rate_2",
    "f_v",  # voltage reset scale [1]
    "delta_v",  # voltage reset offset [mV]
    "t_ref",  # refractory / spike-cut duration [ms]
    "a_v",  # threshold voltage-coupling gain [1/ms]
    "b_v",  # threshold voltage-component decay [1/ms]
    "asc_r",  # ASC multiplier across the spike cut (1.0 in every Allen config)
)

MECHANISM_PARAM_NAMES = (
    "sigma",  # escape-noise width [mV]; the hard mechanism is the sigma -> 0 limit
    "tau_s",  # escape time constant [ms]; lambda = exp((V - Theta)/sigma) / tau_s
    "alpha",  # offset of the survival variable drawn at every reset [1]
)

NOISE_PARAM_NAMES = ("sigma_v",)  # [mV/sqrt(ms)]

COUPLING_PARAM_NAMES = ("tau_syn",)  # synaptic current decay [ms]

EXTRA_PARAM_NAMES = MECHANISM_PARAM_NAMES + NOISE_PARAM_NAMES + COUPLING_PARAM_NAMES

ALL_PARAM_NAMES = PARAM_NAMES + EXTRA_PARAM_NAMES

LEVELS = (1, 2, 3, 4, 5)

LEVEL_NAMES = {
    1: "GLIF1 LIF",
    2: "GLIF2 LIF-R",
    3: "GLIF3 LIF-ASC",
    4: "GLIF4 LIF-R-ASC",
    5: "GLIF5 LIF-R-ASC-A",
}

# parameters each level zeroes out of the full GLIF5 parameter set
LEVEL_ZEROED = {
    1: ("theta_jump", "a_v", "b_v", "f_v", "delta_v", "asc_amp_1", "asc_amp_2"),
    2: ("a_v", "b_v", "asc_amp_1", "asc_amp_2"),
    3: ("theta_jump", "a_v", "b_v", "f_v", "delta_v"),
    4: ("a_v", "b_v"),
    5: (),
}

DEFAULT_PARAMS = {
    "tau_m": 5.192,
    "E_L": -78.737,
    "g_L": 3.214,
    "V_threshold_base": -40.0,
    "theta_decay_rate": 0.34060,
    "theta_jump": 3.722,
    "asc_amp_1": 10.0,
    "asc_amp_2": -25.121,
    "asc_decay_rate_1": 0.23614,
    "asc_decay_rate_2": 0.0068869,
    "f_v": 0.7595,
    "delta_v": 0.28477,
    "t_ref": 6.992,
    "a_v": 0.0010,
    "b_v": 0.0100,
    "asc_r": 1.0,
}

DEFAULT_EXTRA_PARAMS = {
    "sigma": 1.0,
    "tau_s": 1.0,
    "alpha": 3e-2,
    "sigma_v": 0.0,
    "tau_syn": 5.0,
}

DEFAULT_PARAMS_FULL = {**DEFAULT_PARAMS, **DEFAULT_EXTRA_PARAMS}

LIF_PARAMS = {
    "tau_m": 10.0,  # ms
    "E_L": -70.0,  # mV
    "g_L": 10.0,  # nS, i.e. Rm = 100 MOhm
    "V_threshold_base": -55.0,  # mV
    "theta_decay_rate": 1.0,  # unused: the threshold does not move
    "theta_jump": 0.0,
    "f_v": 0.0,
    "delta_v": 5.0,  # reset to E_L - delta_v = -75 mV
    "t_ref": 0.0,
    "asc_r": 1.0,
}
LIF_LEVEL = 2

MECHANISMS = ("hard", "escape")

EXP_CAP = 50.0

NOT_REFRACTORY = -1e30

V, THETA_S, THETA_V, ASC_1, ASC_2 = range(5)

SIGNAL_COLUMNS = {
    "voltage": (V,),
    "theta_s": (THETA_S,),
    "theta_v": (THETA_V,),
    "AScurrents": (ASC_1, ASC_2),
    "threshold": (THETA_S, THETA_V),
}

_STATE_ATTRIBUTES = {
    "threshold": "threshold",
    "theta_s": "theta_s",
    "theta_v": "theta_v",
    "AScurrents": "ascs",
}

RECORDABLE_STATES = tuple(_STATE_ATTRIBUTES)

# What ``run`` records when its caller does not say
DEFAULT_RECORD = frozenset({"spikes", "voltage"})

ALL_COLUMNS = (V, THETA_S, THETA_V, ASC_1, ASC_2)


def _columns_for(record) -> tuple[int, ...]:
    """The ODE state columns the requested signals need, ascending"""
    needed: set[int] = set()
    for name in record:
        needed.update(SIGNAL_COLUMNS.get(name, ()))
    return tuple(sorted(needed))


def _check_mechanism(mechanism: str) -> str:
    if mechanism not in MECHANISMS:
        raise ValueError(
            f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}"
        )
    return mechanism


@dataclasses.dataclass(frozen=True)
class StateLayout:
    i_syn: int
    s: int
    t_ref_end: int
    size: int
    next_forced: int = -1
    mechanisms: tuple[int, ...] = ()

    @classmethod
    def of(
        cls,
        coupled: bool,
        escape: bool,
        forced: bool = False,
        mechanisms: tuple[int, ...] = (),
    ) -> StateLayout:
        size = 5
        i_syn = s = t_ref_end = next_forced = -1
        if coupled:
            i_syn = size
            size += 1
        if escape:
            s = size
            size += 1
        if coupled:
            t_ref_end = size
            size += 1
        if forced:
            next_forced = size
            size += 1
        mechanisms = tuple(int(width) for width in mechanisms)
        return cls(
            i_syn=i_syn,
            s=s,
            t_ref_end=t_ref_end,
            size=size + sum(mechanisms),
            next_forced=next_forced,
            mechanisms=mechanisms,
        )

    @property
    def coupled(self) -> bool:
        return self.i_syn >= 0

    @property
    def escape(self) -> bool:
        return self.s >= 0

    @property
    def forced(self) -> bool:
        return self.next_forced >= 0

    @property
    def mechanism_start(self) -> int:
        """The first mechanism column."""
        return self.size - sum(self.mechanisms)

    def mechanism_columns(self) -> list[tuple[int, int]]:
        """``(start, stop)`` of each mechanism's columns."""
        out, start = [], self.mechanism_start
        for width in self.mechanisms:
            out.append((start, start + width))
            start += width
        return out


def _check_level(level: int) -> int:
    level = int(level)
    if level not in LEVEL_ZEROED:
        raise ValueError(f"unknown GLIF level {level}; expected one of {LEVELS}")
    return level


def apply_level(params: Mapping[str, Any], level: int) -> dict:
    level = _check_level(level)
    out = dict(params)
    for name in LEVEL_ZEROED[level]:
        value = out.get(name)
        out[name] = jnp.zeros_like(value) if eqx.is_array(value) else 0.0
    return out


def trainable_param_names(
    level: int,
    mechanism: str = "hard",
    diffusion: bool = False,
    coupled: bool = False,
) -> tuple[str, ...]:
    level = _check_level(level)
    mechanism = _check_mechanism(mechanism)
    excluded = set(LEVEL_ZEROED[level])
    if mechanism != "escape":
        excluded.update(MECHANISM_PARAM_NAMES)
    if not diffusion:
        excluded.update(NOISE_PARAM_NAMES)
    if not coupled:
        excluded.update(COUPLING_PARAM_NAMES)
    return tuple(name for name in ALL_PARAM_NAMES if name not in excluded)


def to_vector(params: Mapping[str, Any]) -> list:
    return [params[name] for name in PARAM_NAMES]


def from_vector(vector: Sequence) -> dict:
    if len(vector) != len(PARAM_NAMES):
        raise ValueError(
            f"expected {len(PARAM_NAMES)} parameters, got {len(vector)}; "
            f"the canonical order is {PARAM_NAMES}"
        )
    return dict(zip(PARAM_NAMES, vector, strict=False))


_KNOWN_METHODS = {
    "threshold_dynamics_method": {"inf", "spike_component", "three_components_exact"},
    "voltage_reset_method": {"zero", "v_before"},
    "AScurrent_dynamics_method": {"none", "exp"},
    "AScurrent_reset_method": {"none", "sum"},
    "threshold_reset_method": {"inf", "three_components"},
    "voltage_dynamics_method": {"linear_forward_euler"},
}


def _method(config: Mapping, group: str) -> str:
    """Method name of a ``neuron_config`` group, e.g. ``'zero'``"""
    name = config[group]["name"]
    if name not in _KNOWN_METHODS[group]:
        raise ValueError(
            f"unsupported {group} '{name}' -- GLIF handles {_KNOWN_METHODS[group]}"
        )
    return name


def level_of_neuron_config(config: Mapping) -> int:
    """GLIF level (1-5) an Allen ``neuron_config`` describes"""
    threshold = _method(config, "threshold_dynamics_method")
    ascs = _method(config, "AScurrent_dynamics_method")
    if threshold == "three_components_exact":
        return 5
    if threshold == "spike_component":
        return 4 if ascs == "exp" else 2
    return 3 if ascs == "exp" else 1


def from_neuron_config(config: Mapping) -> dict:
    """Allen ``neuron_config`` (SI) -> canonical GLIF parameters (mV / pA / ms)"""
    for group in _KNOWN_METHODS:
        _method(config, group)

    coeffs = config["coeffs"]
    C = config["C"] * coeffs["C"]  # F
    G = coeffs["G"] / config["R_input"]  # S
    asc_tau = config["asc_tau_array"]
    if len(asc_tau) != 2:
        raise ValueError(f"expected 2 after-spike currents, got {len(asc_tau)}")

    thr_dyn = _method(config, "threshold_dynamics_method")
    v_reset = _method(config, "voltage_reset_method")
    asc_dyn = _method(config, "AScurrent_dynamics_method")
    asc_reset = _method(config, "AScurrent_reset_method")

    # membrane, Eq. (1)
    tau_m = (C / G) * 1e3  # s -> ms
    g_L = G * 1e9  # S -> nS (pA / nS = mV)
    E_L = config["El"] * 1e3  # V -> mV (El_reference is the display value)
    V_threshold_base = config["th_inf"] * coeffs["th_inf"] * 1e3

    # spike component of the threshold, Eqs. (2, 6); GLIF1/3 have none
    if thr_dyn == "inf":
        theta_jump, theta_decay_rate = 0.0, 1.0  # rate unused when the jump is 0
    else:
        theta_jump = config["threshold_reset_method"]["params"]["a_spike"] * 1e3
        theta_decay_rate = (
            config["threshold_dynamics_method"]["params"]["b_spike"] / 1e3
        )

    # voltage component of the threshold, Eq. (4); GLIF5 only
    if thr_dyn == "three_components_exact":
        tp = config["threshold_dynamics_method"]["params"]
        a_v = tp["a_voltage"] * coeffs["a"] / 1e3  # 1/s -> 1/ms
        b_v = tp["b_voltage"] * coeffs["b"] / 1e3
    else:
        a_v, b_v = 0.0, 0.0

    # after-spike currents, Eqs. (3, 7); GLIF1/2 have none
    if asc_dyn == "none":
        asc_amp_1 = asc_amp_2 = 0.0
    else:
        amp = config["asc_amp_array"]
        ca = coeffs["asc_amp_array"]
        asc_amp_1 = amp[0] * ca[0] * 1e12  # A -> pA
        asc_amp_2 = amp[1] * ca[1] * 1e12
    asc_decay_rate_1 = 1.0 / asc_tau[0] / 1e3  # 1/s -> 1/ms
    asc_decay_rate_2 = 1.0 / asc_tau[1] / 1e3
    if asc_reset == "sum":
        # scalar r; both entries are equal in every published config
        asc_r = float(config["AScurrent_reset_method"]["params"]["r"][0])
    else:
        asc_r = 1.0

    # voltage reset, Eq. (5). Allen's 'v_before' is a*v + b; ours is f_v*v - delta_v
    if v_reset == "zero":
        f_v, delta_v = 0.0, 0.0
    else:
        rp = config["voltage_reset_method"]["params"]
        f_v = rp["a"]
        delta_v = -rp["b"] * 1e3  # sign flip: b is a +offset, delta_v is a -offset

    t_ref = (config["spike_cut_length"] + 1) * config["dt"] * 1e3  # steps * s -> ms

    return {
        "tau_m": tau_m,
        "E_L": E_L,
        "g_L": g_L,
        "V_threshold_base": V_threshold_base,
        "theta_decay_rate": theta_decay_rate,
        "theta_jump": theta_jump,
        "asc_amp_1": asc_amp_1,
        "asc_amp_2": asc_amp_2,
        "asc_decay_rate_1": asc_decay_rate_1,
        "asc_decay_rate_2": asc_decay_rate_2,
        "f_v": f_v,
        "delta_v": delta_v,
        "t_ref": t_ref,
        "a_v": a_v,
        "b_v": b_v,
        "asc_r": asc_r,
    }


def _default_neuron_config(level: int, dt: float = 5e-5) -> dict:
    level = _check_level(level)
    spiking = level in (2, 4, 5)
    ascs = level in (3, 4, 5)
    return {
        "type": "GLIF",
        "dt": dt,
        "El": 0.0,
        "El_reference": 0.0,
        "C": 1e-11,
        "R_input": 1e8,
        "th_inf": 0.02,
        "init_threshold": 0.02,
        "init_voltage": 0.0,
        "init_AScurrents": [0.0, 0.0],
        "spike_cut_length": 0,
        "extrapolation_method_name": "endpoints",
        "dt_multiplier": 1,
        "th_adapt": None,
        "asc_amp_array": [0.0, 0.0],
        "asc_tau_array": [0.01, 0.01],
        "coeffs": {
            "a": 1,
            "b": 1,
            "C": 1,
            "G": 1,
            "th_inf": 1,
            "asc_amp_array": [1.0, 1.0],
        },
        "voltage_dynamics_method": {"name": "linear_forward_euler", "params": {}},
        "voltage_reset_method": (
            {"name": "v_before", "params": {"a": 0.0, "b": 0.0}}
            if spiking
            else {"name": "zero", "params": {}}
        ),
        "threshold_dynamics_method": (
            {
                "name": "three_components_exact",
                "params": {
                    "a_spike": 0.0,
                    "b_spike": 1e3,
                    "a_voltage": 0.0,
                    "b_voltage": 0.0,
                },
            }
            if level == 5
            else (
                {"name": "spike_component", "params": {"a_spike": 0.0, "b_spike": 1e3}}
                if spiking
                else {"name": "inf", "params": {}}
            )
        ),
        "threshold_reset_method": (
            {"name": "three_components", "params": {"a_spike": 0.0, "b_spike": 1e3}}
            if spiking
            else {"name": "inf", "params": {}}
        ),
        "AScurrent_dynamics_method": (
            {"name": "exp", "params": {}} if ascs else {"name": "none", "params": {}}
        ),
        "AScurrent_reset_method": (
            {"name": "sum", "params": {"r": [1.0, 1.0]}}
            if ascs
            else {"name": "none", "params": {}}
        ),
    }


def to_neuron_config(
    params: Mapping[str, Any],
    template: Mapping | None = None,
    level: int | None = None,
) -> dict:
    params = {name: float(params[name]) for name in PARAM_NAMES}
    if template is None:
        if level is None:
            raise ValueError("pass either a template neuron_config or a level")
        config = _default_neuron_config(level)
    else:
        config = copy.deepcopy(dict(template))
        for group in _KNOWN_METHODS:
            _method(config, group)

    coeffs = config["coeffs"]
    G = params["g_L"] * 1e-9  # nS -> S
    config["R_input"] = coeffs["G"] / G
    config["C"] = (params["tau_m"] * 1e-3 * G) / coeffs["C"]
    config["El"] = params["E_L"] / 1e3
    config["th_inf"] = params["V_threshold_base"] / (coeffs["th_inf"] * 1e3)

    if _method(config, "threshold_dynamics_method") != "inf":
        a_spike = params["theta_jump"] / 1e3
        b_spike = params["theta_decay_rate"] * 1e3
        config["threshold_dynamics_method"]["params"]["a_spike"] = a_spike
        config["threshold_dynamics_method"]["params"]["b_spike"] = b_spike
        config["threshold_reset_method"]["params"]["a_spike"] = a_spike
        config["threshold_reset_method"]["params"]["b_spike"] = b_spike

    if _method(config, "threshold_dynamics_method") == "three_components_exact":
        tp = config["threshold_dynamics_method"]["params"]
        tp["a_voltage"] = params["a_v"] * 1e3 / coeffs["a"]
        tp["b_voltage"] = params["b_v"] * 1e3 / coeffs["b"]

    if _method(config, "AScurrent_dynamics_method") != "none":
        ca = coeffs["asc_amp_array"]
        config["asc_amp_array"] = [
            params["asc_amp_1"] / (ca[0] * 1e12),
            params["asc_amp_2"] / (ca[1] * 1e12),
        ]
    config["asc_tau_array"] = [
        1.0 / (params["asc_decay_rate_1"] * 1e3),
        1.0 / (params["asc_decay_rate_2"] * 1e3),
    ]
    if _method(config, "AScurrent_reset_method") == "sum":
        config["AScurrent_reset_method"]["params"]["r"] = [
            params["asc_r"],
            params["asc_r"],
        ]

    if _method(config, "voltage_reset_method") != "zero":
        rp = config["voltage_reset_method"]["params"]
        rp["a"] = params["f_v"]
        rp["b"] = -params["delta_v"] / 1e3

    config["spike_cut_length"] = max(
        0, round(params["t_ref"] / (config["dt"] * 1e3)) - 1
    )

    return config


class GlifSolution(eqx.Module):
    ts: Any  # (times,) in ms
    ys: Any  # (*samples, cells, times, len(columns))
    threshold: Any  # (*samples, cells, times) in mV, or None when not requested
    spike_times: Any  # (*samples, cells, max_spikes) in ms, inf-padded
    yT: Any  # (*samples, cells, state) final state
    saturated: Any  # (*samples, cells) spike budget exhausted before t1
    solver_ok: Any  # (*samples, cells) every inner solve succeeded
    segment_ts: Any = None  # (*samples, cells, nodes) in ms
    segment_ys: Any = None  # (*samples, cells, nodes, len(columns))
    columns: tuple = eqx.field(static=True, default=ALL_COLUMNS)

    def column(self, index: int):
        """The sampled trace of one ODE state column"""
        try:
            position = self.columns.index(index)
        except ValueError:
            raise AttributeError(
                f"state column {index} was not sampled; the solve recorded "
                f"{self.columns}"
            ) from None
        return self.ys[..., position]

    @property
    def v(self):
        return self.column(V)

    @property
    def theta_s(self):
        return self.column(THETA_S)

    @property
    def theta_v(self):
        return self.column(THETA_V)

    @property
    def ascs(self):
        return jnp.stack([self.column(ASC_1), self.column(ASC_2)], axis=-1)

    @property
    def num_spikes(self):
        return jnp.sum(jnp.isfinite(self.spike_times), axis=-1)


def _for_population(value, n: int, offset: int, rows=None):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.shape[0] == n and rows is None:
        return array

    if rows is not None:
        rows = np.asarray(rows, dtype=int)
        if array.shape[0] == n and len(rows) == n:
            return array
        if array.shape[0] <= int(rows.max(initial=-1)):
            raise ValueError(
                f"a per-cell parameter must cover the population: needed "
                f"{int(rows.max(initial=-1)) + 1} values, got {array.shape[0]}"
            )
        return array[rows]

    if array.shape[0] < offset + n:
        raise ValueError(
            f"a per-cell parameter must cover the population: needed "
            f"{offset + n} values, got {array.shape[0]}"
        )
    return array[offset : offset + n]


def _fallback_key(key):
    return jr.PRNGKey(0) if key is None else key


def _threshold_of(p, y):
    return p["V_threshold_base"] + y[..., THETA_S] + y[..., THETA_V]


def _intensity(p, y):
    sigma = jnp.where(p["sigma"] > 0, p["sigma"], 1e-12)
    excess = (y[..., V] - _threshold_of(p, y)) / sigma
    return jnp.exp(jnp.minimum(excess, EXP_CAP)) / p["tau_s"]


def _draw_survival(p, key, shape):
    return jnp.log(jr.uniform(key, shape, minval=1e-10)) - p["alpha"]


def _active(layout: StateLayout, t, y):
    if layout.t_ref_end < 0:
        return 1.0
    return (t >= y[..., layout.t_ref_end]).astype(y.dtype)


class GlifMechanism(eqx.Module):
    """A extension mechanism inserted into the GLIF cell."""

    states: tuple[str, ...] = eqx.field(static=True, default=(), kw_only=True)
    inputs: tuple[str, ...] = eqx.field(static=True, default=(), kw_only=True)

    def initial(self, n_cells: int):
        """``[cells, len(states)]`` at t0"""
        return jnp.zeros((n_cells, len(self.states)))

    def drift(self, t, v, states, inputs):
        """``(current [cells] in pA into V, d states / dt [cells, len(states)])``"""
        raise NotImplementedError(f"{type(self).__name__} has no drift")

    def on_spike(self, fired, states):
        """The states after the cells marked in ``fired [cells]`` spiked.

        Applied at the spike time, before the refractory cut (the cell's own jumps
        land at its end, Allen's ordering).  Only the fired rows are kept, so the
        rule need not mask.
        """
        return states

    def presynaptic(self, states):
        """``[cells]`` scaling each cell's outgoing synaptic jumps, or None.

        Read from the states just before :meth:`on_spike`.
        """


MECHANISM_INPUTS = ("current", "irradiance", "conductance", "current_density")


def _mechanism_modes(mechanisms) -> tuple[str, ...]:
    modes = (mode for m in mechanisms for mode in m.inputs if mode != "current")
    return tuple(dict.fromkeys(modes))


def _check_mechanisms(mechanisms) -> None:
    names: list[str] = []
    for mechanism in mechanisms:
        if not isinstance(mechanism, GlifMechanism):
            raise TypeError(
                f"a GLIF mechanism is a GlifMechanism, not {type(mechanism).__name__}"
            )
        unknown = set(mechanism.inputs) - set(MECHANISM_INPUTS)
        if unknown:
            raise ValueError(
                f"{type(mechanism).__name__} reads {sorted(unknown)}; a GLIF "
                f"mechanism reads {MECHANISM_INPUTS}"
            )
        names.extend(mechanism.states)
    own = {*RECORDABLE_STATES, *DEFAULT_RECORD, "segments"}
    clashes = sorted({n for n in names if names.count(n) > 1 or n in own})
    if clashes:
        raise ValueError(
            f"mechanism states {clashes} are named twice; a state is recorded by "
            "its name, so the names must be unique and not the cell's own"
        )


def _jumps(mechanism: GlifMechanism) -> bool:
    return type(mechanism).on_spike is not GlifMechanism.on_spike


def _mechanism_terms(mechanisms, layout: StateLayout, t, v, states, inputs):
    """The summed current and the stacked derivatives, on ``[cells]`` / ``[cells, k]``."""
    current, derivatives = 0.0, []
    start = layout.mechanism_start
    for mechanism, (lo, hi) in zip(mechanisms, layout.mechanism_columns(), strict=True):
        given = {mode: inputs[mode] for mode in mechanism.inputs}
        i, d = mechanism.drift(t, v, states[:, lo - start : hi - start], given)
        current = current + jnp.broadcast_to(i, v.shape)
        derivatives.append(jnp.broadcast_to(d, (*v.shape, hi - lo)))
    return current, jnp.concatenate(derivatives, axis=-1)


def _mechanism_spiked(mechanisms, layout: StateLayout, fired, states):
    out, start = [], layout.mechanism_start
    for mechanism, (lo, hi) in zip(mechanisms, layout.mechanism_columns(), strict=True):
        own = states[:, lo - start : hi - start]
        if _jumps(mechanism):
            own = jnp.where(fired[:, None], mechanism.on_spike(fired, own), own)
        out.append(own)
    return jnp.concatenate(out, axis=-1)


def _across_cut(
    mechanisms, layout: StateLayout, t_event, t_resume, n, v, states, inputs_at
):
    """One cell's mechanism states over the cut, on the hold grid: ``([n, k], [k])``.

    A fixed-step RK4 from the spike (after :meth:`GlifMechanism.on_spike`) with V
    held at its reset value, so the samples are the hold the event loop records
    and the last step lands on the resume.
    """
    fired = jnp.ones((1,), bool)
    m0 = _mechanism_spiked(mechanisms, layout, fired, states[None])
    h = (t_resume - t_event) / n

    def f(t, m):
        inputs = {mode: jnp.reshape(x, (1,)) for mode, x in inputs_at(t).items()}
        return _mechanism_terms(mechanisms, layout, t, v[None], m, inputs)[1]

    def step(m, k):
        t = t_event + k * h
        k1 = f(t, m)
        k2 = f(t + h / 2, m + h / 2 * k1)
        k3 = f(t + h / 2, m + h / 2 * k2)
        k4 = f(t + h, m + h * k3)
        return m + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), m[0]

    final, trajectory = jax.lax.scan(step, m0, jnp.arange(n))
    return trajectory, final[0]


def _cell_axes(tree, n_cells: int):
    """vmap axes splitting the per-cell leaves of ``tree`` into one cell each"""
    return jax.tree_util.tree_map(
        lambda x: (
            0 if eqx.is_array(x) and x.ndim >= 1 and x.shape[0] == n_cells else None
        ),
        tree,
    )


def _drift(p, layout: StateLayout, t, y, current, mechanisms=(), inputs=None):
    v = y[..., V]
    asc_1 = y[..., ASC_1]
    asc_2 = y[..., ASC_2]

    total_current = current + asc_1 + asc_2
    if layout.coupled:
        total_current = total_current + y[..., layout.i_syn]
    if mechanisms:
        # on the cell axis even for the one cell of the uncoupled solve
        cells = jnp.reshape(v, (-1,))
        given = {"current": current, **(inputs or {})}
        given = {
            mode: jnp.broadcast_to(x, v.shape).reshape(-1) for mode, x in given.items()
        }
        width = sum(layout.mechanisms)
        states = jnp.reshape(y[..., layout.mechanism_start :], (cells.shape[0], width))
        injected, derivatives = _mechanism_terms(
            mechanisms, layout, t, cells, states, given
        )
        total_current = total_current + jnp.reshape(injected, v.shape)

    active = _active(layout, t, y)

    columns = [
        active * (-(v - p["E_L"]) + total_current / p["g_L"]) / p["tau_m"],  # Eq. (1)
        -p["theta_decay_rate"] * y[..., THETA_S],  # Eq. (2)
        active * (p["a_v"] * (v - p["E_L"]) - p["b_v"] * y[..., THETA_V]),  # Eq. (4)
        -p["asc_decay_rate_1"] * asc_1,  # Eq. (3)
        -p["asc_decay_rate_2"] * asc_2,
    ]
    if layout.coupled:
        columns.append(-y[..., layout.i_syn] / p["tau_syn"])
    if layout.escape:
        columns.append(active * _intensity(p, y))
    if layout.coupled:
        columns.append(jnp.zeros_like(v))  # t_ref_end only moves at an event
    if layout.forced:
        columns.append(jnp.zeros_like(v))  # next_forced only moves at an event

    if not mechanisms:
        return jnp.stack(columns, axis=-1)

    derivatives = jnp.reshape(derivatives, (*v.shape, width))
    return jnp.concatenate([jnp.stack(columns, axis=-1), derivatives], axis=-1)


FAR_FUTURE = 1e30


def _next_forced(forced, after):
    return jnp.min(jnp.where(forced > after, forced, FAR_FUTURE))


def _voltage_reset(p, y):
    return p["E_L"] + p["f_v"] * (y[..., V] - p["E_L"]) - p["delta_v"]


def _diffusion_terms(p, layout: StateLayout, key, t0, t1, tol, n_cells=None):
    if n_cells is None:
        vf = jnp.zeros((layout.size, 1)).at[V, 0].set(p["sigma_v"])
        shape = (1,)
    else:
        index = jnp.arange(n_cells)
        vf = (
            jnp.zeros((n_cells, layout.size, n_cells))
            .at[index, V, index]
            .set(p["sigma_v"])
        )
        shape = (n_cells,)

    path = BrownianPath(t0 - 1.0, t1 + 1.0, tol=tol, shape=shape, key=key)
    return [diffrax.ControlTerm(lambda t, y, args: vf, path)]


def _solve_cell(
    p,
    stim_ts,
    stim_ys,
    y0,
    key,
    *,
    t0,
    t1,
    dt,
    layout,
    config,
    n_out,
    columns,
    diffusion,
    segments=False,
    forced=None,
    mechanisms=(),
    inputs=None,
):
    trace = bool(columns) or segments
    current = diffrax.LinearInterpolation(ts=stim_ts, ys=stim_ys)
    series = {
        mode: diffrax.LinearInterpolation(ts=ts, ys=ys)
        for mode, (ts, ys) in (inputs or {}).items()
    }

    def inputs_at(t):
        return {mode: x.evaluate(t) for mode, x in series.items()}

    def drift(t, y, args):
        if not mechanisms:
            return _drift(p, layout, t, y, current.evaluate(t))
        return _drift(p, layout, t, y, current.evaluate(t), mechanisms, inputs_at(t))

    def across_cut(t_event, y, t_resume, n):
        def given(t):
            return {"current": current.evaluate(t), **inputs_at(t)}

        start = layout.mechanism_start
        return _across_cut(
            mechanisms,
            layout,
            t_event,
            t_resume,
            n,
            _voltage_reset(p, y),
            y[start:],
            given,
        )

    if layout.forced:

        def cond_fn(t, y, args, **kwargs):
            return t - y[layout.next_forced]

    elif layout.escape:

        def cond_fn(t, y, args, **kwargs):
            return y[layout.s]

    else:

        def cond_fn(t, y, args, **kwargs):
            # V > Theta_inf + Theta_s + Theta_v
            return y[V] - _threshold_of(p, y)

    def transition(t_event, y, args, mask, key):
        t_resume = jnp.minimum(t_event + p["t_ref"], t1)
        cut = t_resume - t_event
        reset = [
            _voltage_reset(p, y),  # Eq. (5)
            y[THETA_S] * jnp.exp(-p["theta_decay_rate"] * cut) + p["theta_jump"],  # (6)
            y[THETA_V],  # Eq. (8)
            y[ASC_1] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_1"] * cut)
            + p["asc_amp_1"],  # Eq. (7)
            y[ASC_2] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_2"] * cut)
            + p["asc_amp_2"],
        ]
        if layout.escape:
            reset.append(_draw_survival(p, key, ()))
        if layout.forced:
            reset.append(_next_forced(forced, t_resume))
        if not mechanisms:
            return t_resume, jnp.stack(reset)
        _, carried = across_cut(t_event, y, t_resume, config.points_per_segment)
        return t_resume, jnp.concatenate([jnp.stack(reset), carried])

    def hold_fn(t_event, y, t_resume, ts, args, mask):
        """The refractory window"""
        cut = ts - t_event
        ones = jnp.ones_like(cut)
        held = [
            ones * _voltage_reset(p, y),
            y[THETA_S] * jnp.exp(-p["theta_decay_rate"] * cut),
            ones * y[THETA_V],
            y[ASC_1] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_1"] * cut),
            y[ASC_2] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_2"] * cut),
        ]
        if layout.escape:
            held.append(ones * y[layout.s])
        if layout.forced:
            held.append(ones * y[layout.next_forced])
        if not mechanisms:
            return jnp.stack(held, axis=1)

        trajectory, _ = across_cut(t_event, y, t_resume, ts.shape[0])
        return jnp.concatenate([jnp.stack(held, axis=1), trajectory], axis=1)

    extra_terms = None
    if diffusion:
        key, bm_key = jr.split(key)
        extra_terms = _diffusion_terms(
            p, layout, bm_key, t0, t1, tol=config.dt_solver / 2
        )

    solution = event_solve(
        drift=drift,
        cond_fn=cond_fn,
        transition=transition,
        hold_fn=hold_fn,
        y0=y0,
        t0=t0,
        t1=t1,
        dt=dt,
        extra_terms=extra_terms,
        key=key if layout.escape else None,
        config=config,
        trace=trace,
    )

    if not trace:
        # spikes only: nothing was sampled, so there is nothing to resample
        return (
            jnp.zeros((n_out, 0)),
            solution.event_times,
            solution.y1,
            solution.saturated,
            solution.solver_ok,
        )

    ts_out = t0 + jnp.arange(n_out) * dt
    ys = solution.ys[:, jnp.asarray(columns, dtype=jnp.int32)]
    out = (
        resample(solution.ts, ys, ts_out),
        solution.event_times,
        solution.y1,
        solution.saturated,
        solution.solver_ok,
    )
    if not segments:
        return out

    return (*out, solution.ts, ys)


def _solve_network(
    p,
    stim_ts,
    stim_ys,
    y0,
    key,
    *,
    t0,
    t1,
    dt,
    layout,
    config,
    n_out,
    columns,
    network,
    spike_cells,
    diffusion,
    mechanisms=(),
    inputs=None,
):
    n_cells = int(network.shape[0])
    n_spiking = len(spike_cells)
    spike_index = jnp.asarray(spike_cells, dtype=jnp.int32)
    current = diffrax.LinearInterpolation(ts=stim_ts, ys=stim_ys)
    series = {
        mode: diffrax.LinearInterpolation(ts=ts, ys=ys)
        for mode, (ts, ys) in (inputs or {}).items()
    }

    def drift(t, y, args):
        if not mechanisms:
            return _drift(p, layout, t, y, current.evaluate(t))
        given = {mode: x.evaluate(t) for mode, x in series.items()}
        return _drift(p, layout, t, y, current.evaluate(t), mechanisms, given)

    if layout.escape:

        def spike_cond(t, y, args, n=0, **kwargs):
            return y[n, layout.s]

    else:

        def spike_cond(t, y, args, n=0, **kwargs):
            return y[n, V] - (p["V_threshold_base"][n] + y[n, THETA_S] + y[n, THETA_V])

    def resume_cond(t, y, args, n=0, **kwargs):
        return t - y[n, layout.t_ref_end]

    if config.vectorized_events:
        # the same 2 * n_spiking conditions, as one vector the loop reduces to one
        def cond_fn(t, y, args, **kwargs):
            if layout.escape:
                spike = y[spike_index, layout.s]
            else:
                spike = y[spike_index, V] - (
                    p["V_threshold_base"][spike_index]
                    + y[spike_index, THETA_S]
                    + y[spike_index, THETA_V]
                )
            return jnp.concatenate([spike, t - y[spike_index, layout.t_ref_end]])

    else:
        cond_fn = [functools.partial(spike_cond, n=n) for n in spike_cells] + [
            functools.partial(resume_cond, n=n) for n in spike_cells
        ]

    def widen(bits):
        """A per-condition mask, back on the one-column-per-cell grid"""
        return jnp.zeros(n_cells, bool).at[spike_index].set(bits)

    def transition(t_event, y, args, mask, key):
        fired = widen(mask[:n_spiking])
        resumed = widen(mask[n_spiking:])
        cut = p["t_ref"]

        def undecayed(jump, rate):
            return jump * jnp.exp(jnp.minimum(rate * cut, EXP_CAP))

        theta_s = jnp.where(
            fired,
            y[:, THETA_S] + undecayed(p["theta_jump"], p["theta_decay_rate"]),
            y[:, THETA_S],
        )
        asc_1 = jnp.where(
            fired,
            y[:, ASC_1] * p["asc_r"] + undecayed(p["asc_amp_1"], p["asc_decay_rate_1"]),
            y[:, ASC_1],
        )
        asc_2 = jnp.where(
            fired,
            y[:, ASC_2] * p["asc_r"] + undecayed(p["asc_amp_2"], p["asc_decay_rate_2"]),
            y[:, ASC_2],
        )

        columns_out = [
            jnp.where(fired, _voltage_reset(p, y), y[:, V]),  # Eq. (5)
            theta_s,  # Eq. (6)
            y[:, THETA_V],  # Eq. (8)
            asc_1,  # Eq. (7)
            asc_2,
        ]

        # w[pre, post], so a post-synaptic cell collects the column of every
        # cell that just fired.  Weights are synaptic current jumps, in pA.
        drive = fired.astype(y.dtype)
        for mechanism, (lo, hi) in zip(
            mechanisms, layout.mechanism_columns(), strict=True
        ):
            scale = mechanism.presynaptic(y[:, lo:hi])
            if scale is not None:
                drive = drive * scale
        arriving = drive @ network
        columns_out.append(y[:, layout.i_syn] + arriving)

        if layout.escape:
            columns_out.append(
                jnp.where(fired, _draw_survival(p, key, (n_cells,)), y[:, layout.s])
            )

        t_ref_end = jnp.where(
            fired & (p["t_ref"] > 0), t_event + p["t_ref"], y[:, layout.t_ref_end]
        )
        columns_out.append(jnp.where(resumed, NOT_REFRACTORY, t_ref_end))

        # the loop resumes immediately: the cut is per cell, not global
        if not mechanisms:
            return t_event, jnp.stack(columns_out, axis=-1)
        states = y[:, layout.mechanism_start :]
        carried = _mechanism_spiked(mechanisms, layout, fired, states)
        return t_event, jnp.concatenate(
            [jnp.stack(columns_out, axis=-1), carried], axis=-1
        )

    extra_terms = None
    if diffusion:
        key, bm_key = jr.split(key)
        extra_terms = _diffusion_terms(
            p, layout, bm_key, t0, t1, tol=config.dt_solver / 2, n_cells=n_cells
        )

    trace = bool(columns)
    solution = event_solve(
        drift=drift,
        cond_fn=cond_fn,
        transition=transition,
        y0=y0,
        t0=t0,
        t1=t1,
        dt=dt,
        extra_terms=extra_terms,
        key=key,
        config=config,
        trace=trace,
    )

    if trace:
        ts_out = t0 + jnp.arange(n_out) * dt
        ys = solution.ys[:, :, jnp.asarray(columns, dtype=jnp.int32)]
        # (times, cells, columns) -> (cells, times, columns)
        ys = jnp.transpose(resample(solution.ts, ys, ts_out), (1, 0, 2))
    else:
        # spikes only: nothing was sampled, so there is nothing to resample
        ys = jnp.zeros((n_cells, n_out, 0))

    marks = jnp.zeros((*solution.event_masks.shape[:1], n_cells), bool)
    marks = marks.at[:, spike_index].set(solution.event_masks[:, :n_spiking])
    spike_times = jnp.where(marks.T, solution.event_times[None, :], jnp.inf)

    cells = jnp.zeros((n_cells,), bool)
    return (
        ys,
        spike_times,
        solution.y1,
        cells | solution.saturated,
        cells | solution.solver_ok,
    )


class GlifNeurons(eqx.Module):
    tau_m: Any
    E_L: Any
    g_L: Any
    V_threshold_base: Any
    theta_decay_rate: Any
    theta_jump: Any
    asc_amp_1: Any
    asc_amp_2: Any
    asc_decay_rate_1: Any
    asc_decay_rate_2: Any
    f_v: Any
    delta_v: Any
    t_ref: Any
    a_v: Any
    b_v: Any
    asc_r: Any
    sigma: Any
    tau_s: Any
    alpha: Any
    sigma_v: Any
    tau_syn: Any

    network: Any  # (cells, cells) signed weights in pA, or None when unconnected
    mechanisms: tuple

    n_cells: int = eqx.field(static=True)
    mechanism: str = eqx.field(static=True)
    layout: StateLayout = eqx.field(static=True)
    read_out_neurons: tuple = eqx.field(static=True)
    diffusion: bool = eqx.field(static=True)
    config: SolverConfig = eqx.field(static=True)
    current_scale: float = eqx.field(static=True)
    padded_spikes = True

    def __init__(
        self,
        n_cells: int,
        params: Mapping[str, Any] | None = None,
        level: int = 5,
        mechanism: str = "hard",
        config: SolverConfig | None = None,
        current_scale: float = 1e3,
        network=None,
        read_out_neurons: Sequence[int] | None = None,
        diffusion: bool = False,
        mechanisms: Sequence[GlifMechanism] = (),
    ):
        _check_mechanism(mechanism)

        values = dict(DEFAULT_PARAMS_FULL)
        if params:
            unknown = set(params) - set(ALL_PARAM_NAMES)
            if unknown:
                raise KeyError(
                    f"unknown GLIF parameters {sorted(unknown)}; "
                    f"expected {ALL_PARAM_NAMES}"
                )
            values.update(params)
        values = apply_level(values, level)

        n_cells = int(n_cells)
        for name in ALL_PARAM_NAMES:
            setattr(
                self,
                name,
                jnp.broadcast_to(
                    jnp.asarray(values[name], dtype=jnp.result_type(float)), (n_cells,)
                ),
            )

        if network is not None:
            network = np.asarray(network)
            if network.shape != (n_cells, n_cells):
                raise ValueError(
                    f"expected a ({n_cells}, {n_cells}) connectivity matrix, "
                    f"got {tuple(network.shape)}"
                )
            network = (
                jnp.asarray(network, dtype=jnp.result_type(float))
                if network.any()
                else None
            )
        self.network = network

        read_out_neurons = tuple(int(n) for n in (read_out_neurons or ()))
        if read_out_neurons and network is None:
            raise ValueError(
                "read-out neurons are a network concept: they are the cells the "
                "rest of the network reads without their spiking back into it, "
                "and an unconnected GLIF has no such distinction"
            )
        if any(not 0 <= n < n_cells for n in read_out_neurons):
            raise ValueError(
                f"read-out neurons must be cell indices below {n_cells}, "
                f"got {read_out_neurons}"
            )
        self.read_out_neurons = read_out_neurons

        mechanisms = tuple(mechanisms)
        _check_mechanisms(mechanisms)
        self.mechanisms = mechanisms

        self.n_cells = n_cells
        self.mechanism = mechanism
        self.layout = StateLayout.of(
            network is not None,
            mechanism == "escape",
            mechanisms=tuple(len(m.states) for m in mechanisms),
        )
        self.diffusion = bool(diffusion)
        self.config = config if config is not None else SolverConfig()
        self.current_scale = float(current_scale)

    @property
    def cell_param_names(self) -> tuple[str, ...]:
        return ALL_PARAM_NAMES

    def trainable_params(self, level: int = 5) -> tuple[str, ...]:
        return trainable_param_names(
            level,
            self.mechanism,
            diffusion=self.diffusion,
            coupled=self.network is not None,
        )

    @property
    def input_mode(self) -> str | tuple[str, ...]:
        """``"current"``, and every mode a mechanism reads besides"""
        modes = _mechanism_modes(self.mechanisms)
        return ("current", *modes) if modes else "current"

    @property
    def state_columns(self) -> dict[str, int]:
        """The column of each mechanism state, by name"""
        return dict(
            zip(
                (name for m in self.mechanisms for name in m.states),
                range(self.layout.mechanism_start, self.layout.size),
                strict=True,
            )
        )

    @property
    def spike_cells(self) -> tuple[int, ...]:
        read_out = set(self.read_out_neurons)
        return tuple(n for n in range(self.n_cells) if n not in read_out)

    @classmethod
    def from_neuron_configs(cls, configs: Sequence[Mapping], **kwargs) -> GlifNeurons:
        configs = list(configs)
        if not configs:
            raise ValueError("need at least one neuron_config")
        per_cell = [from_neuron_config(config) for config in configs]
        params = {
            name: jnp.asarray([float(cell[name]) for cell in per_cell])
            for name in PARAM_NAMES
        }
        kwargs.setdefault("level", 5)  # the configs already encode their level
        return cls(len(configs), params, **kwargs)

    def params(self) -> dict:
        """The per-cell parameter arrays"""
        return {name: getattr(self, name) for name in ALL_PARAM_NAMES}

    def initial_state(
        self, key=None, num_samples: int = 1, layout=None, forced_spikes=None
    ):
        layout = self.layout if layout is None else layout
        zeros = jnp.zeros_like(self.E_L)
        columns = [self.E_L, zeros, zeros, zeros, zeros]
        if layout.coupled:
            columns.append(zeros)
        if layout.escape:
            shape = (num_samples, self.n_cells)
            draw = _draw_survival(self.params(), _fallback_key(key), shape)
            columns.append(draw if num_samples > 1 else draw[0])
        if layout.coupled:
            columns.append(jnp.full_like(zeros, NOT_REFRACTORY))
        if layout.forced:
            columns.append(jnp.minimum(jnp.min(forced_spikes, axis=-1), FAR_FUTURE))
        for mechanism in self.mechanisms:
            width = len(mechanism.states)
            initial = jnp.broadcast_to(
                jnp.asarray(mechanism.initial(self.n_cells), dtype=zeros.dtype),
                (self.n_cells, width),
            )
            columns.extend(initial[:, j] for j in range(width))

        return jnp.stack(jnp.broadcast_arrays(*columns), axis=-1)

    def to_neuron_configs(self, templates=None, level=None) -> list[dict]:
        """Write the parameters back as Allen ``neuron_config`` dicts"""
        arrays = {name: np.asarray(getattr(self, name)) for name in PARAM_NAMES}
        configs = []
        for index in range(self.n_cells):
            cell = {name: float(arrays[name][index]) for name in PARAM_NAMES}
            template = None if templates is None else templates[index]
            configs.append(to_neuron_config(cell, template=template, level=level))
        return configs

    def solve(
        self,
        input_current=None,
        t0: float = 0.0,
        t1: float = 100.0,
        dt: float = 0.1,
        y0=None,
        stimulus_dt: float | None = None,
        config: SolverConfig | None = None,
        record=None,
        key=None,
        num_samples: int = 1,
        diffusion: bool | None = None,
        forced_spikes=None,
        inputs: Mapping[str, Any] | None = None,
    ) -> GlifSolution:
        """Integrate from ``t0`` to ``t1``.

        ``inputs`` is ``{mode: [steps, cells]}`` for the modes the mechanisms read,
        sampled every ``stimulus_dt`` like the current and interpolated the same
        way; a mode they read but are not given is zero.
        """
        return self._solve(
            input_current=input_current,
            t0=float(t0),
            t1=float(t1),
            dt=float(dt),
            y0=y0,
            stimulus_dt=stimulus_dt,
            config=config,
            record=None if record is None else frozenset(record),
            key=key,
            num_samples=int(num_samples),
            diffusion=diffusion,
            forced_spikes=forced_spikes,
            inputs=dict(inputs) if inputs else {},
        )

    @eqx.filter_jit
    def _solve(
        self,
        input_current,
        t0: float,
        t1: float,
        dt: float,
        y0,
        stimulus_dt: float | None,
        config: SolverConfig | None,
        record,
        key,
        num_samples: int,
        diffusion: bool | None,
        forced_spikes,
        inputs,
    ) -> GlifSolution:
        t0, t1, dt = float(t0), float(t1), float(dt)
        config = self.config if config is None else config
        layout = self.layout
        if forced_spikes is not None:
            if self.network is not None:
                raise NotImplementedError(
                    "forced spikes are only implemented for the uncoupled solver"
                )
            forced_spikes = jnp.asarray(forced_spikes, dtype=jnp.result_type(float))
            if forced_spikes.ndim == 1:
                forced_spikes = forced_spikes[None, :]
            if forced_spikes.ndim != 2 or forced_spikes.shape[0] != self.n_cells:
                raise ValueError(
                    f"forced_spikes must be (cells={self.n_cells}, k), "
                    f"got {tuple(forced_spikes.shape)}"
                )
            forced_spikes = jnp.sort(forced_spikes, axis=-1)  # inf padding sorts last
            layout = StateLayout.of(
                False, False, forced=True, mechanisms=layout.mechanisms
            )
        n_out = round((t1 - t0) / dt) + 1
        stimulus_dt = dt if stimulus_dt is None else float(stimulus_dt)
        series = self._input_series(inputs, t0, t1, stimulus_dt)

        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        named = self.state_columns
        if record is None:
            columns, want_threshold = ALL_COLUMNS + tuple(named.values()), True
        else:
            record = frozenset(record)
            columns, want_threshold = _columns_for(record), "threshold" in record
            if named:
                chosen = {named[name] for name in record if name in named}
                columns = tuple(sorted({*columns, *chosen}))

        want_segments = bool(record) and "segments" in record

        if input_current is None:
            stim = jnp.zeros((2, self.n_cells))
            stimulus_dt = t1 - t0
        else:
            stim = self._as_series(input_current, "a stimulus")

        stim = stim * self.current_scale  # -> pA
        stim_ts = t0 + jnp.arange(stim.shape[0]) * stimulus_dt

        key, state_key = jr.split(_fallback_key(key))
        if y0 is None:
            y0 = self.initial_state(
                state_key, num_samples, layout=layout, forced_spikes=forced_spikes
            )
        y0 = jnp.asarray(y0)
        if y0.shape[-1] != layout.size:
            raise ValueError(
                f"expected an initial state with {layout.size} columns for "
                f"a {self.mechanism!r} "
                f"{'network' if layout.coupled else 'population'}"
                f"{' under forcing' if layout.forced else ''}"
                f"{' with mechanisms' if layout.mechanisms else ''}, "
                f"got {y0.shape[-1]}"
            )
        if y0.ndim == 2:
            y0 = jnp.broadcast_to(y0, (num_samples, *y0.shape))

        shared = {
            "t0": t0,
            "t1": t1,
            "dt": dt,
            "layout": layout,
            "config": config,
            "n_out": n_out,
            "columns": columns,
            "diffusion": self.diffusion if diffusion is None else bool(diffusion),
        }

        if want_segments and self.network is not None:
            raise NotImplementedError(
                "record='segments' is only implemented for the uncoupled solver; the network path "
                "does not return its native grid yet"
            )

        if self.network is None and not self.mechanisms:

            def solve_sample(y0_sample, sample_key):
                return jax.vmap(
                    lambda p, s, y, k, f: _solve_cell(
                        p, stim_ts, s, y, k, forced=f, segments=want_segments, **shared
                    ),
                    in_axes=(0, 0, 0, 0, None if forced_spikes is None else 0),
                )(
                    self.params(),
                    stim.T,
                    y0_sample,
                    jr.split(sample_key, self.n_cells),
                    forced_spikes,
                )

        elif self.network is None:
            input_ts = {mode: ts for mode, (ts, _) in series.items()}

            def solve_cell(p, s, y, k, f, m, x):
                given = {mode: (input_ts[mode], x[mode]) for mode in x}
                return _solve_cell(
                    p,
                    stim_ts,
                    s,
                    y,
                    k,
                    forced=f,
                    segments=want_segments,
                    mechanisms=m,
                    inputs=given,
                    **shared,
                )

            def solve_sample(y0_sample, sample_key):
                return jax.vmap(
                    solve_cell,
                    in_axes=(
                        0,
                        0,
                        0,
                        0,
                        None if forced_spikes is None else 0,
                        _cell_axes(self.mechanisms, self.n_cells),
                        0,
                    ),
                )(
                    self.params(),
                    stim.T,
                    y0_sample,
                    jr.split(sample_key, self.n_cells),
                    forced_spikes,
                    self.mechanisms,
                    {mode: ys.T for mode, (_, ys) in series.items()},
                )

        else:

            def solve_sample(y0_sample, sample_key):
                return _solve_network(
                    self.params(),
                    stim_ts,
                    stim,
                    y0_sample,
                    sample_key,
                    network=self.network,
                    spike_cells=self.spike_cells,
                    mechanisms=self.mechanisms,
                    inputs=series,
                    **shared,
                )

        sample_keys = jr.split(key, num_samples)
        if num_samples == 1:
            outputs = solve_sample(y0[0], sample_keys[0])
        else:
            outputs = jax.vmap(solve_sample)(y0, sample_keys)
        segment_ts = segment_ys = None
        if want_segments:
            ys, spike_times, yT, saturated, solver_ok, segment_ts, segment_ys = outputs
        else:
            ys, spike_times, yT, saturated, solver_ok = outputs

        columns = tuple(columns)
        threshold = None
        if want_threshold:
            # Theta_inf + Theta_s + Theta_v, Eqs. (2, 4)
            theta_s = ys[..., columns.index(THETA_S)]
            theta_v = ys[..., columns.index(THETA_V)]
            threshold = self.V_threshold_base[:, None] + theta_s + theta_v

        return GlifSolution(
            ts=t0 + jnp.arange(n_out) * dt,
            ys=ys,
            threshold=threshold,
            spike_times=spike_times,
            yT=yT,
            saturated=saturated,
            solver_ok=solver_ok,
            segment_ts=segment_ts,
            segment_ys=segment_ys,
            columns=columns,
        )

    def _as_series(self, value, what: str):
        """``[steps, cells]``, at least two steps"""
        array = jnp.asarray(value)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != self.n_cells:
            raise ValueError(
                f"expected {what} of shape (steps, {self.n_cells}), "
                f"got {tuple(array.shape)}"
            )
        if array.shape[0] < 2:
            array = jnp.repeat(array, 2, axis=0)
        return array

    def _input_series(self, inputs, t0: float, t1: float, stimulus_dt: float):
        """``{mode: (ts, [steps, cells])}`` for every mode a mechanism reads"""
        modes = _mechanism_modes(self.mechanisms)
        unread = sorted(set(inputs) - set(modes))
        if unread:
            raise ValueError(
                f"no mechanism of this GLIF reads {unread}"
                + (f"; they read {list(modes)}" if modes else "")
            )
        series = {}
        for mode in modes:
            if inputs.get(mode) is None:
                ys, step = jnp.zeros((2, self.n_cells)), t1 - t0
            else:
                ys, step = (
                    self._as_series(inputs[mode], f"a {mode!r} input"),
                    stimulus_dt,
                )
            series[mode] = (t0 + jnp.arange(ys.shape[0]) * step, ys)
        return series

    def run(
        self,
        input_current=None,
        noise=None,
        t0: float = 0.0,
        t1: float = 100.0,
        dt: float = 0.1,
        y0=None,
        dt_solver: float | None = None,
        key=None,
        record=None,
        num_samples: int = 1,
        **kwargs,
    ):
        """Simulate, returning ``(spike_row_ids, spike_times, ids, voltage, _, _, yT, states)``.

        ``spike_times`` is the solver's ``(..., cells, k)`` rectangle with non-finite padding, and
        ``spike_row_ids`` labels its rows. Nothing here syncs to the host.
        """
        module, diffusion = self._with_noise(noise)
        num_samples = int(num_samples)

        inputs = kwargs.pop("inputs", None)
        if isinstance(input_current, Mapping):
            inputs = {**(inputs or {}), **input_current}
            input_current = inputs.pop("current", None)

        record = DEFAULT_RECORD if record is None else frozenset(record)

        config = module.config
        if dt_solver is not None:
            config = dataclasses.replace(config, dt_solver=float(dt_solver))

        solution = module.solve(
            input_current=input_current,
            t0=t0,
            t1=t1,
            dt=dt,
            y0=y0,
            config=config,
            stimulus_dt=kwargs.pop("stimulus_dt", None),
            record=record,
            key=key,
            num_samples=num_samples,
            diffusion=diffusion,
            inputs=inputs,
        )

        ids = jnp.arange(module.n_cells)
        voltage = solution.v if "voltage" in record else None
        states = {
            name: (ids, getattr(solution, attribute))
            for name, attribute in _STATE_ATTRIBUTES.items()
            if name in record
        }
        for name, column in module.state_columns.items():
            if name in record:
                states[name] = (ids, solution.column(column))

        if "spikes" not in record:
            return None, None, ids, voltage, None, None, solution.yT, states

        return (
            ids,
            solution.spike_times,
            ids,
            voltage,
            None,
            None,
            solution.yT,
            states,
        )

    def _with_noise(self, noise) -> tuple[GlifNeurons, bool | None]:
        if not noise:
            return self, None
        if not isinstance(noise, Mapping) or set(noise) - {"sigma_v"}:
            raise ValueError(
                f"GLIF takes its noise as {{'sigma_v': amplitude}} in mV/sqrt(ms), "
                f"got {noise!r}"
            )
        sigma_v = jnp.broadcast_to(
            jnp.asarray(noise["sigma_v"], dtype=self.sigma_v.dtype), self.sigma_v.shape
        )
        return eqx.tree_at(lambda m: m.sigma_v, self, sigma_v), True


class GlifNeuronJAX(eqx.Module):
    neurons: GlifNeurons

    def __init__(
        self,
        params=None,
        level: int = 5,
        config: SolverConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = SolverConfig(
                max_segment_span=math.inf, points_per_segment=50, max_rate_per_ms=1.0
            )
        self.neurons = GlifNeurons(
            1, params, level=level, config=config, current_scale=1.0, **kwargs
        )

    @classmethod
    def from_dict(cls, neuron_config, **kwargs) -> GlifNeuronJAX:
        return cls(from_neuron_config(neuron_config), **kwargs)

    @property
    def p(self) -> dict:
        return {name: float(value[0]) for name, value in self.neurons.params().items()}

    def __repr__(self):
        return f"GlifNeuronJAX({', '.join(f'{k}={v:g}' for k, v in self.p.items())})"

    def run(self, stimulus_pA, native_rate, dt: float | None = None) -> dict:
        stim = jnp.asarray(stimulus_pA, dtype=jnp.result_type(float)).reshape(-1, 1)
        n_samples = stim.shape[0]
        if n_samples < 2:
            raise ValueError(f"stimulus must have >= 2 samples, got {n_samples}")
        if native_rate <= 0:
            raise ValueError(f"native_rate must be positive, got {native_rate}")

        stimulus_dt = 1000.0 / native_rate
        t1 = (n_samples - 1) * stimulus_dt
        dt = stimulus_dt if dt is None else float(dt)

        solution = self.neurons.solve(
            input_current=stim, t0=0.0, t1=t1, dt=dt, stimulus_dt=stimulus_dt
        )
        spikes = solution.spike_times[0]
        spikes = spikes[jnp.isfinite(spikes)]

        return {
            "times": solution.ts,
            "voltage": solution.v[0],
            "threshold": solution.threshold[0],
            "theta_s": solution.theta_s[0],
            "theta_v": solution.theta_v[0],
            "AScurrents": solution.ascs[0],
            "spike_times": spikes,
            "n_spikes": int(spikes.shape[0]),
            "duration": 1000.0 * n_samples / native_rate,
        }


def cell_populations(env) -> np.ndarray:
    gids = np.asarray(env.simulated_gids(everywhere=True))
    names = np.full(len(gids), "", dtype=object)
    for name, (start, count, *_) in env.system.population_ranges.items():
        names[(gids >= int(start)) & (gids < int(start) + int(count))] = name
    return names.astype(str)


class GLIF(Model):
    def __init__(
        self,
        level: int = 5,
        mechanism: str = "hard",
        params: Mapping[str, Any] | None = None,
        config: SolverConfig | None = None,
        current_scale: float = 1e3,
        read_out_neurons: Sequence[int] | None = None,
        diffusion: bool = False,
    ):
        self.level = _check_level(level)
        self.mechanism = _check_mechanism(mechanism)
        self.params = dict(params) if params else None
        self.config = config
        self.current_scale = float(current_scale)
        self.read_out_neurons = (
            None
            if read_out_neurons is None
            else tuple(int(n) for n in read_out_neurons)
        )
        self.diffusion = bool(diffusion)

    @classmethod
    def from_neuron_config(cls, neuron_config: Mapping, **kwargs) -> GLIF:
        kwargs.setdefault("level", level_of_neuron_config(neuron_config))
        return cls(params=from_neuron_config(neuron_config), **kwargs)

    @classmethod
    def leaky_integrate_and_fire(cls, **kwargs) -> GLIF:
        """The plain leaky integrator, as a point in the GLIF space"""
        kwargs.setdefault("level", LIF_LEVEL)
        kwargs["params"] = {**LIF_PARAMS, **(kwargs.get("params") or {})}
        return cls(**kwargs)

    def trainable_params(self) -> tuple[str, ...]:
        return trainable_param_names(
            self.level, self.mechanism, diffusion=self.diffusion
        )

    def __repr__(self):
        return f"GLIF(level={self.level}, mechanism={self.mechanism!r})"

    def recordable_states(self) -> tuple[str, ...]:
        return RECORDABLE_STATES

    def prepare_stimulus(self, stimulus):
        if stimulus.input_mode not in MECHANISM_INPUTS:
            raise ValueError(
                f"GLIF reads the 'current' input mode, and the modes of "
                f"{MECHANISM_INPUTS[1:]} its mechanisms declare, not "
                f"{stimulus.input_mode!r}"
            )
        return stimulus

    def cell_params(self, level: int | None = None) -> dict:
        return apply_level(
            {**DEFAULT_PARAMS_FULL, **(self.params or {})},
            self.level if level is None else level,
        )

    def diffrax_module(self, env, key=None) -> GlifNeurons:
        weights = getattr(env, "_weights", None) or None

        gids = np.asarray(env.simulated_gids(everywhere=True))
        network = np.asarray(env.system.connectivity_matrix(weights=weights, gids=gids))

        return GlifNeurons(
            len(gids),
            self.params,
            level=self.level,
            mechanism=self.mechanism,
            config=self.config,
            current_scale=self.current_scale,
            network=network,
            read_out_neurons=self.read_out_neurons if network.any() else None,
            diffusion=self.diffusion,
            mechanisms=tuple(self.diffrax_mechanisms(env)),
        )

    def diffrax_default_noise(self, system: str):
        return {}

    def diffrax_default_weights(self, system: str):
        return {}

    @property
    def brian2_param_units(self) -> dict:
        """GLIF keeps nS / pA on every backend, so the brian2 handle reads them so"""
        import brian2 as b2

        return {
            "g_L": b2.nS,
            "asc_amp_1": b2.pA,
            "asc_amp_2": b2.pA,
        }

    def brian2_population_group(
        self, population_name, n, offset, coordinates, prng, rows=None
    ):
        """The same equations, as a brian2 ``NeuronGroup``.

        Every parameter is a group state variable rather than a namespace
        constant, so the per-cell parameter API reaches it: ``env.cells[gid]
        .set_params({"tau_m": 12.0})`` addresses one row.  The units are
        brian2's, so the values are the canonical mV / pA / ms ones scaled into
        SI.
        """
        import brian2 as b2

        if self.mechanism != "hard":
            raise NotImplementedError(
                f"the {self.mechanism!r} mechanism root-finds an integrated "
                f"escape rate, which needs the event-driven solver; run it on "
                f"the diffrax backend, or use mechanism='hard' here"
            )

        params = self.cell_params()

        group = b2.NeuronGroup(
            n,
            """
            dv/dt = (-(v - E_L) + (I + I_noise + asc_1 + asc_2
                                   + stim_i(t, stim_index))/g_L
                     + stim_v(t, stim_index))/tau_m : volt (unless refractory)
            dtheta_s/dt = -theta_decay_rate*theta_s : volt
            dtheta_v/dt = a_v*(v - E_L) - b_v*theta_v : volt (unless refractory)
            dasc_1/dt = -asc_decay_rate_1*asc_1 : amp
            dasc_2/dt = -asc_decay_rate_2*asc_2 : amp
            I : amp
            I_noise : amp
            noise_amplitude : 1
            # which column of the stimulus array is this neuron's; the backend
            # sets it from the gid, because a selection makes the group hold a
            # scattered subset and the group index no longer names the column
            stim_index : integer (constant)
            tau_m : second
            E_L : volt
            g_L : siemens
            V_threshold_base : volt
            theta_decay_rate : Hz
            theta_jump : volt
            asc_amp_1 : amp
            asc_amp_2 : amp
            asc_decay_rate_1 : Hz
            asc_decay_rate_2 : Hz
            f_v : 1
            delta_v : volt
            t_ref : second
            a_v : Hz
            b_v : Hz
            asc_r : 1
            """,
            threshold="v > V_threshold_base + theta_s + theta_v",
            # The jumps belong at the *end* of the spike cut, undecayed -- Allen's
            # ordering, which the diffrax path gets from its analytic hold.  brian2
            # integrates the cut instead, so each jump is pre-multiplied by the
            # decay it is about to receive and the two backends agree.
            reset="""
            v = E_L + f_v*(v - E_L) - delta_v
            theta_s += theta_jump*exp(theta_decay_rate*t_ref)
            asc_1 = asc_1*asc_r + asc_amp_1*exp(asc_decay_rate_1*t_ref)
            asc_2 = asc_2*asc_r + asc_amp_2*exp(asc_decay_rate_2*t_ref)
            """,
            refractory="t_ref",
            method="euler",
            name=population_name,
        )

        units = {
            "tau_m": b2.ms,
            "E_L": b2.mV,
            "g_L": b2.nS,
            "V_threshold_base": b2.mV,
            "theta_decay_rate": 1 / b2.ms,
            "theta_jump": b2.mV,
            "asc_amp_1": b2.pA,
            "asc_amp_2": b2.pA,
            "asc_decay_rate_1": 1 / b2.ms,
            "asc_decay_rate_2": 1 / b2.ms,
            "f_v": 1,
            "delta_v": b2.mV,
            "t_ref": b2.ms,
            "a_v": 1 / b2.ms,
            "b_v": 1 / b2.ms,
            "asc_r": 1,
        }
        for name, unit in units.items():
            setattr(group, name, _for_population(params[name], n, offset, rows) * unit)

        group.v = _for_population(params["E_L"], n, offset, rows) * b2.mV

        return group

    def brian2_connection_synapse(self, pre_group, post_group):
        import brian2 as b2

        return b2.Synapses(
            pre_group,
            post_group,
            """
            w : 1
            multiplier: 1
            distance: 1
            prefix: 1
            """,
            on_pre="I += prefix * w * multiplier * pA",
        )

    def brian2_noise_op(self, population_group, prng):
        import brian2 as b2

        return population_group.run_regularly(
            "I_noise = noise_amplitude*randn()*pA", dt=1 * b2.ms
        )

    def brian2_noise_configure(self, population_group, level=1.0):
        population_group.noise_amplitude = level
