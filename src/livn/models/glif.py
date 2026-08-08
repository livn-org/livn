from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Mapping, Optional, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from livn.models.eventloop import SolverConfig, event_solve
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

MECHANISMS = ("hard", "escape")


def _check_mechanism(mechanism: str) -> str:
    if mechanism not in MECHANISMS:
        raise ValueError(
            f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}"
        )
    if mechanism != "hard":
        raise NotImplementedError(
            f"the {mechanism!r} mechanism is not implemented yet; use 'hard'"
        )
    return mechanism


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


def trainable_param_names(level: int) -> tuple[str, ...]:
    level = _check_level(level)
    excluded = set(LEVEL_ZEROED[level])
    return tuple(name for name in PARAM_NAMES if name not in excluded)


def to_vector(params: Mapping[str, Any]) -> list:
    return [params[name] for name in PARAM_NAMES]


def from_vector(vector: Sequence) -> dict:
    if len(vector) != len(PARAM_NAMES):
        raise ValueError(
            f"expected {len(PARAM_NAMES)} parameters, got {len(vector)}; "
            f"the canonical order is {PARAM_NAMES}"
        )
    return dict(zip(PARAM_NAMES, vector))


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
    template: Optional[Mapping] = None,
    level: Optional[int] = None,
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
        0, int(round(params["t_ref"] / (config["dt"] * 1e3))) - 1
    )

    return config


class GlifSolution(eqx.Module):
    """Result of :meth:`GlifNeurons.solve`, on a uniform time grid.

    The ODE state is laid out as ``[v, theta_s, theta_v, asc_1, asc_2]``; the
    properties below name its columns.
    """

    ts: Any  # (times,) in ms
    ys: Any  # (cells, times, 5)
    threshold: Any  # (cells, times) in mV
    spike_times: Any  # (cells, max_spikes) in ms, inf-padded
    yT: Any  # (cells, 5) final state
    saturated: Any  # (cells,) spike budget exhausted before t1
    solver_ok: Any  # (cells,) every inner solve succeeded

    @property
    def v(self):
        return self.ys[..., 0]

    @property
    def theta_s(self):
        return self.ys[..., 1]

    @property
    def theta_v(self):
        return self.ys[..., 2]

    @property
    def ascs(self):
        return self.ys[..., 3:]

    @property
    def num_spikes(self):
        return jnp.sum(jnp.isfinite(self.spike_times), axis=-1)


def _solve_cell(p, stim_ts, stim_ys, y0, *, t0, t1, dt, mechanism, config, n_out):
    current = diffrax.LinearInterpolation(ts=stim_ts, ys=stim_ys)

    def drift(t, y, args):
        v, theta_s, theta_v, asc_1, asc_2 = y
        total_current = current.evaluate(t) + asc_1 + asc_2
        return jnp.stack(
            [
                (-(v - p["E_L"]) + total_current / p["g_L"]) / p["tau_m"],  # Eq. (1)
                -p["theta_decay_rate"] * theta_s,  # Eq. (2)
                p["a_v"] * (v - p["E_L"]) - p["b_v"] * theta_v,  # Eq. (4)
                -p["asc_decay_rate_1"] * asc_1,  # Eq. (3)
                -p["asc_decay_rate_2"] * asc_2,
            ]
        )

    def cond_fn(t, y, args, **kwargs):
        # V > Theta_inf + Theta_s + Theta_v
        return y[0] - (p["V_threshold_base"] + y[1] + y[2])

    def transition(t_event, y, args, mask, key):
        t_resume = jnp.minimum(t_event + p["t_ref"], t1)
        cut = t_resume - t_event
        return t_resume, jnp.stack(
            [
                p["E_L"] + p["f_v"] * (y[0] - p["E_L"]) - p["delta_v"],  # Eq. (5)
                y[1] * jnp.exp(-p["theta_decay_rate"] * cut) + p["theta_jump"],  # (6)
                y[2],  # Eq. (8)
                y[3] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_1"] * cut)
                + p["asc_amp_1"],  # Eq. (7)
                y[4] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_2"] * cut)
                + p["asc_amp_2"],
            ]
        )

    def hold_fn(t_event, y, t_resume, ts, args, mask):
        """The refractory window"""
        cut = ts - t_event
        ones = jnp.ones_like(cut)
        return jnp.stack(
            [
                ones * (p["E_L"] + p["f_v"] * (y[0] - p["E_L"]) - p["delta_v"]),
                y[1] * jnp.exp(-p["theta_decay_rate"] * cut),
                ones * y[2],
                y[3] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_1"] * cut),
                y[4] * p["asc_r"] * jnp.exp(-p["asc_decay_rate_2"] * cut),
            ],
            axis=1,
        )

    if mechanism != "hard":
        raise NotImplementedError(
            f"the {mechanism!r} mechanism is not implemented yet; use 'hard'"
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
        config=config,
    )

    ts_out = t0 + jnp.arange(n_out) * dt
    return (
        solution.sample(ts_out),
        solution.event_times,
        solution.y1,
        solution.saturated,
        solution.solver_ok,
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

    n_cells: int = eqx.field(static=True)
    mechanism: str = eqx.field(static=True)
    config: SolverConfig = eqx.field(static=True)
    current_scale: float = eqx.field(static=True)

    def __init__(
        self,
        n_cells: int,
        params: Optional[Mapping[str, Any]] = None,
        level: int = 5,
        mechanism: str = "hard",
        config: Optional[SolverConfig] = None,
        current_scale: float = 1e3,
    ):
        _check_mechanism(mechanism)

        values = dict(DEFAULT_PARAMS)
        if params:
            unknown = set(params) - set(PARAM_NAMES)
            if unknown:
                raise KeyError(
                    f"unknown GLIF parameters {sorted(unknown)}; expected {PARAM_NAMES}"
                )
            values.update(params)
        values = apply_level(values, level)

        n_cells = int(n_cells)
        for name in PARAM_NAMES:
            setattr(
                self,
                name,
                jnp.broadcast_to(
                    jnp.asarray(values[name], dtype=jnp.result_type(float)), (n_cells,)
                ),
            )

        self.n_cells = n_cells
        self.mechanism = mechanism
        self.config = config if config is not None else SolverConfig()
        self.current_scale = float(current_scale)

    @classmethod
    def from_neuron_configs(cls, configs: Sequence[Mapping], **kwargs) -> "GlifNeurons":
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
        return {name: getattr(self, name) for name in PARAM_NAMES}

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
        stimulus_dt: Optional[float] = None,
        config: Optional[SolverConfig] = None,
    ) -> GlifSolution:
        t0, t1, dt = float(t0), float(t1), float(dt)
        config = self.config if config is None else config
        n_out = int(round((t1 - t0) / dt)) + 1
        stimulus_dt = dt if stimulus_dt is None else float(stimulus_dt)

        if input_current is None:
            stim = jnp.zeros((2, self.n_cells))
            stimulus_dt = t1 - t0
        else:
            stim = jnp.asarray(input_current)
            if stim.ndim == 1:
                stim = stim[:, None]
            if stim.ndim != 2 or stim.shape[1] != self.n_cells:
                raise ValueError(
                    f"expected a stimulus of shape (steps, {self.n_cells}), "
                    f"got {tuple(stim.shape)}"
                )
            if stim.shape[0] < 2:
                stim = jnp.repeat(stim, 2, axis=0)

        stim = stim * self.current_scale  # -> pA
        stim_ts = t0 + jnp.arange(stim.shape[0]) * stimulus_dt

        if y0 is None:
            # rest: [E_L, 0, 0, 0, 0] per cell
            zeros = jnp.zeros_like(self.E_L)
            y0 = jnp.stack([self.E_L, zeros, zeros, zeros, zeros], axis=1)
        y0 = jnp.asarray(y0)

        solve = jax.vmap(
            lambda p, s, y: _solve_cell(
                p,
                stim_ts,
                s,
                y,
                t0=t0,
                t1=t1,
                dt=dt,
                mechanism=self.mechanism,
                config=config,
                n_out=n_out,
            ),
            in_axes=(0, 0, 0),
        )
        ys, spike_times, yT, saturated, solver_ok = solve(self.params(), stim.T, y0)

        threshold = self.V_threshold_base[:, None] + ys[..., 1] + ys[..., 2]

        return GlifSolution(
            ts=t0 + jnp.arange(n_out) * dt,
            ys=ys,
            threshold=threshold,
            spike_times=spike_times,
            yT=yT,
            saturated=saturated,
            solver_ok=solver_ok,
        )

    def run(
        self,
        input_current=None,
        noise=None,
        t0: float = 0.0,
        t1: float = 100.0,
        dt: float = 0.1,
        y0=None,
        dt_solver: Optional[float] = None,
        key=None,
        unroll: Optional[str] = None,
        **kwargs,
    ):
        del key  # the hard mechanism is deterministic
        if noise:
            raise NotImplementedError("GLIF does not support noise yet")

        config = self.config
        if dt_solver is not None:
            config = dataclasses.replace(config, dt_solver=float(dt_solver))

        solution = self.solve(
            input_current=input_current,
            t0=t0,
            t1=t1,
            dt=dt,
            y0=y0,
            config=config,
            stimulus_dt=kwargs.pop("stimulus_dt", None),
        )

        ids = jnp.arange(self.n_cells)
        spike_times = solution.spike_times
        fired = jnp.isfinite(spike_times)

        n_out = solution.ts.shape[0]
        if unroll == "mask":
            index = jnp.round((spike_times - float(t0)) / float(dt))
            index = jnp.where(fired, jnp.clip(index, 0, n_out - 1), n_out).astype(
                jnp.int32
            )
            mask = jax.vmap(
                # out-of-range indices (the padding) are dropped
                lambda idx: jnp.zeros(n_out, bool).at[idx].set(True, mode="drop")
            )(index)
            return mask, None, ids, solution.v, None, None, solution.yT

        cell_ids = jnp.broadcast_to(ids[:, None], spike_times.shape)
        if unroll == "padded":
            return (
                cell_ids.reshape(-1),
                spike_times.reshape(-1),
                ids,
                solution.v,
                None,
                None,
                solution.yT,
            )

        keep = np.asarray(fired).reshape(-1)
        it = np.asarray(cell_ids).reshape(-1)[keep]
        tt = np.asarray(spike_times).reshape(-1)[keep]
        order = np.argsort(tt, kind="stable")

        return it[order], tt[order], ids, solution.v, None, None, solution.yT


class GlifNeuronJAX(eqx.Module):
    neurons: GlifNeurons

    def __init__(
        self,
        params=None,
        level: int = 5,
        config: Optional[SolverConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = SolverConfig(
                max_segment_span=math.inf, points_per_segment=50, max_rate=1.0
            )
        self.neurons = GlifNeurons(
            1, params, level=level, config=config, current_scale=1.0, **kwargs
        )

    @classmethod
    def from_dict(cls, neuron_config, **kwargs) -> "GlifNeuronJAX":
        return cls(from_neuron_config(neuron_config), **kwargs)

    @property
    def p(self) -> dict:
        return {name: float(value[0]) for name, value in self.neurons.params().items()}

    def __repr__(self):
        return f"GlifNeuronJAX({', '.join(f'{k}={v:g}' for k, v in self.p.items())})"

    def run(self, stimulus_pA, native_hz, dt: Optional[float] = None) -> dict:
        stim = jnp.asarray(stimulus_pA, dtype=jnp.result_type(float)).reshape(-1, 1)
        n_samples = stim.shape[0]
        if n_samples < 2:
            raise ValueError(f"stimulus must have >= 2 samples, got {n_samples}")
        if native_hz <= 0:
            raise ValueError(f"native_hz must be positive, got {native_hz}")

        stimulus_dt = 1000.0 / native_hz
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
            "spike_times": spikes / 1000.0,  # ms -> s
            "n_spikes": int(spikes.shape[0]),
            "duration_s": n_samples / native_hz,
        }


class GLIF(Model):
    def __init__(
        self,
        level: int = 5,
        mechanism: str = "hard",
        params: Optional[Mapping[str, Any]] = None,
        config: Optional[SolverConfig] = None,
        current_scale: float = 1e3,
    ):
        self.level = _check_level(level)
        self.mechanism = _check_mechanism(mechanism)
        self.params = dict(params) if params else None
        self.config = config
        self.current_scale = float(current_scale)

    @classmethod
    def from_neuron_config(cls, neuron_config: Mapping, **kwargs) -> "GLIF":
        kwargs.setdefault("level", level_of_neuron_config(neuron_config))
        return cls(params=from_neuron_config(neuron_config), **kwargs)

    def trainable_params(self) -> tuple[str, ...]:
        return trainable_param_names(self.level)

    def __repr__(self):
        return f"GLIF(level={self.level}, mechanism={self.mechanism!r})"

    def prepare_stimulus(self, stimulus):
        if stimulus.input_mode != "current":
            raise ValueError(
                f"GLIF only supports the 'current' input mode, got {stimulus.input_mode!r}"
            )
        return stimulus

    def diffrax_module(self, env, key=None) -> GlifNeurons:
        return GlifNeurons(
            len(env.active_gids()),
            self.params,
            level=self.level,
            mechanism=self.mechanism,
            config=self.config,
            current_scale=self.current_scale,
        )

    def diffrax_default_noise(self, system: str):
        return {}

    def diffrax_default_weights(self, system: str):
        return {}
