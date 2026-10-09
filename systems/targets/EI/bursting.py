from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from machinable.config import Field as ConfigField

from systems.targets.EI.base import Culture, _band_constraint, _max_constraint
from systems.targets.EI.features import feature
from systems.targets.EI.units import Stat, read_unit_targets


class BurstingCulture(Culture):
    class Config(Culture.Config):
        evoked_repeats: int = ConfigField(default=8, ge=1)

    LEADER_RANGES: ClassVar[dict | None] = {
        "fraction": [0.02, 0.3],
        "bias": [0.0, 20.0],
        "rest_noise": [0.0, 1.0],
        "efferent": [1.0, 12.0, "log"],
        "delay_sd": [0.0, 30.0],
        "input_spread": [0.0, 0.6],
        "input_power": [0.5, 4.0, "log"],
    }
    LEADER_SELECTION: ClassVar[str] = "input"
    UNIT_FEATURES: ClassVar[tuple] = (
        "unit_participation",
        "unit_between_rate_hz",
        "order_spread_ms",
        "unit_spikes_per_burst",
        "unit_active_between_fraction",
        "between_time_rho",
    )
    SEARCH_OBJECTIVES: ClassVar[tuple] = (
        "unit_participation",
        "order_spread_ms",
        "mean_channel_correlation",
        "pop_autocorr_tau",
    )
    GATED_FEATURES: ClassVar[tuple] = (
        "isi_cv",
        "unit_participation",
        "unit_between_rate_hz",
        "order_spread_ms",
        "unit_spikes_per_burst",
        "unit_active_between_fraction",
        "between_time_rho",
    )
    GATED_RESPONSE: ClassVar[tuple] = ("response_latency_ms", "response_duration_ms")
    MAX_RECRUITMENT_MISS = 1.5
    THRESHOLD_SIGMA = 5.0
    DEAD_TIME_MS = 3.0

    def _measure(self, observation, condition: str) -> None:
        super()._measure(observation, condition)
        missing = [n for n in self.SEARCH_OBJECTIVES if n not in self._targets]
        if missing:
            raise ValueError(
                f"{type(self).__name__} searches on {missing}, which this "
                "observation does not measure"
            )
        self.skip_objectives = tuple(
            name
            for name in dict.fromkeys((*self.skip_objectives, *self._targets))
            if name not in self.SEARCH_OBJECTIVES
        )
        self._build_gate_bands()
        self.channel_scale = self._channel_scale(
            read_unit_targets(observation).n_channels
        )

    def _score_units(self, observation) -> None:
        sidecar = read_unit_targets(observation)
        block = sidecar.at(self.config.spec.scale)
        full = sidecar.require(sidecar, "features", self.UNIT_FEATURES)
        for name in self.UNIT_FEATURES:
            self._set_target(name, block.features.get(name, full[name]))
        if not block.channel_features:
            raise ValueError(
                f"{observation}'s unit targets have no `channel_features`; "
                "rewrite them from the sort"
            )
        for name, stat in block.channel_features.items():
            self._set_target(name, stat)

    def _set_target(self, name: str, stat: Stat) -> None:
        self._targets[name] = float(stat.median)
        self.feature_bands[name] = (float(stat.q_lo), float(stat.q_hi))

    def _deliver_evoked(self, observation) -> None:
        super()._deliver_evoked(observation)
        repeats = int(self.config.evoked_repeats)
        if self.stimulus is None or repeats <= 1:
            return
        from systems.targets.EI.base import Protocol
        from systems.targets.schema import read_target

        block = read_target(observation, "evoked")
        self.stimulus = Protocol.from_block(
            block, repeats=repeats, trial_ms=float(block.window_ms)
        )

    def _build_gate_bands(self) -> None:
        self.gate_bands: dict[str, tuple[float, float]] = {}
        for name in (*self.GATED_FEATURES, *self.GATED_RESPONSE):
            if name not in self.feature_bands:
                continue
            lo, hi = self.feature_bands[name]
            slack = max((hi - lo) / 2.0, feature(name).gate_slack)
            lo, hi = lo - slack, hi + slack
            if not feature(name).signed:
                lo = max(lo, 0.0)
            self.gate_bands[name] = (float(lo), float(hi))

    def objective_names(self) -> list[str]:
        return [n for n in self.SEARCH_OBJECTIVES if n not in self.skip_objectives]

    def _resting_gates(self) -> list[str]:
        return [n for n in self.GATED_FEATURES if n in self.gate_bands]

    def _response_gates(self) -> list[str]:
        if self.stimulus is None:
            return []
        return [
            "recruitment_miss",
            *(n for n in self.GATED_RESPONSE if n in self.gate_bands),
        ]

    def _all_constraint_names(self) -> list[str]:
        return [
            *super()._all_constraint_names(),
            *(f"{n}_band" for n in self._resting_gates()),
            *(
                n if n == "recruitment_miss" else f"{n}_band"
                for n in self._response_gates()
            ),
        ]

    def _gate(self, name: str) -> tuple[float, float]:
        value = float(self.metrics.get(name, float("nan")))
        return float(_band_constraint(value, *self.gate_bands[name])), value

    def compute_constraints(self, env) -> dict:
        result = super().compute_constraints(env)
        for name in self._resting_gates():
            key = f"{name}_band"
            if key not in self.skip_constraints:
                result[key] = self._gate(name)
        return result

    def _response_constraints(self) -> dict:
        result = {}
        for name in self._response_gates():
            if name == "recruitment_miss":
                value = float(self.metrics.get(name, float("nan")))
                gate = (float(_max_constraint(value, self.MAX_RECRUITMENT_MISS)), value)
                key = name
            else:
                gate, key = self._gate(name), f"{name}_band"
            if key not in self.skip_constraints:
                result[key] = gate
        return result

    def __call__(self, env, params=None, directory=None):
        self.record_resting(env)

        objectives = self.compute_objectives(env)
        constraints = self.compute_constraints(env)

        if self.stimulus is not None and self._admits_a_sweep(constraints):
            self.record_evoked(env)
            self._threshold_objective(env)
            self.metrics.update(self._response_shape(env))
        constraints.update(self._response_constraints())

        self._keep_spikes(env, params, directory, constraints)

        return objectives, constraints

    def observed_feature_names(self) -> list[str]:
        objectives = set(self.objective_names())
        names = [
            *super().observed_feature_names(),
            *self._resting_gates(),
            *self._response_gates(),
        ]
        return [n for n in dict.fromkeys(names) if n not in objectives]

    @property
    def noise_uv(self) -> float:
        cached = getattr(self, "_noise_uv", None)
        if cached is None:
            stated = read_unit_targets(self.config.observation).noise_uv
            if not stated:
                raise ValueError(
                    f"{self.config.observation}'s unit targets state no `noise_uv`"
                )
            cached = self._noise_uv = float(stated)
        return cached

    def readout_radius(self, profile: dict) -> float:
        threshold = self.THRESHOLD_SIGMA * self.noise_uv
        distances = np.asarray(profile["distance_um"], dtype=float)
        troughs = np.asarray(profile["trough_uv"], dtype=float)
        above = np.flatnonzero(troughs >= threshold)
        if not len(above):
            return float(distances[1])
        return float(distances[min(int(above[-1]) + 1, len(distances) - 1)])

    def io(self):
        mea = super().io()
        if mea is None:
            return None
        from importlib import import_module

        from livn.io import MEA

        path, options = self.model_spec()
        module, name = path.rsplit(".", 1)
        profile = getattr(import_module(module), name)(
            **options
        ).extracellular_profile()
        kwargs = mea.serialize()
        kwargs.pop("spike_profile", None)
        kwargs.update(
            output_radius=self.readout_radius(profile),
            noise_uv=self.noise_uv,
            threshold_sigma=float(self.THRESHOLD_SIGMA),
            dead_time_ms=float(self.DEAD_TIME_MS),
        )
        return MEA(**kwargs)

    def _units(self, env) -> dict[int, int]:
        key = id(env.io)
        cached = getattr(self, "_units_cache", None)
        if cached is None or cached[0] != key:
            units = env.io.strongest_units(
                env.active_neuron_coordinates(), env.recording_amplitudes()
            )
            self._units_cache = (key, units)
        return self._units_cache[1]

    def compute_objectives(self, env) -> dict:
        self._unit_readout = True
        try:
            return super().compute_objectives(env)
        finally:
            self._unit_readout = False

    def _readout(self, env, data):
        if not getattr(self, "_unit_readout", False):
            return super()._readout(env, data)
        units = self._units(env)
        ids = np.asarray(
            data.spike_ids if data.spike_ids is not None else [], dtype=np.int64
        )
        times = np.asarray(
            data.spike_times if data.spike_times is not None else [], dtype=np.float64
        )
        channel_of = {int(gid): int(channel) for channel, gid in units.items()}
        keep = np.isin(ids, list(channel_of))
        it = np.array([channel_of[int(g)] for g in ids[keep]], dtype=np.int64)
        tt = times[keep]
        order = np.argsort(tt, kind="stable")
        proxy = SimpleNamespace(
            comm=env.comm,
            system=SimpleNamespace(gids=sorted(units)),
            io=env.io,
            voltage_recording_dt=getattr(env, "voltage_recording_dt", None),
        )
        return proxy, data.add_spikes(it[order], tt[order])

    def system_spec(self):
        spec = super().system_spec()
        if self.config.spec.boundary is not None:
            return spec
        from livn.system import Monolayer

        x0, x1 = spec["kwargs"]["area_kwargs"]["x_range"]
        y0, y1 = spec["kwargs"]["area_kwargs"]["y_range"]
        ceiling = min(x1 - x0, y1 - y0) / float(Monolayer.MIN_EXTENT_IN_SIGMA)
        connectivity = spec["kwargs"].setdefault("connectivity", {})
        if float(connectivity.get("sigma", 0.0)) > ceiling:
            connectivity["sigma"] = float(np.floor(ceiling))
        return spec

    def _channel_scale(self, culture_channels: int) -> float | None:
        if float(self.config.spec.scale) == 1.0 or culture_channels <= 0:
            return None
        io = self.io()
        electrodes = 0 if io is None else len(io.electrode_coordinates)
        return culture_channels / electrodes if electrodes > 0 else None

    def _threshold_objective(self, network) -> tuple:
        result = super()._threshold_objective(network)
        scale = getattr(self, "channel_scale", None)
        simulated = self.metrics.get("threshold") or {}
        if not scale or not simulated.get("probabilities"):
            return result
        from livn.decoding import recruitment_threshold
        from systems.targets.EI.base import recruitment_miss, threshold_miss

        curve = {
            float(a): float(p) / scale
            for a, p in zip(
                simulated["amplitudes_mv"], simulated["probabilities"], strict=True
            )
        }
        counted = recruitment_threshold(
            curve, recruited=float(simulated.get("recruited", 0.5))
        )
        self.metrics["threshold"] = {**counted, "channel_scale": scale}
        miss = threshold_miss(self.stimulus_threshold, counted)
        curve_miss = recruitment_miss(self.stimulus_threshold, counted)
        self.metrics["threshold_miss"] = miss
        self.metrics["recruitment_miss"] = curve_miss
        return (1e3 if np.isnan(curve_miss) else float(curve_miss), miss)

    def _unit_metrics(self, network, data, duration: int) -> dict:
        if float(self.config.spec.scale) == 1.0:
            return super()._unit_metrics(network, data, duration)
        from livn.decoding import RecruitmentOrder, merged_spikes
        from livn.utils import P

        units = self._units(network)
        gids = np.asarray(sorted(set(units.values())), dtype=np.int64)
        comm = getattr(network, "comm", None)
        it, tt = merged_spikes(data, network)
        result = None
        if P.is_root(comm=comm):
            it = np.asarray(it if it is not None else [], dtype=np.int64)
            tt = np.asarray(tt if tt is not None else [], dtype=np.float64)
            start, count = network.system.population_ranges["EXC"]
            population = (it >= start) & (it < start + count)
            order = RecruitmentOrder(duration=duration)
            peaks = order.burst_peaks(tt[population], int(count))
            keep = np.isin(it, gids)
            result = order._decode(it[keep], tt[keep], peaks=peaks) or {}
        return P.broadcast(result, comm=comm)
