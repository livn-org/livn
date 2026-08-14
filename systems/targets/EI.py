from types import SimpleNamespace

import numpy as np

from livn.decoding import (
    ActiveFraction,
    AvalancheAnalysis,
    BurstRate,
    GatherAndMerge,
    ISICV,
    MeanFiringRate,
    PairwiseChannelCorrelation,
    PeakSynchrony,
    PerUnitFiringRate,
    PopulationAutocorrTau,
    PopulationRateMetrics,
    Slice,
    Stability,
)
from livn.utils import P
from livn.env.logging import with_progress_logging
from systems.targets.protocol import TuningTargets


def _max_constraint(value, max_val, scale=None):
    """+1 when value <= max_val, <0 otherwise. NaN -> -10."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -10.0
    v = float(value)
    s = max(abs(max_val), 1e-6) if scale is None else max(float(scale), 1e-6)
    if v <= max_val:
        return 1.0 + (max_val - v) / s
    return -1.0 - (v - max_val) / s


def _min_constraint(value, min_val, scale=None):
    """+1 when value >= min_val, <0 otherwise. NaN -> -10."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -10.0
    v = float(value)
    s = max(abs(min_val), 1e-6) if scale is None else max(float(scale), 1e-6)
    if v >= min_val:
        return 1.0 + (v - min_val) / s
    return -1.0 - (min_val - v) / s


def _band_constraint(value, lo, hi, edge_slope=2.0, inside_penalty=0.1):
    """Band feasibility constraint."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -2.0
    v = float(value)
    if v < lo:
        return -((lo - v) / max(abs(lo), 1e-6)) * edge_slope
    if v > hi:
        return -((v - hi) / max(abs(hi), 1e-6)) * edge_slope
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return 1.0 - (abs(v - center) / max(half, 1e-6)) * inside_penalty


class Spontaneous(TuningTargets):
    MAX_NEURON_RATE_HZ = 50.0
    MIN_MEAN_RATE_HZ = 0.2
    MAX_MEAN_RATE_HZ = 15.0
    SYNCHRONY_BAND = (0.02, 0.25)
    MAX_SYNC_PEAK = 0.2
    MIN_ACTIVE_FRACTION = 0.5
    POP_TAU_BAND_MS = (10.0, 500.0)
    MAX_BURST_RATE_HZ = 0.2
    BRANCHING_RATIO_BAND = (0.5, 1.5)
    MIN_AVALANCHE_R2 = 0.5
    MAX_POP_RATE_PER_UNIT_HZ = 20.0
    MIN_POP_RATE_PER_UNIT_HZ = 0.05
    STABILITY_MARGIN = 5.0

    def __init__(
        self,
        targets: dict | None = None,
        duration: float = 30_000.0,
        warmup: float = 1000.0,
        overrides: dict | None = None,
        skip_objectives: tuple = (),
        skip_constraints: tuple = (),
        readout: str = "neurons",
        system: str | None = None,
        section_aliases: dict | None = None,
        feature_bands: dict | None = None,
        mea: dict | None = None,
    ):
        self._targets = {
            "mfr": 1.0,
            "isi_cv": 1.2,
            "active_fraction": 1.0,
            **(targets or {}),
        }
        self.system = system
        self.section_aliases = dict(section_aliases or {})
        self.feature_bands = {
            name: (float(lo), float(hi))
            for name, (lo, hi) in (feature_bands or {}).items()
        }
        self.mea = mea
        self.skip_objectives = tuple(skip_objectives)
        self.skip_constraints = tuple(skip_constraints)
        self.readout = readout

        for name, value in (overrides or {}).items():
            if value is None:
                continue
            if name == "targets":
                self._targets = {**self._targets, **value}
            elif hasattr(self, name):
                setattr(self, name, tuple(value) if isinstance(value, list) else value)
            else:
                raise ValueError(f"Unknown override: {name}")

        self.recording_duration = duration + warmup
        self.warmup_duration = warmup
        self.min_spike_count_for_metrics = 150
        self._env = None
        self._weight_space_cache: dict[str, list] | None = None
        self._reset_state()

    def _reset_state(self):
        self.response_data: tuple | None = None
        self.metrics: dict = {}
        self.objectives: dict = {}

    def io(self):
        if not self.mea:
            return None
        from livn.io import MEA

        return MEA.from_json(self.mea)

    def init(self, env):
        if self.readout == "channels" and not len(getattr(env.io, "channel_ids", ())):
            raise RuntimeError("readout='channels' needs an `mea`.")
        self._env = env
        return with_progress_logging(env)

    def objective_names(self) -> list[str]:
        return [n for n in self._targets if n not in self.skip_objectives]

    def constraint_names(self) -> list[str]:
        return [
            name
            for name in self._all_constraint_names()
            if name not in self.skip_constraints
        ]

    def _all_constraint_names(self) -> list[str]:
        return [
            "not_runaway",
            "not_quiescent",
            "is_stable",
            "max_firing_rate",
            "synchrony",
            "max_synchronous_peak",
            "min_mean_firing_rate",
            "max_mean_firing_rate",
            "active_fraction_floor",
            "pop_autocorr_tau_band",
            "burst_rate_cap",
            "branching_ratio_band",
            "avalanche_r2",
        ]

    def targets(self) -> dict[str, float]:
        return self._targets.copy()

    def rank_solutions(self, best):
        y = best.get("y")
        if y is None or len(y) == 0:
            return best

        span = (y.max() - y.min()).replace(0.0, 1.0)
        worst = ((y - y.min()) / span).max(axis=1)

        f = best.get("f")
        outside = np.zeros(len(y), dtype=int)
        if f is not None and self.feature_bands:
            for name, (lo, hi) in self.feature_bands.items():
                if name in getattr(f, "columns", []):
                    outside += (~f[name].between(lo, hi)).to_numpy().astype(int)

        c = best.get("c")
        infeasible = np.zeros(len(y), dtype=int)
        if c is not None and len(getattr(c, "columns", [])):
            infeasible = (c.to_numpy() < 0).sum(axis=1).astype(int)

        order = np.lexsort((worst.to_numpy(), infeasible, outside))
        ranked = {}
        for key, value in best.items():
            if hasattr(value, "iloc") and len(value) == len(y):
                ranked[key] = value.iloc[order].reset_index(drop=True)
            elif isinstance(value, np.ndarray) and len(value) == len(y):
                ranked[key] = value[order]
            else:
                ranked[key] = value
        return ranked

    def _env_for_naming(self, model):
        if self._env is not None:
            return self._env
        if not self.system and model is None:
            return None

        from livn.env import Env

        return Env(self.system or 1, model=model)

    def _section_names(self, env, population: str, section: str) -> str:
        if section in self.section_aliases:
            return self.section_aliases[section]
        if env is None:
            return section
        return env.destination_sections().get(population, {}).get(section, section)

    def _weight_space(self, model) -> dict[str, list]:
        if self._weight_space_cache is not None:
            return self._weight_space_cache

        env = self._env_for_naming(model)

        system = getattr(env, "system", None) if env is not None else None
        if system is None and self.system:
            from livn.system import resolve

            system = resolve(self.system)

        found = None
        if system is not None:
            try:
                found = system.synapse_projections()
            except (OSError, KeyError, TypeError, ValueError, AttributeError):
                found = None
        if found:
            found = [
                (post, pre, self._section_names(env, post, section), mech, syn_type)
                for post, pre, section, mech, syn_type in found
            ]
        if not found:
            populations = ["EXC", "INH"]
            if model is not None and hasattr(model, "ignored_populations"):
                ignored = set(model.ignored_populations())
                populations = [p for p in populations if p not in ignored]
            found = [
                (
                    post,
                    pre,
                    self._section_names(env, post, "soma" if pre == "INH" else "dend"),
                    mechanism,
                    "inhibitory" if pre == "INH" else "excitatory",
                )
                for pre in populations
                for post in populations
                for mechanism in (["GABA_A"] if pre == "INH" else ["AMPA", "NMDA"])
            ]

        ignored = set()
        if model is not None and hasattr(model, "ignored_populations"):
            ignored = set(model.ignored_populations())

        depressing = bool(getattr(model, "short_term_depression", False))

        default_ranges = {
            "excitatory": [0.001, 1000.0],
            "inhibitory": [0.001, 800.0],
        }
        mechanism_ranges = {"NMDA": [0.001, 300.0]}
        depression_ranges = {
            "tau_rec": [50.0, 3000.0],
            # burst period the mechanism can impose
            "U": [0.05, 0.9],
            # how fast resources deplete, i.e. how many spikes a burst lasts
        }

        weights = {}
        for post, pre, section, mechanism, syn_type in found:
            if pre in ignored or post in ignored:
                continue
            low, high = mechanism_ranges.get(
                mechanism, default_ranges.get(syn_type, [0.001, 10.0])
            )
            weights[f"{post}_{pre}-{section}-{mechanism}-weight"] = [
                low,
                high,
                self.transform_log1p,
            ]

            if depressing and mechanism == "AMPA":
                for name, (dlo, dhi) in depression_ranges.items():
                    key = f"{post}-{section}-{mechanism}-{name}"
                    if key in weights:
                        continue
                    bounds = [dlo, dhi]
                    if dhi / dlo >= 10.0:
                        bounds.append(self.transform_log10)
                    weights[key] = bounds

        self._weight_space_cache = weights
        return weights

    def _noise_space(self, model):
        return {
            "noise-g_e0": [0.01, 1.0, self.transform_log10],
            "noise-g_i0": [0.01, 1.5, self.transform_log10],
            "noise-std_e": [0.005, 0.5, self.transform_log10],
            "noise-std_i": [0.05, 0.4],
            "noise-tau_e": [1.0, 40.0, self.transform_log10],
            "noise-tau_i": [4.0, 20.0],
        }

    def __call__(self, env):
        self.record(env)
        return self.compute_objectives(env), self.compute_constraints(env)

    def record(self, env, return_data=False):
        self._reset_state()
        total_duration = int(self.warmup_duration + self.recording_duration)

        env.record_spikes()
        self.response_data = env.run(total_duration, root_only=False)

        if return_data:
            return GatherAndMerge(
                duration=total_duration, voltages=False, membrane_currents=False
            )(self.response_data, env)

    def compute_objectives(self, env) -> dict:
        targets = self.targets()
        result: dict = {}
        d = int(self.recording_duration)

        recording_slice = Slice(
            start=self.warmup_duration,
            stop=self.warmup_duration + self.recording_duration,
        )
        recording_data = recording_slice(self.response_data)
        env, recording_data = self._readout(env, recording_data)
        it = recording_data.spike_ids

        local_count = int(len(it) if it is not None else 0)
        total_spike_count = P.reduce_sum(
            np.array(local_count, dtype=np.int64), comm=env.comm, all=True
        )
        total_spike_count = int(
            getattr(total_spike_count, "item", lambda: total_spike_count)()
        )
        self.metrics["total_spikes"] = total_spike_count
        enough_spikes = total_spike_count >= self.min_spike_count_for_metrics
        self.metrics["enough_spikes_for_network_metrics"] = enough_spikes

        mfr_result = MeanFiringRate(duration=d)(recording_data, env) or {}
        mfr = float(mfr_result.get("rate_hz", 0.0))
        self.metrics["mfr"] = mfr

        eps = 1e-3
        mfr_target = float(targets["mfr"])
        mfr_obj = float(np.log((max(mfr, 0.0) + eps) / (mfr_target + eps)) ** 2)
        result["mfr"] = (mfr_obj, mfr)

        gids = getattr(env.system, "gids", None)
        n_units = max(len(gids) if gids is not None else 0, 1)
        stability_result = Stability(
            duration=d,
            tail_window=1000.0,
            max_rate_hz=self.MAX_POP_RATE_PER_UNIT_HZ * n_units * self.STABILITY_MARGIN,
            min_rate_hz=self.MIN_POP_RATE_PER_UNIT_HZ * n_units / self.STABILITY_MARGIN,
        )(recording_data, env)
        self.metrics["stability_result"] = stability_result
        self.metrics["is_stable"] = (
            bool(stability_result["is_stable"]) if stability_result else False
        )

        pop_rate = float((stability_result or {}).get("global_mean_hz", 0.0))
        self.metrics["pop_rate_hz"] = pop_rate
        self.metrics["pop_rate_per_unit_hz"] = pop_rate / n_units

        per_unit = PerUnitFiringRate(duration=d)(recording_data, env) or {}
        self.metrics["per_unit_rates_hz"] = per_unit.get("per_unit_rates_hz", {})
        self.metrics["max_neuron_firing_rate"] = float(per_unit.get("max_rate_hz", 0.0))

        isi_result = ISICV(duration=d, min_spikes_per_unit=5)(recording_data, env) or {}
        isi_cv = float(isi_result.get("isi_cv", 0.0))
        self.metrics["isi_cv"] = isi_cv
        self.metrics["isi_cv_n_units_used"] = int(isi_result.get("n_units_used", 0))

        isi_target = float(targets["isi_cv"])
        result["isi_cv"] = ((isi_cv - isi_target) ** 2, isi_cv)

        pop_metrics = (
            PopulationRateMetrics(duration=d, bin_size=100.0)(recording_data, env) or {}
        )
        self.metrics["coefficient_of_variation"] = float(
            pop_metrics.get("coefficient_of_variation", 0.0)
        )
        self.metrics["fano_factor"] = float(pop_metrics.get("fano_factor", 0.0))

        tau_result = (
            PopulationAutocorrTau(duration=d, bin_size=10.0, max_lag=5000.0)(
                recording_data, env
            )
            or {}
        )
        self.metrics["pop_autocorr_tau"] = float(
            tau_result.get("pop_autocorr_tau", 10.0)
        )

        if enough_spikes:
            corr_result = (
                PairwiseChannelCorrelation(duration=d, bin_size=10.0, min_units=2)(
                    recording_data, env
                )
                or {}
            )
            self.metrics["mean_channel_correlation"] = float(
                corr_result.get("mean_pairwise_correlation", 0.0)
            )
        else:
            self.metrics["mean_channel_correlation"] = float("nan")

        peak_result = PeakSynchrony(duration=d, bin_size=2.0)(recording_data, env) or {}
        self.metrics["max_synchronous_peak"] = float(
            peak_result.get("max_synchronous_peak", 0.0)
        )

        burst_result = (
            BurstRate(
                duration=d,
                bin_size=50.0,
                mad_k=3.0,
                min_floor_fraction=0.10,
                min_floor=2.0,
            )(recording_data, env)
            or {}
        )
        self.metrics["burst_rate"] = float(burst_result.get("burst_rate_hz", 0.0))

        active_result = (
            ActiveFraction(duration=d, min_spikes=1)(recording_data, env) or {}
        )
        active_fraction = float(active_result.get("active_fraction", 0.0))
        self.metrics["active_fraction"] = active_fraction

        af_target = float(targets["active_fraction"])
        af_obj = (af_target - active_fraction) ** 2
        result["active_fraction"] = (af_obj, active_fraction)

        avalanche_result = None
        if total_spike_count > 0:
            n_bins_target = max(50, total_spike_count // 15)
            adaptive_bin_width = max(4.0, min(d / n_bins_target, 50.0))
            avalanche_result = AvalancheAnalysis(
                duration=d, bin_width=adaptive_bin_width
            )(recording_data, env)

        sigma = float((avalanche_result or {}).get("branching_ratio", 0.0) or 0.0)
        r2 = float((avalanche_result or {}).get("size_power_law_r2", 0.0) or 0.0)
        self.metrics["branching_ratio"] = sigma
        self.metrics["avalanche_r2"] = r2
        self.metrics["avalanche_result"] = avalanche_result

        result = {
            name: value
            for name, value in result.items()
            if name not in self.skip_objectives
        }
        self.objectives = result
        return result

    def _readout(self, env, data):
        if self.readout != "channels":
            return env, data

        _, per_channel = env.channel_recording(data.spike_ids, data.spike_times)
        if per_channel:
            it = np.concatenate(
                [np.full(len(t), c, dtype=np.int64) for c, t in per_channel.items()]
            )
            tt = np.concatenate([np.asarray(t) for t in per_channel.values()])
            order = np.argsort(tt, kind="stable")
            it, tt = it[order], tt[order]
        else:
            it, tt = np.array([], dtype=np.int64), np.array([])

        proxy = SimpleNamespace(
            comm=env.comm,
            system=SimpleNamespace(gids=list(env.io.channel_ids)),
            io=env.io,
            voltage_recording_dt=getattr(env, "voltage_recording_dt", None),
        )
        return proxy, data.add_spikes(it, tt)

    def compute_constraints(self, env) -> dict:
        result: dict = {}
        m = self.metrics
        stability_result = m.get("stability_result")

        if stability_result:
            tail_mean = stability_result["tail_mean_hz"]
            max_rate = stability_result.get("max_rate_hz", 20.0)
            min_rate = stability_result.get("min_rate_hz", 0.05)

            if stability_result["is_runaway"]:
                runaway_c = -1.0 - (tail_mean - max_rate) / 10.0
            else:
                runaway_c = 1.0 + (max_rate - tail_mean) / 10.0
            if stability_result["is_quiescent"]:
                quiescent_c = -1.0 - (min_rate - tail_mean) / 0.1
            else:
                quiescent_c = 1.0 + (tail_mean - min_rate) / 0.1

            result["not_runaway"] = (float(runaway_c), float(tail_mean))
            result["not_quiescent"] = (float(quiescent_c), float(tail_mean))
            result["is_stable"] = (
                1.0 if stability_result["is_stable"] else -1.0,
                float(stability_result["is_stable"]),
            )
        else:
            result["not_runaway"] = (-10.0, 0.0)
            result["not_quiescent"] = (-10.0, 0.0)
            result["is_stable"] = (-10.0, 0.0)

        max_neuron_rate = m.get("max_neuron_firing_rate", float("nan"))
        result["max_firing_rate"] = (
            float(_max_constraint(max_neuron_rate, self.MAX_NEURON_RATE_HZ)),
            float(max_neuron_rate),
        )

        mean_sync = m.get("mean_channel_correlation", float("nan"))
        try:
            mean_sync_f = float(np.clip(float(mean_sync), -1.0, 1.0))
        except (TypeError, ValueError):
            mean_sync_f = float("nan")
        sync_c = _band_constraint(
            mean_sync_f, self.SYNCHRONY_BAND[0], self.SYNCHRONY_BAND[1]
        )
        result["synchrony"] = (
            float(np.clip(sync_c, -10.0, 10.0)),
            mean_sync_f,
        )

        peak_sync = m.get("max_synchronous_peak", float("nan"))
        if peak_sync is None or (isinstance(peak_sync, float) and np.isnan(peak_sync)):
            peak_c = -1.0
        elif peak_sync <= self.MAX_SYNC_PEAK:
            peak_c = 1.0 + (self.MAX_SYNC_PEAK - peak_sync)
        else:
            peak_c = -1.0 - (peak_sync - self.MAX_SYNC_PEAK) * 10.0
        result["max_synchronous_peak"] = (float(peak_c), float(peak_sync))

        mean_rate = m.get("mfr", float("nan"))
        result["min_mean_firing_rate"] = (
            float(_min_constraint(mean_rate, self.MIN_MEAN_RATE_HZ)),
            float(mean_rate),
        )
        result["max_mean_firing_rate"] = (
            float(_max_constraint(mean_rate, self.MAX_MEAN_RATE_HZ)),
            float(mean_rate),
        )

        active_fraction = m.get("active_fraction", float("nan"))
        result["active_fraction_floor"] = (
            float(
                _min_constraint(active_fraction, self.MIN_ACTIVE_FRACTION, scale=1.0)
            ),
            float(active_fraction),
        )

        pop_tau = m.get("pop_autocorr_tau", float("nan"))
        result["pop_autocorr_tau_band"] = (
            float(_band_constraint(pop_tau, *self.POP_TAU_BAND_MS)),
            float(pop_tau),
        )

        burst_rate = m.get("burst_rate", float("nan"))
        result["burst_rate_cap"] = (
            float(
                _max_constraint(
                    burst_rate, self.MAX_BURST_RATE_HZ, scale=self.MAX_BURST_RATE_HZ
                )
            ),
            float(burst_rate),
        )

        sigma = m.get("branching_ratio", float("nan"))
        result["branching_ratio_band"] = (
            float(_band_constraint(sigma, *self.BRANCHING_RATIO_BAND)),
            float(sigma),
        )

        avalanche_r2 = m.get("avalanche_r2", float("nan"))
        result["avalanche_r2"] = (
            float(_min_constraint(avalanche_r2, self.MIN_AVALANCHE_R2, scale=1.0)),
            float(avalanche_r2),
        )

        return {
            name: value
            for name, value in result.items()
            if name not in self.skip_constraints
        }

    def describe_params(self, decoded):
        env_params = self.set_params(dict(decoded))
        weights = {
            k: v
            for k, v in env_params.items()
            if "-weight" in k and not k.startswith("noise-")
        }
        noise_keys = {"std_e", "std_i", "g_e0", "g_i0", "tau_e", "tau_i"}
        noise = {
            k.replace("noise-", "", 1): v
            for k, v in env_params.items()
            if k.startswith("noise-") and k.replace("noise-", "", 1) in noise_keys
        }
        protocol = {k: v for k, v in decoded.items() if k not in env_params}
        return {
            "All decoded params": dict(decoded),
            "Weights (neuron_default_weights)": weights,
            "Noise (neuron_default_noise)": noise,
            "Protocol-specific params": protocol,
        }
