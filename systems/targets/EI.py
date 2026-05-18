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
    """Baseline target for spontaneous in-vitro dynamics.

    Optimizer-facing surface:
        Objectives (3, smooth & bounded):
            mfr               : log-error toward 1 Hz mean firing rate
            isi_cv            : quadratic toward target ~1.2 (mildly irregular)
            active_fraction   : (1 - active_fraction)^2, push every unit to fire

        Constraints:
            not_runaway / not_quiescent / is_stable      (Stability decoder)
            max_firing_rate                              (per-unit cap)
            min_mean_firing_rate / max_mean_firing_rate  (pop-mean band)
            synchrony                                    (mean-pairwise band)
            max_synchronous_peak                         (peak co-firing cap)
            active_fraction_floor                        (>= 50% units active)
            pop_autocorr_tau_band                        (10..500 ms band)
            burst_rate_cap                               (<= 0.2 Hz upper bound)
            branching_ratio_band                         (sigma in [0.7, 1.3])
            avalanche_r2                                 (>= 0.5 power-law gate)
    """

    DEFAULT_TARGETS = {
        "mfr": 1.0,
        "isi_cv": 1.2,
        "active_fraction": 1.0,
    }
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

    def __init__(
        self,
        targets: dict | None = None,
        duration: float = 30_000.0,
        warmup: float = 1000.0,
    ):
        self._targets = {**self.DEFAULT_TARGETS, **(targets or {})}
        self.recording_duration = duration + warmup
        self.warmup_duration = warmup
        self.min_spike_count_for_metrics = 150
        self._reset_state()

    def _reset_state(self):
        self.response_data: tuple | None = None
        self.metrics: dict = {}
        self.objectives: dict = {}

    def init(self, env):
        return with_progress_logging(env)

    def objective_names(self) -> list[str]:
        return list(self._targets.keys())

    def constraint_names(self) -> list[str]:
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

    def _weight_space(self, model) -> dict[str, list]:
        populations = ["EXC", "INH"]
        if model is not None and hasattr(model, "ignored_populations"):
            ignored = set(model.ignored_populations())
            populations = [p for p in populations if p not in ignored]

        default_ranges = {"EXC": [0.001, 10.0], "INH": [0.001, 8.0]}
        mechanism_ranges = {"NMDA": [0.001, 3.0]}

        weights = {}
        for pre in populations:
            for post in populations:
                if pre == "EXC":
                    sections = ["hillock"]
                    mechanisms = ["AMPA", "NMDA"]
                else:
                    sections = ["soma"]
                    mechanisms = ["GABA_A"]
                for section in sections:
                    for mechanism in mechanisms:
                        key = f"{pre}_{post}-{section}-{mechanism}-weight"
                        if mechanism in mechanism_ranges:
                            low, high = mechanism_ranges[mechanism]
                        else:
                            low, high = default_ranges.get(pre, default_ranges["EXC"])
                        weights[key] = [low, high, self.transform_log1p]

        return weights

    def _noise_space(self, model):
        return {
            "noise-g_e0": [1.0, 10.0],
            "noise-g_i0": [0.01, 1.5],
            "noise-std_e": [0.005, 0.5],
            "noise-std_i": [0.05, 0.4],
            "noise-tau_e": [1.0, 40.0],
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
            it, tt, iv, vv, im, mp = self.response_data
            return GatherAndMerge(
                duration=total_duration, voltages=False, membrane_currents=False
            )(env, it, tt, iv, vv, im, mp)

    def compute_objectives(self, env) -> dict:
        targets = self.targets()
        result: dict = {}
        d = int(self.recording_duration)

        recording_slice = Slice(
            start=self.warmup_duration,
            stop=self.warmup_duration + self.recording_duration,
        )
        recording_data = recording_slice(env, *self.response_data)
        it, _tt, _iv, _vv, _im, _mp = recording_data

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

        mfr_result = MeanFiringRate(duration=d)(env, *recording_data) or {}
        mfr = float(mfr_result.get("rate_hz", 0.0))
        self.metrics["mfr"] = mfr

        eps = 1e-3
        mfr_target = float(targets["mfr"])
        mfr_obj = float(np.log((max(mfr, 0.0) + eps) / (mfr_target + eps)) ** 2)
        result["mfr"] = (mfr_obj, mfr)

        stability_result = Stability(
            duration=d, tail_window=1000.0, max_rate_hz=20.0, min_rate_hz=0.05
        )(env, *recording_data)
        self.metrics["stability_result"] = stability_result
        self.metrics["is_stable"] = (
            bool(stability_result["is_stable"]) if stability_result else False
        )

        per_unit = PerUnitFiringRate(duration=d)(env, *recording_data) or {}
        self.metrics["per_unit_rates_hz"] = per_unit.get("per_unit_rates_hz", {})
        self.metrics["max_neuron_firing_rate"] = float(per_unit.get("max_rate_hz", 0.0))

        isi_result = (
            ISICV(duration=d, min_spikes_per_unit=5)(env, *recording_data) or {}
        )
        isi_cv = float(isi_result.get("isi_cv", 0.0))
        self.metrics["isi_cv"] = isi_cv
        self.metrics["isi_cv_n_units_used"] = int(isi_result.get("n_units_used", 0))

        isi_target = float(targets["isi_cv"])
        result["isi_cv"] = ((isi_cv - isi_target) ** 2, isi_cv)

        pop_metrics = (
            PopulationRateMetrics(duration=d, bin_size=100.0)(env, *recording_data)
            or {}
        )
        self.metrics["coefficient_of_variation"] = float(
            pop_metrics.get("coefficient_of_variation", 0.0)
        )
        self.metrics["fano_factor"] = float(pop_metrics.get("fano_factor", 0.0))

        tau_result = (
            PopulationAutocorrTau(duration=d, bin_size=10.0, max_lag=5000.0)(
                env, *recording_data
            )
            or {}
        )
        self.metrics["pop_autocorr_tau"] = float(
            tau_result.get("pop_autocorr_tau", 10.0)
        )

        if enough_spikes:
            corr_result = (
                PairwiseChannelCorrelation(duration=d, bin_size=10.0, min_units=2)(
                    env, *recording_data
                )
                or {}
            )
            self.metrics["mean_channel_correlation"] = float(
                corr_result.get("mean_pairwise_correlation", 0.0)
            )
        else:
            self.metrics["mean_channel_correlation"] = float("nan")

        peak_result = (
            PeakSynchrony(duration=d, bin_size=2.0)(env, *recording_data) or {}
        )
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
            )(env, *recording_data)
            or {}
        )
        self.metrics["burst_rate"] = float(burst_result.get("burst_rate_hz", 0.0))

        active_result = (
            ActiveFraction(duration=d, min_spikes=1)(env, *recording_data) or {}
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
            )(env, *recording_data)

        sigma = float((avalanche_result or {}).get("branching_ratio", 0.0) or 0.0)
        r2 = float((avalanche_result or {}).get("size_power_law_r2", 0.0) or 0.0)
        self.metrics["branching_ratio"] = sigma
        self.metrics["avalanche_r2"] = r2
        self.metrics["avalanche_result"] = avalanche_result

        self.objectives = result
        return result

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

        return result
