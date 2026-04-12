import numpy as np

from livn.decoding import (
    GatherAndMerge,
    Slice,
    MeanFiringRate,
    Stability,
    LFP,
    AvalancheAnalysis,
)
from livn.utils import P
from systems.targets.protocol import TuningTargets


class Spontaneous(TuningTargets):
    DEFAULT_TARGETS = {
        "mfr": 1.0,  # Mean Firing Rate (Hz): Baseline activity ~0.5-1.5 Hz typical for organoids
        "lfp_delta": 25.0,  # LFP Power in Delta band (1-4 Hz) - should dominate
        "lfp_theta": 18.0,  # LFP Power in Theta band (4-8 Hz)
        "lfp_alpha": 15.0,  # LFP Power in Alpha band (8-13 Hz)
        "lfp_beta": 10.0,  # LFP Power in Beta band (13-30 Hz)
        "lfp_gamma_max": 0.25,  # Cap relative gamma power (fraction of broadband)
        "delta_theta_ratio": 1.6,  # Delta/Theta ratio (target: delta > theta)
        "spectral_slope": -1.5,  # 1/f spectral slope (target: -1 to -2)
        "burst_rate": 0.1,  # Network bursts per second (~1 every 10s)
        "burst_participation": 0.4,  # Fraction of neurons participating in bursts
        "branching_ratio": 1.0,  # Criticality: sigma ~ 1.0
        "avalanche_power_law": 0.6,  # Criticality: looser R^2 target while tuning
    }

    def __init__(
        self,
        targets: dict | None = None,
        duration: float = 30000.0,
        warmup: float = 2000.0,
    ):
        super().__init__()
        self._targets = {**self.DEFAULT_TARGETS, **(targets or {})}
        self.max_firing_rate_limit = 50.0
        self.min_mean_firing_rate_limit = 0.2
        self.max_mean_firing_rate_limit = 15.0
        self.min_spike_count_for_metrics = 150
        self.recording_duration = duration
        self.warmup_duration = warmup

        self._reset_state()

    def _reset_state(self):
        self.response_data: tuple | None = None

        self.channel_spikes = {}
        self.channel_times = {}
        self.voltage = None  # [n_neurons, T]
        self.metrics = {}
        self.objectives = {}  # (objective_value, feature_value)

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
        ]

    def targets(self) -> dict[str, float]:
        return self._targets.copy()

    def _weight_space(self, populations: list[str] | None = None) -> dict[str, list]:
        if populations is None:
            populations = ["EXC", "INH"]

        default_ranges = {
            "EXC": [0.001, 20.0],
            "INH": [0.001, 12.0],
        }
        mechanism_ranges = {
            "NMDA": [0.001, 3.0],
        }

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

    def _noise_space(self) -> dict[str, list]:
        return {
            "noise-g_e0": [1.0, 5.0],
            "noise-g_i0": [0.005, 2.0],
            "noise-std_e": [0.005, 0.5],
            "noise-std_i": [0.001, 0.4],
            "noise-tau_e": [1.0, 40.0],
            "noise-tau_i": [4.0, 30.0],
        }

    def __call__(self, env):
        self.record(env)

        return self.compute_objectives(env), self.compute_constraints(env)

    def record(self, env, return_data=False):
        self._reset_state()
        comm = getattr(env, "comm", None)

        total_duration = int(self.warmup_duration + self.recording_duration)

        env.record_spikes()
        env.record_voltage()
        env.record_membrane_current(dt=0.5)

        self.response_data = env.run(total_duration, root_only=False)

        it, tt, iv, vv, im, mp = self.response_data

        cit, ct = env.channel_recording(it, tt)
        cit, ct, iv, vv = P.gather(cit, ct, iv, vv, comm=comm)

        if P.is_root(comm=comm):
            cit, ct, iv, vv = P.merge(cit, ct, iv, vv)
            self.channel_spikes = dict(cit)
            self.channel_times = dict(ct)
            self.voltage = vv

        if return_data:
            return GatherAndMerge(duration=total_duration)(env, it, tt, iv, vv, im, mp)

    def compute_objectives(self, env) -> dict:
        targets = self.targets()
        result = {}

        recording_slice = Slice(
            start=self.warmup_duration,
            stop=self.warmup_duration + self.recording_duration,
        )
        recording_data = recording_slice(env, *self.response_data)
        it, tt, iv, vv, im, mp = recording_data

        local_spike_count = int(len(it) if it is not None else 0)
        total_spike_count = P.reduce_sum(
            np.array(local_spike_count, dtype=np.int64), comm=env.comm, all=True
        )
        if hasattr(total_spike_count, "item"):
            total_spike_count = int(total_spike_count.item())
        else:
            total_spike_count = int(total_spike_count)
        self.metrics["total_spikes"] = total_spike_count
        enough_spikes = total_spike_count >= self.min_spike_count_for_metrics
        self.metrics["enough_spikes_for_network_metrics"] = enough_spikes

        gathered_it, gathered_tt = P.gather(it, tt, comm=env.comm)
        if P.is_root(comm=env.comm):
            global_spike_indices, global_spike_times = P.merge(gathered_it, gathered_tt)
            if global_spike_indices is not None:
                global_spike_indices = np.asarray(global_spike_indices)
            if global_spike_times is not None:
                global_spike_times = np.asarray(global_spike_times)
        else:
            global_spike_indices = None
            global_spike_times = None

        mfr_result = MeanFiringRate(duration=self.recording_duration)(
            env, *recording_data
        )
        mfr = mfr_result["rate_hz"] if mfr_result else 0.0
        self.metrics["mfr"] = mfr

        pathology_penalty = 0.0
        if 0.0 <= mfr < self.min_mean_firing_rate_limit:
            pathology_penalty += (self.min_mean_firing_rate_limit - mfr) ** 2 * 50.0
        elif mfr > 3.0:
            pathology_penalty += (mfr - 3.0) ** 2 * 10.0

        mfr_target = targets["mfr"]

        if mfr < self.min_mean_firing_rate_limit:
            mfr_objective = (0.3 - mfr) ** 2 * 3.0 + (mfr - mfr_target) ** 2
        elif mfr > self.max_mean_firing_rate_limit:
            mfr_objective = (mfr - self.max_mean_firing_rate_limit) ** 2 * 3.0 + (
                mfr - mfr_target
            ) ** 2
        else:
            mfr_objective = (mfr - mfr_target) ** 2
        result["mfr"] = (mfr_objective, mfr)

        stability_result = Stability(
            duration=self.recording_duration,
            tail_window=1000,
            max_rate_hz=20.0,
            min_rate_hz=0.05,
        )(env, *recording_data)

        if stability_result:
            is_stable = stability_result["is_stable"]
        else:
            is_stable = False

        self.metrics["is_stable"] = is_stable
        self.metrics["stability_result"] = stability_result

        lfp_result = LFP(
            duration=self.recording_duration,
            downsample_hz=1000,
            lowpass_hz=100,
            compute_band_power={
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 80.0),
            },
        )(env, *recording_data)

        if lfp_result and lfp_result.get("n_channels", 0) > 0:
            band_power_result = lfp_result

            delta_power = band_power_result.get("delta", 0.0)
            theta_power = band_power_result.get("theta", 0.0)
            alpha_power = band_power_result.get("alpha", 0.0)
            beta_power = band_power_result.get("beta", 0.0)
            gamma_power = band_power_result.get("gamma", 0.0)

            delta_target = targets.get("lfp_delta")
            theta_target = targets.get("lfp_theta")
            alpha_target = targets.get("lfp_alpha")
            beta_target = targets.get("lfp_beta")
            gamma_target = targets.get("lfp_gamma_max")

            broadband = band_power_result.get("broadband", 1.0)
            if broadband > 0:
                # Relative powers
                delta_relative = delta_power / broadband
                theta_relative = theta_power / broadband
                alpha_relative = alpha_power / broadband
                beta_relative = beta_power / broadband
                gamma_relative = gamma_power / broadband

                eps = 1e-6
                band_weights = {"delta": 1.0, "theta": 1.0, "alpha": 0.8, "beta": 0.8}

                def _rel_log_objective(
                    rel_pct: float, target_pct: float, weight: float = 1.0
                ):
                    rel_pct = max(rel_pct, eps)
                    target_pct = max(target_pct, eps)
                    err = np.log(rel_pct / target_pct)
                    return (err**2) * weight

                if delta_target is not None:
                    rel_pct = delta_relative * 100
                    result["lfp_delta"] = (
                        _rel_log_objective(
                            rel_pct, delta_target, band_weights["delta"]
                        ),
                        rel_pct,
                    )

                if theta_target is not None:
                    rel_pct = theta_relative * 100
                    result["lfp_theta"] = (
                        _rel_log_objective(
                            rel_pct, theta_target, band_weights["theta"]
                        ),
                        rel_pct,
                    )

                if alpha_target is not None:
                    rel_pct = alpha_relative * 100
                    result["lfp_alpha"] = (
                        _rel_log_objective(
                            rel_pct, alpha_target, band_weights["alpha"]
                        ),
                        rel_pct,
                    )

                if beta_target is not None:
                    rel_pct = beta_relative * 100
                    result["lfp_beta"] = (
                        _rel_log_objective(rel_pct, beta_target, band_weights["beta"]),
                        rel_pct,
                    )

                if gamma_target is not None:
                    gamma_obj = 0.0
                    if gamma_relative > gamma_target:
                        gamma_obj = (gamma_relative - gamma_target) ** 2 * 50.0
                    result["lfp_gamma_max"] = (gamma_obj, float(gamma_relative))

                if targets.get("delta_theta_ratio") is not None:
                    if theta_relative > 0:
                        actual_ratio = delta_relative / theta_relative
                    else:
                        actual_ratio = 10.0 if delta_relative > 0 else 0.0

                    ratio_target = targets["delta_theta_ratio"]
                    if actual_ratio < 1.0:
                        ratio_objective = (actual_ratio - ratio_target) ** 2 * 15.0
                    else:
                        ratio_objective = (actual_ratio - ratio_target) ** 2

                    result["delta_theta_ratio"] = (ratio_objective, actual_ratio)
                    self.metrics["delta_theta_ratio"] = actual_ratio

                if targets.get("spectral_slope") is not None:
                    band_freqs = [2.5, 6.0, 10.5, 21.5, 55.0]
                    band_powers = [
                        delta_relative,
                        theta_relative,
                        alpha_relative,
                        beta_relative,
                        gamma_relative,
                    ]

                    valid_pairs = [
                        (f, p) for f, p in zip(band_freqs, band_powers) if p > 1e-10
                    ]

                    if len(valid_pairs) >= 3:
                        freqs = np.array([f for f, _ in valid_pairs])
                        powers = np.array([p for _, p in valid_pairs])

                        log_freqs = np.log10(freqs)
                        log_powers = np.log10(powers)
                        slope, _ = np.polyfit(log_freqs, log_powers, 1)

                        slope_target = targets["spectral_slope"]
                        slope_objective = (slope - slope_target) ** 2 * 2.0

                        result["spectral_slope"] = (slope_objective, slope)
                        self.metrics["spectral_slope"] = slope
                    else:
                        result["spectral_slope"] = (20.0, 0.0)
                        self.metrics["spectral_slope"] = None

            else:
                for key, target in [
                    ("lfp_delta", delta_target),
                    ("lfp_theta", theta_target),
                    ("lfp_alpha", alpha_target),
                    ("lfp_beta", beta_target),
                ]:
                    if target is not None:
                        result[key] = (200.0, 0.0)

                if gamma_target is not None:
                    result["lfp_gamma_max"] = (200.0, 0.0)
                if targets.get("delta_theta_ratio") is not None:
                    result["delta_theta_ratio"] = (200.0, 0.0)
                if targets.get("spectral_slope") is not None:
                    result["spectral_slope"] = (200.0, 0.0)

                delta_relative = 0.0
                theta_relative = 0.0
        else:
            for key in [
                "lfp_delta",
                "lfp_theta",
                "lfp_alpha",
                "lfp_beta",
                "lfp_gamma_max",
                "delta_theta_ratio",
                "spectral_slope",
            ]:
                if key in targets:
                    result[key] = (1000.0, 0.0)
            band_power_result = None

        self.metrics["lfp_result"] = lfp_result
        self.metrics["band_power"] = band_power_result

        channel_times = self.channel_times
        mean_correlation_value = None

        if enough_spikes:
            if P.is_root(comm=env.comm):
                if len(channel_times) > 1:
                    try:
                        bin_width = 20
                        n_bins = int(self.recording_duration / bin_width)
                        spike_matrix = []

                        for channel_id in sorted(channel_times.keys()):
                            times = np.array(channel_times[channel_id])
                            valid_times = times[
                                (times >= self.warmup_duration)
                                & (
                                    times
                                    < self.warmup_duration + self.recording_duration
                                )
                            ]
                            valid_times = valid_times - self.warmup_duration

                            if len(valid_times) > 0:
                                counts, _ = np.histogram(
                                    valid_times,
                                    bins=n_bins,
                                    range=(0, self.recording_duration),
                                )
                                spike_matrix.append(counts)

                        if len(spike_matrix) > 1:
                            spike_matrix = np.array(spike_matrix)
                            correlations = []
                            for i in range(spike_matrix.shape[0]):
                                for j in range(i + 1, spike_matrix.shape[0]):
                                    std_i = np.std(spike_matrix[i])
                                    std_j = np.std(spike_matrix[j])
                                    if std_i > 0 and std_j > 0:
                                        mean_i = np.mean(spike_matrix[i])
                                        mean_j = np.mean(spike_matrix[j])
                                        cov = np.mean(
                                            (spike_matrix[i] - mean_i)
                                            * (spike_matrix[j] - mean_j)
                                        )
                                        corr = cov / (std_i * std_j)
                                        correlations.append(corr)

                            if correlations:
                                mean_correlation_value = float(np.mean(correlations))
                            else:
                                mean_correlation_value = 0.0
                        else:
                            mean_correlation_value = 0.0
                    except Exception:
                        mean_correlation_value = None
                else:
                    mean_correlation_value = 0.0
        else:
            if P.is_root(comm=env.comm):
                mean_correlation_value = np.nan

        mean_correlation_value = P.broadcast(mean_correlation_value, comm=env.comm)
        if mean_correlation_value is None:
            mean_correlation_value = np.nan
        self.metrics["mean_channel_correlation"] = mean_correlation_value

        # Calculate population peak synchrony (fraction of active neurons firing in 2ms bin)
        peak_sync_val = None
        if P.is_root(comm=env.comm):
            if (
                global_spike_times is not None
                and global_spike_indices is not None
                and len(global_spike_times) > 0
            ):
                n_active_neurons = len(np.unique(global_spike_indices))
                # Use 2 ms bin for fine-grained synchrony detection
                sync_bin_width = 2.0
                n_sync_bins = int(self.recording_duration / sync_bin_width)
                if n_sync_bins > 0:
                    pop_counts, _ = np.histogram(
                        global_spike_times,
                        bins=n_sync_bins,
                        range=(0, self.recording_duration),
                    )
                    max_pop_count = float(np.max(pop_counts))
                    # Fraction of ACTIVE neurons firing in the same bin
                    if n_active_neurons > 0:
                        peak_sync_val = max_pop_count / n_active_neurons
                    else:
                        peak_sync_val = 0.0
                else:
                    peak_sync_val = 0.0
            else:
                peak_sync_val = 0.0

        peak_sync_val = P.broadcast(peak_sync_val, comm=env.comm)
        if peak_sync_val is None:
            peak_sync_val = np.nan
        self.metrics["max_synchronous_peak"] = peak_sync_val

        max_neuron_rate = None
        if P.is_root(comm=env.comm):
            if global_spike_indices is not None and len(global_spike_indices) > 0:
                unique_neurons, spike_counts = np.unique(
                    global_spike_indices, return_counts=True
                )
                firing_rates = (spike_counts / self.recording_duration) * 1000.0
                max_neuron_rate = float(np.max(firing_rates))

        max_neuron_rate = P.broadcast(max_neuron_rate, comm=env.comm)
        if max_neuron_rate is None:
            max_neuron_rate = np.nan
        self.metrics["max_neuron_firing_rate"] = max_neuron_rate

        if (
            targets.get("burst_rate") is not None
            or targets.get("burst_participation") is not None
        ):
            burst_rate_entry = None
            burst_participation_entry = None
            if P.is_root(comm=env.comm):
                if (
                    global_spike_times is not None
                    and global_spike_indices is not None
                    and len(global_spike_times) > 100
                ):
                    burst_bin_width = 50.0
                    min_spike_count = 8
                    n_burst_bins = int(self.recording_duration / burst_bin_width)
                    spike_counts, _ = np.histogram(
                        global_spike_times,
                        bins=n_burst_bins,
                        range=(0, self.recording_duration),
                    )

                    mean_count = float(np.mean(spike_counts))
                    std_count = float(np.std(spike_counts))
                    burst_threshold = mean_count + 1.5 * std_count

                    burst_bins = (spike_counts > burst_threshold) & (
                        spike_counts >= min_spike_count
                    )
                    n_bursts = int(
                        np.sum(
                            np.diff(np.concatenate([[False], burst_bins, [False]])) == 1
                        )
                    )
                    burst_rate = n_bursts / (self.recording_duration / 1000.0)

                    if targets.get("burst_rate") is not None:
                        burst_rate_target = targets["burst_rate"]
                        burst_rate_objective = (
                            burst_rate - burst_rate_target
                        ) ** 2 * 10.0
                        burst_rate_entry = (burst_rate_objective, float(burst_rate))

                    if targets.get("burst_participation") is not None:
                        n_neurons = len(np.unique(global_spike_indices))
                        if n_neurons > 0 and n_bursts > 0:
                            burst_windows = np.where(burst_bins)[0]
                            neurons_in_bursts = set()
                            for bin_idx in burst_windows:
                                t_start = bin_idx * burst_bin_width
                                t_end = (bin_idx + 1) * burst_bin_width
                                mask = (global_spike_times >= t_start) & (
                                    global_spike_times < t_end
                                )
                                neurons_in_window = global_spike_indices[mask]
                                neurons_in_bursts.update(neurons_in_window.tolist())
                            burst_participation = len(neurons_in_bursts) / n_neurons
                        elif n_neurons > 0:
                            burst_participation = 0.0
                        else:
                            burst_participation = np.nan

                        participation_target = targets["burst_participation"]
                        if np.isnan(burst_participation):
                            burst_participation_entry = (np.nan, np.nan)
                        else:
                            participation_objective = (
                                burst_participation - participation_target
                            ) ** 2 * 5.0
                            burst_participation_entry = (
                                participation_objective,
                                float(burst_participation),
                            )
                else:
                    if targets.get("burst_rate") is not None:
                        burst_rate_entry = (np.nan, np.nan)
                    if targets.get("burst_participation") is not None:
                        burst_participation_entry = (np.nan, np.nan)

            if targets.get("burst_rate") is not None:
                if burst_rate_entry is None:
                    burst_rate_entry = (np.nan, np.nan)
                burst_rate_entry = P.broadcast(burst_rate_entry, comm=env.comm)
                result["burst_rate"] = burst_rate_entry
                self.metrics["burst_rate"] = (
                    burst_rate_entry[1] if burst_rate_entry else np.nan
                )

            if targets.get("burst_participation") is not None:
                if burst_participation_entry is None:
                    burst_participation_entry = (np.nan, np.nan)
                burst_participation_entry = P.broadcast(
                    burst_participation_entry, comm=env.comm
                )
                result["burst_participation"] = burst_participation_entry
                self.metrics["burst_participation"] = (
                    burst_participation_entry[1]
                    if burst_participation_entry
                    else np.nan
                )

        if enough_spikes:
            total_spikes = len(it)
            n_bins_target = max(50, total_spikes // 15)
            adaptive_bin_width = self.recording_duration / n_bins_target
            adaptive_bin_width = max(4.0, min(adaptive_bin_width, 50.0))

            avalanche_result = AvalancheAnalysis(
                duration=self.recording_duration, bin_width=adaptive_bin_width
            )(env, *recording_data)

            if avalanche_result:
                sigma = avalanche_result.get("branching_ratio", 0.0)
                r_squared = avalanche_result.get("size_power_law_r2", 0.0)

                self.metrics["branching_ratio"] = sigma
                self.metrics["avalanche_r2"] = r_squared
                self.metrics["avalanche_result"] = avalanche_result

                br_target = targets.get("branching_ratio", 1.0)
                if sigma < 0.8:
                    br_objective = (sigma - br_target) ** 2 * 10.0
                elif sigma > 1.2:
                    br_objective = (sigma - br_target) ** 2 * 10.0
                else:
                    br_objective = (sigma - br_target) ** 2
                result["branching_ratio"] = (br_objective, sigma)

                pl_target = targets.get("avalanche_power_law", 0.6)
                if r_squared >= pl_target:
                    pl_objective = 0.0
                else:
                    pl_objective = (pl_target - r_squared) ** 2 * 5.0

                result["avalanche_power_law"] = (pl_objective, r_squared)

            else:
                result["branching_ratio"] = (50.0, 0.0)
                result["avalanche_power_law"] = (50.0, 0.0)
        else:
            result["branching_ratio"] = (50.0, 0.0)
            result["avalanche_power_law"] = (50.0, 0.0)

        if pathology_penalty > 0:
            for key in result:
                obj_val, feat_val = result[key]
                result[key] = (obj_val + pathology_penalty, feat_val)

        self.objectives = result
        return result

    def compute_constraints(self, env) -> dict:
        result = {}
        stability_result = self.metrics.get("stability_result")

        if stability_result:
            is_runaway = stability_result["is_runaway"]
            is_quiescent = stability_result["is_quiescent"]
            tail_mean = stability_result["tail_mean_hz"]
            max_rate = stability_result.get("max_rate_hz", 20.0)
            min_rate = stability_result.get("min_rate_hz", 0.05)

            if is_runaway:
                runaway_constraint = -1.0 - (tail_mean - max_rate) / 10.0
            else:
                runaway_constraint = 1.0 + (max_rate - tail_mean) / 10.0

            if is_quiescent:
                quiescent_constraint = -1.0 - (min_rate - tail_mean) / 0.1
            else:
                quiescent_constraint = 1.0 + (tail_mean - min_rate) / 0.1

            result["not_runaway"] = (runaway_constraint, tail_mean)
            result["not_quiescent"] = (quiescent_constraint, tail_mean)

            result["is_stable"] = (
                1.0 if stability_result["is_stable"] else -1.0,
                float(stability_result["is_stable"]),
            )
        else:
            result["not_runaway"] = (-10.0, 0.0)
            result["not_quiescent"] = (-10.0, 0.0)
            result["is_stable"] = (-10.0, 0.0)

        max_limit = getattr(self, "max_firing_rate_limit", 50.0)
        max_neuron_rate = self.metrics.get("max_neuron_firing_rate", np.nan)
        if max_neuron_rate is None or np.isnan(max_neuron_rate):
            max_rate_constraint = -10.0
        elif max_neuron_rate <= max_limit:
            max_rate_constraint = 1.0 + (max_limit - max_neuron_rate) / max(
                max_limit, 1e-6
            )
        else:
            max_rate_constraint = -1.0 - (max_neuron_rate - max_limit) / max(
                max_limit, 1e-6
            )

        result["max_firing_rate"] = (
            max_rate_constraint,
            float(max_neuron_rate) if max_neuron_rate is not None else np.nan,
        )

        mean_sync = self.metrics.get("mean_channel_correlation", np.nan)

        if mean_sync is None:
            mean_sync = np.nan
        try:
            mean_sync_float = float(mean_sync)
            mean_sync_float = np.clip(mean_sync_float, -10.0, 10.0)
        except (ValueError, TypeError):
            mean_sync_float = np.nan

        if np.isnan(mean_sync_float) or np.isinf(mean_sync_float):
            synchrony_constraint = -2.0
            mean_sync_value = float("nan")
        else:
            mean_sync = float(np.clip(mean_sync_float, -1.0, 1.0))
            mean_sync_value = mean_sync
            lower_pref = 0.05
            upper_pref = 0.5
            penalty_scale = 0.3
            if mean_sync < lower_pref:
                deficit = (lower_pref - mean_sync) / max(lower_pref, 1e-6)
                synchrony_constraint = 1.0 - deficit * penalty_scale
            elif mean_sync > upper_pref:
                excess = (mean_sync - upper_pref) / max(upper_pref, 1e-6)
                synchrony_constraint = 1.0 - excess * penalty_scale
            else:
                center = (lower_pref + upper_pref) / 2.0
                half_width = (upper_pref - lower_pref) / 2.0
                synchrony_constraint = (
                    1.0
                    - (abs(mean_sync - center) / max(half_width, 1e-6)) * penalty_scale
                )

        synchrony_constraint = float(np.clip(synchrony_constraint, -10.0, 10.0))
        result["synchrony"] = (synchrony_constraint, float(mean_sync_value))

        # Peak Synchrony Constraint
        peak_sync_val = self.metrics.get("max_synchronous_peak", np.nan)
        if np.isnan(peak_sync_val):
            peak_sync_constraint = -1.0
        else:
            # We want to avoid > 30% of the network firing precisely together
            max_tolerated_fraction = 0.3
            if peak_sync_val <= max_tolerated_fraction:
                # Good
                peak_sync_constraint = 1.0 + (max_tolerated_fraction - peak_sync_val)
            else:
                # Bad
                peak_sync_constraint = (
                    -1.0 - (peak_sync_val - max_tolerated_fraction) * 10.0
                )

        result["max_synchronous_peak"] = (peak_sync_constraint, float(peak_sync_val))

        mean_rate = self.metrics.get("mfr", np.nan)
        min_limit = getattr(self, "min_mean_firing_rate_limit", 0.05)
        max_mean_limit = getattr(self, "max_mean_firing_rate_limit", 15.0)

        if mean_rate is None or np.isnan(mean_rate):
            min_mean_constraint = -10.0
            max_mean_constraint = -10.0
        else:
            if mean_rate >= min_limit:
                min_mean_constraint = 1.0 + (mean_rate - min_limit) / max(
                    min_limit, 1e-6
                )
            else:
                min_mean_constraint = -1.0 - (min_limit - mean_rate) / max(
                    min_limit, 1e-6
                )

            if mean_rate <= max_mean_limit:
                max_mean_constraint = 1.0 + (max_mean_limit - mean_rate) / max(
                    max_mean_limit, 1e-6
                )
            else:
                max_mean_constraint = -1.0 - (mean_rate - max_mean_limit) / max(
                    max_mean_limit, 1e-6
                )

        result["min_mean_firing_rate"] = (min_mean_constraint, float(mean_rate))
        result["max_mean_firing_rate"] = (max_mean_constraint, float(mean_rate))

        return result
