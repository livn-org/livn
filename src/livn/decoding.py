from typing import Callable, Optional

import numpy as np
from pydantic import Field, PrivateAttr
from scipy.signal import butter, filtfilt, welch
from livn.types import Decoding
from livn.utils import P


class Slice(Decoding):
    """Slice decoding

    Slices the system response into [start -> stop]

    Args:
        start: Start time in ms (float or int)
        duration: Duration of the slice in ms (float or int)
    """

    start: float = 0.0
    duration: float = Field(validation_alias="stop")

    def __call__(self, env, it, tt, iv, vv, im, mp):
        stop = self.duration

        # spikes
        if it is not None and tt is not None:
            mask = (tt >= self.start) & (tt < stop)
            it = it[mask]
            tt = tt[mask] - self.start

        # voltage [n_neurons, T]
        if iv is not None and vv is not None:
            v_dt = env.voltage_recording_dt
            self._validate_no_information_loss(self.start, v_dt, "start", "voltage")
            self._validate_no_information_loss(stop, v_dt, "duration", "voltage")

            start_idx = int(self.start / v_dt)
            stop_idx = int(stop / v_dt)
            vv = vv[:, start_idx:stop_idx]

        # membrane currents [n_neurons, T]
        if im is not None and mp is not None:
            m_dt = env.membrane_current_recording_dt
            self._validate_no_information_loss(
                self.start, m_dt, "start", "membrane current"
            )
            self._validate_no_information_loss(
                stop, m_dt, "duration", "membrane current"
            )

            start_idx = int(self.start / m_dt)
            stop_idx = int(stop / m_dt)
            mp = mp[:, start_idx:stop_idx]

        return it, tt, iv, vv, im, mp

    def _validate_no_information_loss(
        self, time_ms: float, dt: float, param_name: str, data_type: str
    ) -> None:
        index = time_ms / dt
        if not self._is_integer(index):
            raise ValueError(
                f"{param_name}={time_ms} ms does not align with {data_type} recording dt={dt} ms"
                f" yielding a fractional index {index}"
            )

    @staticmethod
    def _is_integer(value: float, tol: float = 1e-9) -> bool:
        return abs(value - round(value)) < tol


class ChannelRecording(Decoding):
    def setup(self, env):
        env.record_spikes()

    def __call__(self, env, it, tt, iv, vv, im, mp):
        # per-rank electrode potential, sum-reduced for each channel [T, #channels]
        p = P.reduce_sum(env.potential_recording(mp), all=True, comm=env.comm)

        cit, ct = env.channel_recording(it, tt)

        cit, ct, iv, vv = P.gather(cit, ct, iv, vv, comm=env.comm)

        if P.is_root(comm=env.comm):
            cit, ct, iv, vv = P.merge(cit, ct, iv, vv)
            return cit, ct, iv, vv, env.io.channel_ids, p


class GatherAndMerge(Decoding):
    root: int = 0
    spikes: bool = True
    voltages: bool = True
    membrane_currents: bool = True

    def setup(self, env):
        if self.spikes:
            env.record_spikes()
        if self.voltages:
            env.record_voltage()
        if self.membrane_currents:
            env.record_membrane_current()

    def __call__(self, env, it, tt, iv, vv, im, mp):
        if self.spikes:
            it, tt = P.gather(it, tt, comm=env.comm, root=self.root)
        else:
            it, tt = None, None

        if self.voltages:
            iv, vv = P.gather(iv, vv, comm=env.comm, root=self.root)
        else:
            iv, vv = None, None

        if self.membrane_currents:
            im, mp = P.gather(im, mp, comm=env.comm, root=self.root)
        else:
            im, mp = None, None

        if P.is_root(comm=env.comm, root=self.root):
            if self.spikes:
                it, tt = P.merge(it, tt)
            if self.voltages:
                iv, vv = P.merge(iv, vv)
            if self.membrane_currents:
                im, mp = P.merge(im, mp)

            return it, tt, iv, vv, im, mp


class Pipe(Decoding):
    stages: list[Callable] = Field(default_factory=list)

    _context: dict = PrivateAttr(default_factory=dict)  # cleared each __call__
    _state: dict = PrivateAttr(default_factory=dict)  # persists across calls

    @property
    def context(self) -> dict:
        return self._context

    @property
    def state(self) -> dict:
        return self._state

    def setup(self, env):
        for stage in self.stages:
            if hasattr(stage, "setup"):
                stage.setup(env)

    def reset(self, **kwargs):
        for stage in self.stages:
            if hasattr(stage, "reset"):
                return stage.reset(**kwargs)
        return None

    def get_stage(self, stage_type: type):
        for s in self.stages:
            if isinstance(s, stage_type):
                return s
        return None

    def clear(self):
        self._state.clear()

    def __call__(self, env, it, tt, iv, vv, im, mp):
        self._context.clear()
        data = (it, tt, iv, vv, im, mp)
        for stage in self.stages:
            result = stage(env, *data)
            if result is None:
                pass
            elif isinstance(result, tuple):
                data = result
            else:
                data = (result,)
        return data[0] if len(data) == 1 else data

    def __repr__(self):
        stage_reprs = ", ".join(repr(s) for s in self.stages)
        return f"Pipe([{stage_reprs}])"


class MeanFiringRate(Decoding):
    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_count = 0
        local_units = set()
        if tt is not None and it is not None:
            local_count = len(tt)
            local_units = set(it.tolist()) if local_count > 0 else set()

        counts = P.gather(local_count, comm=env.comm)
        unit_sets = P.gather(local_units, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            total_spikes = sum(counts) if counts else local_count
            all_units = set()
            if unit_sets:
                for s in unit_sets:
                    all_units.update(s)
            else:
                all_units = local_units

            n_units = len(all_units) if all_units else 1
            duration_s = self.duration / 1000.0
            rate_hz = total_spikes / (n_units * duration_s + 1e-9)

            result = {
                "rate_hz": float(rate_hz),
                "total_spikes": int(total_spikes),
                "n_units": int(n_units),
                "duration_s": float(duration_s),
            }

        return P.broadcast(result, comm=env.comm)


class ActiveFraction(Decoding):
    min_spikes: int = 1

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_unit_counts = {}
        if tt is not None and it is not None:
            for uid in it:
                uid_int = int(uid)
                local_unit_counts[uid_int] = local_unit_counts.get(uid_int, 0) + 1

        all_counts = P.gather(local_unit_counts, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_counts = {}
            if all_counts:
                for counts in all_counts:
                    for uid, count in counts.items():
                        merged_counts[uid] = merged_counts.get(uid, 0) + count
            else:
                merged_counts = local_unit_counts

            total_units = (
                len(env.system.gids)
                if hasattr(env.system, "gids")
                else len(merged_counts)
            )
            if total_units == 0:
                total_units = 1

            active_units = sum(
                1 for c in merged_counts.values() if c >= self.min_spikes
            )
            silent_units = [
                uid for uid, c in merged_counts.items() if c < self.min_spikes
            ]

            active_fraction = min(1.0, active_units / total_units)

            result = {
                "active_fraction": float(active_fraction),
                "active_units": int(active_units),
                "total_units": int(total_units),
                "silent_units": silent_units,
            }

        return P.broadcast(result, comm=env.comm)


class Stability(Decoding):
    tail_window: float = 1000.0
    max_rate_hz: float = 20.0
    min_rate_hz: float = 0.05
    bin_size: float = 100.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_spikes = tt.tolist() if tt is not None else []
        all_spikes = P.gather(local_spikes, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_spikes = []
            if all_spikes:
                for spikes in all_spikes:
                    merged_spikes.extend(spikes)

            merged_spikes = np.array(merged_spikes) if merged_spikes else np.array([])

            duration_ms = float(self.duration)
            n_bins = max(1, int(duration_ms / self.bin_size))
            bin_edges = np.linspace(0, duration_ms, n_bins + 1)

            if len(merged_spikes) > 0:
                hist, _ = np.histogram(merged_spikes, bins=bin_edges)
                rates = hist / (self.bin_size / 1000.0)  # Convert to Hz
            else:
                rates = np.zeros(n_bins)

            tail_start_bin = max(
                0, int((duration_ms - self.tail_window) / self.bin_size)
            )
            tail_rates = rates[tail_start_bin:]
            tail_mean = float(np.mean(tail_rates)) if len(tail_rates) > 0 else 0.0
            global_mean = float(np.mean(rates)) if len(rates) > 0 else 0.0

            has_nan = bool(np.any(np.isnan(rates)) or np.any(np.isinf(rates)))

            is_runaway = tail_mean > self.max_rate_hz
            is_quiescent = global_mean < self.min_rate_hz
            is_stable = not is_runaway and not is_quiescent and not has_nan

            result = {
                "is_stable": bool(is_stable),
                "is_runaway": bool(is_runaway),
                "is_quiescent": bool(is_quiescent),
                "has_nan": bool(has_nan),
                "tail_mean_hz": tail_mean,
                "global_mean_hz": global_mean,
                "max_rate_hz": float(self.max_rate_hz),
                "min_rate_hz": float(self.min_rate_hz),
            }

        return P.broadcast(result, comm=env.comm)


class LFP(Decoding):
    channels: Optional[list[int]] = None
    downsample_hz: float = 1000.0
    lowpass_hz: Optional[float] = None
    lowpass_order: int = 4
    compute_band_power: bool | dict[str, tuple[float, float]] = False
    nperseg: int = 2048  # Welch window length for band power
    noverlap: Optional[int] = None  # Welch overlap (None = nperseg//2)

    def __call__(self, env, it, tt, iv, vv, im, mp):
        if mp is not None:
            local_potential = env.potential_recording(mp)
            if local_potential is not None:
                local_potential = np.asarray(local_potential, dtype=np.float32)
        else:
            local_potential = None

        potential = P.reduce_sum(local_potential, all=False, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            if potential is not None and potential.size > 0:
                lfp = np.array(potential, dtype=np.float32)  # [n_channels, T]

                n_samples = lfp.shape[1]
                original_dt_ms = self.duration / n_samples if n_samples > 0 else 0.1
                original_hz = 1000.0 / original_dt_ms

                if self.channels is not None:
                    lfp = lfp[self.channels, :]

                if self.lowpass_hz is not None and self.lowpass_hz < original_hz / 2:
                    nyq = original_hz / 2
                    normalized_cutoff = self.lowpass_hz / nyq
                    b, a = butter(self.lowpass_order, normalized_cutoff, btype="low")
                    lfp = filtfilt(b, a, lfp, axis=1).astype(np.float32)

                downsample_factor = int(original_hz / self.downsample_hz)
                if downsample_factor > 1:
                    lfp = lfp[:, ::downsample_factor]
                    sample_rate_hz = original_hz / downsample_factor
                else:
                    sample_rate_hz = original_hz

                result = {
                    "lfp": lfp,
                    "sample_rate_hz": float(sample_rate_hz),
                    "n_channels": int(lfp.shape[0]),
                    "n_samples": int(lfp.shape[1]),
                }

                if self.compute_band_power:
                    if isinstance(self.compute_band_power, dict):
                        bands = self.compute_band_power
                    else:
                        bands = {
                            "delta": (1.0, 4.0),
                            "theta": (4.0, 8.0),
                            "alpha": (8.0, 13.0),
                            "beta": (13.0, 30.0),
                            "gamma": (30.0, 80.0),
                        }

                    result.update(self._compute_band_power(lfp, sample_rate_hz, bands))
            else:
                result = {
                    "lfp": np.array([], dtype=np.float32),
                    "sample_rate_hz": self.downsample_hz,
                    "n_channels": 0,
                    "n_samples": 0,
                }

        return P.broadcast(result, comm=env.comm)

    def _compute_band_power(
        self,
        lfp: np.ndarray,
        sample_rate_hz: float,
        bands: dict[str, tuple[float, float]],
    ) -> dict:
        noverlap = self.noverlap if self.noverlap is not None else self.nperseg // 2

        # average across channels if multi-channel
        if lfp.ndim == 2:
            freqs, psd = welch(
                lfp.mean(axis=0),
                fs=sample_rate_hz,
                nperseg=min(self.nperseg, lfp.shape[1]),
                noverlap=min(noverlap, lfp.shape[1] // 2),
            )
        else:
            freqs, psd = welch(
                lfp,
                fs=sample_rate_hz,
                nperseg=min(self.nperseg, len(lfp)),
                noverlap=min(noverlap, len(lfp) // 2),
            )

        result = {}
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_power = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else 0.0

        for band_name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs <= f_high)
            if np.sum(mask) > 1:
                band_power = float(np.trapezoid(psd[mask], freqs[mask]))
            elif np.sum(mask) == 1:
                band_power = float(psd[mask][0] * df)
            else:
                band_power = 0.0
            result[band_name] = band_power

        result["broadband"] = total_power

        if total_power > 0:
            for band_name in bands:
                result[f"{band_name}_relative"] = result[band_name] / total_power

        return result


class AvalancheAnalysis(Decoding):
    """Analyzes neuronal avalanches

    Detects avalanches as continuous sequences of active bins (bins with >0 spikes),
    separated by silent bins. Computes avalanche statistics (power-law distribution)
    and branching ratio (ideally around 1)

    Args:
        duration: Duration of the analysis window in ms
        bin_width: Time bin size in ms (default: 4.0, approx avg ISI at critical state)

    Returns: dict
        - n_avalanches: Number of detected avalanches
        - mean_size: Mean avalanche size (total spikes per avalanche)
        - mean_duration: Mean avalanche duration in bins
        - branching_ratio: mean(n_{t+1} / n_t) across all avalanche bins
        - size_power_law_exponent: Exponent from power law fit
        - size_power_law_r2: R² goodness of fit for power law
    """

    bin_width: float = 4.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_spikes = tt.tolist() if tt is not None else []
        all_spikes = P.gather(local_spikes, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_spikes = []
            if all_spikes:
                for spikes in all_spikes:
                    merged_spikes.extend(spikes)
            merged_spikes = np.sort(merged_spikes)

            if len(merged_spikes) > 0:
                t_start = merged_spikes[0]
                t_end = merged_spikes[-1]
                if t_end > t_start:
                    n_bins = int((t_end - t_start) / self.bin_width) + 1
                    counts, _ = np.histogram(
                        merged_spikes, bins=n_bins, range=(t_start, t_end)
                    )
                else:
                    counts = np.array([len(merged_spikes)])

                is_active = counts > 0
                is_active_padded = np.concatenate(([0], is_active, [0]))
                diff = np.diff(is_active_padded.astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                sizes = []
                durations = []

                for s, e in zip(starts, ends):
                    avalanche_counts = counts[s:e]
                    sizes.append(np.sum(avalanche_counts))
                    durations.append(e - s)

                ratios = []
                for s, e in zip(starts, ends):
                    if e > s + 1:
                        av_counts = counts[s:e]
                        ancestors = av_counts[:-1]
                        descendants = av_counts[1:]
                        valid = ancestors > 0
                        if np.any(valid):
                            ratios.extend(descendants[valid] / ancestors[valid])

                if ratios:
                    sigma = float(np.mean(ratios))
                else:
                    sigma = 0.0

                # power law fit (R^2) for sizes
                # log(P(s)) ~ -alpha * log(s)
                r_squared = 0.0
                alpha = 0.0
                if len(sizes) > 10:
                    try:
                        max_size = max(sizes)
                        if max_size > 1:
                            bins = np.logspace(0, np.log10(max_size), 20)
                            hist, edges = np.histogram(sizes, bins=bins)
                            centers = (edges[:-1] + edges[1:]) / 2
                            mask = hist > 0
                            if np.sum(mask) > 2:
                                log_x = np.log10(centers[mask])
                                log_y = np.log10(hist[mask] / np.sum(hist))
                                # linear regression: log(P) = -alpha * log(s) + intercept
                                slope, intercept = np.polyfit(log_x, log_y, 1)
                                alpha = -slope  # power law exponent
                                y_pred = slope * log_x + intercept
                                ss_res = np.sum((log_y - y_pred) ** 2)
                                ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                                if ss_tot > 0:
                                    r_squared = 1 - (ss_res / ss_tot)
                    except Exception:
                        pass

                result = {
                    "n_avalanches": len(sizes),
                    "mean_size": float(np.mean(sizes)) if sizes else 0.0,
                    "mean_duration": float(np.mean(durations)) if durations else 0.0,
                    "branching_ratio": float(sigma),
                    "size_power_law_exponent": float(alpha),
                    "size_power_law_r2": float(r_squared),
                }
            else:
                result = {
                    "n_avalanches": 0,
                    "mean_size": 0.0,
                    "mean_duration": 0.0,
                    "branching_ratio": 0.0,
                    "size_power_law_exponent": 0.0,
                    "size_power_law_r2": 0.0,
                }

        return P.broadcast(result, comm=env.comm)
