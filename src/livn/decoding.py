import os
from typing import Callable, Optional, Any

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
        # Per-rank electrode potential, sum-reduced for each channel [n_channels, T].
        if self.membrane_currents and mp is not None:
            p = P.reduce_sum(env.potential_recording(mp), all=True, comm=env.comm)
        else:
            p = None

        if self.spikes:
            cit, ct = env.channel_recording(it, tt)
            cit, ct = P.gather(cit, ct, comm=env.comm, root=self.root)
        else:
            cit, ct = None, None

        if self.voltages:
            iv, vv = P.gather(iv, vv, comm=env.comm, root=self.root)
        else:
            iv, vv = None, None

        if P.is_root(comm=env.comm, root=self.root):
            if self.spikes:
                cit, ct = P.merge(cit, ct)
            if self.voltages:
                iv, vv = P.merge(iv, vv)
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

    def __getstate__(self):
        state = super().__getstate__()
        private = state.get("__pydantic_private__", {})
        if private:
            private = dict(private)
            private["_state"] = {}
            private["_context"] = {}
            state["__pydantic_private__"] = private
        return state

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

    def get_stage(self, stage_type: "type | str") -> Any | None:
        """Return the first matching stage or None

        Note: stage_type may be either a class object or a plain string name
         but matching is done by class name rather than `isinstance` so that
         the lookup is robust against the same module being imported under
         two different paths ('duck-typed override')
        """
        name = stage_type if isinstance(stage_type, str) else stage_type.__name__
        for s in self.stages:
            if type(s).__name__ == name:
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


class PopulationRateMetrics(Decoding):
    """Population spike-count statistics in fixed-width bins.

    Returns dict with:
        mean_count
        std_count
        coefficient_of_variation (= std/mean)
        fano_factor (= var/mean)
        bin_size
        n_bins
    """

    bin_size: float = 100.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_spikes = tt.tolist() if tt is not None else []
        all_spikes = P.gather(local_spikes, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged: list = []
            if all_spikes:
                for s in all_spikes:
                    merged.extend(s)

            duration = float(self.duration)
            n_bins = max(2, int(duration / self.bin_size))
            if merged:
                counts, _ = np.histogram(merged, bins=n_bins, range=(0.0, duration))
                mean_c = float(counts.mean())
                std_c = float(counts.std())
                cv = std_c / mean_c if mean_c > 0 else 0.0
                fano = (std_c**2) / mean_c if mean_c > 0 else 0.0
            else:
                mean_c = std_c = cv = fano = 0.0

            result = {
                "mean_count": float(mean_c),
                "std_count": float(std_c),
                "coefficient_of_variation": float(cv),
                "fano_factor": float(fano),
                "bin_size": float(self.bin_size),
                "n_bins": int(n_bins),
            }

        return P.broadcast(result, comm=env.comm)


class ISICV(Decoding):
    """Mean per-unit interspike-interval coefficient of variation.

    Returns dict with:
        isi_cv (mean across units with >= min_spikes_per_unit)
        n_units_used
    """

    min_spikes_per_unit: int = 5

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_it = it.tolist() if it is not None else []
        local_tt = tt.tolist() if tt is not None else []
        all_it = P.gather(local_it, comm=env.comm)
        all_tt = P.gather(local_tt, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_it: list = []
            merged_tt: list = []
            if all_it and all_tt:
                for ii, ts in zip(all_it, all_tt):
                    merged_it.extend(ii)
                    merged_tt.extend(ts)

            isi_cvs: list = []
            if merged_tt:
                merged_it_arr = np.asarray(merged_it)
                merged_tt_arr = np.asarray(merged_tt)
                unit_ids, inv = np.unique(merged_it_arr, return_inverse=True)
                order = np.argsort(merged_tt_arr)
                t_sorted = merged_tt_arr[order]
                inv_sorted = inv[order]
                for u_idx in range(len(unit_ids)):
                    ts = t_sorted[inv_sorted == u_idx]
                    if len(ts) >= self.min_spikes_per_unit:
                        isi = np.diff(ts)
                        if isi.mean() > 0:
                            isi_cvs.append(float(isi.std() / isi.mean()))

            result = {
                "isi_cv": float(np.mean(isi_cvs)) if isi_cvs else 0.0,
                "n_units_used": int(len(isi_cvs)),
            }

        return P.broadcast(result, comm=env.comm)


class PopulationAutocorrTau(Decoding):
    """Integral autocorrelation timescale of population spike train.

    Estimates tau from the normalized positive-lag ACF of population spike
    counts in `bin_size` bins, using the integral-timescale formula
    ``tau = bin_size * (1 + 2 * sum(positive_lags))`` truncated at the first
    non-positive lag and clamped to ``[bin_size, max_lag]``.

    Use a `bin_size` that is small relative to the expected timescale so
    short-time autocorrelation is actually resolvable.

    Returns dict with:
        pop_autocorr_tau
        bin_size
        max_lag
    """

    bin_size: float = 10.0
    max_lag: float = 5000.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_spikes = tt.tolist() if tt is not None else []
        all_spikes = P.gather(local_spikes, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged = []
            if all_spikes:
                for s in all_spikes:
                    merged.extend(s)

            duration = float(self.duration)
            tau = float(self.bin_size)
            n_bins = max(2, int(duration / self.bin_size))
            if merged and n_bins > 8:
                counts, _ = np.histogram(merged, bins=n_bins, range=(0.0, duration))
                mean_c = float(counts.mean())
                std_c = float(counts.std())
                if std_c > 0:
                    x = counts.astype(np.float64) - mean_c
                    n = len(x)
                    fft_len = 1 << int(np.ceil(np.log2(2 * n)))
                    F = np.fft.rfft(x, n=fft_len)
                    acf_full = np.fft.irfft(F * np.conj(F), n=fft_len)[:n]
                    acf = acf_full / acf_full[0]
                    max_lag = min(n - 1, int(self.max_lag / self.bin_size))
                    pos = acf[1 : max_lag + 1]
                    nonpos = np.where(pos <= 0)[0]
                    cutoff = int(nonpos[0]) if len(nonpos) > 0 else len(pos)
                    if cutoff > 0:
                        tau = float(self.bin_size * (1.0 + 2.0 * pos[:cutoff].sum()))
                    tau = float(np.clip(tau, self.bin_size, self.max_lag))

            result = {
                "pop_autocorr_tau": float(tau),
                "bin_size": float(self.bin_size),
                "max_lag": float(self.max_lag),
            }

        return P.broadcast(result, comm=env.comm)


class BurstRate(Decoding):
    """Network burst rate (Hz) via robust threshold on binned counts.

    A bin is flagged as a burst when its spike count exceeds
    ``max(median + mad_k * 1.4826 * MAD, max(min_floor, min_floor_fraction * n_units))``.
    Contiguous bursting bins are collapsed into a single burst event.

    Returns dict with:
        burst_rate_hz
        n_bursts
        threshold
    """

    bin_size: float = 50.0
    mad_k: float = 5.0
    min_floor_fraction: float = 0.15
    min_floor: float = 3.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_it = it.tolist() if it is not None else []
        local_tt = tt.tolist() if tt is not None else []
        all_it = P.gather(local_it, comm=env.comm)
        all_tt = P.gather(local_tt, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_it: list = []
            merged_tt: list = []
            if all_it and all_tt:
                for ii, ts in zip(all_it, all_tt):
                    merged_it.extend(ii)
                    merged_tt.extend(ts)

            duration = float(self.duration)
            burst_rate = 0.0
            n_bursts = 0
            threshold = 0.0
            if merged_tt:
                n_units = max(1, len(set(merged_it)))
                n_bins = max(1, int(duration / self.bin_size))
                counts, _ = np.histogram(merged_tt, bins=n_bins, range=(0.0, duration))
                median = float(np.median(counts))
                mad = float(np.median(np.abs(counts - median))) or 1.0
                robust_t = median + self.mad_k * 1.4826 * mad
                floor = max(self.min_floor, self.min_floor_fraction * n_units)
                threshold = max(robust_t, floor)
                burst_bins = counts >= threshold
                n_bursts = int(
                    np.sum(np.diff(np.concatenate([[False], burst_bins, [False]])) == 1)
                )
                burst_rate = n_bursts / (duration / 1000.0)

            result = {
                "burst_rate_hz": float(burst_rate),
                "n_bursts": int(n_bursts),
                "threshold": float(threshold),
            }

        return P.broadcast(result, comm=env.comm)


class PeakSynchrony(Decoding):
    """Peak fraction of active units co-firing in a single `bin_size` bin.

    Returns dict with:
        max_synchronous_peak
        max_bin_count
        n_active_units
    """

    bin_size: float = 2.0

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_it = it.tolist() if it is not None else []
        local_tt = tt.tolist() if tt is not None else []
        all_it = P.gather(local_it, comm=env.comm)
        all_tt = P.gather(local_tt, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_it: list = []
            merged_tt: list = []
            if all_it and all_tt:
                for ii, ts in zip(all_it, all_tt):
                    merged_it.extend(ii)
                    merged_tt.extend(ts)

            duration = float(self.duration)
            peak = 0.0
            max_count = 0
            n_active = 0
            if merged_tt:
                n_active = len(set(merged_it))
                n_bins = max(1, int(duration / self.bin_size))
                counts, _ = np.histogram(merged_tt, bins=n_bins, range=(0.0, duration))
                max_count = int(np.max(counts))
                if n_active > 0:
                    peak = float(max_count) / float(n_active)

            result = {
                "max_synchronous_peak": float(peak),
                "max_bin_count": int(max_count),
                "n_active_units": int(n_active),
            }

        return P.broadcast(result, comm=env.comm)


class PairwiseChannelCorrelation(Decoding):
    """Mean pairwise Pearson correlation across per-unit binned spike counts.

    Returns dict with:
        mean_pairwise_correlation
        n_pairs
    """

    bin_size: float = 20.0
    min_units: int = 2

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_it = it.tolist() if it is not None else []
        local_tt = tt.tolist() if tt is not None else []
        all_it = P.gather(local_it, comm=env.comm)
        all_tt = P.gather(local_tt, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged_it: list = []
            merged_tt: list = []
            if all_it and all_tt:
                for ii, ts in zip(all_it, all_tt):
                    merged_it.extend(ii)
                    merged_tt.extend(ts)

            duration = float(self.duration)
            mean_corr = 0.0
            n_pairs = 0
            if merged_tt:
                merged_it_arr = np.asarray(merged_it)
                merged_tt_arr = np.asarray(merged_tt)
                n_bins = max(1, int(duration / self.bin_size))
                unit_ids = np.unique(merged_it_arr)
                rows = []
                for uid in unit_ids:
                    ts = merged_tt_arr[merged_it_arr == uid]
                    counts, _ = np.histogram(ts, bins=n_bins, range=(0.0, duration))
                    rows.append(counts)
                if len(rows) >= self.min_units:
                    M = np.asarray(rows, dtype=np.float64)
                    stds = M.std(axis=1)
                    valid = stds > 0
                    if int(valid.sum()) >= self.min_units:
                        Mv = M[valid]
                        Mv = (Mv - Mv.mean(axis=1, keepdims=True)) / Mv.std(
                            axis=1, keepdims=True
                        )
                        C = (Mv @ Mv.T) / Mv.shape[1]
                        iu = np.triu_indices(C.shape[0], k=1)
                        if iu[0].size > 0:
                            mean_corr = float(np.mean(C[iu]))
                            n_pairs = int(iu[0].size)

            result = {
                "mean_pairwise_correlation": float(mean_corr),
                "n_pairs": int(n_pairs),
            }

        return P.broadcast(result, comm=env.comm)


class PerUnitFiringRate(Decoding):
    """Per-unit firing rates (Hz) and summary statistics.

    Returns dict with:
        per_unit_rates_hz
        max_rate_hz
        min_rate_hz
        mean_rate_hz
        rate_cv
        n_units
    """

    def __call__(self, env, it, tt, iv, vv, im, mp):
        local_it = it.tolist() if it is not None else []
        all_it = P.gather(local_it, comm=env.comm)

        result = None
        if P.is_root(comm=env.comm):
            merged: list = []
            if all_it:
                for ii in all_it:
                    merged.extend(ii)

            duration_s = float(self.duration) / 1000.0
            rates: dict = {}
            if merged:
                arr = np.asarray(merged)
                uids, counts = np.unique(arr, return_counts=True)
                rates = {int(u): float(c / duration_s) for u, c in zip(uids, counts)}

            rate_arr = np.asarray(list(rates.values())) if rates else np.array([])
            result = {
                "per_unit_rates_hz": rates,
                "max_rate_hz": float(rate_arr.max()) if rate_arr.size else 0.0,
                "min_rate_hz": float(rate_arr.min()) if rate_arr.size else 0.0,
                "mean_rate_hz": (float(rate_arr.mean()) if rate_arr.size else 0.0),
                "rate_cv": (
                    float(rate_arr.std() / rate_arr.mean())
                    if rate_arr.size and rate_arr.mean() > 0
                    else 0.0
                ),
                "n_units": int(rate_arr.size),
            }

        return P.broadcast(result, comm=env.comm)


class PopulationFiringRates(Decoding):
    """Mean firing rate (Hz) per simulated population.

    Buckets spike gids by ``system.cells_meta_data.population_ranges`` and divides
    each population's spike count by (simulated cells in that population x seconds).
    The denominator uses env.cells, so it is correct under ``selection()``
    subsampling. Returns ``{"rates_hz": {pop: Hz}}``.
    """

    def __call__(self, env, it, tt, iv, vv, im, mp):
        ranges = env.system.cells_meta_data.population_ranges
        pops = sorted(ranges, key=lambda p: ranges[p][0])  # ascending start gid
        starts = np.array([ranges[p][0] for p in pops], dtype=np.int64)
        ends = np.array([ranges[p][0] + ranges[p][1] for p in pops], dtype=np.int64)

        # local spike counts per population (vectorized gid -> population bucket)
        spikes = np.zeros(len(pops), dtype=np.int64)
        if it is not None and tt is not None and len(it):
            gids = np.asarray(it, dtype=np.int64)
            idx = np.searchsorted(starts, gids, side="right") - 1
            valid = (idx >= 0) & (gids < ends[idx])
            spikes += np.bincount(idx[valid], minlength=len(pops))

        # local simulated cell count per population (correct under selection())
        cells = np.array([len(env.cells.get(p, {})) for p in pops], dtype=np.int64)

        # reduce both across ranks in one Allreduce
        spikes, cells = P.reduce_sum(spikes, cells, all=True, comm=env.comm)

        duration_s = float(self.duration) / 1000.0
        return {
            "rates_hz": {
                p: float(s / (c * duration_s + 1e-9))
                for p, s, c in zip(pops, spikes, cells)
                if c > 0
            }
        }


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


class ArrowDataset(GatherAndMerge):
    """Save raw simulation output to an Arrow file

    Gathers and merges distributed data, then writes each decoded result as an
    independently valid Arrow shard file.  The dataset is readable after
    every call via `datasets.Dataset.from_file()`.

    To turn written shards into a full dataset use `.dataset()`

    Args:
        directory: Directory where the Arrow dataset is written
        spikes: Whether to record / save spike data
        voltages: Whether to record / save voltage traces
        membrane_currents: Whether to record / save membrane currents
        root: MPI root rank for gather operations
    """

    directory: str

    def __call__(self, env, it, tt, iv, vv, im, mp):
        data = super().__call__(env, it, tt, iv, vv, im, mp)
        if data is None:
            return

        it, tt, iv, vv, im, mp = data

        row = {"duration": self.duration}
        if self.spikes:
            row["it"] = it
            row["tt"] = tt
        if self.voltages:
            row["iv"] = iv
            row["vv"] = vv
        if self.membrane_currents:
            row["im"] = im
            row["mp"] = mp

        self._write_shard(row)

        return it, tt, iv, vv, im, mp

    def _write_shard(self, row):
        from datasets.arrow_writer import ArrowWriter

        os.makedirs(self.directory, exist_ok=True)
        idx = self._next_shard_index()
        shard_path = os.path.join(
            self.directory,
            f"data-{idx:05d}.arrow",
        )
        writer = ArrowWriter(path=shard_path, writer_batch_size=1)
        writer.write(row)
        writer.finalize()
        writer.close()

    def _next_shard_index(self) -> int:
        if not os.path.isdir(self.directory):
            return 0
        return len(
            [
                f
                for f in os.listdir(self.directory)
                if f.startswith("data-") and f.endswith(".arrow")
            ]
        )

    def dataset(self):
        import pyarrow as pa
        from datasets import Dataset

        shard_files = sorted(
            f
            for f in os.listdir(self.directory)
            if f.startswith("data-") and f.endswith(".arrow")
        )
        if not shard_files:
            return

        tables = []
        for f in shard_files:
            tables.append(
                pa.ipc.open_stream(os.path.join(self.directory, f)).read_all()
            )

        return Dataset(pa.concat_tables(tables))
