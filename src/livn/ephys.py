from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

from livn.types import Capability, supports
from livn.utils import P

logger = logging.getLogger(__name__)

fi_value_dtype = np.dtype([("frequency", float)])
isi_value_dtype = np.dtype(
    [
        ("first", float),
        ("last", float),
        ("ratio", float),
        ("mean", float),
        ("std", float),
        ("N", int),
    ]
)


def alpha_pulse(t, onset: float, tau: float, amplitude: float) -> np.ndarray:
    """``amplitude * x * exp(1 - x)`` with ``x = (t - onset) / tau``, zero before ``onset``.

    The peak is ``amplitude`` at ``t = onset + tau``.
    """
    t = np.asarray(t, dtype=float)
    x = np.maximum((t - onset) / max(float(tau), 1e-6), 0.0)
    u = amplitude * x * np.exp(1.0 - x)
    return np.where(t < onset, 0.0, u)


def square_step(t, t0: float, t1: float, amplitude: float) -> np.ndarray:
    """``amplitude`` on ``t0 <= t < t1``, zero elsewhere."""
    t = np.asarray(t, dtype=float)
    slack = 1e-6
    return np.where((t >= t0 - slack) & (t < t1 - slack), float(amplitude), 0.0)


def measure_deflection(t, v, t0, t1, stim_amp=None):
    """Measure voltage deflection (min or max, between start and end)."""

    start_index = int(np.argwhere(t >= t0 * 0.999)[0, 0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0, 0])

    deflect_fn = np.argmin
    if stim_amp is not None and (stim_amp > 0):
        deflect_fn = np.argmax

    v_window = v[start_index:end_index]
    peak_index = deflect_fn(v_window) + start_index

    return {
        "t_peak": t[peak_index],
        "v_peak": v[peak_index],
        "peak_index": peak_index,
        "t_baseline": t[start_index],
        "v_baseline": v[start_index],
        "baseline_index": start_index,
        "stim_amp": stim_amp,
    }


#
# Code based on https://www.github.com/AllenInstitute/ipfx/ipfx/subthresh_features.py
#


def fit_membrane_time_constant(t, v, t0, t1, rmse_max_tol=1.0):
    """Fit an exponential to estimate membrane time constant between start and end.

    Returns a, inv_tau, y0 of ``y0 + a * exp(-inv_tau * x)``; np.nan on failure.
    """

    def exp_curve(x, a, inv_tau, y0):
        return y0 + a * np.exp(-inv_tau * x)

    start_index = int(np.argwhere(t >= t0 * 0.999)[0, 0])
    end_index = int(np.argwhere(t >= t1 * 0.999)[0, 0])

    p0 = (v[start_index] - v[end_index], 0.1, v[end_index])
    t_window = (t[start_index:end_index] - t[start_index]).astype(np.float64)
    v_window = v[start_index:end_index].astype(np.float64)
    try:
        popt, _pcov = curve_fit(exp_curve, t_window, v_window, p0=p0)
    except (TypeError, RuntimeError):
        logger.info("Curve fit for membrane time constant failed")
        return np.nan, np.nan, np.nan

    pred = exp_curve(t_window, *popt)
    rmse = np.sqrt(np.mean((pred - v_window) ** 2))

    if rmse > rmse_max_tol:
        logger.debug(
            f"RMSE {rmse:f} for the Curve fit for membrane time constant exceeded the "
            f"maximum tolerance of {rmse_max_tol:f}"
        )
        return np.nan, np.nan, np.nan

    return popt


def measure_time_constant(
    t, v, t0, t1, stim_amp, frac=0.1, baseline_interval=100.0, min_snr=20.0
):
    """Membrane time constant (ms) from a single-exponential fit of the response."""

    if np.max(t) < t0 or np.max(t) < t1:
        logger.debug(
            "measure_time_constant: time series ends before t0 = {t0} or t1 = {t1}"
        )
        return np.nan

    # Assumes this is being done on a hyperpolarizing step
    deflection_results = measure_deflection(t, v, t0, t1, stim_amp)
    v_peak = deflection_results["v_peak"]
    peak_index = deflection_results["peak_index"]
    v_baseline = deflection_results["v_baseline"]
    start_index = deflection_results["baseline_index"]

    # Check that SNR is high enough to proceed
    sig = np.abs(v_baseline - v_peak)
    noise_interval_start_index = int(
        np.argwhere(t >= (t0 - baseline_interval) * 0.999)[0, 0]
    )
    noise = np.std(v[noise_interval_start_index:start_index])

    # a noiseless trace is likely a deterministic model
    snr = np.inf if noise == 0 else sig / noise
    if snr < min_snr:
        logger.error(
            "measure_time_constant: signal-to-noise ratio too low for time "
            f"constant estimate ({snr:g} < {min_snr:g})"
        )
        return np.nan

    search_result = np.flatnonzero(
        v[start_index:] <= frac * (v_peak - v_baseline) + v_baseline
    )

    if not search_result.size:
        logger.error(
            "measure_time_constant: could not find interval for time constant estimate"
        )
        return np.nan

    fit_start_index = search_result[0] + start_index
    fit_end_index = peak_index
    fit_start = t[fit_start_index]
    fit_end = t[fit_end_index]

    if not (fit_start < fit_end):
        logger.error(
            "measure_time_constant: could not find interval for time constant estimate"
        )
        return np.nan

    _a, inv_tau, _y0 = fit_membrane_time_constant(t, v, fit_start, fit_end)

    return 1.0 / inv_tau


def measure_passive(t, v, t0, t1, stim_amp):
    if np.max(t) < t0 or np.max(t) < t1:
        logger.debug(
            f"measure_passive: time series ends at {np.max(t)} before "
            f"t0 = {t0} or t1 = {t1}"
        )
        return {"Rinp": np.nan, "tau": np.nan}

    deflection_results = measure_deflection(t, v, t0, t1, stim_amp=stim_amp)
    v_peak = deflection_results["v_peak"]
    v_baseline = deflection_results["v_baseline"]

    Rinp = (v_peak - v_baseline) / stim_amp
    tau = measure_time_constant(t, v, t0, t1, stim_amp)

    return {"Rinp": Rinp, "tau": tau}


def detect_spikes(T, Y, t0, t1, before_peak=50.0):
    spk_info_dtype = np.dtype(
        [
            ("Vpeak", float),
            ("Tpeak", float),
            ("amplitude", float),
            ("T0", float),
            ("T1", float),
        ]
    )

    dt = np.mean(np.diff(T))
    pre_period_idxs = np.argwhere(t0 - before_peak > T).flat
    pre_peak_info = signal.find_peaks(
        Y[pre_period_idxs], height=-20.0, width=(None, int(before_peak / dt))
    )
    pre_peak_idxs = pre_peak_info[0]
    N_peaks_pre = len(pre_peak_idxs)

    spk_period_idxs = np.argwhere(np.logical_and(t0 - before_peak <= T, t1 >= T)).flat
    T_spk = T[spk_period_idxs]
    Y_spk = Y[spk_period_idxs]
    peak_info = signal.find_peaks(
        Y_spk, height=-20.0, width=(None, int(before_peak / dt))
    )
    peak_idxs = peak_info[0]
    if len(peak_idxs) == 0:
        return N_peaks_pre, 0, None, None

    # Threshold via dV/dt (Atherton and Bevan 2005):
    # 1. take dV/dt; 2. SD of dV/dt for `before_peak` ms before the AP;
    # 3. threshold = mean(dV/dt) + 2*SD(dV/dt).
    dydt = np.gradient(Y_spk, T_spk)
    mean_dydt = np.mean(dydt)
    peak_idx = peak_idxs[0]
    T_peak = T_spk[peak_idx]
    T_before_idxs = np.argwhere(
        np.isclose(T_spk, T_peak - before_peak, rtol=1e-4, atol=1e-4)
    )
    if len(T_before_idxs) == 0:
        return N_peaks_pre, 0, None, None

    T_before_idx = T_before_idxs[0][0]
    sd_dydt = np.std(dydt[T_before_idx:peak_idx])
    dydt_threshold = mean_dydt + 2 * sd_dydt
    try:
        threshold_idx = np.argwhere(dydt[T_before_idx:peak_idx] >= dydt_threshold)[0]
    except Exception:
        logger.debug("Error in threshold computation")
        return N_peaks_pre, 0, None, None

    threshold = Y_spk[T_before_idx:peak_idx][threshold_idx][0]

    period_idxs = np.argwhere(np.logical_and(t0 <= T, t1 >= T)).flat
    T = T[period_idxs]
    Y = Y[period_idxs]

    # Binary-threshold crossings, then split into per-spike intervals.
    threshold_crossings = np.diff(threshold < Y, prepend=False)
    crossing_idx = np.argwhere(threshold_crossings)[:, 0]
    up_crossing_idx = np.argwhere(threshold_crossings)[::2, 0]
    N_peaks = len(up_crossing_idx)

    spk_info = None
    if N_peaks > 0:
        Y_intervals = np.split(Y, crossing_idx[1::2])[:-1]
        T_intervals = np.split(T, crossing_idx[1::2])[:-1]
        peak_idxs = [np.argmax(Y_interval) for Y_interval in Y_intervals]
        N_peaks = len(peak_idxs)
        Y_peaks = []
        T_peaks = []
        peak_amps = []
        for j, (T_interval, peak_idx) in enumerate(
            zip(T_intervals, peak_idxs, strict=False)
        ):
            if len(T_interval) < 2:
                N_peaks -= 1
                continue
            peak_amp = np.max(Y_intervals[j]) - threshold
            peak_amps.append(peak_amp)
            Y_peak = Y_intervals[j][peak_idx]
            Y_peaks.append(Y_peak)
            if peak_idx < len(T_interval):
                T_peaks.append(T_interval[peak_idx])
            else:
                T_peaks.append(T_interval[-1])

        if N_peaks > 0:
            spk_info = np.zeros(shape=(N_peaks), dtype=spk_info_dtype)
            for p in range(N_peaks):
                spk_info[p]["Vpeak"] = Y_peaks[p]
                spk_info[p]["Tpeak"] = T_peaks[p]
                spk_info[p]["T0"] = T_intervals[p][0]
                spk_info[p]["T1"] = T_intervals[p][-1]
                spk_info[p]["amplitude"] = peak_amps[p]

    return N_peaks_pre, N_peaks, threshold, spk_info


def measure_spike_features(iclamp_results, t0, t1):
    N_sweeps = len(iclamp_results)
    pre_spk_cnt = np.zeros(shape=(N_sweeps, 1), dtype=int)
    spk_cnt = np.zeros(shape=(N_sweeps, 1), dtype=int)
    spk_infos = []
    thresholds = np.zeros(shape=(N_sweeps,), dtype=np.float32)
    mean_spike_amplitudes = np.zeros(shape=(N_sweeps,), dtype=np.float32)
    for i in range(N_sweeps):
        if iclamp_results[i] is None:
            spk_infos.append(None)
            continue
        T = iclamp_results[i]["t"]
        Y = iclamp_results[i]["v"]
        N_peaks_pre, N_peaks, threshold, spk_info = detect_spikes(T, Y, t0, t1)
        spk_infos.append(spk_info)
        pre_spk_cnt[i, 0] = N_peaks_pre
        spk_cnt[i, 0] = N_peaks
        if threshold is not None:
            thresholds[i] = threshold
        if spk_info is not None:
            mean_spike_amplitudes[i] = np.mean(spk_info["amplitude"])
    return pre_spk_cnt, spk_cnt, spk_infos, thresholds, mean_spike_amplitudes


def measure_fI(spk_cnt, t0, t1, cur_steps):
    N_inj = len(cur_steps)
    fI_array = np.zeros(shape=(N_inj,), dtype=fi_value_dtype)
    for i in range(N_inj):
        if i >= len(spk_cnt):
            break
        fI_array[i]["frequency"] = spk_cnt[i] * 1000 / (t1 - t0)
    return fI_array


def measure_ISI(cur_steps_amp, spk_infos):
    N = len(cur_steps_amp)
    ISI_array = np.zeros(shape=(N,), dtype=isi_value_dtype)
    for field in ["first", "last", "ratio", "mean", "std"]:
        ISI_array[field] = np.nan

    for i in range(N):
        if i >= len(spk_infos):
            break
        spk_info = spk_infos[i]
        N_peaks = spk_info.shape[0] if spk_info is not None else 0
        ISI_array[i]["N"] = N_peaks

        if N_peaks >= 2:
            ISI = np.diff(spk_info["Tpeak"])
            ISI_array[i]["first"] = ISI[0]
            if ISI.shape[0] > 1:
                ISI_array[i]["last"] = ISI[-1]
                ISI_array[i]["ratio"] = ISI[-1] / ISI[0]
                ISI_array[i]["mean"] = np.mean(ISI)
                ISI_array[i]["std"] = np.std(ISI)

    return ISI_array


def spike_shape(v, t, onset: float, crossing_mv: float = -20.0) -> dict:
    """What a trace did after ``onset``, as numbers and no verdict.

    Returns ``finite`` (no NaN or inf anywhere), ``peak_mV`` (maximum after
    ``onset``), ``end_mV`` (the last sample) and ``n_crossings`` (upward crossings
    of ``crossing_mv`` after ``onset``). A trace that is not finite reports NaN for
    the voltages and -1 crossings, so a caller can tell a failed solve from a
    quiet cell.
    """
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(v)):
        return {
            "finite": False,
            "peak_mV": float("nan"),
            "end_mV": float("nan"),
            "n_crossings": -1,
        }
    after = v[t >= onset]
    if after.size == 0:
        raise ValueError(f"the trace ends at {t[-1]:g} ms, before onset {onset:g} ms")
    up = (after[:-1] < crossing_mv) & (after[1:] >= crossing_mv)
    return {
        "finite": True,
        "peak_mV": float(after.max()),
        "end_mV": float(v[-1]),
        "n_crossings": int(up.sum()),
    }


def threshold_search(
    probe: Callable[[float], object],
    criterion: Callable[[object], bool],
    lo: float,
    hi: float,
    tol: float = 0.02,
) -> float | None:
    """Smallest amplitude at which ``criterion(probe(amplitude))`` holds."""
    if not 0 < lo <= hi:
        raise ValueError(f"need 0 < lo <= hi, got lo={lo}, hi={hi}")

    below, above = None, None
    amp = float(lo)
    while amp <= hi:
        if criterion(probe(amp)):
            above = amp
            break
        below = amp
        amp *= 2.0
    if above is None:
        return None
    if below is None:
        return float(above)
    while (above - below) / above > tol:
        mid = (below * above) ** 0.5
        if criterion(probe(mid)):
            above = mid
        else:
            below = mid
    return float(above)


def _isolated(env):
    names = list(env.weight_names)
    if names:
        env = env.set_weights(dict.fromkeys(names, 0.0)) or env
    return env


def _prepare(env, *, record_dt: float, isolate: bool, gid: int, current: bool):
    if isolate:
        env = _isolated(env)
    gids = [int(gid)] if supports(env, Capability.PER_GID_VOLTAGE) else None
    env = env.record_voltage(dt=record_dt, gids=gids)
    if current:
        env = env.record_membrane_current(dt=record_dt)
    return env.clear(reseed=False)


def _narrowed(run, gid: int):
    from livn.run import Run

    out = Run(t0=run.t0, duration=run.duration)
    for series in ("voltage", "current"):
        if run.channel(series) is None:
            continue
        rows = np.asarray(run.ids(series)).astype(np.int64) == int(gid)
        sections = run.sections(series)
        out = getattr(out, f"add_{series}")(
            np.asarray(run.ids(series))[rows],
            np.asarray(run.values(series))[rows],
            dt=run.channel(series).dt,
            sections=None if sections is None else np.asarray(sections)[rows],
        )
    return out


def _current_stimulus(env, gid: int, u: np.ndarray, dt: float):
    from livn.stimulus import Stimulus

    return Stimulus.from_current(
        np.asarray(u, dtype=np.float32)[:, None],
        dt=dt,
        gids=np.array([gid], dtype=np.int32),
    )


def _run(env, duration_ms: float, stimulus, dt: float):
    try:
        run = env.run(duration_ms, stimulus, dt=dt)
    except ValueError as error:
        if "compiled for" not in str(error):
            raise
        raise ValueError(
            f"{error}. livn.ephys injects current, so on this backend build the "
            "model with input_mode='current', e.g. "
            "ReducedCalciumSomaDendrite(input_mode='current')"
        ) from error
    if P.size(comm=env.comm) > 1:
        run = run.gather(env.comm)
    return run


def _soma_row(run, gid: int, series: str = "voltage") -> int:
    ids = np.asarray(run.ids(series)).astype(np.int64)
    sections = run.sections(series)
    rows = ids == int(gid)
    if sections is not None:
        sections = np.asarray(sections).astype(str)
        rows &= np.char.startswith(sections, "soma")
    found = np.flatnonzero(rows)
    if found.size == 0:
        raise RuntimeError(
            f"the run recorded no {series} row for the soma of gid {gid}"
        )
    return int(found[0])


def current_step(
    env,
    gid: int,
    amplitude_nA: float,
    t0_ms: float,
    t1_ms: float,
    duration_ms: float,
    *,
    dt: float,
    record_dt: float,
    isolate: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Inject a square current into the soma of ``gid`` and return ``(t, v_soma)``.

    The env restarts from t=0 first, so repeated calls on one env are independent
    protocol runs. ``dt`` is the integration step, at which the stimulus is sampled,
    and ``record_dt`` the recording step; ``t`` is on the step the backend recorded at,
    which on diffrax is ``dt``. With ``isolate`` the network's synaptic weights are
    zeroed first.

    Under MPI only the root rank gets a trace; the others get ``(None, None)``.
    """
    env = _prepare(env, record_dt=record_dt, isolate=isolate, gid=gid, current=False)
    # the stimulus is sampled at the integration step: diffrax takes an array as
    # given, where NEURON and brian2 resample it onto their own grid
    t_stim = np.arange(round(duration_ms / dt)) * dt
    u = square_step(t_stim, t0_ms, t1_ms, amplitude_nA)
    run = _run(env, duration_ms, _current_stimulus(env, gid, u, dt), dt)
    if run is None:
        return None, None

    trace = np.asarray(run.voltage)[_soma_row(run, gid)].astype(float)
    # diffrax samples at its integration step, whatever was asked for
    step = run.voltage_dt or record_dt
    return np.arange(trace.size, dtype=float) * step, trace


def spike_template(
    env,
    gid: int,
    amplitude_nA: float,
    pulse_tau_ms: float,
    pre_ms: float,
    post_ms: float,
    *,
    dt: float,
    record_dt: float | None = None,
    isolate: bool = True,
):
    """One spike, evoked by an alpha-shaped current pulse into the soma of ``gid``.

    Under MPI the root rank gets the run and the others ``None``.
    """
    record_dt = dt if record_dt is None else record_dt
    env = _prepare(env, record_dt=record_dt, isolate=isolate, gid=gid, current=True)
    duration = pre_ms + post_ms
    t = np.arange(round(duration / dt)) * dt
    u = alpha_pulse(t, pre_ms, pulse_tau_ms, amplitude_nA)
    run = _run(env, duration, _current_stimulus(env, gid, u, dt), dt)
    if run is None:
        return None
    return _narrowed(run, gid)
