import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

__all__ = [
    "EV_DT",
    "EV_SIGMA",
    "bits_per_spike",
    "ev_ratio",
    "explained_variance_ratio",
    "reconstruction",
    "soft_explained_variance",
    "spike_f1",
    "spikes_to_idx",
    "threshold_spikes",
]

EV_SIGMA = 10.0  # ms, Gaussian width = Teeter et al.'s 10 ms temporal scale
EV_DT = 0.05  # ms, 20 kHz frame for building the spike trains


def _gauss_kernel(sigma, dt):
    sig = sigma / dt
    L = round(sig * 10)
    x = jnp.arange(L) - (L - 1) / 2.0
    k = jnp.exp(-0.5 * (x / sig) ** 2)
    return k / jnp.sum(k)


def _binary_train(spike_idx, steps):
    return jnp.zeros(steps).at[spike_idx].add(1.0)


def _expvar(a, b):
    va, vb = jnp.var(a), jnp.var(b)
    vd = jnp.var(a - b)
    denom = va + vb
    return jnp.where(denom == 0, 1.0, (va + vb - vd) / denom)


def _ev_ratio_traces(model_trace, data_traces, kernel):
    smooth = lambda x: jnp.convolve(x, kernel, mode="same")  # noqa: E731
    dconv = jax.vmap(smooth)(data_traces)  # (R, steps)
    mconv = smooth(model_trace)  # (steps,)
    davg = dconv.mean(0)
    ev_dd = jnp.mean(jax.vmap(lambda d: _expvar(d, davg))(dconv))
    ev_md = jnp.mean(jax.vmap(lambda d: _expvar(d, mconv))(dconv))  # data vs model
    return jnp.where(ev_dd == 0, jnp.nan, ev_md / ev_dd)


def spikes_to_idx(spike_times, steps, dt):
    idx = jnp.round(jnp.asarray(spike_times) / dt).astype(jnp.int32)
    return idx[(idx >= 0) & (idx < steps)]


def ev_ratio(model_idx, data_idx_list, steps, dt, sigma=EV_SIGMA):
    if len(data_idx_list) < 2:
        return None
    kernel = _gauss_kernel(sigma, dt)
    data_traces = jnp.stack(
        [_binary_train(jnp.asarray(idx), steps) for idx in data_idx_list]
    )
    return _ev_ratio_traces(
        _binary_train(jnp.asarray(model_idx), steps), data_traces, kernel
    )


def explained_variance_ratio(
    model_spikes, data_spike_list, duration, dt=EV_DT, sigma=EV_SIGMA
):
    if len(data_spike_list) < 2:
        return None
    steps = round(duration / dt)
    midx = spikes_to_idx(model_spikes, steps, dt)
    didx = [spikes_to_idx(s, steps, dt) for s in data_spike_list]
    r = ev_ratio(midx, didx, steps, dt, sigma=sigma)
    if r is None:
        return None
    r = float(r)
    return r if jnp.isfinite(r) else None


def soft_explained_variance(model_trace, data_traces, dt=EV_DT, sigma=EV_SIGMA):
    model_trace = jnp.asarray(model_trace)
    data_traces = jnp.asarray(data_traces)
    if data_traces.ndim != 2 or data_traces.shape[0] < 2:
        raise ValueError(
            f"data_traces must be (R, steps) with R >= 2, got {data_traces.shape}"
        )
    if model_trace.shape != data_traces.shape[1:]:
        raise ValueError(
            f"model_trace {model_trace.shape} does not match data_traces {data_traces.shape}"
        )
    return _ev_ratio_traces(model_trace, data_traces, _gauss_kernel(sigma, dt))


def _poisson_nll(rates, spikes):
    valid_mask = ~jnp.isnan(spikes)
    rates = jnp.maximum(rates, 1e-9)
    nll = rates - spikes * jnp.log(rates) + gammaln(spikes + 1.0)
    return jnp.sum(jnp.where(valid_mask, nll, 0.0))


def bits_per_spike(rates, spikes):
    nll_model = _poisson_nll(rates, spikes)

    null_rates = jnp.tile(
        jnp.nanmean(spikes, axis=(0, 1), keepdims=True),
        (spikes.shape[0], spikes.shape[1], 1),
    )
    nll_null = _poisson_nll(null_rates, spikes)

    total_spikes = jnp.nansum(spikes)
    return jnp.where(
        total_spikes > 0,
        (nll_null - nll_model) / total_spikes / jnp.log(2),
        0.0,
    )


def reconstruction(pred, obs, axis: int = 0) -> dict:
    """Trace reconstruction error using global MSE and per-channel Pearson correlation.

    Host-side numpy.

    Args:
        pred: simulated trace, e.g. ``(time, channel)``
        obs: recorded trace of the same shape
        axis: the time axis; the correlation is taken along it, one value per remaining entry

    Returns:
        ``{"mse", "corr", "corr_mean"}``: the MSE over every sample, the correlation per channel
        (non-finite values replaced by 0), and its mean.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if pred.shape != obs.shape:
        raise ValueError(f"pred {pred.shape} and obs {obs.shape} must match")

    mse = float(np.mean((pred - obs) ** 2))
    pc = pred - pred.mean(axis=axis, keepdims=True)
    oc = obs - obs.mean(axis=axis, keepdims=True)
    denom = np.sqrt(
        np.maximum(np.sum(pc**2, axis=axis), 1e-12)
        * np.maximum(np.sum(oc**2, axis=axis), 1e-12)
    )
    corr = np.sum(pc * oc, axis=axis) / denom
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return {"mse": mse, "corr": corr, "corr_mean": float(np.mean(corr))}


def threshold_spikes(
    trace,
    dt: float,
    mad_k: float = 5.0,
    refractory: float = 10.0,
    polarity: str = "negative",
):
    """Spike times from a trace by MAD threshold.

    Args:
        trace: ``(time,)`` or ``(time, channel)``
        dt: sample interval in ms
        mad_k: threshold in robust standard deviations
        refractory: dead time after a detection
        polarity: ``"negative"`` or ``"positive"``

    Returns:
        Spike times in ms: a sorted array for 1-D input, a list of them (one per channel) for 2-D.
    """
    if polarity not in ("negative", "positive"):
        raise ValueError(f"polarity must be 'negative' or 'positive', got {polarity!r}")
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim == 2:
        return [
            threshold_spikes(trace[:, c], dt, mad_k, refractory, polarity)
            for c in range(trace.shape[1])
        ]
    if trace.ndim != 1:
        raise ValueError(f"trace must be (time,) or (time, channel), got {trace.shape}")

    mad = float(np.median(np.abs(trace - np.median(trace)))) + 1e-9
    threshold = mad_k * mad * 1.4826
    hit = trace < -threshold if polarity == "negative" else trace > threshold
    refractory_samples = max(1, round(refractory / max(dt, 1e-9)))

    kept = []
    last = -refractory_samples - 1
    for t in np.flatnonzero(hit):
        if t - last > refractory_samples:
            kept.append(t)
            last = t
    return np.asarray(kept, dtype=np.float64) * dt


def spike_f1(pred_times, gt_times, tol: float = 1.0) -> dict:
    """Tolerance-matched spike precision, recall and F1 for one train.

    Args:
        pred_times: predicted spike times in ms
        gt_times: recorded spike times in ms
        tol: matching tolerance in ms

    Returns:
        ``{"precision", "recall", "f1", "tp", "fp", "fn"}``. Ratios are NaN when undefined (no
        predictions, no recorded spikes, both empty). Pool ``tp/fp/fn`` over units before
        recomputing ``f1 = 2 tp / (2 tp + fp + fn)`` for an aggregate score.
    """
    pred = np.sort(np.asarray(pred_times, dtype=np.float64).ravel())
    gt = np.sort(np.asarray(gt_times, dtype=np.float64).ravel())
    tol = max(0.0, float(tol))

    used = np.zeros(gt.shape, dtype=bool)
    tp = 0
    first = 0
    for p in pred:
        while first < gt.size and gt[first] < p - tol:
            first += 1
        best, best_d = -1, np.inf
        j = first
        while j < gt.size and gt[j] <= p + tol:
            if not used[j] and abs(gt[j] - p) < best_d:
                best, best_d = j, abs(gt[j] - p)
            j += 1
        if best >= 0:
            used[best] = True
            tp += 1

    fp = int(pred.size) - tp
    fn = int(gt.size) - tp
    nan = float("nan")
    denom = 2 * tp + fp + fn
    return {
        "precision": tp / (tp + fp) if tp + fp else nan,
        "recall": tp / (tp + fn) if tp + fn else nan,
        "f1": 2 * tp / denom if denom else nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
