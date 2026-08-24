import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

__all__ = [
    "EV_DT",
    "EV_SIGMA",
    "bits_per_spike",
    "ev_ratio",
    "explained_variance_ratio",
    "soft_explained_variance",
    "spikes_to_idx",
]

EV_SIGMA = 0.01  # Gaussian width = Teeter et al.'s 10 ms temporal scale
EV_DT = 5e-5  # 20 kHz frame for building the spike trains


def _gauss_kernel(sigma_s, dt):
    sig = sigma_s / dt
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


def spikes_to_idx(spike_times_s, steps, dt):
    idx = jnp.round(jnp.asarray(spike_times_s) / dt).astype(jnp.int32)
    return idx[(idx >= 0) & (idx < steps)]


def ev_ratio(model_idx, data_idx_list, steps, dt, sigma_s=EV_SIGMA):
    if len(data_idx_list) < 2:
        return None
    kernel = _gauss_kernel(sigma_s, dt)
    data_traces = jnp.stack(
        [_binary_train(jnp.asarray(idx), steps) for idx in data_idx_list]
    )
    return _ev_ratio_traces(
        _binary_train(jnp.asarray(model_idx), steps), data_traces, kernel
    )


def explained_variance_ratio(
    model_spikes, data_spike_list, duration_s, dt=EV_DT, sigma=EV_SIGMA
):
    if len(data_spike_list) < 2:
        return None
    steps = round(duration_s / dt)
    midx = spikes_to_idx(model_spikes, steps, dt)
    didx = [spikes_to_idx(s, steps, dt) for s in data_spike_list]
    r = ev_ratio(midx, didx, steps, dt, sigma_s=sigma)
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
