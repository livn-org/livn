import jax
import jax.numpy as jnp

from livn.models.glif import EXP_CAP

__all__ = [
    "intensity",
    "log_intensity",
    "param_prior",
    "refractory_mask",
    "spike_nll",
    "voltage_mse",
]


def _per_cell(value, like):
    value = jnp.asarray(value)
    if value.ndim == like.ndim - 1:
        return value[..., None]
    return value


def log_intensity(voltage, threshold, sigma, tau_s):
    """Reconstruct ``log lambda(t)`` from a recorded run, in log(1/ms).

    Args:
        voltage: ``(..., cells, times)`` membrane potential in mV
        threshold: ``(..., cells, times)`` threshold in mV, i.e. the ``threshold`` channel
        sigma: escape-noise width in mV, scalar or one per cell
        tau_s: escape time constant in ms, scalar or one per cell

    Returns:
        ``(..., cells, times)`` log intensity
    """
    voltage = jnp.asarray(voltage)
    threshold = jnp.asarray(threshold)
    sigma = _per_cell(sigma, voltage)
    tau_s = _per_cell(tau_s, voltage)
    sigma = jnp.where(sigma > 0, sigma, 1e-12)
    return (voltage - threshold) / sigma - jnp.log(tau_s)


def intensity(voltage, threshold, sigma, tau_s):
    """The escape-rate intensity ``lambda(t)`` in 1/ms"""
    return jnp.exp(
        jnp.minimum(log_intensity(voltage, threshold, sigma, tau_s), EXP_CAP)
    )


def refractory_mask(spike_times, times, t_ref, *, ids=None, n_cells=None):
    """Samples to keep: ``False`` inside ``[t_spike, t_spike + t_ref)`` of any spike.

    Materializes a ``(cells, spikes, times)`` intermediate, so keep ``spikes`` bounded for long runs.

    Args:
        spike_times: ``(cells, k)`` inf-padded, or flat ``(n,)`` alongside ``ids``
        times: ``(times,)`` sample times of the trace, in ms
        t_ref: refractory / spike-cut duration in ms, scalar or one per cell
        ids: cell index per spike, when ``spike_times`` is flat
        n_cells: number of cells, required when ``spike_times`` is flat

    Returns:
        ``(cells, times)`` boolean mask
    """
    times = jnp.asarray(times)
    spike_times = jnp.asarray(spike_times)

    if ids is None:
        if spike_times.ndim != 2:
            raise ValueError(
                f"spike_times must be (cells, k) without ids, got {spike_times.shape}"
            )
        cells = spike_times.shape[0]
        rows = spike_times
    else:
        if n_cells is None:
            raise ValueError("n_cells is required when spike_times is flat")
        cells = int(n_cells)
        ids = jnp.asarray(ids)
        # scatter the flat spikes into one row per cell, padding the rest with +inf
        rows = jnp.full((cells, spike_times.shape[0]), jnp.inf)
        rows = rows.at[ids, jnp.arange(spike_times.shape[0])].set(spike_times)

    width = jnp.broadcast_to(
        jnp.asarray(t_ref, dtype=times.dtype).reshape(-1), (cells,)
    )
    start = rows[:, :, None]
    stop = start + width[:, None, None]
    inside = jnp.isfinite(start) & (times >= start) & (times < stop)

    return ~jnp.any(inside, axis=1)


def voltage_mse(voltage, target, mask=None):
    """Mean squared error between a simulated and a target voltage trace, in mV^2.

    Args:
        voltage: ``(cells, times)`` simulated trace
        target: ``(cells, times)`` recorded trace
        mask: optional boolean of the same shape, ``True`` where a sample counts. Pair with
            :func:`refractory_mask` to drop the spike-cut windows.

    Returns:
        Scalar MSE over the unmasked samples (0.0 when nothing is unmasked).
    """
    voltage = jnp.asarray(voltage)
    target = jnp.asarray(target)
    if voltage.shape != target.shape:
        raise ValueError(
            f"voltage {voltage.shape} and target {target.shape} must have the same shape"
        )

    error = (voltage - target) ** 2
    if mask is None:
        return jnp.mean(error)

    mask = jnp.asarray(mask, dtype=error.dtype)
    total = jnp.sum(mask)
    return jnp.where(
        total > 0, jnp.sum(error * mask) / jnp.where(total > 0, total, 1.0), 0.0
    )


def param_prior(params, reference, weights=None):
    """Squared pull toward a reference parameter set, ``sum ||theta - theta_0||^2``.

    Lets a fit be regularized toward a prior guess (an encoder's prediction, a population mean, or
    the values a previous fit landed on).

    Args:
        params: ``{name: value}`` being fitted
        reference: ``{name: value}`` to pull toward; every name must be present in ``params``
        weights: optional ``{name: weight}``, default 1.0 each. Parameters live on wildly different
            scales (``tau_m`` in ms against ``asc_decay_rate`` in 1/ms), so an unweighted prior is
            rarely the one you want.

    Returns:
        Scalar penalty.
    """
    missing = sorted(set(reference) - set(params))
    if missing:
        raise KeyError(f"reference has parameters that are not being fitted: {missing}")

    weights = weights or {}
    total = jnp.asarray(0.0)
    for name, value in reference.items():
        deviation = jnp.asarray(params[name]) - jnp.asarray(value)
        total = total + jnp.asarray(weights.get(name, 1.0)) * jnp.sum(deviation**2)

    return total


def spike_kernel(sim_times, rec_times, bandwidth: float = 5.0, normalize: bool = True):
    r"""Squared RKHS distance between two spike trains, computed from spike times alone.

    .. math:: d^2 = \sum_{ij} K(s_i, s_j) - 2 \sum_{ij} K(s_i, r_j) + \sum_{ij} K(r_i, r_j)

    with a Gaussian :math:`K` this amounts to :math:`\|f - g\|^2` for :math:`f(t) = \sum_i K(t - s_i)`.

    Args:
        sim_times: ``(cells, k)`` simulated spike times, ``inf``-padded
        rec_times: ``(cells, m)`` recorded spike times, ``inf``-padded
        bandwidth: kernel width in ms, i.e. the timescale on which two spikes count as "the same spike"
        normalize: divide by the recorded train's self-similarity, so the value is comparable across
            cells with very different firing rates and across windows of different length

    Returns:
        scalar, summed over cells
    """
    import jax.numpy as jnp

    sim = jnp.atleast_2d(jnp.asarray(sim_times))
    rec = jnp.atleast_2d(jnp.asarray(rec_times))
    sim_ok, rec_ok = jnp.isfinite(sim), jnp.isfinite(rec)

    sim = jnp.where(sim_ok, sim, 0.0)
    rec = jnp.where(rec_ok, rec, 0.0)

    def gram(a, b, a_ok, b_ok):
        d = (a[:, :, None] - b[:, None, :]) / bandwidth
        k = jnp.exp(-0.5 * d * d)
        return jnp.sum(
            jnp.where(a_ok[:, :, None] & b_ok[:, None, :], k, 0.0), axis=(1, 2)
        )

    ss = gram(sim, sim, sim_ok, sim_ok)
    sr = gram(sim, rec, sim_ok, rec_ok)
    rr = gram(rec, rec, rec_ok, rec_ok)
    d2 = ss - 2.0 * sr + rr
    if normalize:
        d2 = d2 / jnp.maximum(rr, 1.0)
    return jnp.sum(d2)


def _interp_at(ts, values, x):
    """Linear interpolation of ``values(ts)`` at times ``x``, mapped over the leading cell axis."""
    import jax.numpy as jnp

    def one(t, v, q):
        n = t.shape[-1]
        i1 = jnp.clip(jnp.searchsorted(t, q, side="right"), 1, n - 1)
        i0 = i1 - 1
        t0_, t1_ = t[i0], t[i1]
        span = t1_ - t0_
        safe = span > 0
        w = jnp.where(safe, (q - t0_) / jnp.where(safe, span, 1.0), 0.0)
        return v[i0] + (v[i1] - v[i0]) * jnp.clip(w, 0.0, 1.0)

    return jax.vmap(one)(ts, values, x)


def spike_nll(log_rate, ts, spike_times):
    r"""Point-process negative log-likelihood.

    Args:
        log_rate: ``(cells, nodes)`` log intensity at the segment nodes
        ts: ``(cells, nodes)`` the segment node times, in ms
        spike_times: ``(cells, k)`` recorded spike times, inf-padded

    Returns:
        scalar, summed over cells
    """
    import jax.numpy as jnp

    log_rate = jnp.atleast_2d(log_rate)
    ts = jnp.atleast_2d(ts)
    rec = jnp.atleast_2d(jnp.asarray(spike_times))
    ok = jnp.isfinite(rec)
    rec = jnp.where(ok, rec, 0.0)

    lam = jnp.exp(log_rate)
    width = ts[..., 1:] - ts[..., :-1]
    integral = jnp.sum(0.5 * (lam[..., 1:] + lam[..., :-1]) * width, axis=-1)

    at_spikes = _interp_at(ts, log_rate, rec)
    point = jnp.sum(jnp.where(ok, at_spikes, 0.0), axis=-1)
    return jnp.sum(integral - point)
