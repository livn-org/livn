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


def _interpolate(log_rate, times, spike_times, ids):
    """log lambda at each spike time, taking each spike's value from its own cell's trace"""
    if ids is None:
        return jax.vmap(lambda trace, ts: jnp.interp(ts, times, trace))(
            log_rate, spike_times
        )
    return jax.vmap(lambda cell, t: jnp.interp(t, times, log_rate[cell]))(
        ids, spike_times
    )


def spike_nll(log_rate, spike_times, dt, *, ids=None, t0=0.0):
    """Point-process negative log likelihood of a spike train under an intensity.

    ``NLL = integral lambda dt - sum_i log lambda(t_i)``, the natural likelihood for the escape-rate
    mechanism. The integral is a left-rectangle sum over the recording grid; the log-intensity term
    is linearly interpolated at each (continuous) spike time, so the loss is differentiable in the
    spike times and not only in the trace.

    Args:
        log_rate: ``(cells, times)`` log intensity on the recording grid, from :func:`log_intensity`
        spike_times: the spikes to score, in ms. Either ``(cells, k)`` with one inf-padded row per
            cell, or a flat ``(n,)`` array alongside ``ids``. Non-finite entries are padding and are
            skipped.
        dt: recording grid spacing in ms the same ``dt`` that produced ``log_rate``
        ids: cell index per spike, when ``spike_times`` is flat
        t0: absolute time of grid sample 0, in ms

    Returns:
        Scalar NLL, summed over cells.
    """
    log_rate = jnp.asarray(log_rate)
    if log_rate.ndim != 2:
        raise ValueError(f"log_rate must be (cells, times), got {log_rate.shape}")
    spike_times = jnp.asarray(spike_times)
    if ids is None and spike_times.ndim != 2:
        raise ValueError(
            f"spike_times must be (cells, k) without ids, got {spike_times.shape}; "
            "pass ids= for a flat array"
        )
    if ids is not None and spike_times.ndim != 1:
        raise ValueError(
            f"spike_times must be flat when ids is given, got {spike_times.shape}"
        )

    times = t0 + jnp.arange(log_rate.shape[1]) * dt

    integral = jnp.sum(jnp.exp(jnp.minimum(log_rate, EXP_CAP))) * dt

    fired = jnp.isfinite(spike_times)
    # padding is +inf, which jnp.interp would clamp to the last sample rather than ignore
    safe = jnp.where(fired, spike_times, times[0])
    at_spikes = _interpolate(
        log_rate, times, safe, None if ids is None else jnp.asarray(ids)
    )

    return integral - jnp.sum(jnp.where(fired, at_spikes, 0.0))


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
