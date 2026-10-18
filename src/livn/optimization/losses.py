import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Int

from livn.models.glif import EXP_CAP

__all__ = [
    "events_to_mask",
    "intensity",
    "log_intensity",
    "param_prior",
    "refractory_mask",
    "shape_logmag",
    "smoothness",
    "spike_nll",
    "voltage_mse",
]


def _per_cell(value, like):
    value = jnp.asarray(value)
    if value.ndim == like.ndim - 1:
        return value[..., None]
    return value


def log_intensity(voltage, threshold, sigma, tau):
    """Reconstruct ``log lambda(t)`` from a recorded run, in log(1/ms).

    Args:
        voltage: ``(..., cells, times)`` membrane potential in mV
        threshold: ``(..., cells, times)`` threshold in mV, i.e. the ``threshold`` channel
        sigma: escape-noise width in mV, scalar or one per cell
        tau: escape time constant in ms, scalar or one per cell

    Returns:
        ``(..., cells, times)`` log intensity
    """
    voltage = jnp.asarray(voltage)
    threshold = jnp.asarray(threshold)
    sigma = _per_cell(sigma, voltage)
    tau = _per_cell(tau, voltage)
    sigma = jnp.where(sigma > 0, sigma, 1e-12)
    return (voltage - threshold) / sigma - jnp.log(tau)


def intensity(voltage, threshold, sigma, tau):
    """The escape-rate intensity ``lambda(t)`` in 1/ms"""
    return jnp.exp(jnp.minimum(log_intensity(voltage, threshold, sigma, tau), EXP_CAP))


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


def param_prior(params, reference, weights=None, scale: float = 1.0):
    """Squared pull toward a reference parameter set, ``scale * sum w ||theta - theta_0||^2``.

    Args:
        params: the parameters being fitted: a ``{name: value}`` dict, or any pytree
        reference: mirrors ``params``; a dict may name only some of its entries (every name must
            be present in ``params``)
        weights: a scalar, or mirrors ``reference`` (a dict may omit names, which weigh 1.0).
            Parameters live on wildly different scales (``tau_m`` in ms against
            ``asc_decay_rate`` in 1/ms), so an unweighted prior is rarely the one you want.
        scale: overall factor

    Returns:
        Scalar penalty.
    """
    return scale * _pull(params, reference, weights)


def _pull(theta, reference, weights):
    if isinstance(reference, dict):
        missing = sorted(set(reference) - set(theta))
        if missing:
            raise KeyError(
                f"reference has parameters that are not being fitted: {missing}"
            )
        total = jnp.asarray(0.0)
        for name, ref in reference.items():
            w = weights.get(name) if isinstance(weights, dict) else weights
            total = total + _pull(theta[name], ref, w)
        return total

    w = 1.0 if weights is None else weights
    if jtu.treedef_is_leaf(jtu.tree_structure(w)):
        terms = jtu.tree_map(
            lambda t, r: (
                jnp.asarray(w) * jnp.sum((jnp.asarray(t) - jnp.asarray(r)) ** 2)
            ),
            theta,
            reference,
        )
    else:
        terms = jtu.tree_map(
            lambda t, r, ww: (
                jnp.asarray(ww) * jnp.sum((jnp.asarray(t) - jnp.asarray(r)) ** 2)
            ),
            theta,
            reference,
            w,
        )
    return sum(jtu.tree_leaves(terms), jnp.asarray(0.0))


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


def shape_logmag(
    pred: Float[Array, "*batch time channel"],
    target: Float[Array, "*batch time channel"],
    mask: Bool[Array, "*batch time channel"] | None = None,
    log_mag_weight: float = 1.0,
    eps: float = 1e-8,
    return_terms: bool = False,
):
    r"""Scale-invariant trace loss: RMS-normalised shape plus log-magnitude.

    .. math:: \mathcal{L} = \Bigl\| \tfrac{\hat V}{\mathrm{RMS}(\hat V)} - \tfrac{V}{\mathrm{RMS}(V)} \Bigr\|^2
              + \lambda\,(\log \mathrm{RMS}(\hat V) - \log \mathrm{RMS}(V))^2

    The RMS is taken over time, per (batch, channel), counting only masked samples. The shape term
    is averaged over all masked samples in the batch, the log-magnitude term over all
    (batch, channel) pairs. Both are exactly zero for a scaled copy and for equal RMS respectively.

    Args:
        pred: ``(*batch, time, channel)`` simulated trace
        target: same shape, recorded trace
        mask: optional boolean of the same shape (or broadcastable to it), ``True`` where a sample
            counts, e.g. from :func:`events_to_mask`. A (batch, channel) with no unmasked sample
            has RMS ``sqrt(eps)`` on both sides and contributes nothing.
        log_mag_weight: ``lambda``, the weight of the log-magnitude term
        eps: added to the mean square before the square root, and to the RMS before the log
        return_terms: return ``(shape, logmag)`` instead of the weighted sum

    Returns:
        Scalar loss.
    """
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} and target {target.shape} must match")

    if mask is None:
        w = jnp.ones_like(pred)
    else:
        w = jnp.broadcast_to(jnp.asarray(mask, dtype=pred.dtype), pred.shape)

    per_bc = jnp.maximum(jnp.sum(w, axis=-2, keepdims=True), 1.0)
    pred_rms = jnp.sqrt(jnp.sum(w * pred**2, axis=-2, keepdims=True) / per_bc + eps)
    tgt_rms = jnp.sqrt(jnp.sum(w * target**2, axis=-2, keepdims=True) / per_bc + eps)

    shape = jnp.sum(w * (pred / pred_rms - target / tgt_rms) ** 2) / jnp.maximum(
        jnp.sum(w), 1.0
    )
    logmag = jnp.mean((jnp.log(pred_rms + eps) - jnp.log(tgt_rms + eps)) ** 2)

    if return_terms:
        return shape, logmag
    return shape + log_mag_weight * logmag


def smoothness(
    currents: Float[Array, "*batch time unit"],
    dt: float,
    sigma: float = 0.75,
    scale: float = 1.0,
    warmup_samples: int = 0,
):
    """High-pass energy of a trace: ``mean((x - gaussian_lowpass(x))**2) / scale**2``.

    Args:
        currents: ``(*batch, time, unit)`` trace
        dt: sample interval in ms
        sigma: Gaussian width of the lowpass
        scale: normalisation, the loss is divided by ``max(scale**2, 1e-8)``
        warmup_samples: leading samples dropped before the lowpass, as for the other trace losses

    Returns:
        Scalar loss.
    """
    x = jnp.asarray(currents)[..., warmup_samples:, :]
    sigma_samples = max(float(sigma) / max(float(dt), 1e-9), 1e-6)
    radius = max(1, round(3.0 * sigma_samples))
    offsets = jnp.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = jnp.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel = kernel / jnp.maximum(jnp.sum(kernel), 1e-12)

    pad = [(0, 0)] * (x.ndim - 2) + [(radius, radius), (0, 0)]
    padded = jnp.pad(x, pad, mode="edge")
    # the kernel is symmetric, so convolution equals correlation
    flat = jnp.moveaxis(padded, -2, -1).reshape(-1, padded.shape[-2])
    low = jax.vmap(lambda row: jnp.convolve(row, kernel, mode="valid"))(flat)
    low = jnp.moveaxis(
        low.reshape(*padded.shape[:-2], x.shape[-1], x.shape[-2]), -1, -2
    )

    return jnp.mean((x - low) ** 2) / jnp.maximum(float(scale) ** 2, 1e-8)


def events_to_mask(
    events: Float[Array, "*batch unit spike"],
    shape: tuple[int, int],
    dt: float,
    window: float,
    channel_radius: int = 0,
    channels: Int[Array, "unit ..."] | None = None,
) -> Bool[Array, "*batch time channel"]:
    """Boolean ``(time, channel)`` mask of a window around every spike, on its unit's channel.

    Fixed shapes and no host round trip, so it can sit inside the jitted loss. A spike at ``t``
    marks samples ``s - h .. s + h`` (inclusive) with ``s = round(t / dt)`` and
    ``h = max(1, round(window / dt / 2))``, on channels ``c - channel_radius .. c +
    channel_radius`` clipped to the array, for each channel ``c`` of its unit.

    Args:
        events: ``(*batch, unit, spike)`` spike times in ms relative to the window start,
            ``inf``-padded (the ``livn.optimization.data`` batch format)
        shape: ``(time, channel)`` of the trace the mask is applied to
        dt: sample interval in ms
        window: total window width in ms
        channel_radius: channels either side of the unit's channel to include
        channels: ``(unit,)`` channel index per unit, or ``(unit, m)`` with ``-1`` padding for
            units seen on several channels. Defaults to unit ``u`` on channel ``u``.

    Returns:
        ``(*batch, time, channel)`` boolean mask. Spikes outside the window clip or drop out.
    """
    T, C = shape
    events = jnp.asarray(events)
    U = events.shape[-2]
    half = max(1, round(window / max(dt, 1e-9) / 2))
    if channels is None:
        channels = jnp.arange(U)
    channels = jnp.asarray(channels)
    if channels.ndim == 1:
        channels = channels[:, None]
    # (unit, channel): does the unit's window reach this channel
    reach = (channels[:, :, None] >= 0) & (
        jnp.abs(jnp.arange(C)[None, None, :] - channels[:, :, None]) <= channel_radius
    )
    reach = jnp.any(reach, axis=1).astype(jnp.float32)

    def one(ev):
        ok = jnp.isfinite(ev)
        s = jnp.round(jnp.where(ok, ev, 0.0) / dt)
        # clip in float first so huge times cannot overflow the int cast
        lo = jnp.clip(s - half, 0, T).astype(jnp.int32)
        hi = jnp.clip(s + half + 1, 0, T).astype(jnp.int32)
        lo = jnp.where(ok, lo, 0)
        hi = jnp.where(ok, hi, 0)
        # difference array: +1 where a window opens, -1 where it closes
        unit = jnp.broadcast_to(jnp.arange(U)[:, None], ev.shape)
        diff = jnp.zeros((U, T + 1), jnp.int32)
        diff = diff.at[unit, lo].add(1).at[unit, hi].add(-1)
        covered = jnp.cumsum(diff, axis=-1)[:, :T] > 0
        return (covered.astype(jnp.float32).T @ reach) > 0

    flat = events.reshape(-1, *events.shape[-2:])
    out = jax.vmap(one)(flat)
    return out.reshape(*events.shape[:-2], T, C)
