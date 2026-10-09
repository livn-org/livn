from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from livn.decoding import (
    ISICV,
    ActiveFraction,
    AvalancheAnalysis,
    BurstAnatomy,
    BurstRate,
    MeanFiringRate,
    PairwiseChannelCorrelation,
    PeakSynchrony,
    PerUnitFiringRate,
    PopulationAutocorrTau,
    PopulationRateMetrics,
)
from livn.utils import P

BURST_MIN_FLOOR_FRACTION = 0.10
MIN_SPIKES_FOR_CORRELATION = 150

ANATOMY_FEATURES = (
    "burst_width_ms",
    "spikes_per_unit_per_burst",
    "burst_onset_peak",
    "burst_interval_cv",
    "units_recruited_per_burst",
)

CHANNEL_FEATURES = (
    "mfr",
    "isi_cv",
    "burst_rate",
    "pop_autocorr_tau",
    "mean_channel_correlation",
    "max_synchronous_peak",
    "fano_factor",
    "coefficient_of_variation",
    "max_neuron_firing_rate",
    "active_fraction",
    "branching_ratio",
)

ORDER_FEATURES = (
    "unit_participation",
    "unit_between_rate_hz",
    "order_spread_ms",
    "unit_spikes_per_burst",
    "unit_active_between_fraction",
    "between_time_rho",
)

ASYNC_FEATURES = (
    "unit_rate_median",
    "unit_rate_cv",
    "unit_top10_share",
    "unit_corr_median",
    "coordination_excess",
)


def channel_env(units, comm=None, **extra) -> SimpleNamespace:
    """The `env` the decoders need to read spikes keyed by `units`."""
    return SimpleNamespace(
        comm=comm,
        system=SimpleNamespace(gids=[int(u) for u in units]),
        io=None,
        voltage_recording_dt=None,
        **extra,
    )


def resting_features(data, env, duration: float) -> dict:
    d = int(duration)
    nan = float("nan")

    def get(result, key, default):
        value = (result or {}).get(key, default)
        return default if value is None else float(value)

    local = 0 if data.spike_ids is None else len(data.spike_ids)
    total = P.reduce_sum(np.array(local, dtype=np.int64), comm=env.comm, all=True)
    total = int(getattr(total, "item", lambda: total)())

    out: dict = {"total_spikes": total}
    out["mfr"] = get(MeanFiringRate(duration=d)(data, env), "rate_hz", 0.0)
    per_unit = PerUnitFiringRate(duration=d)(data, env) or {}
    out["per_unit_rates_hz"] = per_unit.get("per_unit_rates_hz", {})
    out["max_neuron_firing_rate"] = get(per_unit, "max_rate_hz", 0.0)
    isi = ISICV(duration=d, min_spikes_per_unit=5)(data, env) or {}
    out["isi_cv"] = get(isi, "isi_cv", 0.0)
    out["isi_cv_n_units_used"] = int(isi.get("n_units_used", 0))
    pop = PopulationRateMetrics(duration=d, bin_size=100.0)(data, env) or {}
    out["coefficient_of_variation"] = get(pop, "coefficient_of_variation", 0.0)
    out["fano_factor"] = get(pop, "fano_factor", 0.0)
    out["pop_autocorr_tau"] = get(
        PopulationAutocorrTau(duration=d, bin_size=10.0, max_lag=5000.0)(data, env),
        "pop_autocorr_tau",
        nan,
    )
    out["mean_channel_correlation"] = (
        get(
            PairwiseChannelCorrelation(duration=d, bin_size=10.0, min_units=2)(
                data, env
            ),
            "mean_pairwise_correlation",
            0.0,
        )
        if total >= MIN_SPIKES_FOR_CORRELATION
        else nan
    )
    out["max_synchronous_peak"] = get(
        PeakSynchrony(duration=d, bin_size=2.0)(data, env),
        "max_synchronous_peak",
        0.0,
    )
    out["burst_rate"] = get(
        BurstRate(
            duration=d,
            bin_size=50.0,
            mad_k=3.0,
            min_floor_fraction=BURST_MIN_FLOOR_FRACTION,
            min_floor=2.0,
        )(data, env),
        "burst_rate_hz",
        0.0,
    )
    anatomy = BurstAnatomy(duration=d)(data, env) or {}
    out["burst_anatomy"] = anatomy
    for name in ANATOMY_FEATURES:
        out[name] = get(anatomy, name, nan)
    out["n_bursts"] = int(anatomy.get("n_bursts", 0))
    out["active_fraction"] = get(
        ActiveFraction(duration=d, min_spikes=1)(data, env), "active_fraction", 0.0
    )

    avalanche = None
    if total > 0:
        width = max(4.0, min(d / max(50, total // 15), 50.0))
        avalanche = AvalancheAnalysis(duration=d, bin_width=width)(data, env)
    out["avalanche_result"] = avalanche
    out["branching_ratio"] = get(avalanche, "branching_ratio", 0.0)
    out["avalanche_r2"] = get(avalanche, "size_power_law_r2", 0.0)
    return out
