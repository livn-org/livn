from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    floor: float | None = None
    scale: float | None = None
    gate_floor: float | None = None
    signed: bool = False

    @property
    def gate_slack(self) -> float:
        if self.gate_floor is not None:
            return float(self.gate_floor)
        return float(self.floor or 0.0)


FEATURES: dict[str, Feature] = {
    # channels: bursts and population structure
    "isi_cv": Feature(gate_floor=0.05),
    "fano_factor": Feature(floor=0.1),
    "pop_autocorr_tau": Feature(floor=1.0),
    "burst_rate": Feature(floor=0.05),
    "max_synchronous_peak": Feature(floor=0.05),
    "burst_width_ms": Feature(floor=5.0),
    "spikes_per_unit_per_burst": Feature(floor=0.1),
    "burst_onset_peak": Feature(floor=0.05),
    "burst_interval_cv": Feature(floor=0.05),
    "units_recruited_per_burst": Feature(floor=0.05),
    # the stimulus response
    "response_latency_ms": Feature(floor=5.0),
    "response_duration_ms": Feature(floor=10.0),
    # units against bursts (`livn.decoding.RecruitmentOrder`)
    "unit_participation": Feature(floor=0.05),
    "unit_between_rate_hz": Feature(floor=0.05),
    "order_spread_ms": Feature(floor=2.0),
    "unit_spikes_per_burst": Feature(floor=0.1),
    "unit_active_between_fraction": Feature(floor=0.02),
    "between_time_rho": Feature(scale=0.2, gate_floor=0.05, signed=True),
    # units without bursts (`livn.decoding.UnitCoordination`)
    "unit_rate_median": Feature(floor=0.05),
    "unit_rate_cv": Feature(floor=0.05),
    "unit_top10_share": Feature(gate_floor=0.02),
    "unit_corr_median": Feature(scale=0.01, gate_floor=0.005, signed=True),
    "coordination_excess": Feature(floor=0.05),
}


def feature(name: str) -> Feature:
    return FEATURES.get(name, Feature())
