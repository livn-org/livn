import hashlib
import json
import logging
import math
import os
import re
import time
import zlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
from machinable.config import Field as ConfigField
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from livn.decoding import (
    GatherAndMerge,
    PopulationActiveFraction,
    RecruitmentCurve,
    Slice,
    Stability,
    StimulusResponse,
)
from livn.env.logging import with_progress_logging
from livn.policy import PulseSweepPolicy
from livn.utils import P, sentinel
from systems.targets.EI import measure
from systems.targets.EI.features import feature
from systems.targets.EI.measure import resting_features
from systems.targets.protocol import Sizing, Target, digest, note
from systems.targets.schema import FREE_RUNNING, read_target

logger = logging.getLogger(__name__)


def logged(document: dict) -> set[tuple[str, str | None]]:
    experiments = (document.get("metadata") or {}).get("experiments") or {}
    found = set()
    for name, recording in (document.get("recordings") or {}).items():
        channels = [c["channel"] for c in (recording.get("channels") or ())]
        date = re.match(r"\d{4}-(\d{2}-\d{2})_", name)
        if not channels or date is None:
            continue
        array = f"MEA_{chr(ord('A') + min(channels) // 128)}"
        entry = ((experiments.get(date.group(1)) or {}).get("samples") or {}).get(
            array
        ) or {}
        if entry.get("sample"):
            found.add((entry["sample"], entry.get("composition")))
    return found


def composition_of(document: dict) -> str | None:
    seeded = {composition for _, composition in logged(document) if composition}
    return seeded.pop() if len(seeded) == 1 else None


def geometry(metadata: dict, scale: float = 1.0, guard: float | None = None) -> dict:
    pos = metadata["geometry"]["pos"]
    xs = [p[1] for p in pos]
    ys = [p[2] for p in pos]
    pitch = float(metadata["geometry"]["pitch"][0])

    recorded = pitch / 2.0
    margin = recorded if guard is None else float(guard)

    def _box(m, s):
        a0, a1 = min(xs) - m, max(xs) + m
        b0, b1 = min(ys) - m, max(ys) + m
        if s != 1.0:
            cx, cy = (a0 + a1) / 2.0, (b0 + b1) / 2.0
            hw, hh = (a1 - a0) * s / 2.0, (b1 - b0) * s / 2.0
            a0, a1, b0, b1 = cx - hw, cx + hw, cy - hh, cy + hh
        return (a0, b0), (a1, b1)

    (x0, y0), (x1, y1) = _box(margin, scale)
    (ix0, iy0), (ix1, iy1) = _box(recorded, scale)

    electrodes = [
        [int(i), float(x), float(y)]
        for i, x, y in pos
        if ix0 <= x <= ix1 and iy0 <= y <= iy1
    ]
    return {
        "area": ((x0, y0), (x1, y1)),
        "interior": ((ix0, iy0), (ix1, iy1)),
        "electrodes": electrodes,
        "margin": float(margin - recorded) * float(scale),
    }


def _max_constraint(value, max_val, scale=None):
    """+1 when value <= max_val, <0 otherwise. NaN -> -10."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -10.0
    v = float(value)
    s = max(abs(max_val), 1e-6) if scale is None else max(float(scale), 1e-6)
    if v <= max_val:
        return 1.0 + (max_val - v) / s
    return -1.0 - (v - max_val) / s


def _min_constraint(value, min_val, scale=None):
    """+1 when value >= min_val, <0 otherwise. NaN -> -10."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -10.0
    v = float(value)
    s = max(abs(min_val), 1e-6) if scale is None else max(float(scale), 1e-6)
    if v >= min_val:
        return 1.0 + (v - min_val) / s
    return -1.0 - (min_val - v) / s


def _band_constraint(value, lo, hi, edge_slope=2.0, inside_penalty=0.1):
    """Band feasibility constraint."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return -2.0
    v = float(value)
    if v < lo:
        return -((lo - v) / max(abs(lo), 1e-6)) * edge_slope
    if v > hi:
        return -((v - hi) / max(abs(hi), 1e-6)) * edge_slope
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return 1.0 - (abs(v - center) / max(half, 1e-6)) * inside_penalty


class Protocol(PulseSweepPolicy):
    amplitudes: tuple[float, ...] = (300.0, 400.0, 500.0, 600.0)
    """Recruitment curve."""

    trial_ms: float = Field(default=6000.0, gt=0)
    """Spacing between pulses."""

    pre_ms: float = Field(default=1000.0, gt=0)
    """Baseline taken from this long before each pulse."""

    post_ms: float = Field(default=500.0, gt=0)
    """The response proper, over which gain and latency are measured."""

    recovery_ms: float = Field(default=3500.0, ge=0)
    """Quiet time required between one response ending and the next baseline."""

    electrode: int | None = None
    """Driving channels"""

    uA_per_mv: float = Field(default=1.0, gt=0)
    """Microamps the electrode delivers per millivolt of input."""

    def _render(self, start_ms: float, stop_ms: float, dt: float, strict: bool):
        return super()._render(start_ms, stop_ms, dt, strict) * self.uA_per_mv

    @staticmethod
    def _extended(amplitudes: tuple, probe_to_mv: float | None) -> tuple:
        if probe_to_mv is None or probe_to_mv <= amplitudes[-1]:
            return amplitudes

        step = amplitudes[-1] - amplitudes[-2]
        if step <= 0:
            raise ValueError(
                f"the measured sweep ends {amplitudes[-2]:g}, {amplitudes[-1]:g} "
                "mV, which does not rise, so there is no spacing to continue"
            )
        extra = []
        at = amplitudes[-1] + step
        while at <= probe_to_mv + 1e-9:
            extra.append(float(at))
            at += step
        return (*amplitudes, *extra)

    @classmethod
    def from_block(
        cls, block, probe_to_mv: float | None = None, **overrides
    ) -> "Protocol":
        amplitudes = block.amplitudes_mv
        if len(amplitudes) < 2:
            raise ValueError(
                f"{block.condition!r} block carries {len(amplitudes)} < 2 amplitude(s)"
            )
        if not block.onsets_ms:
            raise ValueError(
                f"{block.condition!r} block records no pulse times, so "
                "there is nowhere to put the stimulus"
            )
        if min(amplitudes) <= 0:
            raise ValueError(
                f"{block.condition!r} block carries a {min(amplitudes):g} mV"
            )

        # the array's calibration: a 300 mV command drives 8 uA
        overrides.setdefault("uA_per_mv", 8.0 / 300.0)

        return cls(
            amplitudes=cls._extended(tuple(amplitudes), probe_to_mv),
            onset_ms=float(block.onsets_ms[0]),
            **overrides,
        )

    @model_validator(mode="after")
    def _the_measurement_fits_the_trial(self):
        if self.electrode is not None and not self.channels:
            self.channels = [self.electrode]
        if self.onset_ms < self.pre_ms:
            raise ValueError(
                f"the pulse falls {self.onset_ms:g} ms into its trial but the "
                f"baseline is {self.pre_ms:g} ms, so there is not enough before "
                "it to measure the response against"
            )
        if self.onset_ms + self.post_ms > self.trial_ms:
            raise ValueError(
                f"a {self.post_ms:g} ms response after a pulse at "
                f"{self.onset_ms:g} ms runs past the {self.trial_ms:g} ms trial"
            )
        if self.recovery_ms and self.quiet_ms < self.recovery_ms:
            raise ValueError(
                f"a {self.trial_ms:g} ms trial leaves {self.quiet_ms:g} ms "
                f"between the end of one response and the start of the next "
                f"pulse's baseline, and the network needs {self.recovery_ms:g} "
                f"ms to be back where it started. Give the trial at least "
                f"{self.pre_ms + self.post_ms + self.recovery_ms:g} ms, or "
                "lower `recovery_ms` if this preparation really does settle "
                "faster"
            )
        return self

    @property
    def quiet_ms(self) -> float:
        """Undriven time between a response ending and the next baseline."""
        return self.trial_ms - self.pre_ms - self.post_ms


RECRUITED = 0.5
CENSORED_SLOPE = 20.0
MAX_CENSORED_DECADES = 1.0
_P_FLOOR = 1e-3


def _logit(p: float) -> float:
    p = min(max(float(p), _P_FLOOR), 1.0 - _P_FLOOR)
    return math.log(p / (1.0 - p))


def _recruitment_slope(amplitudes: list, probabilities: list) -> float | None:
    if any(a <= 0 for a in amplitudes) or len(amplitudes) < 2:
        return None
    xs = [math.log10(a) for a in amplitudes]
    ys = [_logit(p) for p in probabilities]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return slope if slope > 0.0 else None


def _past_the_end(
    bracket: dict, amplitudes: list, step: float, *, above: bool
) -> float:
    probabilities = [float(p) for p in bracket.get("probabilities") or ()]
    if not probabilities or len(probabilities) != len(amplitudes):
        return math.log10(step)

    criterion = _logit(float(bracket.get("recruited", RECRUITED)))
    shortfall = criterion - _logit(probabilities[-1 if above else 0])
    if not above:
        shortfall = -shortfall

    slope = _recruitment_slope(amplitudes, probabilities) or CENSORED_SLOPE
    return min(max(shortfall, 0.0) / slope, MAX_CENSORED_DECADES)


def recruitment_miss(measured: dict, simulated: dict) -> float:
    if not measured or not simulated:
        return float("nan")

    m_amplitudes = [float(a) for a in measured.get("amplitudes_mv") or ()]
    m_probabilities = [float(p) for p in measured.get("probabilities") or ()]
    s_amplitudes = [float(a) for a in simulated.get("amplitudes_mv") or ()]
    s_probabilities = [float(p) for p in simulated.get("probabilities") or ()]
    if len(m_amplitudes) != len(m_probabilities) or not m_amplitudes:
        return float("nan")
    if len(s_amplitudes) != len(s_probabilities) or not s_amplitudes:
        return float("nan")

    simulated_at = dict(zip(s_amplitudes, s_probabilities, strict=True))
    shared = [
        (a, p)
        for a, p in zip(m_amplitudes, m_probabilities, strict=True)
        if a in simulated_at
    ]
    if not shared:
        return float("nan")

    squares = [(_logit(simulated_at[a]) - _logit(p)) ** 2 for a, p in shared]
    return math.sqrt(sum(squares) / len(squares))


def threshold_miss(measured: dict, simulated: dict) -> float:
    if not measured or not simulated:
        return float("nan")

    def point(bracket: dict) -> float:
        amplitudes = [float(a) for a in bracket.get("amplitudes_mv") or ()]
        step = (
            amplitudes[-1] / amplitudes[-2]
            if len(amplitudes) >= 2 and amplitudes[-2] > 0
            else 1.0
        )

        censored = bracket.get("censored")
        if censored == "above":
            past = _past_the_end(bracket, amplitudes, step, above=True)
            return float(bracket["highest_tested_mv"]) * 10.0**past
        if censored == "below":
            past = _past_the_end(bracket, amplitudes, step, above=False)
            return float(bracket["above_mv"]) / 10.0**past
        return math.sqrt(float(bracket["below_mv"]) * float(bracket["above_mv"]))

    difference = math.log10(point(simulated)) - math.log10(point(measured))
    censored = measured.get("censored")
    if censored == "above":
        return max(0.0, -difference)
    if censored == "below":
        return max(0.0, difference)
    return abs(difference)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = 1.0
    sigma: float | None = 300.0
    degree: float = 20.0
    boundary: float | None = None
    cells: int = 2600
    inhibitory_fraction: float | None = None
    size_cv: float = 0.2
    weight_cv: float = 0.7


class Culture(Target):
    RATIO_SUFFIX = "_ratio"
    STRUCTURE_PREFIX = "system-"
    COMPOSITION_KEY = "inhibitory_fraction"
    ANATOMY_FEATURES: ClassVar[tuple] = measure.ANATOMY_FEATURES
    RESPONSE_FEATURES: ClassVar[tuple] = (
        "response_gain",
        "evoked_rate_hz",
        "response_peak_hz",
        "response_latency_ms",
        "response_probability",
        "response_duration_ms",
    )
    RESPONSE_OBJECTIVES: ClassVar[tuple] = (
        "response_latency_ms",
        "response_duration_ms",
    )
    MEASURED_FEATURES: ClassVar[tuple] = (
        "mfr",
        "isi_cv",
        "active_fraction",
        "mean_channel_correlation",
        "max_synchronous_peak",
        "max_neuron_firing_rate",
        "pop_rate_hz",
        "pop_rate_per_unit_hz",
        "pop_autocorr_tau",
        "burst_rate",
        "branching_ratio",
        "avalanche_r2",
        "fano_factor",
        "coefficient_of_variation",
        *ANATOMY_FEATURES,
        *RESPONSE_FEATURES,
    )
    MIN_SPIKE_COUNT_FOR_METRICS = measure.MIN_SPIKES_FOR_CORRELATION
    EXCITATORY_CELLS: ClassVar[dict] = {
        "synapse_type": "excitatory",
        "transmitter": "cholinergic",
        "soma_only": False,
    }
    INHIBITORY_CELLS: ClassVar[dict] = {
        "synapse_type": "inhibitory",
        "transmitter": "glycinergic",
        "soma_only": True,
    }
    INHIBITORY_DEGREE: ClassVar[dict] = {"INH->EXC": 40.0, "EXC->INH": 4.0}
    DEGREE_REFERENCE: ClassVar[dict] = {
        "EXC->EXC": 1.0,
        "INH->EXC": 0.5,
        "EXC->INH": 0.5,
    }
    SYNAPSE_OVERRIDES: ClassVar[dict] = {
        "EXC->EXC": {
            "NMDA": {
                "e": 0,
                "g_unit": 0.0005,
                "tau_decay": 80.0,
                "tau_rise": 0.5,
                "weight": 0.0,
            }
        }
    }
    ELECTRODE_RADIUS_UM = 50.0
    ELECTRODE_HEIGHT_UM = 5.0
    OBJECTIVES: ClassVar[tuple] = (
        "fano_factor",
        "burst_rate",
        "pop_autocorr_tau",
        "max_synchronous_peak",
        "burst_width_ms",
        "spikes_per_unit_per_burst",
        "burst_onset_peak",
        "burst_interval_cv",
        "units_recruited_per_burst",
    )
    OBSERVED: ClassVar[tuple] = (
        "active_fraction",
        "pop_autocorr_tau",
        "burst_onset_peak",
        "units_recruited_per_burst",
    )
    OBSERVED_WHEN_BURSTING: ClassVar[tuple] = (
        "fano_factor",
        "mean_channel_correlation",
        "max_synchronous_peak",
    )
    ANATOMY_MIN_WINDOW_FRACTION = 0.1
    BURST_OBJECTIVES: ClassVar[tuple] = (
        "fano_factor",
        "pop_autocorr_tau",
        "burst_rate",
        "max_synchronous_peak",
        "burst_width_ms",
        "spikes_per_unit_per_burst",
        "burst_onset_peak",
        "burst_interval_cv",
        "units_recruited_per_burst",
    )
    MEASURED_GATES: ClassVar[tuple] = (
        "MAX_NEURON_RATE_HZ",
        "MIN_MEAN_RATE_HZ",
        "MAX_MEAN_RATE_HZ",
        "SYNCHRONY_BAND",
        "MAX_SYNC_PEAK",
        "MIN_SYNC_PEAK",
        "MIN_ACTIVE_FRACTION",
        "POP_TAU_BAND_MS",
        "MAX_BURST_RATE_HZ",
        "MIN_BURST_RATE_HZ",
        "BRANCHING_RATIO_BAND",
        "MIN_AVALANCHE_R2",
        "MAX_POP_RATE_PER_UNIT_HZ",
        "MIN_POP_RATE_PER_UNIT_HZ",
    )
    max_neuron_rate_hz = 50.0
    min_mean_rate_hz = 0.2
    max_mean_rate_hz = 15.0
    synchrony_band = (0.02, 0.25)
    max_sync_peak = 0.2
    min_sync_peak = 0.0
    min_active_fraction = 0.5
    pop_tau_band_ms = (10.0, 500.0)
    max_burst_rate_hz = 0.2
    min_burst_rate_hz = 0.0
    branching_ratio_band = (0.5, 1.5)
    min_avalanche_r2 = 0.5
    max_pop_rate_per_unit_hz = 20.0
    min_pop_rate_per_unit_hz = 0.05

    LIVENESS: ClassVar[tuple] = ("not_runaway", "not_quiescent", "is_stable")
    MIN_POPULATION_ACTIVE = 0.05
    STABILITY_MARGIN = 5.0
    STABILITY_TAIL_MS = 5000.0

    BURST_MIN_FLOOR_FRACTION = measure.BURST_MIN_FLOOR_FRACTION
    NOISE_STD = 0.0003
    NOISE_TOTAL_RANGE: ClassVar[list] = [0.0005, 0.002]
    NOISE_RATIO_RANGE: ClassVar[list] = [8.0, 20.0]
    NOISE_STD_FRACTION_RANGE: ClassVar[list] = [0.05, 1.0]

    NOISE_TAU_RANGES: ClassVar[dict] = {
        "tau_e": [1.0, 100.0],
        "tau_i": [4.0, 100.0],
    }
    EXC_WEIGHT_RANGE: ClassVar[list] = [0.15, 1.0]
    INH_WEIGHT_RANGE: ClassVar[list] = [0.05, 100.0]
    NMDA_RATIO_RANGE: ClassVar[list] = [0.2, 1.0]
    RATIO_RANGES: ClassVar[dict] = {
        "excitatory": [0.01, 1000.0],
        "inhibitory": [0.01, 1000.0],
    }
    DEPRESSION_RANGES: ClassVar[dict] = {
        "tau_rec": [300.0, 3000.0],
        "U": [0.05, 0.5],
    }
    LEADER_RANGES: ClassVar[dict | None] = None
    UNIT_FEATURES: ClassVar[tuple] = ()
    LEADER_SELECTION: ClassVar[str] = "hash"
    STRUCTURE_RANGES: ClassVar[dict] = {
        "EXC->EXC": [5.0, 200.0],
        "INH->EXC": [6.3, 252.0],
        "EXC->INH": [0.65, 25.2],
        "inhibitory_fraction": [0.02, 0.7],
        "sigma": [150.0, 400.0],
    }

    class Config(Target.Config):
        model_config = ConfigDict(extra="forbid")

        sizing: Sizing = Sizing(
            min_ranks_per_worker=2,
            nprocs_per_worker=2,
            n_initial=25,
        )

        observation: str = ConfigField(identifying=False)
        _normalise = field_validator("observation")(
            lambda path: os.path.normpath(path) if path else path
        )
        spec: Spec = Spec()
        skip_objectives: list[str] = ConfigField(default_factory=list)
        structure: dict[str, tuple[float, float]] | bool = True
        save_spikes: bool | Literal["feasible", "all"] = "all"

        @model_validator(mode="after")
        def _consistent(self):
            if self.structure is True:
                self.structure = Culture.STRUCTURE_RANGES
            if self.structure:
                self.structure = {
                    k: (float(v[0]), float(v[1])) for k, v in self.structure.items()
                }
                bad = {
                    k: v for k, v in self.structure.items() if not 0.0 < v[0] <= v[1]
                }
                if bad:
                    raise ValueError(
                        f"structural bounds must satisfy 0 < lo <= hi; got {bad}"
                    )
            if self.save_spikes is True:
                self.save_spikes = "feasible"
            return self

    def measurement(self) -> dict:
        return {
            name: {
                "ei_targets": (block.get("pooled") or {}).get("ei_targets"),
                "summary": (block.get("pooled") or {}).get("summary"),
                "threshold": (block.get("pooled") or {}).get("threshold"),
            }
            for name, block in sorted((self.document.get("conditions") or {}).items())
        }

    RUNTIME_CONSTANTS: ClassVar[tuple] = (
        "REBUILD_BUDGET",
        "REBUILD_LEAK_BYTES_PER_CELL",
        "RUNTIME_CONSTANTS",
    )

    def definition(self) -> dict:
        import dataclasses

        from systems.targets.EI.features import FEATURES

        def plain(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, Mapping):
                return {str(k): plain(v) for k, v in value.items()}
            if callable(value):
                return f"{getattr(value, '__module__', '')}.{value.__qualname__}"
            raise TypeError(f"{value!r} is not part of a problem definition")

        cls = type(self)
        stated = {
            name: plain(getattr(cls, name))
            for name in dir(cls)
            if name.isupper()
            and name not in self.RUNTIME_CONSTANTS
            and not isinstance(getattr(cls, name), type)
        }
        stated["FEATURES"] = {k: dataclasses.asdict(v) for k, v in FEATURES.items()}
        return {"target": f"{cls.__module__}.{cls.__qualname__}", **stated}

    def on_compute_predicate(self):
        stated = {k: v for k, v in self.settings.items() if k != "observation"}
        return {
            "culture": self.sample,
            "measurement": digest(self.measurement()),
            "problem": digest(stated),
            "definition": digest(self.definition()),
        }

    def version_fit(self, observation: str, **options):
        return {"observation": observation, **options}

    def system_spec(self):
        spec = self.config.spec
        metadata = self.document["metadata"]
        fraction = spec.inhibitory_fraction
        composed = (self.config.structure or {}).get(self.COMPOSITION_KEY)
        if composed is not None and fraction is None:
            fraction = math.sqrt(float(composed[0]) * float(composed[1]))
            self._note(
                f"the composition is searched, so the base spec is drawn at an "
                f"inhibitory fraction of {fraction:.3f}"
            )
        if fraction is None:
            fraction = self.plated_inhibitory_fraction

        scale = float(spec.scale)
        pitch = float(metadata["geometry"]["pitch"][0])
        boundary = spec.boundary
        guard = pitch / 2.0 + (0.0 if boundary is None else float(boundary))
        geo = geometry(metadata, scale=scale, guard=guard)
        (x0, y0), (x1, y1) = geo["area"]
        (ix0, iy0), (ix1, iy1) = geo["interior"]
        widening = ((x1 - x0) * (y1 - y0)) / ((ix1 - ix0) * (iy1 - iy0))
        total = max(1, round(int(spec.cells) * scale**2 * widening))

        share = {"EXC": 1.0 - float(fraction), "INH": float(fraction)}
        degrees = {"INH->INH": 0.0, "default": 0.0}
        for projection, value in (
            {"EXC->EXC": float(spec.degree)} | self.INHIBITORY_DEGREE
        ).items():
            pre, post = projection.split("->")
            degrees[projection] = (
                0.0
                if share[pre] <= 0.0 or share[post] <= 0.0
                else value * share[pre] / self.DEGREE_REFERENCE[projection]
            )
        inhibitory = int(total * share["INH"])
        if degrees["INH->EXC"] > max(inhibitory, 0) > 0:
            raise ValueError(
                f"INH->EXC={degrees['INH->EXC']:.0f} needs at least that many "
                f"inhibitory cells, but {total} cells at ratio "
                f"{share['INH']:g} gives {inhibitory}"
            )

        connectivity = {
            "mean_degree": degrees,
            "degree_rule": "fixed_probability",
            "degree_reference": dict(self.DEGREE_REFERENCE),
        }
        if spec.sigma is not None:
            connectivity["sigma"] = float(spec.sigma)

        z = float(self.ELECTRODE_HEIGHT_UM)
        radius = float(self.ELECTRODE_RADIUS_UM)
        mea = {
            "electrode_coordinates": [
                [float(i), float(x), float(y), z]
                for i, x, y in geometry(metadata, scale=scale)["electrodes"]
            ],
            "input_radius": radius,
            "output_radius": radius,
        }
        sample = self.sample
        return {
            "cls": "livn.system.Monolayer",
            "kwargs": {
                "total_cells": total,
                "populations": {
                    "EXC": self.EXCITATORY_CELLS | {"ratio": share["EXC"]},
                    "INH": self.INHIBITORY_CELLS | {"ratio": share["INH"]},
                },
                "connectivity": connectivity,
                "area": "rectangle",
                "area_kwargs": {"x_range": [x0, x1], "y_range": [y0, y1]},
                "boundary": None if boundary is None else geo["margin"],
                "synapse_overrides": self.SYNAPSE_OVERRIDES,
                "seed": zlib.crc32(sample.encode()) % 100_000,
                "name": f"{sample}@{metadata['name']}",
                "mea": mea,
            },
        }

    @property
    def plated_inhibitory_fraction(self) -> float:
        composition = composition_of(self.document)
        if composition is None:
            raise ValueError(
                f"{self.config.observation!r} states no composition for {self.sample!r}"
            )
        if composition == "E":
            return 0.0
        try:
            excitatory, inhibitory = (float(p) for p in composition.split("/"))
        except ValueError:
            raise ValueError(
                f"composition {composition!r} is neither 'E' nor 'a/b'"
            ) from None
        return inhibitory / (excitatory + inhibitory)

    def model_spec(self):
        return [
            "livn.models.rcsd.ReducedCalciumSomaDendrite",
            {
                "size_cv": float(self.config.spec.size_cv),
                "weight_cv": float(self.config.spec.weight_cv),
            },
        ]

    @property
    def document(self) -> dict:
        if "document" not in self._cache:
            with open(self.config.observation) as f:
                self._cache["document"] = json.load(f)
        return self._cache["document"]

    @property
    def condition(self) -> str:
        if "condition" not in self._cache:
            blocks = self.document.get("conditions") or {}
            name = next((n for n in FREE_RUNNING if blocks.get(n)), None)
            if name is None:
                raise ValueError(
                    f"{self.config.observation!r} holds no free-running block "
                    f"({' or '.join(FREE_RUNNING)}) to fit the resting features "
                    f"to; it has {sorted(blocks)}"
                )
            if name != "spontaneous":
                self._note(
                    f"{os.path.basename(self.config.observation)} has no "
                    f"'spontaneous' block, so the resting features come from {name!r}"
                )
            self._cache["condition"] = name
        return self._cache["condition"]

    @property
    def sample(self) -> str:
        named = {sample for sample, _ in logged(self.document)}
        if len(named) != 1:
            raise ValueError(
                f"{self.config.observation!r} reads as {sorted(named) or 'no'} "
                "sample(s) of the experiment log, so a network cannot be drawn"
            )
        return next(iter(named))

    def _configure(self):
        self._targets = {"mfr": 1.0, "isi_cv": 1.2, "active_fraction": 1.0}
        self.feature_bands: dict[str, tuple[float, float]] = {}
        self.mea = None
        self.stimulus = None
        self.stimulus_threshold: dict = {}
        self.response_kwargs: dict = {}
        self.response_blank_ms = (0.0, 0.0)
        self.skip_objectives: tuple[str, ...] = tuple(self.config.skip_objectives)
        self.recording_duration = 20_000.0
        self.warmup_duration = 1_000.0
        self.skip_constraints = ("avalanche_r2",)
        self.structure = dict(self.config.structure or {}) or None
        self.save_spikes = self.config.save_spikes

        self._env = None
        self._sigma_ceiling = sentinel
        self._depression_keys: list[str] = []
        self._weight_space_cache: dict[str, list] | None = None
        self._weight_space_model = False
        self._weight_reference: str | None = None
        self._reset_state()

        self._measure(self.config.observation, self.condition)
        self._gates_from_bands()

    def _measure(self, observation: str, condition: str) -> None:
        from livn.system import resolve

        spec = self.system
        if isinstance(spec, Mapping):
            self.mea = spec.get("kwargs", {}).get("mea")
        if self.mea is None:
            raise ValueError(
                "the spec carries no `mea`, so there are no electrodes to read "
                "the culture's channel features on"
            )

        block = read_target(observation, condition)
        measured = block.ei_targets
        for name, value in measured.items():
            if not name.isupper():
                continue
            if name not in self.MEASURED_GATES:
                raise ValueError(
                    f"{os.path.basename(observation)} states a gate {name!r} "
                    f"that {type(self).__name__} does not read; add it to "
                    "MEASURED_GATES or re-extract the document"
                )
            setattr(self, name.lower(), value)

        targets = dict(measured["targets"])
        lo, hi = measured.get("SYNCHRONY_BAND") or (0.0, 1.0)
        if lo > 0.0 and hi >= 0.01:  # a correlation measurably above zero
            targets["mean_channel_correlation"] = float((lo + hi) / 2.0)
        self._targets = {**self._targets, **targets}

        blocks = len(block.summary.recordings)
        if blocks < 2:
            note(
                f"{os.path.basename(observation)} pools one recording block, so "
                "its quantiles never saw the drift between blocks; bands taken "
                "from the extremes of that block instead"
            )
        self.feature_bands = {}
        for name, stat in block.summary.features.items():
            if stat is None or name in self.skip_constraints:
                continue
            lo, hi = stat.q_lo, stat.q_hi
            if lo is None or hi is None:
                continue
            if blocks < 2 and stat.min is not None and stat.max is not None:
                lo, hi = min(lo, stat.min), max(hi, stat.max)
            lo, hi = float(lo), float(hi)
            if name == "pop_autocorr_tau":
                # no narrower than the decoder resolves: four 10 ms bins
                floor = 40.0
                if hi - lo < floor:
                    centre = 0.5 * (lo + hi)
                    lo, hi = centre - floor / 2.0, centre + floor / 2.0
                lo = max(0.0, lo)
            self.feature_bands[name] = (lo, hi)

        self._score_anatomy(observation, condition)
        self._score_units(observation)
        self._deliver_evoked(observation)

        extracted = (
            ((self.document.get("conditions") or {}).get(condition) or {})
            .get("config", {})
            .get("channels")
        )
        electrodes = [
            int(c[0]) for c in (self.mea or {}).get("electrode_coordinates", ())
        ]
        if extracted is not None and electrodes and len(extracted) > len(electrodes):
            self._note(
                f"{os.path.basename(observation)} was extracted on "
                f"{len(extracted)} channels and this window reads "
                f"{len(electrodes)}. Rate features compare; count-based ones "
                "(Fano, burst rate, correlation, the anatomy family) do not. "
                f"Re-extract with channels= the array's ids for {electrodes}."
            )

        system = resolve(spec)
        self._note(f"{system!r}, {len(electrodes)} electrodes, uuid {system.uuid}")

    def _gates_from_bands(self) -> None:
        bands = self.feature_bands

        def widened(name):
            lo, hi = bands[name]
            slack = (hi - lo) / 2.0
            return max(lo - slack, 0.0), hi + slack

        if "mean_channel_correlation" in bands:
            self.synchrony_band = widened("mean_channel_correlation")
        if "max_synchronous_peak" in bands:
            self.min_sync_peak, self.max_sync_peak = widened("max_synchronous_peak")
        if "pop_autocorr_tau" in bands:
            self.pop_tau_band_ms = widened("pop_autocorr_tau")
        if "burst_rate" in bands:
            self.min_burst_rate_hz, self.max_burst_rate_hz = widened("burst_rate")
        if "mfr" in bands:
            self.min_mean_rate_hz, self.max_mean_rate_hz = widened("mfr")
            self.min_pop_rate_per_unit_hz, self.max_pop_rate_per_unit_hz = widened(
                "mfr"
            )
        if "max_neuron_firing_rate" in bands:
            self.max_neuron_rate_hz = widened("max_neuron_firing_rate")[1]
        if "branching_ratio" in bands:
            self.branching_ratio_band = widened("branching_ratio")
        if "active_fraction" in bands:
            self.min_active_fraction = widened("active_fraction")[0]

    def _score_anatomy(self, observation: str, condition: str) -> None:
        summary = read_target(observation, condition).summary
        features = summary.features
        floor = self.ANATOMY_MIN_WINDOW_FRACTION * max(int(summary.n_windows), 1)

        def measured(name: str) -> bool:
            spec = features.get(name)
            if spec is None or spec.median is None:
                return False
            if name in self.ANATOMY_FEATURES and int(spec.n) < floor:
                self._note(
                    f"{os.path.basename(observation)} measures {name!r} in "
                    f"{spec.n} of {summary.n_windows} windows, under the "
                    f"{self.ANATOMY_MIN_WINDOW_FRACTION:.0%} this protocol "
                    "asks for; treating the culture as non-bursting."
                )
                return False
            return True

        scored = [name for name in self.OBJECTIVES if measured(name)]
        missing = [name for name in self.OBJECTIVES if name not in scored]
        if scored and missing:
            self._note(
                f"{os.path.basename(observation)} measures no {missing} in its "
                f"{condition!r} block, so they are not scored. Re-extract it if "
                "its culture bursts."
            )
        for name in scored:
            self._targets[name] = float(features[name].median)

        # measured and banded, but not steering the search
        observed = list(self.OBSERVED)
        if any(name in self.ANATOMY_FEATURES for name in scored):
            observed += list(self.OBSERVED_WHEN_BURSTING)
        else:
            self._note(
                f"{os.path.basename(observation)} measures no burst anatomy, so "
                f"{list(self.OBSERVED_WHEN_BURSTING)} are scored instead -- they "
                "are the only description of population structure this culture "
                "has, and nothing is left for them to duplicate."
            )
        self.skip_objectives = tuple(
            dict.fromkeys(self.skip_objectives + tuple(observed))
        )

    def _score_units(self, observation) -> None:
        """Targets from a spike sort of the recording; this target scores none."""

    def _unit_metrics(self, network, data, duration: int) -> dict:
        from livn.decoding import RecruitmentOrder

        units = network.io.strongest_units(
            network.active_neuron_coordinates(), network.recording_amplitudes()
        )
        gids = np.asarray(sorted(set(units.values())), dtype=np.int64)
        it = np.asarray(
            data.spike_ids if data.spike_ids is not None else [], dtype=np.int64
        )
        tt = np.asarray(
            data.spike_times if data.spike_times is not None else [], dtype=np.float64
        )
        keep = np.isin(it, gids)
        return (
            RecruitmentOrder(duration=duration)(
                data.add_spikes(it[keep], tt[keep]), network
            )
            or {}
        )

    def _deliver_evoked(self, observation: str) -> None:
        why = None
        try:
            block = read_target(observation, "evoked")
        except (KeyError, ValueError) as reason:
            why = str(reason)
        else:
            if block.threshold is None:
                why = "its recruitment curve was too short to read"
        if why is not None:
            self._note(
                f"{os.path.basename(observation)} has no evoked block to fit "
                f"({why}); fitting the free-running block alone."
            )
            return

        self.stimulus = Protocol.from_block(
            block, repeats=1, trial_ms=float(block.window_ms)
        )
        self.stimulus_threshold = block.threshold.model_dump()
        self._targets.setdefault("stimulus_threshold", 0.0)
        self._adopt_response(block)

    def _adopt_response(self, block) -> None:
        config = (
            (self.document.get("conditions") or {}).get(block.condition) or {}
        ).get("config") or {}
        stated = config.get("response") or {}
        self.response_kwargs = {
            name: float(stated[name])
            for name in ("pre_ms", "post_ms", "bin_size", "tail_ms", "threshold_k")
            if stated.get(name) is not None
        }
        blank = config.get("blank") or (0.0, 0.0)
        self.response_blank_ms = (float(blank[0]), float(blank[1]))

        for name in self.RESPONSE_FEATURES:
            spec = (block.response or {}).get(name) or {}
            target, band = spec.get("target"), spec.get("band")
            if target is None or not np.isfinite(float(target)):
                continue
            if band is not None:
                self.feature_bands[name] = (float(band[0]), float(band[1]))
            if name in self.RESPONSE_OBJECTIVES:
                self._targets[name] = float(target)

    def _reset_state(self):
        self.spec: dict | None = None
        self.response_data: tuple | None = None
        self.metrics: dict = {}
        self.objectives: dict = {}
        self.curve: dict[float, float] = {}
        self.simulated_ms: int = 0
        self.evoked_recorded: bool = False

    DETECTION_THRESHOLD = 0.2

    def io(self):
        if not self.mea:
            return None
        from livn.io import MEA

        mea = MEA.from_json(self.mea)
        mea.detection_threshold = float(self.DETECTION_THRESHOLD)
        return mea

    def init(self, env):
        if not len(getattr(env.io, "channel_ids", ())):
            raise RuntimeError("the channel readout needs an `mea`.")
        self._env = env
        return with_progress_logging(env)

    def objective_names(self) -> list[str]:
        return [n for n in self._targets if n not in self.skip_objectives]

    def observed_feature_names(self) -> list[str]:
        objectives = set(self.objective_names())
        return [
            name
            for name in self.MEASURED_FEATURES
            if name not in objectives and name in self.feature_bands
        ]

    def observed_features(self) -> dict[str, float]:
        values = {}
        for name in self.observed_feature_names():
            value = self.metrics.get(name, float("nan"))
            values[name] = float(value) if value is not None else float("nan")
        return values

    def constraint_names(self) -> list[str]:
        return [
            name
            for name in self._all_constraint_names()
            if name not in self.skip_constraints
        ]

    def _all_constraint_names(self) -> list[str]:
        return [
            "not_runaway",
            "not_quiescent",
            "is_stable",
            "max_firing_rate",
            "synchrony",
            "max_synchronous_peak",
            "min_mean_firing_rate",
            "max_mean_firing_rate",
            "active_fraction_floor",
            "populations_active",
            "pop_autocorr_tau_band",
            "burst_rate_band",
            "branching_ratio_band",
            "avalanche_r2",
        ]

    def targets(self) -> dict[str, float]:
        return self._targets.copy()

    def rank_solutions(self, best):
        y = best.get("y")
        if y is None or len(y) == 0:
            return best

        span = (y.max() - y.min()).replace(0.0, 1.0)
        worst = ((y - y.min()) / span).max(axis=1)

        f = best.get("f")
        outside = np.zeros(len(y), dtype=int)
        if f is not None and self.feature_bands:
            for name, (lo, hi) in self.feature_bands.items():
                if name in getattr(f, "columns", []):
                    outside += (~f[name].between(lo, hi)).to_numpy().astype(int)

        c = best.get("c")
        infeasible = np.zeros(len(y), dtype=int)
        if c is not None and len(getattr(c, "columns", [])):
            infeasible = (c.to_numpy() < 0).sum(axis=1).astype(int)

        order = np.lexsort((worst.to_numpy(), infeasible, outside))
        ranked = {}
        for key, value in best.items():
            if hasattr(value, "iloc") and len(value) == len(y):
                ranked[key] = value.iloc[order].reset_index(drop=True)
            elif isinstance(value, np.ndarray) and len(value) == len(y):
                ranked[key] = value[order]
            else:
                ranked[key] = value
        return ranked

    def _env_for_naming(self, model):
        if self._env is not None:
            return self._env
        if not self.system and model is None:
            return None

        from livn.env import Env

        return Env(self.system or 1, model=model)

    def _section_names(self, env, population: str, section: str, model=None) -> str:
        namer = getattr(model, "section_name", None)
        if callable(namer):
            return str(namer(population, section))
        if env is None:
            return section
        return env.destination_sections().get(population, {}).get(section, section)

    def _model_for(self, model):
        if model is not None:
            return model
        return getattr(getattr(self, "_env", None), "model", None)

    def _weight_space(self, model) -> dict[str, list]:
        model = self._model_for(model)
        cached = self._weight_space_cache

        if cached is not None and (
            getattr(self, "_weight_space_model", False) or model is None
        ):
            return cached
        self._depression_keys = []

        env = self._env_for_naming(model)

        system = getattr(env, "system", None) if env is not None else None
        if system is None and self.system:
            from livn.system import resolve

            system = resolve(self.system)

        found = None
        if system is not None:
            try:
                found = system.synapse_projections()
            except (OSError, KeyError, TypeError, ValueError, AttributeError):
                found = None
        if found:
            found = [
                (
                    post,
                    pre,
                    self._section_names(env, post, section, model),
                    mech,
                    syn_type,
                )
                for post, pre, section, mech, syn_type in found
            ]
        if not found:
            populations = ["EXC", "INH"]
            if model is not None and hasattr(model, "ignored_populations"):
                ignored = set(model.ignored_populations())
                populations = [p for p in populations if p not in ignored]
            found = [
                (
                    post,
                    pre,
                    self._section_names(
                        env, post, "soma" if pre == "INH" else "dend", model
                    ),
                    mechanism,
                    "inhibitory" if pre == "INH" else "excitatory",
                )
                for pre in populations
                for post in populations
                for mechanism in (["GABA_A"] if pre == "INH" else ["AMPA", "NMDA"])
            ]

        ignored = set()
        if model is not None and hasattr(model, "ignored_populations"):
            ignored = set(model.ignored_populations())

        depressing = "AMPA" in self._depressing_receptors(model)

        default_ranges = {
            "excitatory": list(self.EXC_WEIGHT_RANGE),
            "inhibitory": list(self.INH_WEIGHT_RANGE),
        }
        mechanism_ranges = {"NMDA": list(self.NMDA_RATIO_RANGE)}
        depression_ranges = {k: list(v) for k, v in self.DEPRESSION_RANGES.items()}

        reference = None
        for post, pre, section, mechanism, syn_type in found:
            if pre in ignored or post in ignored:
                continue
            if post == pre and syn_type == "excitatory" and mechanism == "AMPA":
                reference = f"{post}_{pre}-{section}-{mechanism}-weight"
                break
        self._weight_reference = reference

        weights = {}
        for post, pre, section, mechanism, syn_type in found:
            if pre in ignored or post in ignored:
                continue
            key = f"{post}_{pre}-{section}-{mechanism}-weight"

            if reference is not None and key != reference:
                low, high = mechanism_ranges.get(
                    mechanism, self.RATIO_RANGES.get(syn_type, [0.01, 100.0])
                )
                weights[key + self.RATIO_SUFFIX] = [low, high, self.transform_log10]
                continue

            low, high = mechanism_ranges.get(
                mechanism, default_ranges.get(syn_type, [0.001, 10.0])
            )
            weights[key] = [low, high, self.transform_log10]

            if depressing and mechanism == "AMPA":
                for name, (dlo, dhi) in depression_ranges.items():
                    key = f"{post}-{section}-{mechanism}-{name}"
                    if key in weights or key in self._depression_keys:
                        continue
                    bounds = [dlo, dhi]
                    if dhi / dlo >= 10.0:
                        bounds.append(self.transform_log10)
                    weights[key] = bounds

        self._weight_space_cache = weights
        self._weight_space_model = model is not None
        return weights

    @staticmethod
    def _depressing_receptors(model) -> tuple[str, ...]:
        mechanisms = getattr(model, "neuron_synapse_mechanisms", None)
        rules = getattr(model, "neuron_synapse_rules", None)
        if not callable(mechanisms) or not callable(rules):
            return ()
        table = rules()
        return tuple(
            receptor
            for receptor, name in mechanisms().items()
            if "U" in (table.get(name) or {}).get("mech_params", ())
        )

    def _resolve_depression(self, decoded: dict, model) -> dict:
        model = self._model_for(model)
        self._weight_space(model)
        if not any(name.endswith("-weight") for name in decoded):
            return decoded
        resolved = dict(decoded)
        mirrors = [r for r in self._depressing_receptors(model) if r != "AMPA"]
        if not mirrors:
            return resolved
        for key, value in list(resolved.items()):
            head, _, param = key.rpartition("-")
            if param not in self.DEPRESSION_RANGES or not head.endswith("-AMPA"):
                continue
            for receptor in mirrors:
                resolved[f"{head[: -len('AMPA')]}{receptor}-{param}"] = value
        return resolved

    def decode_params(self, params: dict, model=None, strict: bool = False) -> dict:
        model = self._model_for(model)
        transforms, _, _ = self._space_metadata(model)
        undecodable = sorted(
            name
            for name in params
            if name.rpartition("-")[2] in self.DEPRESSION_RANGES
            and name not in transforms
        )
        if undecodable:
            raise ValueError(
                f"{undecodable} have no transform in this target's space, so they "
                "would reach the synapses at their encoded value"
            )
        decoded = super().decode_params(params, model=model, strict=strict)
        if self.LEADER_RANGES and self.LEADER_SELECTION == "input":
            decoded["leaders-by_input"] = 1.0
        decoded = self._resolve_noise_drive(decoded)
        decoded = self._resolve_noise_std(decoded)
        decoded = self._resolve_depression(decoded, model)

        suffix = self.RATIO_SUFFIX
        ratios = [name for name in decoded if name.endswith(suffix)]
        if not ratios:
            return self._resolve_release(decoded)

        self._weight_space(model)  # populates `_weight_reference`
        reference = self._weight_reference
        if reference is None or reference not in decoded:
            raise ValueError(
                f"{sorted(ratios)} are relative to {reference!r}, which is not "
                "in this vector; the ratios cannot be resolved to conductances. "
                "The target is probably built differently from the run that "
                "produced them."
            )

        scale = float(decoded[reference])
        resolved = {k: v for k, v in decoded.items() if not k.endswith(suffix)}
        for name in ratios:
            resolved[name[: -len(suffix)]] = float(decoded[name]) * scale
        return self._resolve_release(resolved)

    def _gain_ceiling(self, model, lo: float, hi: float) -> float:
        declares = getattr(model, "stimulus_bounds", None)
        bounds = declares("extracellular") if declares else None
        if not bounds:
            return hi

        peak = max(self.stimulus.amplitudes) * self.stimulus.uA_per_mv
        if peak <= 0:
            return hi
        allowed = min(abs(float(b)) for b in bounds) / peak
        allowed *= 1.0 - 1e-6
        if allowed >= hi:
            return hi
        if allowed <= lo:
            raise ValueError(
                f"a {max(self.stimulus.amplitudes):g} mV pulse is {peak:g} uA, "
                f"which reaches this model's extracellular bound at a gain of "
                f"{allowed:g}"
            )
        logger.info(
            "[space] a %g mV pulse is %g uA, so gains above %g leave the "
            "%s mV this model is defined over; capping the search at it "
            "instead of %g",
            max(self.stimulus.amplitudes),
            peak,
            allowed,
            list(bounds),
            hi,
        )
        return allowed

    def _protocol_space(self, model) -> dict[str, list]:
        space = {}
        if self.stimulus is not None:
            from livn.io import DEFAULT_STIMULATION_GAIN

            span = 10.0
            lo = DEFAULT_STIMULATION_GAIN / span
            space["io-volume_conductor-stimulation_gain"] = [
                lo,
                self._gain_ceiling(model, lo, DEFAULT_STIMULATION_GAIN * span),
                self.transform_log10,
            ]

        space.update(self._structure_space())

        for name, box in (self.LEADER_RANGES or {}).items():
            lo, hi = float(box[0]), float(box[1])
            log = len(box) > 2 and box[2] == "log"
            space[f"leaders-{name}"] = (
                [lo, hi, self.transform_log10] if log else [lo, hi]
            )

        return space

    REBUILD_LEAK_BYTES_PER_CELL = 580.0
    REBUILD_BUDGET = 400

    def worker_memory(self, system, ranks: int = 1, selection: str | None = None):
        if not self.structure:
            return super().worker_memory(system, ranks, selection)

        corner = self.restructured(
            {
                name: bounds[1]
                for name, bounds in self.structure.items()
                if name != "sigma"
            }
        )
        base = super().worker_memory(corner or system, ranks, selection)

        from livn.system import resolve

        cells = sum(resolve(corner or system).population_counts.values())
        return base + self.REBUILD_LEAK_BYTES_PER_CELL * cells * self.REBUILD_BUDGET

    def _base_degrees(self) -> dict[str, float]:
        base = self.system
        if not (isinstance(base, Mapping) and "cls" in base):
            return {}
        connectivity = (base.get("kwargs") or {}).get("connectivity") or {}
        degrees = connectivity.get("mean_degree")
        if not isinstance(degrees, Mapping):
            return {}
        return {str(k): float(v) for k, v in degrees.items()}

    def _periodic_sigma_ceiling(self) -> float | None:
        base = self.system
        if not (isinstance(base, Mapping) and "cls" in base):
            return None
        if self._sigma_ceiling is sentinel:
            from livn.system import resolve

            system = resolve(base)
            period = getattr(system, "period", None)
            floor = float(getattr(type(system), "MIN_EXTENT_IN_SIGMA", 0.0) or 0.0)
            self._sigma_ceiling = (
                min(period) / floor if period and floor > 0.0 else None
            )
        return self._sigma_ceiling

    def _structure_space(self) -> dict[str, list]:
        if not self.structure:
            return {}

        degrees = self._base_degrees()
        composed = self.COMPOSITION_KEY in self.structure
        ceiling = self._periodic_sigma_ceiling()
        space = {}
        for name, (lo, hi) in self.structure.items():
            if "->" in name:
                if name not in degrees:
                    raise ValueError(
                        f"structure names {name!r}, which is not a projection "
                        f"of this system; it has {sorted(degrees)}"
                    )
                if degrees[name] <= 0.0 and not composed:
                    continue
            lo, hi = float(lo), float(hi)
            if name == "sigma" and ceiling is not None and hi > ceiling:
                if lo >= ceiling:
                    raise ValueError(
                        f"this culture is periodic and {min(lo, hi):g} um is "
                        f"already past the {ceiling:g}"
                    )
                hi = ceiling
            space[f"{self.STRUCTURE_PREFIX}{name}"] = [
                lo,
                hi,
                self.transform_log10,
            ]
        return space

    def _constructible_sigma(self, sigma: float) -> float:
        ceiling = self._periodic_sigma_ceiling()
        if ceiling is None:
            return sigma
        return min(sigma, ceiling * (1.0 - 1e-9))

    @staticmethod
    def _degree_at(projection: str, value: float, connectivity: dict, fraction):
        if fraction is None:
            return value
        if connectivity.get("degree_rule") == "fixed_degree":
            return value
        reference = (connectivity.get("degree_reference") or {}).get(projection)
        if not reference:
            return value

        pre = projection.split("->")[0]
        share = float(fraction) if pre == "INH" else 1.0 - float(fraction)
        if share <= 0.0:
            return 0.0
        return value * share / float(reference)

    def restructured(self, structural: dict) -> dict | None:
        if not structural:
            return None

        base = self.system
        if not (isinstance(base, Mapping) and "cls" in base):
            raise ValueError(
                f"{sorted(structural)} ask for a different system, but this "
                f"target was handed {base!r} rather than a spec"
            )

        from livn.types import _plain

        spec = _plain(base)
        kwargs = spec.setdefault("kwargs", {})
        connectivity = kwargs.setdefault("connectivity", {})
        degrees = dict(connectivity.get("mean_degree") or {})

        fraction = structural.get(self.COMPOSITION_KEY)
        if fraction is not None:
            kwargs["populations"] = self._composed(kwargs.get("populations"), fraction)

        share = fraction if fraction is not None else self._spec_fraction(kwargs)

        for name, value in structural.items():
            if name == self.COMPOSITION_KEY:
                continue
            if name == "sigma":
                connectivity["sigma"] = self._constructible_sigma(float(value))
            elif name == "velocity":
                connectivity["velocity"] = float(value)
            elif "->" in name:
                if name not in degrees:
                    raise ValueError(
                        f"{name!r} is not a projection of this system; it has "
                        f"{sorted(degrees)}"
                    )
                if fraction is not None or degrees[name] > 0.0:
                    degrees[name] = self._degree_at(
                        name, float(value), connectivity, share
                    )
            else:
                raise ValueError(
                    f"{name!r} is not a structural parameter; expected "
                    f"{self.COMPOSITION_KEY!r}, 'sigma', 'velocity', or a "
                    "projection such as 'EXC->EXC'"
                )
        if degrees:
            connectivity["mean_degree"] = self._realisable(
                degrees, kwargs.get("total_cells"), kwargs.get("populations")
            )
        return spec

    @staticmethod
    def _spec_fraction(kwargs: dict) -> float | None:
        """The inhibitory fraction a spec is built at, if it names both."""
        populations = kwargs.get("populations") or {}
        if not {"EXC", "INH"} <= set(populations):
            return None
        ratios = {
            p: float((populations[p] or {}).get("ratio") or 0.0) for p in ("EXC", "INH")
        }
        total = ratios["EXC"] + ratios["INH"]
        return ratios["INH"] / total if total > 0.0 else None

    def _composed(self, populations, fraction: float) -> dict:
        populations = {p: dict(v) for p, v in (populations or {}).items()}
        missing = {"EXC", "INH"} - set(populations)
        if missing:
            raise ValueError(
                f"a searched composition needs both populations named, and "
                f"{sorted(missing)} is absent. A spec that omits one cannot "
                "grow it: name it at ratio 0 instead"
            )
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"inhibitory fraction must be in [0, 1], got {fraction}")
        for name, share in (("EXC", 1.0 - fraction), ("INH", fraction)):
            populations[name]["ratio"] = share
            # a `count` would override the ratio and silently pin the split
            populations[name].pop("count", None)
        return populations

    @staticmethod
    def _realisable(degrees: dict, total_cells, populations) -> dict:
        if not total_cells or not populations:
            return degrees
        counts = {
            name: int(float(total_cells) * float((spec or {}).get("ratio") or 0.0))
            for name, spec in populations.items()
        }
        capped = {}
        for name, value in degrees.items():
            pre = name.split("->")[0] if "->" in name else None
            available = counts.get(pre)
            capped[name] = (
                float(min(float(value), float(available)))
                if available is not None
                else float(value)
            )
        return capped

    def system_for(self, params: dict) -> dict | None:
        return self.spec

    def set_params(self, params: dict) -> dict:
        structural = {}
        remaining = {}
        prefix = self.STRUCTURE_PREFIX
        for name, value in params.items():
            if name.startswith(prefix):
                structural[name[len(prefix) :]] = float(value)
            else:
                remaining[name] = value
        self.spec = self.restructured(structural)
        return remaining

    def _resolve_release(self, decoded: dict) -> dict:
        suffix = "-U"
        for name, value in list(decoded.items()):
            if not name.endswith(suffix):
                continue
            post, _, rest = name[: -len(suffix)].partition("-")
            if not rest:
                continue
            try:
                release = float(value)
            except (TypeError, ValueError):
                continue
            if not release > 0.0:
                continue
            # `<post>-<section>-<mech>-U` governs every `<post>_<pre>-<section>-<mech>-weight`
            for weight_name in list(decoded):
                if not weight_name.endswith(f"-{rest}-weight"):
                    continue
                if weight_name.split("_", 1)[0] != post:
                    continue
                decoded[weight_name] = float(decoded[weight_name]) / release
        return decoded

    def _noise_space(self, model):
        drive = {
            "noise-g_total": [*self.NOISE_TOTAL_RANGE, self.transform_log10],
            "noise-g_ratio": [*self.NOISE_RATIO_RANGE, self.transform_log10],
            "noise-std_fraction": [
                *self.NOISE_STD_FRACTION_RANGE,
                self.transform_log10,
            ],
        }
        tau_e_lo, tau_e_hi = self.NOISE_TAU_RANGES["tau_e"]
        tau_i_lo, tau_i_hi = self.NOISE_TAU_RANGES["tau_i"]
        return {
            **drive,
            "noise-tau_e": [float(tau_e_lo), float(tau_e_hi), self.transform_log10],
            "noise-tau_i": [float(tau_i_lo), float(tau_i_hi)],
        }

    def _resolve_noise_drive(self, decoded: dict) -> dict:
        """`(total, ratio)` back into the two conductances the env takes.

            g_e0 = total / (1 + ratio)      g_i0 = total * ratio / (1 + ratio)

        so `total` is `g_e0 + g_i0` and `ratio` is `g_i0 / g_e0` exactly.
        """
        total = decoded.get("noise-g_total")
        ratio = decoded.get("noise-g_ratio")
        if total is None and ratio is None:
            return decoded
        if total is None or ratio is None:
            missing = "noise-g_total" if total is None else "noise-g_ratio"
            raise ValueError(
                f"{missing!r} is missing; the background is parameterised as a "
                "pair and neither half resolves to a conductance alone"
            )
        total, ratio = float(total), float(ratio)
        consumed = ("noise-g_total", "noise-g_ratio", "noise-std_fraction")
        resolved = {k: v for k, v in decoded.items() if k not in consumed}
        resolved["noise-g_e0"] = total / (1.0 + ratio)
        resolved["noise-g_i0"] = total * ratio / (1.0 + ratio)
        fraction = decoded.get("noise-std_fraction")
        if fraction is not None:
            resolved["noise-std_e"] = float(fraction) * resolved["noise-g_e0"]
            resolved["noise-std_i"] = float(fraction) * resolved["noise-g_i0"]
        return resolved

    def _resolve_noise_std(self, decoded: dict) -> dict:
        if not any(name.startswith("noise-") for name in decoded):
            return decoded
        if "noise-std_e" in decoded and "noise-std_i" in decoded:
            return decoded
        return {**decoded, "noise-std_e": self.NOISE_STD, "noise-std_i": self.NOISE_STD}

    def __call__(self, env, params=None, directory=None):
        self.record_resting(env)

        objectives = self.compute_objectives(env)
        constraints = self.compute_constraints(env)

        if (
            self.stimulus is not None
            and "stimulus_threshold" in objectives
            and self._admits_a_sweep(constraints)
        ):
            self.record_evoked(env)
            objectives["stimulus_threshold"] = self._threshold_objective(env)
            objectives.update(self._score_response(env))

        self._keep_spikes(env, params, directory, constraints)

        return objectives, constraints

    def _score_response(self, env) -> dict:
        self.metrics.update(self._response_shape(env))
        targets = self.targets()
        scored = {
            name: (
                self._log_ratio_objective(
                    float(self.metrics.get(name, float("nan"))),
                    float(targets[name]),
                    feature(name).floor,
                ),
                float(self.metrics.get(name, float("nan"))),
            )
            for name in self.RESPONSE_OBJECTIVES
            if name in targets and name not in self.skip_objectives
        }
        self.objectives = {**self.objectives, **scored}
        return scored

    def _keep_spikes(self, env, params, directory, constraints) -> None:
        if not self.save_spikes or directory is None or params is None:
            return

        feasible = all(
            float(v[0] if isinstance(v, (list, tuple)) else v) >= 0.0
            for v in constraints.values()
        )

        feasible = P.broadcast(feasible, comm=getattr(env, "comm", None))
        if self.save_spikes == "feasible" and not feasible:
            return
        data = self.response_data
        if data is None:
            return

        gathered = data.gather(comm=getattr(env, "comm", None), root=0)
        if not P.is_root(comm=getattr(env, "comm", None)):
            return

        key = hashlib.md5(
            json.dumps({k: float(v) for k, v in sorted(params.items())}).encode()
        ).hexdigest()[:16]
        out = os.path.join(directory, "spikes")
        os.makedirs(out, exist_ok=True)
        gids = getattr(getattr(env, "system", None), "gids", None)
        np.savez_compressed(
            os.path.join(out, f"spikes-{key}.npz"),
            spike_ids=np.asarray(gathered.spike_ids, dtype=np.int64),
            spike_times=np.asarray(gathered.spike_times, dtype=np.float64),
            meta=json.dumps(
                {
                    "parameters": {k: float(v) for k, v in params.items()},
                    "constraints": {
                        k: float(v[0] if isinstance(v, (list, tuple)) else v)
                        for k, v in constraints.items()
                    },
                    "feasible": bool(feasible),
                    "simulated_ms": float(self.simulated_ms or 0.0),
                    "n_cells": 0 if gids is None else len(gids),
                }
            ),
        )

    def _admits_a_sweep(self, constraints: dict) -> bool:
        return all(
            float(constraints[name][0]) >= 0.0
            for name in self.LIVENESS
            if name in constraints
        )

    def record(self, env, return_data=False):
        self.record_resting(env)
        self.record_evoked(env)

        if return_data:
            return GatherAndMerge(
                duration=self.simulated_ms, voltages=False, membrane_currents=False
            )(self.response_data, env)
        return None

    def record_resting(self, env):
        self._reset_state()
        duration = int(self.warmup_duration + self.recording_duration)

        env.record_spikes()
        t0 = time.time()
        self.response_data = env.run(duration, root_only=False)
        self.simulated_ms = duration
        self._log_simulated("free-running", duration, t0)

    def record_evoked(self, env):
        """Deliver the pulse sweep after whatever has been recorded so far."""
        if self.stimulus is None:
            return

        evoked_duration = math.ceil(self.stimulus.duration_ms)
        t0 = time.time()

        electrode = self.stimulus.electrode
        if electrode is None:
            channel_ids = np.asarray(env.io.channel_ids)
            distances = np.asarray(env.io.distances(env.active_neuron_coordinates()))
            within = distances[distances[:, -1] <= float(env.io.input_radius)]
            if within.size == 0:
                electrode = 0
            else:
                channels, counts = np.unique(
                    within[:, 0].astype(np.int64), return_counts=True
                )
                driving = int(channels[int(np.argmax(counts))])
                found = np.flatnonzero(channel_ids == driving)
                electrode = int(found[0]) if found.size else 0

        dt = 0.1
        trial_ms = float(self.stimulus.trial_ms)
        for _at, amplitude in self.stimulus.schedule(start_ms=0.0):
            trial = self.stimulus.for_array(
                len(env.io.channel_ids),
                [electrode],
                start_ms=0.0,
                total_ms=trial_ms,
                amplitudes=(float(amplitude),),
                repeats=1,
                order=(),
                dt=dt,
            )
            piece = env.run(trial_ms, stimulus=trial, root_only=False)
            self.response_data = (
                piece
                if self.response_data is None
                else self.response_data.concat(piece)
            )

        remainder = evoked_duration - self.stimulus.duration_ms
        if remainder > 1e-9:
            self.response_data = self.response_data.concat(
                env.run(remainder, root_only=False)
            )

        self.evoked_recorded = True
        self.simulated_ms += evoked_duration
        self._log_simulated("evoked", evoked_duration, t0)

    def _log_simulated(self, phase: str, duration: float, started: float):
        local = 0
        if self.response_data is not None and self.response_data.spike_ids is not None:
            local = len(self.response_data.spike_ids)
        logger.info(
            "[phase] simulated %d ms of %s in %.0f s (%d spikes on this rank)",
            duration,
            phase,
            time.time() - started,
            local,
        )

    @property
    def stimulus_start(self) -> float:
        return float(self.warmup_duration + self.recording_duration)

    def compute_objectives(self, env) -> dict:
        targets = self.targets()
        result: dict = {}
        d = int(self.recording_duration)
        _measure_started = time.time()

        recording_slice = Slice(
            start=self.warmup_duration,
            stop=self.warmup_duration + self.recording_duration,
        )
        recording_data = recording_slice(self.response_data)
        liveness = (
            PopulationActiveFraction(duration=d, bin_size=float(d))(recording_data, env)
            or {}
        )
        self.metrics["population_liveness"] = liveness.get("mean_active_fraction", {})

        network = env
        if self.UNIT_FEATURES:
            units = self._unit_metrics(network, recording_data, d)
            self.metrics["recruitment_order"] = units
            for name in self.UNIT_FEATURES:
                value = float(units.get(name, float("nan")))
                self.metrics[name] = value
                if feature(name).scale is not None:
                    scale = float(feature(name).scale)
                    measured = 0.0 if np.isnan(value) else value
                    objective = float(((measured - float(targets[name])) / scale) ** 2)
                else:
                    objective = self._log_ratio_objective(
                        value, float(targets[name]), feature(name).floor
                    )
                result[name] = (objective, value)
        env, recording_data = self._readout(env, recording_data)

        measured = resting_features(recording_data, env, d)
        total_spike_count = measured.pop("total_spikes")
        self.metrics["total_spikes"] = total_spike_count
        self.metrics["enough_spikes_for_network_metrics"] = (
            total_spike_count >= self.MIN_SPIKE_COUNT_FOR_METRICS
        )
        self.metrics.update(measured)
        if np.isnan(self.metrics["pop_autocorr_tau"]):
            self.metrics["pop_autocorr_tau"] = 10.0

        mfr = float(self.metrics["mfr"])
        eps = 1e-3
        mfr_target = float(targets["mfr"])
        mfr_obj = float(np.log((max(mfr, 0.0) + eps) / (mfr_target + eps)) ** 2)
        result["mfr"] = (mfr_obj, mfr)

        gids = getattr(env.system, "gids", None)
        n_units = max(len(gids) if gids is not None else 0, 1)
        stability_result = Stability(
            duration=d,
            tail_window=float(self.STABILITY_TAIL_MS),
            max_rate_hz=self.max_pop_rate_per_unit_hz * n_units * self.STABILITY_MARGIN,
            min_rate_hz=self.min_pop_rate_per_unit_hz * n_units / self.STABILITY_MARGIN,
        )(recording_data, env)
        self.metrics["stability_result"] = stability_result
        self.metrics["is_stable"] = (
            bool(stability_result["is_stable"]) if stability_result else False
        )
        pop_rate = float((stability_result or {}).get("global_mean_hz", 0.0))
        self.metrics["pop_rate_hz"] = pop_rate
        self.metrics["pop_rate_per_unit_hz"] = pop_rate / n_units

        isi_cv = float(self.metrics["isi_cv"])
        result["isi_cv"] = ((isi_cv - float(targets["isi_cv"])) ** 2, isi_cv)

        active_fraction = float(self.metrics["active_fraction"])
        af_obj = (float(targets["active_fraction"]) - active_fraction) ** 2
        result["active_fraction"] = (af_obj, active_fraction)

        if "mean_channel_correlation" in targets:
            sync = float(self.metrics["mean_channel_correlation"])
            sync_target = float(targets["mean_channel_correlation"])
            result["mean_channel_correlation"] = (
                1e3 if np.isnan(sync) else float((sync - sync_target) ** 2),
                sync,
            )

        if self.stimulus is not None:
            result["stimulus_threshold"] = self._threshold_objective(network)

        for names, unmeasured in (
            (self.BURST_OBJECTIVES, 0.0),
            (self.RESPONSE_OBJECTIVES, float("nan")),
        ):
            for name in names:
                if name not in targets:
                    continue
                value = float(self.metrics.get(name, unmeasured))
                result[name] = (
                    self._log_ratio_objective(
                        value, float(targets[name]), feature(name).floor
                    ),
                    value,
                )

        result = {
            name: value
            for name, value in result.items()
            if name not in self.skip_objectives
        }
        self.objectives = result
        logger.info(
            "[phase] measured %d channel-level spikes in %.0f s",
            self.metrics.get("total_spikes", 0) or 0,
            time.time() - _measure_started,
        )
        return result

    @staticmethod
    def _log_ratio_objective(value: float, target: float, floor: float) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 1e3
        return float(np.log((max(float(value), 0.0) + floor) / (target + floor)) ** 2)

    def _response_shape(self, network) -> dict:
        if not self.evoked_recorded or self.stimulus is None:
            return {}

        proxy, stimulated = self._readout(
            network,
            Slice(
                start=self.stimulus_start,
                stop=self.stimulus_start + self.stimulus.duration_ms,
            )(self.response_data),
        )
        schedule = self.stimulus.schedule(0.0)
        blanked = self._blanked(stimulated, [float(at) for at, _ in schedule])

        duration = int(self.stimulus.duration_ms)
        measured = [
            one
            for one in (
                StimulusResponse(
                    duration=duration, onset_ms=float(at), **self.response_kwargs
                )(blanked, proxy)
                for at, _amplitude in schedule
            )
            if one
        ]
        return self._pooled_response(measured)

    @classmethod
    def _pooled_response(cls, measured: list[dict]) -> dict:
        if not measured:
            return {}
        pooled: dict = {"response_pulses": float(len(measured))}
        for name in cls.RESPONSE_FEATURES:
            values = np.asarray(
                [one[name] for one in measured if name in one], dtype=float
            )
            values = values[np.isfinite(values)]
            if values.size:
                pooled[name] = float(np.median(values))
        return pooled

    def _blanked(self, data, onsets: list[float]):
        """The recording without the artefact window around each pulse."""
        lo, hi = self.response_blank_ms
        times = np.asarray(getattr(data, "spike_times", ()), dtype=float)
        if hi <= lo or times.size == 0:
            return data

        from livn.run import Run

        ids = np.asarray(data.spike_ids)
        keep = np.ones(times.shape, dtype=bool)
        for at in onsets:
            keep &= ~((times >= at + lo) & (times <= at + hi))
        if keep.all():
            return data
        return Run(t0=data.t0, duration=data.duration).add_spikes(
            ids[keep], times[keep]
        )

    def _threshold_objective(self, network) -> tuple:
        if not self.evoked_recorded:
            return (1e3, float("nan"))

        proxy, stimulated = self._readout(
            network,
            Slice(
                start=self.stimulus_start,
                stop=self.stimulus_start + self.stimulus.duration_ms,
            )(self.response_data),
        )

        simulated = (
            RecruitmentCurve(
                duration=int(self.stimulus.duration_ms),
                schedule=self.stimulus.schedule(0.0),
                pre_ms=float(self.stimulus.pre_ms),
                post_ms=float(self.stimulus.post_ms),
            )(stimulated, proxy)
            or {}
        )
        self.curve = simulated.pop("curve", {})
        self.metrics["recruitment_curve"] = dict(self.curve)
        self.metrics["threshold"] = simulated
        self.metrics["threshold_censored"] = self.stimulus_threshold.get("censored")
        miss = threshold_miss(self.stimulus_threshold, simulated)
        curve_miss = recruitment_miss(self.stimulus_threshold, simulated)
        self.metrics["threshold_miss"] = miss
        self.metrics["recruitment_miss"] = curve_miss

        scored = curve_miss
        return (1e3 if np.isnan(scored) else float(scored), miss)

    def _readout(self, env, data):
        _, per_channel = env.channel_recording(data.spike_ids, data.spike_times)
        if per_channel:
            it = np.concatenate(
                [np.full(len(t), c, dtype=np.int64) for c, t in per_channel.items()]
            )
            tt = np.concatenate([np.asarray(t) for t in per_channel.values()])
            order = np.argsort(tt, kind="stable")
            it, tt = it[order], tt[order]
        else:
            it, tt = np.array([], dtype=np.int64), np.array([])

        proxy = SimpleNamespace(
            comm=env.comm,
            system=SimpleNamespace(gids=list(env.io.channel_ids)),
            io=env.io,
            voltage_recording_dt=getattr(env, "voltage_recording_dt", None),
        )
        return proxy, data.add_spikes(it, tt)

    def compute_constraints(self, env) -> dict:
        result: dict = {}
        m = self.metrics
        stability_result = m.get("stability_result")

        if stability_result:
            tail_mean = stability_result["tail_mean_hz"]
            max_rate = stability_result.get("max_rate_hz", 20.0)
            min_rate = stability_result.get("min_rate_hz", 0.05)

            if stability_result["is_runaway"]:
                runaway_c = -1.0 - (tail_mean - max_rate) / 10.0
            else:
                runaway_c = 1.0 + (max_rate - tail_mean) / 10.0
            if stability_result["is_quiescent"]:
                quiescent_c = -1.0 - (min_rate - tail_mean) / 0.1
            else:
                quiescent_c = 1.0 + (tail_mean - min_rate) / 0.1

            result["not_runaway"] = (float(runaway_c), float(tail_mean))
            result["not_quiescent"] = (float(quiescent_c), float(tail_mean))
            result["is_stable"] = (
                1.0 if stability_result["is_stable"] else -1.0,
                float(stability_result["is_stable"]),
            )
        else:
            result["not_runaway"] = (-10.0, 0.0)
            result["not_quiescent"] = (-10.0, 0.0)
            result["is_stable"] = (-10.0, 0.0)

        max_neuron_rate = m.get("max_neuron_firing_rate", float("nan"))
        result["max_firing_rate"] = (
            float(_max_constraint(max_neuron_rate, self.max_neuron_rate_hz)),
            float(max_neuron_rate),
        )

        mean_sync = m.get("mean_channel_correlation", float("nan"))
        try:
            mean_sync_f = float(np.clip(float(mean_sync), -1.0, 1.0))
        except (TypeError, ValueError):
            mean_sync_f = float("nan")
        sync_c = _band_constraint(
            mean_sync_f, self.synchrony_band[0], self.synchrony_band[1]
        )
        result["synchrony"] = (
            float(np.clip(sync_c, -10.0, 10.0)),
            mean_sync_f,
        )

        peak_sync = m.get("max_synchronous_peak", float("nan"))
        result["max_synchronous_peak"] = (
            float(_band_constraint(peak_sync, self.min_sync_peak, self.max_sync_peak)),
            float(peak_sync),
        )

        mean_rate = m.get("mfr", float("nan"))
        result["min_mean_firing_rate"] = (
            float(_min_constraint(mean_rate, self.min_mean_rate_hz)),
            float(mean_rate),
        )
        result["max_mean_firing_rate"] = (
            float(_max_constraint(mean_rate, self.max_mean_rate_hz)),
            float(mean_rate),
        )

        liveness = m.get("population_liveness") or {}
        worst = min(liveness.values()) if liveness else float("nan")
        result["populations_active"] = (
            float(_min_constraint(worst, self.MIN_POPULATION_ACTIVE, scale=1.0)),
            float(worst),
        )

        active_fraction = m.get("active_fraction", float("nan"))
        result["active_fraction_floor"] = (
            float(
                _min_constraint(active_fraction, self.min_active_fraction, scale=1.0)
            ),
            float(active_fraction),
        )

        pop_tau = m.get("pop_autocorr_tau", float("nan"))
        result["pop_autocorr_tau_band"] = (
            float(_band_constraint(pop_tau, *self.pop_tau_band_ms)),
            float(pop_tau),
        )

        burst_rate = m.get("burst_rate", float("nan"))
        result["burst_rate_band"] = (
            float(
                _band_constraint(
                    burst_rate, self.min_burst_rate_hz, self.max_burst_rate_hz
                )
            ),
            float(burst_rate),
        )

        sigma = m.get("branching_ratio", float("nan"))
        result["branching_ratio_band"] = (
            float(_band_constraint(sigma, *self.branching_ratio_band)),
            float(sigma),
        )

        avalanche_r2 = m.get("avalanche_r2", float("nan"))
        result["avalanche_r2"] = (
            float(_min_constraint(avalanche_r2, self.min_avalanche_r2, scale=1.0)),
            float(avalanche_r2),
        )

        return {
            name: value
            for name, value in result.items()
            if name not in self.skip_constraints
        }

    def describe_params(self, decoded):
        env_params = self.set_params(dict(decoded))
        weights = {
            k: v
            for k, v in env_params.items()
            if "-weight" in k and not k.startswith("noise-")
        }
        noise_keys = {"std_e", "std_i", "g_e0", "g_i0", "tau_e", "tau_i"}
        noise = {
            k.replace("noise-", "", 1): v
            for k, v in env_params.items()
            if k.startswith("noise-") and k.replace("noise-", "", 1) in noise_keys
        }
        protocol = {k: v for k, v in decoded.items() if k not in env_params}
        return {
            "All decoded params": dict(decoded),
            "Weights (neuron_default_weights)": weights,
            "Noise (neuron_default_noise)": noise,
            "Protocol-specific params": protocol,
        }
