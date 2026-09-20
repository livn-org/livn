import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
from machinable.config import Field as ConfigField
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from livn.decoding import (
    ISICV,
    ActiveFraction,
    AvalancheAnalysis,
    BurstAnatomy,
    BurstRate,
    GatherAndMerge,
    MeanFiringRate,
    PairwiseChannelCorrelation,
    PeakSynchrony,
    PerUnitFiringRate,
    PopulationActiveFraction,
    PopulationAutocorrTau,
    PopulationRateMetrics,
    RecruitmentCurve,
    Slice,
    Stability,
    StimulusResponse,
)
from livn.env.logging import with_progress_logging
from livn.policy import PulseSweepPolicy
from livn.utils import P, sentinel
from systems.targets.protocol import Sizing, Target, digest

logger = logging.getLogger(__name__)


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
        return math.log10(step)  # no curve to read; one rung, as it always was

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
    if censored == "above":  # the culture is at least this hard to drive
        return max(0.0, -difference)
    if censored == "below":  # and at most this easy
        return max(0.0, difference)
    return abs(difference)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = 1.0
    sigma: float | None = 300.0
    degree: float = 20.0
    boundary: float | None = None
    cells: int | None = None
    inhibitory_fraction: float | None = None
    size_cv: float = 0.2


class Culture(Target):
    RATIO_SUFFIX = "_ratio"
    STRUCTURE_PREFIX = "system-"
    COMPOSITION_KEY = "inhibitory_fraction"
    ADAPTATION_PARAMS: ClassVar[dict] = {
        "cells-EXC:soma.gmax_KCa": "soma_gmax_KCa",
        "cells-EXC:dend.gmax_KCa": "dend_gmax_KCa",
        "cells-EXC:soma.kCa_Ca_conc": "soma_kCa_Caconc",
        "cells-EXC:dend.kCa_Ca_conc": "dend_kCa_Caconc",
        "cells-EXC:dend.gmax_CaN": "dend_gmax_CaN",
    }
    ANATOMY_FEATURES: ClassVar[tuple] = (
        "burst_width_ms",
        "spikes_per_unit_per_burst",
        "burst_onset_peak",
        "burst_interval_cv",
        "units_recruited_per_burst",
    )
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
    RESPONSE_OBJECTIVE_EPS: ClassVar[dict] = {
        "response_latency_ms": 5.0,
        "response_duration_ms": 10.0,
    }
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
    READOUT = "channels"
    MIN_SPIKE_COUNT_FOR_METRICS = 150

    # --- what is scored
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
    BURST_OBJECTIVE_EPS: ClassVar[dict] = {
        "fano_factor": 0.1,
        "pop_autocorr_tau": 1.0,
        "burst_rate": 0.05,
        "max_synchronous_peak": 0.05,
        "burst_width_ms": 5.0,
        "spikes_per_unit_per_burst": 0.1,
        "burst_onset_peak": 0.05,
        "burst_interval_cv": 0.05,
        "units_recruited_per_burst": 0.05,
    }
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

    # --- the gates no document states
    LIVENESS: ClassVar[tuple] = ("not_runaway", "not_quiescent", "is_stable")
    MIN_POPULATION_ACTIVE = 0.05
    STABILITY_MARGIN = 5.0
    STABILITY_TAIL_MS = 5000.0
    BURST_MIN_FLOOR_FRACTION = 0.3
    NOISE_STD = 0.0003
    NOISE_TOTAL_RANGE: ClassVar[list] = [0.0005, 0.002]
    NOISE_RATIO_RANGE: ClassVar[list] = [8.0, 20.0]
    NOISE_TAU_RANGES: ClassVar[dict] = {
        "tau_e": [1.0, 100.0],
        "tau_i": [4.0, 100.0],
    }
    EXC_WEIGHT_RANGE: ClassVar[list] = [0.15, 1.0]
    INH_WEIGHT_RANGE: ClassVar[list] = [0.05, 100.0]
    NMDA_RATIO_RANGE: ClassVar[list] = [4.0, 35.0]
    RATIO_RANGES: ClassVar[dict] = {
        "excitatory": [0.01, 1000.0],
        "inhibitory": [0.01, 1000.0],
    }
    DEPRESSION_RANGES: ClassVar[dict] = {
        "tau_rec": [300.0, 3000.0],
        "U": [0.04, 0.5],
    }
    ADAPTATION_DECADES = 0.5
    ADAPTATION_CENTRE: ClassVar[dict] = {
        "cells-EXC:dend.gmax_CaN": 40.0,
        "cells-EXC:dend.gmax_KCa": 3.0,
        "cells-EXC:soma.kCa_Ca_conc": 0.5,
        "cells-EXC:dend.kCa_Ca_conc": 0.5,
    }
    STRUCTURE_RANGES: ClassVar[dict] = {
        "EXC->EXC": [5.0, 100.0],
        "INH->EXC": [12.6, 126.0],
        "EXC->INH": [1.3, 12.6],
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

    def on_compute_predicate(self):
        stated = {k: v for k, v in self.settings.items() if k != "observation"}
        return {
            "culture": self.sample,
            "measurement": digest(self.measurement()),
            "problem": digest(stated),
        }

    def version_fit(self, observation: str, **options):
        return {"observation": observation, **options}

    def system_spec(self):
        spec = self.config.spec
        fraction = spec.inhibitory_fraction
        composed = (self.config.structure or {}).get(self.COMPOSITION_KEY)
        if composed is not None and fraction is None:
            fraction = math.sqrt(float(composed[0]) * float(composed[1]))
            self._note(
                f"the composition is searched, so the base spec is drawn at an "
                f"inhibitory fraction of {fraction:.3f} -- the middle of "
                f"{list(composed)} -- rather than at the log's label. This "
                "decides which synapses the weight space has and nothing else: "
                "every evaluation is run at the fraction its own vector asks "
                "for, and the label is what the result is checked against."
            )

        from systems.targets.culture import spec as culture_spec

        return culture_spec(
            self.document["metadata"],
            self.sample,
            cells=spec.cells,
            excitatory_degree=float(spec.degree),
            scale=float(spec.scale),
            inhibitory_fraction=fraction,
            sigma=None if spec.sigma is None else float(spec.sigma),
            boundary=spec.boundary,
        )

    def model_spec(self):
        return [
            "livn.models.rcsd.ReducedCalciumSomaDendrite",
            {"size_cv": float(self.config.spec.size_cv)},
        ]

    @property
    def document(self) -> dict:
        if "document" not in self._cache:
            with open(self.config.observation) as f:
                self._cache["document"] = json.load(f)
        return self._cache["document"]

    @property
    def condition(self) -> str:
        from systems.targets.observation import free_running_block

        if "condition" not in self._cache:
            name = free_running_block(self.document)
            if name != "spontaneous":
                self._note(
                    f"{os.path.basename(self.config.observation)} has no "
                    f"'spontaneous' block, so the resting features come from "
                    f"{name!r} -- which mixes quiet and active windows, and "
                    "whose widened bands admit an asynchronous network."
                )
            self._cache["condition"] = name
        return self._cache["condition"]

    @property
    def sample(self) -> str:
        from systems.targets.observation import sample_of

        try:
            return sample_of(self.document)
        except ValueError as _ex:
            raise ValueError(f"{self.config.observation!r}: {_ex}") from _ex

    def _configure(self):
        self._targets = {"mfr": 1.0, "isi_cv": 1.2, "active_fraction": 1.0}
        self.feature_bands: dict[str, tuple[float, float]] = {}
        self.mea = None
        self.stimulus = None
        self.stimulus_threshold: dict = {}
        self.response_kwargs: dict = {}
        self.response_blank_ms = (0.0, 0.0)
        self.skip_objectives: tuple[str, ...] = ()
        self.recording_duration = 20_000.0
        self.warmup_duration = 1_000.0
        self.skip_constraints = ("avalanche_r2",)
        self.structure = dict(self.config.structure or {}) or None
        self.save_spikes = self.config.save_spikes

        self._env = None
        self._sigma_ceiling = sentinel
        self._depression_keys: list[str] = []
        self._weight_space_cache: dict[str, list] | None = None
        self._weight_reference: str | None = None
        self._reset_state()

        self._measure(self.config.observation, self.condition)

    def _measure(self, observation: str, condition: str) -> None:
        from livn.system import resolve
        from systems.targets.observation import measured_options

        spec = self.system
        if isinstance(spec, Mapping):
            self.mea = spec.get("kwargs", {}).get("mea")

        options = measured_options(
            observation,
            condition,
            self.mea,
            resolve(spec),
            readout=self.READOUT,
            skip_constraints=list(self.skip_constraints),
        )

        for name, value in options.items():
            if not name.isupper():
                continue
            if name not in self.MEASURED_GATES:
                raise ValueError(
                    f"{os.path.basename(observation)} states a gate {name!r} "
                    f"that {type(self).__name__} does not read; add it to "
                    "MEASURED_GATES or re-extract the document"
                )
            setattr(self, name.lower(), value)

        self._targets = {**self._targets, **options["targets"]}
        self.feature_bands = {
            name: (float(lo), float(hi))
            for name, (lo, hi) in options["feature_bands"].items()
        }

        self._score_anatomy(observation, condition)
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

    def _score_anatomy(self, observation: str, condition: str) -> None:
        from systems.targets.schema import read_target

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

    def _deliver_evoked(self, observation: str) -> None:
        from systems.targets.schema import read_target

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

    def io(self):
        if not self.mea:
            return None
        from livn.io import MEA

        return MEA.from_json(self.mea)

    def init(self, env):
        if self.READOUT == "channels" and not len(getattr(env.io, "channel_ids", ())):
            raise RuntimeError("readout='channels' needs an `mea`.")
        if self.stimulus is not None and self.READOUT != "channels":
            raise RuntimeError(
                f"a stimulated target reads out through the array, not "
                f"{self.READOUT!r}; pass readout='channels' with the recording "
                "set's `mea`"
            )
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

    def _weight_space(self, model) -> dict[str, list]:
        if self._weight_space_cache is not None:
            return self._weight_space_cache

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
        self._weight_space(model)  # populates `_depression_keys`
        if not any(name.endswith("-weight") for name in decoded):
            return decoded  # a vector with no synapses in it, e.g. a cell fit
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
        decoded = super().decode_params(params, model=model, strict=strict)
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

        if model is None:
            model = getattr(self._env, "model", None)

        fitted = {}
        if model is not None and hasattr(model, "params"):
            try:
                fitted = model.params("BoothRinzelKiehn-MN") or {}
            except (KeyError, ValueError, TypeError):
                fitted = {}
        if not fitted:
            raise ValueError(
                "the model exposes no 'BoothRinzelKiehn-MN' parameters to "
                "centre the adaptation bounds on; the search would silently "
                "drop these dimensions"
            )

        span = 10.0**self.ADAPTATION_DECADES
        for key, name in self.ADAPTATION_PARAMS.items():
            value = fitted.get(name)
            if value is None or float(value) <= 0.0:
                raise ValueError(
                    f"{name!r} is {value!r}; a log-scaled bound needs a "
                    "positive fitted value to centre on"
                )
            value = float(value) * float(self.ADAPTATION_CENTRE.get(key, 1.0))
            space[key] = [value / span, value * span, self.transform_log10]
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
                        f"already past the {ceiling:g} um its box can carry "
                        "(a torus must be at least four sigma across or the "
                        "kernel wraps onto itself). Lower the sigma "
                        "range, widen the area, or give the spec a guard"
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

    def restructured(self, structural: dict) -> dict | None:
        if not structural:
            return None

        base = self.system
        if not (isinstance(base, Mapping) and "cls" in base):
            raise ValueError(
                f"{sorted(structural)} ask for a different system, but this "
                f"target was handed {base!r} rather than a spec, so there is "
                "nothing to restructure. Fit from a spec, or drop `structure`."
            )

        from livn.types import _plain

        spec = _plain(base)
        kwargs = spec.setdefault("kwargs", {})
        connectivity = kwargs.setdefault("connectivity", {})
        degrees = dict(connectivity.get("mean_degree") or {})

        fraction = structural.get(self.COMPOSITION_KEY)
        if fraction is not None:
            kwargs["populations"] = self._composed(kwargs.get("populations"), fraction)

        for name, value in structural.items():
            if name == self.COMPOSITION_KEY:
                continue
            if name == "sigma":
                connectivity["sigma"] = self._constructible_sigma(float(value))
            elif "->" in name:
                if name not in degrees:
                    raise ValueError(
                        f"{name!r} is not a projection of this system; it has "
                        f"{sorted(degrees)}"
                    )
                if fraction is not None or degrees[name] > 0.0:
                    degrees[name] = float(value)
            else:
                raise ValueError(
                    f"{name!r} is not a structural parameter; expected "
                    f"{self.COMPOSITION_KEY!r}, 'sigma', or a projection such "
                    "as 'EXC->EXC'"
                )
        if degrees:
            connectivity["mean_degree"] = self._realisable(
                degrees, kwargs.get("total_cells"), kwargs.get("populations")
            )
        return spec

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
            return decoded  # a vector with no background at all, e.g. a cell fit
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
                    floor,
                ),
                float(self.metrics.get(name, float("nan"))),
            )
            for name, floor in self.RESPONSE_OBJECTIVE_EPS.items()
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
        env, recording_data = self._readout(env, recording_data)
        it = recording_data.spike_ids

        local_count = int(len(it) if it is not None else 0)
        total_spike_count = P.reduce_sum(
            np.array(local_count, dtype=np.int64), comm=env.comm, all=True
        )
        total_spike_count = int(
            getattr(total_spike_count, "item", lambda: total_spike_count)()
        )
        self.metrics["total_spikes"] = total_spike_count
        enough_spikes = total_spike_count >= self.MIN_SPIKE_COUNT_FOR_METRICS
        self.metrics["enough_spikes_for_network_metrics"] = enough_spikes

        mfr_result = MeanFiringRate(duration=d)(recording_data, env) or {}
        mfr = float(mfr_result.get("rate_hz", 0.0))
        self.metrics["mfr"] = mfr

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

        per_unit = PerUnitFiringRate(duration=d)(recording_data, env) or {}
        self.metrics["per_unit_rates_hz"] = per_unit.get("per_unit_rates_hz", {})
        self.metrics["max_neuron_firing_rate"] = float(per_unit.get("max_rate_hz", 0.0))

        isi_result = ISICV(duration=d, min_spikes_per_unit=5)(recording_data, env) or {}
        isi_cv = float(isi_result.get("isi_cv", 0.0))
        self.metrics["isi_cv"] = isi_cv
        self.metrics["isi_cv_n_units_used"] = int(isi_result.get("n_units_used", 0))

        isi_target = float(targets["isi_cv"])
        result["isi_cv"] = ((isi_cv - isi_target) ** 2, isi_cv)

        pop_metrics = (
            PopulationRateMetrics(duration=d, bin_size=100.0)(recording_data, env) or {}
        )
        self.metrics["coefficient_of_variation"] = float(
            pop_metrics.get("coefficient_of_variation", 0.0)
        )
        self.metrics["fano_factor"] = float(pop_metrics.get("fano_factor", 0.0))

        tau_result = (
            PopulationAutocorrTau(duration=d, bin_size=10.0, max_lag=5000.0)(
                recording_data, env
            )
            or {}
        )
        self.metrics["pop_autocorr_tau"] = float(
            tau_result.get("pop_autocorr_tau", 10.0)
        )

        if enough_spikes:
            corr_result = (
                PairwiseChannelCorrelation(duration=d, bin_size=10.0, min_units=2)(
                    recording_data, env
                )
                or {}
            )
            self.metrics["mean_channel_correlation"] = float(
                corr_result.get("mean_pairwise_correlation", 0.0)
            )
        else:
            self.metrics["mean_channel_correlation"] = float("nan")

        peak_result = PeakSynchrony(duration=d, bin_size=2.0)(recording_data, env) or {}
        self.metrics["max_synchronous_peak"] = float(
            peak_result.get("max_synchronous_peak", 0.0)
        )

        burst_result = (
            BurstRate(
                duration=d,
                bin_size=50.0,
                mad_k=3.0,
                min_floor_fraction=float(self.BURST_MIN_FLOOR_FRACTION),
                min_floor=2.0,
            )(recording_data, env)
            or {}
        )
        self.metrics["burst_rate"] = float(burst_result.get("burst_rate_hz", 0.0))

        anatomy = BurstAnatomy(duration=d)(recording_data, env)
        self.metrics["burst_anatomy"] = anatomy or {}
        for name in self.ANATOMY_FEATURES:
            self.metrics[name] = float((anatomy or {}).get(name, float("nan")))

        active_result = (
            ActiveFraction(duration=d, min_spikes=1)(recording_data, env) or {}
        )
        active_fraction = float(active_result.get("active_fraction", 0.0))
        self.metrics["active_fraction"] = active_fraction

        af_target = float(targets["active_fraction"])
        af_obj = (af_target - active_fraction) ** 2
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

        for table, unmeasured in (
            (self.BURST_OBJECTIVE_EPS, 0.0),
            (self.RESPONSE_OBJECTIVE_EPS, float("nan")),
        ):
            for name, floor in table.items():
                if name not in targets:
                    continue
                value = float(self.metrics.get(name, unmeasured))
                result[name] = (
                    self._log_ratio_objective(value, float(targets[name]), floor),
                    value,
                )

        avalanche_result = None
        if total_spike_count > 0:
            n_bins_target = max(50, total_spike_count // 15)
            adaptive_bin_width = max(4.0, min(d / n_bins_target, 50.0))
            avalanche_result = AvalancheAnalysis(
                duration=d, bin_width=adaptive_bin_width
            )(recording_data, env)

        sigma = float((avalanche_result or {}).get("branching_ratio", 0.0) or 0.0)
        r2 = float((avalanche_result or {}).get("size_power_law_r2", 0.0) or 0.0)
        self.metrics["branching_ratio"] = sigma
        self.metrics["avalanche_r2"] = r2
        self.metrics["avalanche_result"] = avalanche_result

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
        if self.READOUT != "channels":
            return env, data

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
