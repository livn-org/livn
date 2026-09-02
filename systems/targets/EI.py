import hashlib
import json
import logging
import math
import os
import time
from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from livn.decoding import (
    ISICV,
    ActiveFraction,
    AvalancheAnalysis,
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
)
from livn.env.logging import with_progress_logging
from livn.policy import PulseSweepPolicy
from livn.utils import P
from systems.targets.protocol import TuningTargets

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

    REFERENCE_MV: ClassVar[float] = 300.0
    UA_AT_REFERENCE: ClassVar[float] = 8.0

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

        overrides.setdefault("uA_per_mv", cls.UA_AT_REFERENCE / cls.REFERENCE_MV)

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


class Culture(TuningTargets):
    MIN_RANKS_PER_WORKER = 2

    MAX_NEURON_RATE_HZ = 50.0
    MIN_MEAN_RATE_HZ = 0.2
    MAX_MEAN_RATE_HZ = 15.0
    SYNCHRONY_BAND = (0.02, 0.25)
    MAX_SYNC_PEAK = 0.2
    MIN_SYNC_PEAK = 0.0
    MIN_ACTIVE_FRACTION = 0.5
    MIN_POPULATION_ACTIVE = 0.05
    POP_TAU_BAND_MS = (10.0, 500.0)
    MAX_BURST_RATE_HZ = 0.2
    MIN_BURST_RATE_HZ = 0.0
    BRANCHING_RATIO_BAND = (0.5, 1.5)
    MIN_AVALANCHE_R2 = 0.5
    MAX_POP_RATE_PER_UNIT_HZ = 20.0
    MIN_POP_RATE_PER_UNIT_HZ = 0.05
    STABILITY_MARGIN = 5.0
    IGNITION_SUFFIX = "_ignition"
    IGNITION_RANGE: ClassVar[list] = [0.0005, 0.1]
    ADAPTATION_DECADES = 1.0
    ADAPTATION_POPULATION = "EXC"
    ADAPTATION_PARAMS: ClassVar[dict] = {
        "cells-EXC:soma.gmax_KCa": "soma_gmax_KCa",
        "cells-EXC:dend.gmax_KCa": "dend_gmax_KCa",
        "cells-EXC:soma.kCa_Ca_conc": "soma_kCa_Caconc",
        "cells-EXC:dend.kCa_Ca_conc": "dend_kCa_Caconc",
    }
    RATIO_SUFFIX = "_ratio"
    RATIO_RANGES: ClassVar[dict] = {
        "excitatory": [0.01, 1000.0],
        "inhibitory": [0.01, 1000.0],
    }
    NOISE_TOTAL_RANGE: ClassVar[list] = [0.005, 0.2]
    NOISE_RATIO_RANGE: ClassVar[list] = [1.0, 30.0]
    NOISE_STD_FRACTION_RANGE: ClassVar[list] = [0.05, 1.0]
    NOISE_BALANCE_SUFFIX = "_ratio"
    NOISE_BALANCE_PARAMS: ClassVar[tuple] = ("g_e0", "g_i0")
    NOISE_BALANCE_RANGE: ClassVar[list] = [0.1, 10.0]
    NOISE_BALANCE_POPULATION = "INH"
    SIZE_CV_SUFFIX = "-size_cv"
    SIZE_CV_RANGE: ClassVar[list] = [0.0, 0.8]
    """Cell-to-cell spread in linear size, per population."""
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
    )
    RELEASE_PARAM = "U"
    STIMULUS_GAIN_DECADES = 1.0
    ACTIVE_MIN_SPIKES = 1
    THRESHOLD_OBJECTIVE = "curve"
    GAIN_CEILING_MARGIN = 1e-6

    def __init__(
        self,
        targets: dict | None = None,
        duration: float = 30_000.0,
        warmup: float = 1000.0,
        overrides: dict | None = None,
        skip_objectives: tuple = (),
        skip_constraints: tuple = (),
        readout: str = "neurons",
        system: str | None = None,
        section_aliases: dict | None = None,
        feature_bands: dict | None = None,
        mea: dict | None = None,
        adaptation: bool = False,
        heterogeneity: bool = False,
        noise_balance: bool = False,
        noise_parameterisation: str = "balance",
        ignition: bool = False,
        stimulus: dict | None = None,
        stimulus_threshold: dict | None = None,
        gate_stimulus: bool = True,
        save_spikes: bool | Literal["feasible", "all"] = "all",
    ):
        self._targets = {
            "mfr": 1.0,
            "isi_cv": 1.2,
            "active_fraction": 1.0,
            **(targets or {}),
        }
        if stimulus is not None:
            self._targets.setdefault("stimulus_threshold", 0.0)
        self.system = system
        self.section_aliases = dict(section_aliases or {})
        self.feature_bands = {
            name: (float(lo), float(hi))
            for name, (lo, hi) in (feature_bands or {}).items()
        }
        self.mea = mea
        self.adaptation = bool(adaptation)
        self.heterogeneity = bool(heterogeneity)
        if noise_parameterisation not in {"balance", "conductances"}:
            raise ValueError(
                f"unknown noise_parameterisation {noise_parameterisation!r}; "
                "expected 'balance' (total conductance and I:E ratio) or "
                "'conductances' (g_e0 and g_i0 directly)"
            )
        self.noise_parameterisation = noise_parameterisation
        self.noise_balance = bool(noise_balance)
        self.ignition = bool(ignition)
        self.stimulus = Protocol(**stimulus) if stimulus else None
        self.stimulus_threshold = dict(stimulus_threshold or {})
        self.gate_stimulus = bool(gate_stimulus)
        if save_spikes not in (True, False, "feasible", "all"):
            raise ValueError(
                f"save_spikes={save_spikes!r}; expected 'all' (every evaluation), "
                "'feasible' (only those satisfying every constraint), or a bool"
            )
        self.save_spikes = "feasible" if save_spikes is True else save_spikes
        self.skip_objectives = tuple(skip_objectives)
        self.skip_constraints = tuple(skip_constraints)
        self.readout = readout

        for name, value in (overrides or {}).items():
            if value is None:
                continue
            if name == "targets":
                self._targets = {**self._targets, **value}
            elif hasattr(self, name):
                setattr(self, name, tuple(value) if isinstance(value, list) else value)
            else:
                raise ValueError(f"Unknown override: {name}")

        self.recording_duration = duration
        self.warmup_duration = warmup
        self.min_spike_count_for_metrics = 150
        self._env = None
        self._weight_space_cache: dict[str, list] | None = None
        self._weight_reference: str | None = None
        self._reset_state()

    def _reset_state(self):
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
        if self.readout == "channels" and not len(getattr(env.io, "channel_ids", ())):
            raise RuntimeError("readout='channels' needs an `mea`.")
        if self.stimulus is not None and self.readout != "channels":
            raise RuntimeError(
                f"a stimulated target reads out through the array, not "
                f"{self.readout!r}; pass readout='channels' with the recording "
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
        if section in self.section_aliases:
            return self.section_aliases[section]
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

        depressing = bool(getattr(model, "short_term_depression", False))

        default_ranges = {
            "excitatory": [0.05, 100.0],
            "inhibitory": [0.05, 100.0],
        }
        mechanism_ranges = {"NMDA": [0.05, 5.0]}
        depression_ranges = {
            "tau_rec": [50.0, 3000.0],
            # burst period the mechanism can impose
            "U": [0.05, 0.9],
            # how fast resources deplete, i.e. how many spikes a burst lasts
        }

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

            if self.ignition:
                low, high = self.IGNITION_RANGE
                weights[key + self.IGNITION_SUFFIX] = [low, high, self.transform_log10]
            else:
                low, high = mechanism_ranges.get(
                    mechanism, default_ranges.get(syn_type, [0.001, 10.0])
                )
                weights[key] = [low, high, self.transform_log10]

            if depressing and mechanism == "AMPA":
                for name, (dlo, dhi) in depression_ranges.items():
                    key = f"{post}-{section}-{mechanism}-{name}"
                    if key in weights:
                        continue
                    bounds = [dlo, dhi]
                    if dhi / dlo >= 10.0:
                        bounds.append(self.transform_log10)
                    weights[key] = bounds

        self._weight_space_cache = weights
        return weights

    def decode_params(self, params: dict, model=None) -> dict:
        decoded = super().decode_params(params, model=model)
        decoded = self._resolve_size(decoded)
        decoded = self._resolve_noise_drive(decoded)
        decoded = self._resolve_ignition(decoded, model)
        decoded = self._resolve_noise_balance(decoded)

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
        allowed *= 1.0 - self.GAIN_CEILING_MARGIN
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

            span = 10.0**self.STIMULUS_GAIN_DECADES
            lo = DEFAULT_STIMULATION_GAIN / span
            space["io-volume_conductor-stimulation_gain"] = [
                lo,
                self._gain_ceiling(model, lo, DEFAULT_STIMULATION_GAIN * span),
                self.transform_log10,
            ]

        space.update(self._size_space())

        if not self.adaptation:
            return space

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
                "adaptation=True but the model exposes no "
                "'BoothRinzelKiehn-MN' parameters to centre the bounds on; "
                "the search would silently drop these dimensions"
            )

        span = 10.0**self.ADAPTATION_DECADES
        for key, name in self.ADAPTATION_PARAMS.items():
            value = fitted.get(name)
            if value is None or float(value) <= 0.0:
                raise ValueError(
                    f"{name!r} is {value!r}; a log-scaled bound needs a "
                    "positive fitted value to centre on"
                )
            value = float(value)
            space[key] = [value / span, value * span, self.transform_log10]
        return space

    def _size_space(self) -> dict[str, list]:
        if not self.heterogeneity:
            return {}
        populations = ["EXC", "INH"]
        env = getattr(self, "_env", None)
        if env is not None:
            populations = list(env.active_populations())
        lo, hi = self.SIZE_CV_RANGE
        return {f"{p}{self.SIZE_CV_SUFFIX}": [lo, hi] for p in populations}

    def _resolve_size(self, decoded: dict) -> dict:
        suffix = self.SIZE_CV_SUFFIX
        cvs = {
            name[: -len(suffix)]: float(value)
            for name, value in decoded.items()
            if name.endswith(suffix)
        }
        if not cvs:
            return decoded

        resolved = {k: v for k, v in decoded.items() if not k.endswith(suffix)}
        env = getattr(self, "_env", None)
        if env is None:  # describe_params, with nothing built to apply them to
            return resolved

        model = env.model
        ranges = env.system.population_ranges
        scaled = {}
        for population, cv in cvs.items():
            start, count = ranges.get(population, (0, 0))
            if not count:
                continue
            gids = list(range(int(start), int(start) + int(count)))
            factors = model.size_scales(population, gids, cv=cv)
            params = model.params(
                model._inh_params_name()
                if population == "INH"
                else "BoothRinzelKiehn-MN"
            )
            for key in ("global_diam", "axon_diam"):
                base = params.get(key)
                if base is None:
                    continue
                scaled.setdefault(key, {}).update(
                    dict(zip(gids, float(base) * factors, strict=True))
                )

        order = [int(g) for g in env.cells.gids]
        for key, by_gid in scaled.items():
            if len(by_gid) < len(order):  # a population left un-searched
                continue
            resolved[f"cells-{key}"] = [float(by_gid[g]) for g in order]
        return resolved

    def _resolve_ignition(self, decoded: dict, model) -> dict:
        self._weight_space(model)  # populates `_weight_reference`
        reference = self._weight_reference
        if reference is None:
            return decoded
        key = reference + self.IGNITION_SUFFIX
        if key not in decoded:
            return decoded

        background = decoded.get("noise-g_e0")
        if not background:
            raise ValueError(
                f"{key!r} is a product with the background drive, but "
                "'noise-g_e0' is not in this vector, so it cannot be resolved "
                "to a weight. The target is probably built differently from "
                "the run that produced it."
            )
        decoded[reference] = float(decoded.pop(key)) / float(background)
        return decoded

    def _resolve_release(self, decoded: dict) -> dict:
        suffix = f"-{self.RELEASE_PARAM}"
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
        if self.noise_parameterisation == "balance":
            drive = {
                "noise-g_total": [*self.NOISE_TOTAL_RANGE, self.transform_log10],
                "noise-g_ratio": [*self.NOISE_RATIO_RANGE, self.transform_log10],
                "noise-std_fraction": [
                    *self.NOISE_STD_FRACTION_RANGE,
                    self.transform_log10,
                ],
            }
        else:
            drive = {
                "noise-g_e0": [0.0002, 0.02, self.transform_log10],
                "noise-g_i0": [0.002, 0.06, self.transform_log10],
                "noise-std_e": [0.0001, 0.05, self.transform_log10],
                "noise-std_i": [0.0005, 0.05, self.transform_log10],
            }
        space = {
            **drive,
            "noise-tau_e": [1.0, 40.0, self.transform_log10],
            "noise-tau_i": [4.0, 40.0],
        }
        if self.noise_balance:
            lo, hi = self.NOISE_BALANCE_RANGE
            population = self.NOISE_BALANCE_POPULATION
            for name in self.NOISE_BALANCE_PARAMS:
                key = f"noise-{population}-{name}{self.NOISE_BALANCE_SUFFIX}"
                space[key] = [lo, hi, self.transform_log10]
        return space

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

    def _resolve_noise_balance(self, decoded: dict) -> dict:
        suffix = self.NOISE_BALANCE_SUFFIX
        ratios = {
            name: value
            for name, value in decoded.items()
            if name.startswith("noise-") and name.endswith(suffix)
        }
        if not ratios:
            return decoded

        resolved = {k: v for k, v in decoded.items() if k not in ratios}
        for name, value in ratios.items():
            _, population, parameter = name[: -len(suffix)].split("-", 2)
            shared = decoded.get(f"noise-{parameter}")
            if shared is None:
                raise ValueError(
                    f"{name!r} is a multiple of 'noise-{parameter}', which is "
                    "not in this vector, so it cannot be resolved to a "
                    "conductance. The target is probably built differently "
                    "from the run that produced it."
                )
            resolved[f"noise-{population}-{parameter}"] = float(shared) * float(value)
        return resolved

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

        self._keep_spikes(env, params, directory, constraints)

        return objectives, constraints

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
        """Whether the free-running window leaves this candidate feasible."""
        if not self.gate_stimulus:
            return True
        return all(float(value) >= 0.0 for value, _ in constraints.values())

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
        enough_spikes = total_spike_count >= self.min_spike_count_for_metrics
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
            tail_window=1000.0,
            max_rate_hz=self.MAX_POP_RATE_PER_UNIT_HZ * n_units * self.STABILITY_MARGIN,
            min_rate_hz=self.MIN_POP_RATE_PER_UNIT_HZ * n_units / self.STABILITY_MARGIN,
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
                min_floor_fraction=0.10,
                min_floor=2.0,
            )(recording_data, env)
            or {}
        )
        self.metrics["burst_rate"] = float(burst_result.get("burst_rate_hz", 0.0))

        active_result = (
            ActiveFraction(duration=d, min_spikes=int(self.ACTIVE_MIN_SPIKES))(
                recording_data, env
            )
            or {}
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

        scored = curve_miss if self.THRESHOLD_OBJECTIVE == "curve" else miss
        return (1e3 if np.isnan(scored) else float(scored), miss)

    def _readout(self, env, data):
        if self.readout != "channels":
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
            float(_max_constraint(max_neuron_rate, self.MAX_NEURON_RATE_HZ)),
            float(max_neuron_rate),
        )

        mean_sync = m.get("mean_channel_correlation", float("nan"))
        try:
            mean_sync_f = float(np.clip(float(mean_sync), -1.0, 1.0))
        except (TypeError, ValueError):
            mean_sync_f = float("nan")
        sync_c = _band_constraint(
            mean_sync_f, self.SYNCHRONY_BAND[0], self.SYNCHRONY_BAND[1]
        )
        result["synchrony"] = (
            float(np.clip(sync_c, -10.0, 10.0)),
            mean_sync_f,
        )

        peak_sync = m.get("max_synchronous_peak", float("nan"))
        result["max_synchronous_peak"] = (
            float(_band_constraint(peak_sync, self.MIN_SYNC_PEAK, self.MAX_SYNC_PEAK)),
            float(peak_sync),
        )

        mean_rate = m.get("mfr", float("nan"))
        result["min_mean_firing_rate"] = (
            float(_min_constraint(mean_rate, self.MIN_MEAN_RATE_HZ)),
            float(mean_rate),
        )
        result["max_mean_firing_rate"] = (
            float(_max_constraint(mean_rate, self.MAX_MEAN_RATE_HZ)),
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
                _min_constraint(active_fraction, self.MIN_ACTIVE_FRACTION, scale=1.0)
            ),
            float(active_fraction),
        )

        pop_tau = m.get("pop_autocorr_tau", float("nan"))
        result["pop_autocorr_tau_band"] = (
            float(_band_constraint(pop_tau, *self.POP_TAU_BAND_MS)),
            float(pop_tau),
        )

        burst_rate = m.get("burst_rate", float("nan"))
        result["burst_rate_band"] = (
            float(
                _band_constraint(
                    burst_rate, self.MIN_BURST_RATE_HZ, self.MAX_BURST_RATE_HZ
                )
            ),
            float(burst_rate),
        )

        sigma = m.get("branching_ratio", float("nan"))
        result["branching_ratio_band"] = (
            float(_band_constraint(sigma, *self.BRANCHING_RATIO_BAND)),
            float(sigma),
        )

        avalanche_r2 = m.get("avalanche_r2", float("nan"))
        result["avalanche_r2"] = (
            float(_min_constraint(avalanche_r2, self.MIN_AVALANCHE_R2, scale=1.0)),
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
