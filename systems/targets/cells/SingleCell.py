from __future__ import annotations

import os
import warnings

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from livn.env import Env
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.stimulus import Stimulus
from livn.utils import import_object_by_path

from . import ephys


class StepTarget(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    current: list[float] = Field(default_factory=list, alias="I")
    current_factor: float = Field(1.0, alias="I_factor")
    t: list[float] | None = None
    mean: list[float | None] | None = None
    lower: list[float | None] | None = None
    upper: list[float | None] | None = None
    V_hold: float | None = None


class HoldTarget(BaseModel):
    val: float


class SingleCellTargets(BaseModel):
    Rin: StepTarget
    tau0: StepTarget
    threshold: float = -30.0
    V_hold: HoldTarget
    V_rest: HoldTarget
    f_I: StepTarget
    spike_amp: StepTarget
    spike_adaptation: StepTarget


class SingleCellNumerics(BaseModel):
    tstop: float
    celsius: float = 36.0


class DriveTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    levels: list[float] = [0.3, 1.0, 3.0, 6.0]
    duration: float = 2000.0
    seed: int = 20260811
    min_spikes: float = 5.0
    min_top_fraction: float = 0.5
    std_fraction: float = 0.33
    tau_e: float = 33.0
    tau_i: float = 28.5
    mean: list[float | None] | None = None
    lower: list[float | None] | None = None
    upper: list[float | None] | None = None


class SingleCellOptConfig(BaseModel):
    Template: str
    Population: str = "EXC"
    Numerics: SingleCellNumerics
    Targets: SingleCellTargets
    Drive: DriveTarget = DriveTarget()
    Parameters: dict[str, float] = {}
    Space: dict[str, list[float]] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "SingleCellOptConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


class SingleCellModel(ReducedCalciumSomaDendrite):
    def __init__(
        self,
        template: str,
        threshold: float = -30.0,
        v_rest: float = -60.0,
        dend_type: str = "hillock",
        celsius: float = 36.0,
        population: str = "EXC",
    ):
        super().__init__()
        self._template_path = template
        self._cell_threshold = float(threshold)
        self._cell_v_rest = float(v_rest)
        self._cell_dend_type = dend_type
        self._celsius = float(celsius)
        self._population = population

    def neuron_celsius(self) -> float:
        return self._celsius

    def neuron_cells(self):
        from livn.backend.neuron.cells import ReducedCell

        template_class = import_object_by_path(self._template_path)
        thr, vr, dt = self._cell_threshold, self._cell_v_rest, self._cell_dend_type

        def make(morphology=None):
            cell = template_class()  # defaults; target reconfigures per eval
            return ReducedCell(cell, threshold=thr, v_rest=vr, dend_type=dt)

        return {self._population: make}


def range_distance(x, lb, ub):
    """0 inside [lb, ub], else distance to the nearer edge (NaN-safe -> inf)."""
    if x is None or np.isnan(x):
        return np.inf
    if x < lb:
        return lb - x
    if x > ub:
        return x - ub
    return 0.0


class SingleCell:
    """Generic single-cell current-clamp tuning target."""

    def __init__(
        self,
        config: SingleCellOptConfig | dict | str = os.path.join(
            os.path.dirname(__file__), "motoneuron.yaml"
        ),
        population: str | None = None,
        sim_dt: float = 0.0125,
        record_dt: float = 0.05,
    ):
        self.cfg = self._resolve_config(config)
        self.population = population or self.cfg.Population
        self.sim_dt = float(sim_dt)
        self.record_dt = float(record_dt)
        self._decoded: dict | None = None
        self._parse()

    @staticmethod
    def _resolve_config(config) -> SingleCellOptConfig:
        if isinstance(config, SingleCellOptConfig):
            return config
        if isinstance(config, str):
            return SingleCellOptConfig.from_yaml(config)
        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
        except ImportError:
            pass
        return SingleCellOptConfig(**dict(config))

    def _parse(self) -> None:
        cfg = self.cfg
        self.template_path = cfg.Template

        self.tstop = float(cfg.Numerics.tstop)
        self.celsius = float(cfg.Numerics.celsius)

        tc = cfg.Targets
        self.v_hold = float(tc.V_hold.val)
        self.v_rest = float(tc.V_rest.val)
        self.target_threshold = float(tc.threshold)

        rin = tc.Rin
        self.rin_amp = float(rin.current[0]) * float(rin.current_factor)
        self.rin_t = (float(rin.t[0]), float(rin.t[1]))
        self.target_rn = (float(rin.lower[0]), float(rin.upper[0]))

        tau0 = tc.tau0
        self.target_tau = (float(tau0.lower[0]), float(tau0.upper[0]))

        fI = tc.f_I
        self.fI_amps = np.asarray(fI.current, dtype=float) * float(fI.current_factor)
        self.fI_t = (float(fI.t[0]), float(fI.t[1]))

        # optional holding potential for the f-I sweeps (default: v_hold). Set to
        # the RMP when the f-I data was recorded from rest rather than a clamp.
        self.fI_hold = float(fI.V_hold if fI.V_hold is not None else self.v_hold)
        self.fI_mean = np.asarray(fI.mean or [], dtype=float)
        self.fI_lb = np.asarray(
            fI.lower if fI.lower is not None else (fI.mean or []), dtype=float
        )
        self.fI_ub = np.asarray(
            fI.upper if fI.upper is not None else (fI.mean or []), dtype=float
        )

        sa = tc.spike_amp
        self.spk_amp_lb = np.asarray(sa.lower or [], dtype=float)
        self.spk_amp_ub = np.asarray(sa.upper or [], dtype=float)

        sad = tc.spike_adaptation
        self.spk_adapt_lb = np.asarray(sad.lower or [], dtype=float)
        self.spk_adapt_ub = np.asarray(sad.upper or [], dtype=float)

        # No explicit first-ISI floor in the MN config -> 0 (any repetitive
        # firing passes; silent sweeps fail, as in the reference optimization).
        self.first_ISI_lower = np.zeros_like(self.fI_amps)

        self.drive = cfg.Drive

        self.fixed = dict(cfg.Parameters)
        self.space = {k: [float(v[0]), float(v[1])] for k, v in cfg.Space.items()}

    def _log_scaled(self) -> set:
        return {k for k, (lo, hi) in self.space.items() if lo > 0 and hi / lo >= 10.0}

    def search_space(self, model=None) -> dict:
        log_scaled = self._log_scaled()
        return {
            k: (
                [float(np.log10(lo)), float(np.log10(hi))]
                if k in log_scaled
                else [float(lo), float(hi)]
            )
            for k, (lo, hi) in self.space.items()
        }

    def _from_search(self, x) -> dict:
        """Optimizer coordinates back to biophysical values."""
        log_scaled = self._log_scaled()
        return {
            k: (10.0 ** float(v) if k in log_scaled else float(v))
            for k, v in dict(x).items()
        }

    def objective_names(self) -> list[str]:
        return [
            "rn_error",
            "tau_error",
            "fI_error",
            "spike_amplitude_error",
            "ISI_adaptation_error",
        ] + (["drive_error"] if self._drive_bands() is not None else [])

    def feature_bands(self) -> dict:
        bands = {
            "rn_error": self.target_rn,
            "tau_error": self.target_tau,
        }
        if self.spk_amp_lb.size and self.spk_amp_ub.size:
            bands["spike_amplitude_error"] = (
                float(np.mean(self.spk_amp_lb)),
                float(np.mean(self.spk_amp_ub)),
            )
        if self.spk_adapt_lb.size and self.spk_adapt_ub.size:
            bands["ISI_adaptation_error"] = (
                float(np.mean(self.spk_adapt_lb)),
                float(np.mean(self.spk_adapt_ub)),
            )

        fI_width = float(np.mean(self.fI_ub - self.fI_lb)) if self.fI_ub.size else 0.0
        if fI_width > 1e-9:
            bands["fI_error"] = (float(np.mean(self.fI_lb)), float(np.mean(self.fI_ub)))
        elif self.fI_mean.size:
            m = float(np.mean(self.fI_mean))
            bands["fI_error"] = (0.8 * m, 1.2 * m)

        drive = self._drive_bands()
        if drive is not None:
            bands["drive_error"] = (
                float(np.nanmean(drive[0])),
                float(np.nanmean(drive[1])),
            )
        return bands

    def objective_scales(self) -> dict:
        fI_w = float(np.mean(self.fI_ub - self.fI_lb)) if self.fI_ub.size else 0.0
        if fI_w <= 1e-9:  # mean-only f-I bands (no lower/upper) -> relative tol
            fI_w = 0.2 * float(np.mean(self.fI_mean)) if self.fI_mean.size else 5.0
        spk_w = (
            float(np.mean(self.spk_amp_ub - self.spk_amp_lb))
            if self.spk_amp_ub.size
            else 20.0
        )
        isi_w = (
            float(np.mean(self.spk_adapt_ub - self.spk_adapt_lb)) * 100.0
            if self.spk_adapt_ub.size
            else 100.0
        )
        widths = {
            "rn_error": max(self.target_rn[1] - self.target_rn[0], 1.0),
            "tau_error": max(self.target_tau[1] - self.target_tau[0], 1.0),
            "fI_error": max(fI_w, 1.0),
            "spike_amplitude_error": max(spk_w, 1.0),
            "ISI_adaptation_error": max(isi_w, 1.0),
        }
        bands = self._drive_bands()
        if bands is not None:
            widths["drive_error"] = max(float(np.nanmean(bands[1] - bands[0])), 1.0)
        return widths

    def _drive_constraints(self, counts) -> dict:
        if not self.drive.enabled:
            return {}
        top, best = float(counts[-1]), float(np.max(counts))
        return {
            "drive_sustained": (best - self.drive.min_spikes, best),
            "drive_no_inversion": (top - self.drive.min_top_fraction * best, top),
        }

    def _drive_rates(self, counts) -> np.ndarray:
        """Spike counts over the sweep as rates in Hz."""
        seconds = max(self.drive.duration, 1e-9) / 1000.0
        return np.asarray(counts, dtype=float) / seconds

    def _drive_bands(self) -> tuple[np.ndarray, np.ndarray] | None:
        """`(lower, upper)` per level, or None when the sweep is unscored."""
        d = self.drive
        if not d.enabled:
            return None

        def _arr(values):
            return np.asarray(
                [np.nan if v is None else float(v) for v in values], float
            )

        if d.lower is not None and d.upper is not None:
            lower, upper = _arr(d.lower), _arr(d.upper)
        elif d.mean is not None:
            mean = _arr(d.mean)
            lower, upper = 0.7 * mean, 1.3 * mean  # +/-30%, as the f-I bands do
        else:
            return None
        if np.all(np.isnan(lower)) or np.all(np.isnan(upper)):
            return None
        if len(lower) != len(d.levels) or len(upper) != len(d.levels):
            raise ValueError(
                f"Drive has {len(d.levels)} levels but "
                f"{len(lower)}/{len(upper)} band entries"
            )
        return lower, upper

    def _drive_objective(self, counts) -> tuple[float, float] | None:
        bands = self._drive_bands()
        if bands is None:
            return None
        rates = self._drive_rates(counts)
        lower, upper = bands
        n = min(len(rates), len(lower))
        scored = [
            range_distance(rates[i], lower[i], upper[i]) ** 2
            for i in range(n)
            if not (np.isnan(lower[i]) or np.isnan(upper[i]))
        ]
        if not scored:
            return None
        return _finite(float(np.mean(scored))), float(np.mean(rates)) if n else 0.0

    def constraint_names(self) -> list[str]:
        return (
            []
            if not self.drive.enabled
            else [
                "drive_sustained",
                "drive_no_inversion",
            ]
        ) + [
            "monotonic_fI",
            "rn_constr",
            "tau_constr",
            "spike_amplitude_constr",
            "first_ISI_constr",
            "ISI_adaptation_constr",
            "pre_spk_count",
            "initial_v_constr",
        ]

    def transform_params(self, x) -> dict:
        # x: {param: value} from the optimizer. Merge with fixed parameters and
        # stash for __call__ (applied to the cell via env.cells, not set_params).
        self._decoded = self._resolve(x)
        return {}

    def decode_params(self, x, model=None) -> dict:
        return self._resolve(x)

    def _resolve(self, x) -> dict:
        decoded = dict(self.fixed)
        decoded.update(self._from_search(x))
        return self._couple_from_axial_resistance(decoded)

    @staticmethod
    def _couple_from_axial_resistance(decoded: dict) -> dict:
        if "Ra" not in decoded or "gc" in decoded:
            return decoded

        import math

        diam = float(decoded["global_diam"])  # um
        total = float(decoded["Ltotal"])  # um
        pp = float(decoded["pp"])  # somatic fraction of the length

        cross_cm2 = math.pi * (diam * 1e-4 / 2.0) ** 2
        r_axial = float(decoded["Ra"]) * (total * 1e-4 / 2.0) / cross_cm2 * 1e-6

        area_soma = math.pi * diam * (pp * total) * 1e-8  # cm2
        area_dend = math.pi * diam * ((1.0 - pp) * total) * 1e-8
        decoded = dict(decoded)
        decoded["gc"] = pp / 2e3 * (1.0 / area_soma + 1.0 / area_dend) / r_axial
        return decoded

    def set_params(self, params: dict) -> dict:
        # single-cell biophysics are not env weights/noise; nothing to route
        return {}

    def describe_params(self, decoded: dict) -> dict:
        """Split decoded params into searched vs fixed for the inspect() report."""
        searched = {k: decoded[k] for k in self.space if k in decoded}
        fixed = {k: v for k, v in decoded.items() if k not in self.space}
        return {"Searched params": searched, "Fixed params": fixed}

    def rank_solutions(self, best):
        import pandas as pd

        y = best.get("y")
        if not isinstance(y, pd.DataFrame) or len(y) <= 1:
            return best

        c = best.get("c")
        if isinstance(c, pd.DataFrame) and len(c.columns):
            feasible = (c.to_numpy() >= 0).all(axis=1)
        else:
            feasible = np.ones(len(y), dtype=bool)

        if not feasible.any():
            violated = (
                c.columns[(c.to_numpy() < 0).any(axis=0)].tolist()
                if isinstance(c, pd.DataFrame) and len(c.columns)
                else []
            )
            warnings.warn(
                f"no point on this front is feasible; returning the "
                f"best-ranked one, which violates "
                f"{', '.join(violated) or 'at least one constraint'}",
                stacklevel=2,
            )

        rank_sum = y.rank(axis=0, method="min").sum(axis=1).to_numpy()

        # feasible first (~feasible: 0 for feasible), then lowest rank-sum
        idx = np.asarray(y.index)[np.lexsort((rank_sum, ~feasible))]
        return {
            k: (v.loc[idx].reset_index(drop=True) if isinstance(v, pd.DataFrame) else v)
            for k, v in best.items()
        }

    def build_env(self, system, model=None, comm=None, subworld_size=None):
        cell_model = SingleCellModel(
            template=self.template_path,
            threshold=self.target_threshold,
            v_rest=self.v_rest,
            celsius=self.celsius,
            population=self.population,
        )
        if isinstance(system, int):
            system = {self.population: system}
        env = Env(system, model=cell_model, comm=comm, subworld_size=subworld_size)
        if isinstance(system, (str, os.PathLike)):
            env.selection(1)
        env.init()
        env.record_voltage(dt=self.record_dt)
        env.v_init = self.v_hold  # initialize at the holding potential each run
        return env

    def init(self, env):
        env.record_voltage(dt=self.record_dt)
        return env

    def _cell(self, env):
        for _pop, cells in env.cells.items():
            for gid, cell in cells.items():
                return int(gid), cell
        raise RuntimeError("SingleCell: no cell built (selection failed?)")

    def _apply_params(self, cell) -> None:
        # reconfigure the underlying template in place (BRK / V1In both expose
        # set_parameters/geometry/biophys on flat params)
        impl = cell._template
        impl.set_parameters(self._decoded)
        # geometry() recomputes nseg from each section's current Ra/cm via the
        # d_lambda rule. A fresh build (and the reference MN_nrn.hoc) run this with
        # NEURON-default Ra/cm, but a reconfigured cell still carries the previous
        # biophys values -> a different nseg, which changes tau/f-I. Reset to the
        # NEURON defaults so re-applied cells discretize identically to a fresh one.
        for sec in getattr(impl, "sections", []):
            sec.Ra = 35.4
            sec.cm = 1.0
        impl.geometry()
        impl.biophys()
        cell._v_rest = self.v_hold  # hold at v_hold during init_ic pinning

    def _pin_holding(
        self,
        env,
        gid,
        cell,
        v_target=None,
        bracket=(-0.5, 0.5),
        settle_ms=600.0,
        n_iter=10,
    ):
        """Solve for the steady-state holding current at ``v_target`` (default
        ``v_hold``)."""
        if v_target is None:
            v_target = self.v_hold
        soma = cell._template.soma
        n = int(round(settle_ms / self.record_dt))
        zeros = np.zeros((n, 1), dtype=np.float32)
        gid_arr = np.array([gid], dtype=np.int32)
        tail = max(1, int(round(300.0 / self.record_dt)))

        def probe(ic_val):
            cell.init_ic = lambda v=None, _x=ic_val: setattr(soma, "ic_constant", _x)
            env.clear(reseed=False)
            stim = Stimulus.from_current(zeros, dt=self.record_dt, gids=gid_arr)
            _s, _t, vid, v, _c, _cc = env.run(settle_ms, stim, dt=self.sim_dt)
            vid = np.asarray(vid)
            v = np.asarray(v)
            trace = v[int(np.where(vid == gid)[0][0])]
            spiked = bool(np.max(trace) >= 0.0)
            return float(np.mean(trace[-tail:])), spiked

        # lo: too depolarized / spiking (root is at higher ic); hi: hyperpolarized,
        # non-spiking, settled below v_target (root at lower ic).
        lo, hi = float(bracket[0]), float(bracket[1])
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            mean_v, spiked = probe(mid)
            if spiked or mean_v > v_target:  # too depolarized -> raise ic
                lo = mid
            else:  # too hyperpolarized -> lower ic
                hi = mid
        solved = 0.5 * (lo + hi)

        cell.init_ic = lambda v=None, _x=float(solved): setattr(soma, "ic_constant", _x)
        return solved

    def _hold_at(self, env, gid, cell, v):
        """(Re)initialize at membrane potential ``v`` and pin the holding current
        so the cell settles there. Used to hold the passive protocol at v_hold and
        the f-I sweeps at their own holding potential (fI_hold)."""
        env.clear(reseed=False)
        env.v_init = v  # allowed: clear() reset t to 0
        self._pin_holding(env, gid, cell, v_target=v)

    def _run_step(self, env, gid, amp, t0, t1, tstop):
        env.clear(reseed=False)
        n = int(round(tstop / self.record_dt))
        cur = np.zeros((n, 1), dtype=np.float32)
        cur[int(round(t0 / self.record_dt)) : int(round(t1 / self.record_dt)), 0] = amp
        stim = Stimulus.from_current(
            cur, dt=self.record_dt, gids=np.array([gid], dtype=np.int32)
        )
        _sid, _st, vid, v, _cid, _c = env.run(tstop, stim, dt=self.sim_dt)
        vid = np.asarray(vid)
        v = np.asarray(v)
        row = int(np.where(vid == gid)[0][0])
        trace = v[row]
        # time axis at the dt we recorded with (record_voltage(dt=record_dt));
        # env.voltage_recording_dt does not reflect the per-call dt.
        t = np.arange(trace.size, dtype=float) * self.record_dt
        return t, trace

    NOISE_KEYS = ("g_e0", "g_i0", "std_e", "std_i")

    @classmethod
    def _drive_noise(cls, env) -> dict:
        try:
            import inspect

            sig = inspect.signature(env.model.neuron_noise_configure)
            return {
                f"noise-{k}": float(sig.parameters[k].default)
                for k in cls.NOISE_KEYS
                if k in sig.parameters
            }
        except Exception:  # pragma: no cover - model without a noise mechanism
            return {}

    @classmethod
    def _resting_noise(cls, env) -> dict:
        state = getattr(env, "_noise_state", None) or {}
        return {
            f"noise-{k}": float(state[k]) if k in state else 0.0 for k in cls.NOISE_KEYS
        }

    def _drive_counts(self, env, gid) -> np.ndarray:
        resting = self._resting_noise(env)
        driving = self._drive_noise(env)
        env.record_spikes()
        counts = []
        try:
            for i, level in enumerate(self.drive.levels):
                env.clear(reseed=False)
                env.set_params(
                    {
                        **driving,
                        "noise-g_e0": float(level),
                        "noise-std_e": float(level) * self.drive.std_fraction,
                        "noise-tau_e": self.drive.tau_e,
                        "noise-tau_i": self.drive.tau_i,
                    }
                )
                env.reseed_noise(self.drive.seed + i)
                run = env.run(self.drive.duration, dt=self.sim_dt)
                ids = getattr(run, "spike_ids", None)
                counts.append(0 if ids is None else int(np.sum(np.asarray(ids) == gid)))
        finally:
            env.clear(reseed=False)
            env.set_params(resting)
        return np.asarray(counts, dtype=float)

    def _measure(self, env) -> dict:
        gid, cell = self._cell(env)
        self._apply_params(cell)

        # --- passive: Rin / tau, held at v_hold ---------------------------
        self._hold_at(env, gid, cell, self.v_hold)
        t, v = self._run_step(
            env, gid, self.rin_amp, self.rin_t[0], self.rin_t[1], self.rin_t[1] + 50.0
        )
        passive = ephys.measure_passive(
            t, v, self.rin_t[0], self.rin_t[1], self.rin_amp
        )
        rn = float(passive["Rinp"])
        tau = float(passive["tau"])

        # holding-voltage error before the step (should sit at v_hold)
        pre = v[t < self.rin_t[0]]
        v_baseline = float(np.mean(pre)) if pre.size else float(v[0])

        # --- f-I sweeps, held at fI_hold (= RMP for cells whose f-I data was
        # recorded from rest, e.g. Renshaw/Perry 2015; defaults to v_hold) ------
        if self.fI_hold != self.v_hold:
            self._hold_at(env, gid, cell, self.fI_hold)
        # only need to simulate a little past the step end (not the full tstop)
        fI_run_ms = self.fI_t[1] + 100.0
        iclamp_results = []
        for amp in self.fI_amps:
            ti, vi = self._run_step(
                env, gid, amp, self.fI_t[0], self.fI_t[1], fI_run_ms
            )
            iclamp_results.append({"t": ti, "v": vi})

        pre_spk_cnt, spk_cnt, spk_infos, thresholds, spk_amps = (
            ephys.measure_spike_features(
                iclamp_results, self.fI_t[0], self.fI_t[1] + 2.0
            )
        )
        ISI = ephys.measure_ISI(self.fI_amps, spk_infos)
        fI = ephys.measure_fI(spk_cnt[:, 0], self.fI_t[0], self.fI_t[1], self.fI_amps)

        drive_counts = (
            self._drive_counts(env, gid)
            if self.drive.enabled
            else np.asarray([], dtype=float)
        )

        return {
            "drive_counts": drive_counts,
            "rn": rn,
            "tau": tau,
            "v_baseline": v_baseline,
            "initial_v_error": v_baseline - self.v_hold,
            "rates": np.asarray(fI["frequency"], dtype=float),
            "spk_amps": np.asarray(spk_amps, dtype=float),
            "ISI": ISI,
            "pre_spk_cnt": pre_spk_cnt,
            "traces": iclamp_results,
            "passive_trace": {"t": t, "v": v},
        }

    def __call__(self, env):
        m = self._measure(env)
        rn = m["rn"]
        tau = m["tau"]
        rates = m["rates"]
        spk_amps = m["spk_amps"]
        ISI = m["ISI"]
        pre_spk_cnt = m["pre_spk_cnt"]
        initial_v_error = m["initial_v_error"]

        # --- objectives ----------------------------------------------------
        rn_obj = range_distance(rn, *self.target_rn) ** 2
        tau_obj = range_distance(tau, *self.target_tau) ** 2

        fI_obj = float(
            np.mean(
                [
                    range_distance(r, lb, ub) ** 2
                    for r, lb, ub in zip(rates, self.fI_lb, self.fI_ub)
                ]
            )
        )

        n_amp = min(len(self.spk_amp_lb), len(spk_amps))
        amp_dists = [
            range_distance(spk_amps[i], self.spk_amp_lb[i], self.spk_amp_ub[i]) ** 2
            for i in range(n_amp)
            if not np.isnan(self.spk_amp_lb[i])
        ]
        spike_amplitude_obj = float(np.mean(amp_dists)) if amp_dists else np.nan

        n_adpt = min(len(self.spk_adapt_lb), len(ISI))
        adapt_dists = [
            range_distance(
                ISI["ratio"][i] * 100.0,
                self.spk_adapt_lb[i] * 100.0,
                self.spk_adapt_ub[i] * 100.0,
            )
            ** 2
            for i in range(n_adpt)
        ]
        ISI_adaptation_obj = float(np.mean(adapt_dists)) if adapt_dists else np.nan

        drive_obj = self._drive_objective(m["drive_counts"])

        objectives = {
            "rn_error": (_finite(rn_obj), rn),
            "tau_error": (_finite(tau_obj), tau),
            "fI_error": (_finite(fI_obj), float(np.mean(rates)) if len(rates) else 0.0),
            "spike_amplitude_error": (
                _finite(spike_amplitude_obj),
                float(np.nanmean(spk_amps)) if spk_amps.size else 0.0,
            ),
            "ISI_adaptation_error": (
                _finite(ISI_adaptation_obj),
                float(np.nanmean(ISI["ratio"])) if len(ISI) else 0.0,
            ),
        }
        if drive_obj is not None:
            objectives["drive_error"] = drive_obj

        # --- constraints (>=0 feasible, <0 infeasible) ---------------------
        rate_diff = np.diff(rates[:-1]) if len(rates) > 2 else np.array([1.0])
        monotonic = 1.0 if np.all(rate_diff > 0) else -1.0
        rn_constr = 1.0 if (rn > 0.0 and rn < 5000.0) else -1.0
        tau_constr = 1.0 if (tau > 0.0 and tau < 1000.0) else -1.0
        spike_amp_constr = -1.0 if np.isnan(spike_amplitude_obj) else 1.0
        first_ISI = ISI["first"]
        first_ISI_constr = 1.0 if np.all(first_ISI > self.first_ISI_lower) else -1.0
        ISI_adapt_constr = -1.0 if np.isnan(ISI_adaptation_obj) else 1.0
        pre_spk_constr = -1.0 if np.sum(pre_spk_cnt) > 0 else 1.0
        initial_v_constr = 1.0 if abs(initial_v_error) < 1.0 else -1.0

        constraints = {
            **self._drive_constraints(m["drive_counts"]),
            "monotonic_fI": (monotonic, float(np.mean(rates)) if len(rates) else 0.0),
            "rn_constr": (rn_constr, rn),
            "tau_constr": (tau_constr, tau),
            "spike_amplitude_constr": (spike_amp_constr, 0.0),
            "first_ISI_constr": (first_ISI_constr, 0.0),
            "ISI_adaptation_constr": (ISI_adapt_constr, 0.0),
            "pre_spk_count": (pre_spk_constr, float(np.sum(pre_spk_cnt))),
            "initial_v_constr": (initial_v_constr, float(initial_v_error)),
        }

        return objectives, constraints

    def plot_measurement(self, env, save_path: str, title: str = "") -> str:
        """Re-simulate the currently-stashed params and render modeled vs target:
        the f-I curve, per-sweep spike amplitude and ISI adaptation against their
        target bands, and the passive properties (Rin, tau) normalized to their
        bands. Returns the path the figure was written to."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = self._measure(env)
        amps = self.fI_amps * 1e3  # nA -> pA for readability
        band = "#c7d3e0"
        c_model, c_target = "#1f4e8c", "#d1622b"
        ok, bad = "#2e8b57", "#c0392b"

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        def _band_panel(ax, y, lb, ub, ylabel, sub, mean=None):
            n = min(len(amps), len(y))
            x = amps[:n]
            if lb is not None and ub is not None and len(lb) >= n and len(ub) >= n:
                ax.fill_between(
                    x,
                    np.asarray(lb)[:n],
                    np.asarray(ub)[:n],
                    color=band,
                    alpha=0.7,
                    label="target band",
                    zorder=1,
                )
            if mean is not None and len(mean) >= n:
                ax.plot(
                    x,
                    np.asarray(mean)[:n],
                    "--o",
                    color=c_target,
                    lw=1.5,
                    ms=5,
                    label="target mean",
                    zorder=2,
                )
            yv = np.asarray(y, dtype=float)[:n]
            ax.plot(x, yv, "-s", color=c_model, lw=2, ms=6, label="model", zorder=3)
            ax.set_xlabel("injected current (pA)")
            ax.set_ylabel(ylabel)
            ax.set_title(sub)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.2)

        # --- f-I -----------------------------------------------------------
        _band_panel(
            axes[0, 0],
            m["rates"],
            self.fI_lb,
            self.fI_ub,
            "firing rate (Hz)",
            "f–I",
            mean=self.fI_mean,
        )
        # --- spike amplitude ----------------------------------------------
        _band_panel(
            axes[0, 1],
            m["spk_amps"],
            self.spk_amp_lb,
            self.spk_amp_ub,
            "AP amplitude (mV)",
            "spike amplitude",
        )
        # --- ISI adaptation ------------------------------------------------
        ratio = np.asarray(m["ISI"]["ratio"], dtype=float)
        _band_panel(
            axes[1, 0],
            ratio,
            self.spk_adapt_lb,
            self.spk_adapt_ub,
            "last/first ISI ratio",
            "ISI adaptation",
        )

        # --- passive (Rin, tau) normalized to their target bands ----------
        ax = axes[1, 1]
        feats = [
            ("Rin", m["rn"], self.target_rn, "MΩ"),
            ("τ", m["tau"], self.target_tau, "ms"),
        ]
        for i, (name, val, (lb, ub), unit) in enumerate(feats):
            yy = len(feats) - 1 - i
            ax.hlines(yy, 0, 1, color=band, lw=12, alpha=0.9, zorder=1)
            span = (ub - lb) or 1.0
            norm = (val - lb) / span
            inb = lb <= val <= ub
            # keep out-of-band markers on-axis (they sit outside the 0/1 band lines)
            ax.plot(
                np.clip(norm, -0.55, 1.9),
                yy,
                "o",
                color=ok if inb else bad,
                ms=13,
                zorder=3,
            )
            ax.text(-0.15, yy, name, ha="right", va="center", fontsize=11)
            ax.text(
                1.15,
                yy,
                f"{val:.1f} {unit}\ntarget {lb:.0f}–{ub:.0f}",
                ha="left",
                va="center",
                fontsize=8,
            )
        ax.axvline(0, ls=":", c="gray", lw=1)
        ax.axvline(1, ls=":", c="gray", lw=1)
        ax.set_xlim(-0.7, 2.1)
        ax.set_ylim(-0.6, len(feats) - 0.4)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["lower", "upper"])
        ax.set_title("passive (normalized to target band)")

        fig.suptitle(title or self.template_path.rsplit(".", 1)[-1], fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        return save_path

    def _extra_steps(self, env, amps_pa):
        """Simulate additional current steps (pA) beyond the f-I target sweeps,
        held the same way (fI_hold). Returns (traces, rates). Reuses the env state
        left by _measure (params applied, cell pinned at fI_hold)."""
        gid, cell = self._cell(env)
        results = []
        for pa in amps_pa:
            ti, vi = self._run_step(
                env,
                gid,
                float(pa) / 1e3,
                self.fI_t[0],
                self.fI_t[1],
                self.fI_t[1] + 100.0,
            )
            results.append({"t": ti, "v": vi})
        _pre, spk_cnt, _si, _th, _sa = ephys.measure_spike_features(
            results, self.fI_t[0], self.fI_t[1] + 2.0
        )
        fI = ephys.measure_fI(
            spk_cnt[:, 0],
            self.fI_t[0],
            self.fI_t[1],
            np.asarray(amps_pa, dtype=float) / 1e3,
        )
        return results, list(np.asarray(fI["frequency"], dtype=float))

    def plot_traces(self, env, save_path: str, title: str = "", extra_pa=None) -> str:
        """Re-simulate the currently-stashed params and plot the raw membrane
        voltage: the passive (Rin) step on top, then each f-I current step stacked
        below (colored by current, annotated with the resulting rate). ``extra_pa``
        is an optional list of additional step amplitudes (pA) to probe beyond the
        f-I target range (e.g. [500, 1000]). Returns the path written to."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = self._measure(env)
        traces = list(m["traces"])
        amps = list(self.fI_amps * 1e3)  # nA -> pA
        rates = list(m["rates"])
        n_fI = len(traces)

        extra_pa = [float(a) for a in (extra_pa or [])]
        if extra_pa:
            ex_traces, ex_rates = self._extra_steps(env, extra_pa)
            traces += ex_traces
            amps += extra_pa
            rates += ex_rates

        pw = m["passive_trace"]
        n = len(traces)
        cmap = plt.get_cmap("viridis")
        c_pass = "#1f4e8c"
        step_col = "#d1622b"

        fig, axes = plt.subplots(
            n + 1, 1, figsize=(9, 1.35 * (n + 1) + 0.6), squeeze=False
        )
        axes = axes[:, 0]

        # --- passive Rin step ---------------------------------------------
        ax = axes[0]
        ax.plot(pw["t"], pw["v"], color=c_pass, lw=1.2)
        ax.axvspan(self.rin_t[0], self.rin_t[1], color=step_col, alpha=0.10, lw=0)
        ax.set_ylabel("mV", fontsize=8)
        ax.set_title(
            f"passive step ({self.rin_amp * 1e3:.0f} pA):  "
            f"Rin={m['rn']:.0f} MΩ,  τ={m['tau']:.1f} ms",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

        # --- current steps, shared voltage scale for comparability --------
        vmin = min(float(np.min(tr["v"])) for tr in traces)
        vmax = max(float(np.max(tr["v"])) for tr in traces)
        pad = 0.05 * (vmax - vmin or 1.0)
        for i, tr in enumerate(traces):
            ax = axes[i + 1]
            color = cmap(i / max(n - 1, 1))
            ax.plot(tr["t"], tr["v"], color=color, lw=0.9)
            ax.axvspan(self.fI_t[0], self.fI_t[1], color=step_col, alpha=0.08, lw=0)
            ax.set_ylim(vmin - pad, vmax + pad)
            is_extra = i >= n_fI
            ax.set_ylabel(
                f"{amps[i]:.0f} pA" + ("*" if is_extra else ""),
                fontsize=8,
                color=step_col if is_extra else "black",
            )
            ax.tick_params(labelsize=7)
            ax.text(
                0.99,
                0.92,
                f"{rates[i]:.0f} Hz",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="0.25",
            )
            if i < n - 1:
                ax.set_xticklabels([])
        axes[-1].set_xlabel("time (ms)", fontsize=9)

        suptitle = (
            title or self.template_path.rsplit(".", 1)[-1]
        ) + " — voltage traces"
        if extra_pa:
            suptitle += "   (* beyond f-I target range)"
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        return save_path


def _finite(x, big: float = 1.0e6) -> float:
    """Map NaN/inf objective values to a large finite penalty for the optimizer."""
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return big
    return x
