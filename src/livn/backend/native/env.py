from __future__ import annotations

import contextlib
import inspect
import logging
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Self

import numpy as np

from livn.backend.native import _lib as L
from livn.backend.native.cells import TEMPLATES, NativeCell, build_cell
from livn.backend.native.synapses import SynapseBuilder
from livn.cells import CellRegistry
from livn.run import Run
from livn.stimulus import Stimulus, check_bounds, chunk_bytes
from livn.types import Capability
from livn.types import Env as EnvProtocol
from livn.utils import NOISE_STREAM_STRIDE

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LIVN_NATIVE_LOGGING", "WARNING"))

DEFAULT_DT = 0.025
NOISE_H = 0.025  # Gfluct3's update interval parameter

SUPPORTED_MODES = ("extracellular", "current", "current_density", "irradiance")

_OPSIN_PARAMS = ("g0", "E", "v0", "k_a", "k_r", "p", "q", "Gd", "Gr0", "phi_m")
_OPSIN_DEFAULTS = {
    "g0": 1.0,
    "E": 0.0,
    "v0": 43.0,
    "k_a": 0.28,
    "k_r": 0.28,
    "p": 0.4,
    "q": 0.4,
    "Gd": 0.0909,
    "Gr0": 0.0002,
    "phi_m": 1e16,
}


class SynapseSite:
    """Attribute access to one point process's parameters and states."""

    __slots__ = ("_env", "_row")

    def __init__(self, env: Env, row: int):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_row", int(row))

    @property
    def row(self) -> int:
        return self._row

    def __getattr__(self, name):
        env = object.__getattribute__(self, "_env")
        row = object.__getattribute__(self, "_row")
        if name in L.SP:
            return float(env._sp[L.SP[name], row])
        if name in L.SS:
            return float(env._ss[L.SS[name], row])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        env = object.__getattribute__(self, "_env")
        row = object.__getattribute__(self, "_row")
        if name in L.SP:
            env._sp[L.SP[name], row] = float(value)
            if name in ("tau_rise", "tau_decay", "U", "tau_rec"):
                L.check(env._lib.rcsd_synapse_refresh(env._sim, row), env._lib)
            return
        if name in L.SS:
            env._ss[L.SS[name], row] = float(value)
            return
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"SynapseSite({self._row})"


class ConnectionHandle:
    """A NetCon's view: ``weight[slot]`` reads and writes the library's slots."""

    __slots__ = ("_env", "_row")

    def __init__(self, env: Env, row: int):
        self._env = env
        self._row = int(row)

    @property
    def weight(self) -> np.ndarray:
        return self._env._w[self._row]

    @property
    def delay(self) -> float:
        return float(self._env.conn.delay[self._row])

    def __repr__(self) -> str:
        return f"ConnectionHandle({self._row})"


class _NoiseSite:
    """What ``neuron_noise_configure`` writes to, standing in for a Gfluct3."""

    def __init__(self, section_name: str):
        self._section_name = section_name
        self.h = NOISE_H
        self.on = 0
        self.std_e = 0.0
        self.std_i = 0.0
        self.g_e0 = 0.0
        self.g_i0 = 0.0
        self.tau_e = 2.728
        self.tau_i = 10.49
        self.E_e = 0.0
        self.E_i = -75.0
        self.D_e = self.D_i = 0.0
        self.exp_e = self.exp_i = 0.0
        self.amp_e = self.amp_i = 0.0

    def get_segment(self):
        return _Segment(self._section_name)


class _Segment:
    def __init__(self, name):
        self.sec = _SectionName(name)


class _SectionName:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _ModeState:
    """One stimulus mode's rows, held blocks and deferred streams."""

    def __init__(self, mode: int, input_mode: str):
        self.mode = mode
        self.input_mode = input_mode
        self.rows: dict[tuple[int, int], int] = {}
        self.row_cells: list[int] = []
        self.row_sections: list[int] = []
        self.dt: float | None = None
        self.block: np.ndarray | None = None
        self.streams: list[dict] = []
        self.bounds: tuple[float, float] | None = None
        self.scale = 1.0
        self.units: str | None = None
        self.dirty = True
        self.window: tuple[int, int] | None = None

    def clear(self) -> None:
        self.rows = {}
        self.row_cells = []
        self.row_sections = []
        self.dt = None
        self.block = None
        self.streams = []
        self.bounds = None
        self.scale = 1.0
        self.units = None
        self.dirty = True
        self.window = None

    def row(self, cell: int, section: int, gid: int) -> int:
        key = (int(gid), int(section))
        found = self.rows.get(key)
        if found is None:
            found = len(self.rows)
            self.rows[key] = found
            self.row_cells.append(int(cell))
            self.row_sections.append(int(section))
            self.dirty = True
        return found

    def extent(self) -> int:
        end = 0 if self.block is None else int(self.block.shape[1])
        for stream in self.streams:
            end = max(end, int(stream["start_step"] + stream["n_steps"]))
        return end

    def chunk_steps(self) -> int:
        itemsize = np.dtype(np.float64).itemsize
        width = max(1, len(self.rows))
        steps = max(1, int(chunk_bytes() // (width * itemsize)))
        for stream in self.streams:
            steps = min(steps, int(stream["chunk_steps"]))
        return max(1, steps)


class Env(EnvProtocol):
    capabilities = frozenset(
        {
            Capability.SIMULATION,
            Capability.NOISE,
            Capability.REPLAYABLE_NOISE,
            Capability.PLASTICITY,
            Capability.PER_GID_VOLTAGE,
            Capability.EXTRACELLULAR_STIMULUS,
        }
    )

    def __init__(
        self,
        system: System | str | int,
        model: Model | None = None,
        io: IO | None = None,
        seed: int | None = 123,
        comm: MPI.Intracomm | None = None,
        subworld_size: int | None = None,
    ):
        from livn.system import resolve

        self._lib = L.load()
        self._sim = None
        self.seed = seed
        self.comm = comm
        self.subworld_size = subworld_size
        self.system = resolve(system, comm=comm)
        self.model = (
            model if model is not None else self.system.default_model(comm=comm)
        )
        self.io = io if io is not None else self.system.default_io(comm=comm)

        self.encoding = None
        self.decoding = None
        self.duration = None

        self._select_spec = None
        self._select_method = "first"
        self._select_bounds = None
        self._selection: dict[str, object] | None = None
        self._selected_gids: set[int] | None = None

        self.cells = CellRegistry(self, comm=comm)
        self._cell_index: dict[int, int] = {}  # gid -> library cell index
        self._index_gid: list[int] = []
        self._thresholds: list[float] = []
        self._pop_code: dict[str, int] = {}
        self._mech_code: dict[str, int] = {}
        self._sectype_code: dict[str, int] = {}
        self._mech_id_to_name: dict[int, str] = {}
        self._wplastic_slot: dict[str, int] = {}
        self._stdp_syn_rows = np.empty(0, dtype=np.int64)
        self._stdp_conn_rows = np.empty(0, dtype=np.int64)
        self.syn = None
        self.conn = None
        self._sp = np.zeros((L.SP_N, 0))
        self._ss = np.zeros((L.SS_N, 0))
        self._w = np.zeros((0, L.NWEIGHT))
        self._input_indices: dict[int, int] = {}

        self._plasticity_enabled = False
        self._flucts: dict[str, tuple[int, int, _NoiseSite]] = {}
        self._noise_state: dict = {}
        self._noise_stream = 0
        self.w_recs: dict[tuple, list] = {}
        self._weight_rows: dict[tuple, tuple[int, int]] = {}
        self._weight_rec_dt = 0.1
        self._weight_recording_active = False
        self._w_rec_times: list | None = None

        self._spike_gids: set[int] = set()
        self.v_recs: dict[tuple[int, int], int] = {}
        self.v_sections: dict[tuple[int, int], str] = {}
        self.v_dt: dict[str, float] = {}
        self.i_recs: dict[tuple[int, int], int] = {}
        self.i_dt: dict[str, float] = {}

        self.t = 0.0
        self._v_init = -75.0
        self._dt: float | None = None
        self._closed = False

        self._refractory_period = (
            float(self.model.neuron_refractory_period())
            if hasattr(self.model, "neuron_refractory_period")
            else 2.0
        )
        self._celsius = (
            float(self.model.neuron_celsius())
            if hasattr(self.model, "neuron_celsius")
            else 36.0
        )

        self._stim: dict[str, _ModeState] = {
            "extracellular": _ModeState(L.STIM_EXTRACELLULAR, "extracellular"),
            "current": _ModeState(L.STIM_CURRENT, "current"),
            "current_density": _ModeState(L.STIM_CURRENT_DENSITY, "current_density"),
            "irradiance": _ModeState(L.STIM_PHOTON_FLUX, "irradiance"),
        }
        self._stim_unbacked_warned = False
        self._opsin_refs: dict[tuple[int, int], int] = {}

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # -- selection ----------------------------------------------------------

    def selection(self, select, method: str = "first", bounds=None) -> Self:
        if self.cells:
            raise RuntimeError("selection() must be called before init()")
        self._select_spec = select
        self._select_method = method
        self._select_bounds = bounds
        return self

    @property
    def selection_name(self) -> str | None:
        spec = self._select_spec
        return spec if isinstance(spec, str) else None

    def _resolve_selection(self, buildable: list[str]) -> None:
        self._selection = self.system.selection(
            self._select_spec,
            populations=buildable,
            seed=self.seed,
            method=self._select_method,
            bounds=self._select_bounds,
        )
        if self._selection is None:
            self._selected_gids = None
        else:
            self._selected_gids = {
                int(g) for gids in self._selection.values() for g in gids
            }

    # -- construction -------------------------------------------------------

    def _cell_types(self) -> dict[str, tuple[str, str]]:
        if hasattr(self.model, "cell_types"):
            return dict(self.model.cell_types())
        raise TypeError(
            f"{type(self.model).__name__} does not say which cell each population "
            "is (no `cell_types()`), so the native backend cannot build it"
        )

    def init(self) -> Self:
        if self._sim is not None:
            self.close()
            self._closed = False
        self._sim = self._lib.rcsd_create(self._celsius, self._v_init)
        if not self._sim:
            raise L.NativeError(self._lib.rcsd_last_error().decode())

        ignored = (
            set(self.model.ignored_populations())
            if hasattr(self.model, "ignored_populations")
            else set()
        )
        types = self._cell_types()
        buildable = [
            p for p in self.system.populations if p not in ignored and p in types
        ]
        self._resolve_selection(buildable)
        if self._selection is None:
            simulated_pops = set(buildable)
        else:
            simulated_pops = {
                p for p in buildable if len(self._selection.get(p, [])) > 0
            }

        cells_by_pop: dict[str, dict[int, NativeCell]] = {}
        for pop in buildable:
            if pop not in simulated_pops:
                continue
            sel = None
            if self._selection is not None:
                sel = {int(g) for g in self._selection.get(pop, [])}
            cells = self._build_population(pop, types[pop], sel)
            cells_by_pop[pop] = cells
            self.cells.add(pop, cells)

        builder = SynapseBuilder(
            self,
            selected_gids=self._selected_gids,
            simulated_pops=simulated_pops,
        )
        (
            self.syn,
            self.conn,
            self._pop_code,
            self._mech_code,
            self._sectype_code,
            self._input_indices,
        ) = builder.build(cells_by_pop)
        self._refresh_views()
        self._index_plastic_synapses()
        self._insert_opsins()
        return self

    def _build_population(self, population: str, cell_type, selection) -> dict:
        from livn.models.rcsd import hold_potential

        template_name, params_name = cell_type
        template_cls = TEMPLATES.get(template_name)
        if template_cls is None:
            raise ValueError(
                f"the native backend has no {template_name!r} template "
                f"(for population {population!r}); it has {sorted(TEMPLATES)}"
            )
        base = self.model.params(params_name)
        coords = self.system.coordinate_array(population)
        available = sorted(int(r[0]) for r in coords) if len(coords) else []
        if selection is not None:
            gids = sorted(int(g) for g in selection)
        elif available:
            gids = available
        else:
            raise RuntimeError(
                f"population {population!r} has no coordinates; a selection() is "
                "required to build it"
            )
        pop_code = len(self._pop_code_for_cells)
        self._pop_code_for_cells.setdefault(population, pop_code)
        soma_type = self.model.section_name(population, "soma")
        dend_type = self.model.section_name(population, "dend")
        tref = self._refractory_period
        cells: dict[int, NativeCell] = {}
        for gid in gids:
            params = self._scaled(base, population, gid)
            template = template_cls(params)
            threshold = float(params["V_threshold"])
            v_hold = hold_potential(params)
            index, sections = build_cell(
                self._lib, self._sim, gid, pop_code, template, threshold, v_hold, tref
            )
            self._cell_index[gid] = index
            self._index_gid.append(gid)
            self._thresholds.append(threshold)
            cells[gid] = NativeCell(
                self, population, gid, index, template, sections, soma_type, dend_type
            )
        return cells

    @property
    def _pop_code_for_cells(self) -> dict[str, int]:
        found = self.__dict__.get("_pop_code_cells")
        if found is None:
            found = self.__dict__["_pop_code_cells"] = {}
        return found

    def _scaled(self, params: dict, population: str, gid: int) -> dict:
        """The parameters with the cell's size factor folded into its diameters."""
        size_cv = getattr(self.model, "size_cv_for", None)
        if size_cv is None or size_cv(population) <= 0.0:
            return dict(params)
        scale = float(self.model.size_scales(population, [gid])[0])
        out = dict(params)
        for key in ("global_diam", "axon_diam"):
            if params.get(key) is not None:
                out[key] = float(params[key]) * scale
        return out

    def _refresh_views(self) -> None:
        lib, sim = self._lib, self._sim
        n_conn = lib.rcsd_connection_count(sim)
        self._sp = L.site_view(lib, sim, lib.rcsd_synapse_params(sim), L.SP_N)
        self._ss = L.site_view(lib, sim, lib.rcsd_synapse_states(sim), L.SS_N)
        self._w = L.view(lib.rcsd_connection_weights(sim), (n_conn, L.NWEIGHT))

    def _index_plastic_synapses(self) -> None:
        rules = self.model.neuron_synapse_rules()
        self._mech_id_to_name = {v: k for k, v in self._mech_code.items()}
        self._wplastic_slot = {}
        plastic_ids: set[int] = set()
        for name, mid in self._mech_code.items():
            netcon_params = rules.get(name, {}).get("netcon_params", {})
            if "w_plastic" in netcon_params:
                self._wplastic_slot[name] = int(netcon_params["w_plastic"])
                plastic_ids.add(mid)
        if self.syn is not None and self.syn.size:
            self._stdp_syn_rows = np.flatnonzero(
                np.isin(self.syn.mech_id, list(plastic_ids))
            )
        else:
            self._stdp_syn_rows = np.empty(0, dtype=np.int64)
        if self.conn is not None and self.conn.size:
            self._stdp_conn_rows = np.flatnonzero(
                np.isin(self.conn.mech_id, list(plastic_ids))
            )
        else:
            self._stdp_conn_rows = np.empty(0, dtype=np.int64)

    def _opsin_config(self) -> dict:
        for name in ("opsin_config", "native_opsin_config", "neuron_opsin_config"):
            hook = getattr(self.model, name, None)
            if callable(hook):
                return hook() or {}
        return {}

    def _insert_opsins(self) -> None:
        cfg = self._opsin_config()
        if not cfg:
            return
        mech_name = cfg.get("mechanism", "RhO3c")
        if mech_name != "RhO3c":
            logger.warning(
                "opsin mechanism %s not available in the native backend", mech_name
            )
            return
        populations = cfg.get("populations", list(self.cells.keys()))
        target_sections = set(cfg.get("sections", ["soma"]))
        params = {**_OPSIN_DEFAULTS, **(cfg.get("params", {}) or {})}
        lib, sim = self._lib, self._sim
        for pop in populations:
            for gid, cell in self.cells.get(pop, {}).items():
                for sec_id, name in enumerate(cell.section_names):
                    if name not in target_sections:
                        continue
                    opsin = L.check(
                        lib.rcsd_add_opsin(sim, cell.index, sec_id, 0.5), lib
                    )
                    L.check(
                        lib.rcsd_opsin_set(
                            sim, opsin, *[float(params[k]) for k in _OPSIN_PARAMS]
                        ),
                        lib,
                    )
                    self._opsin_refs[(int(gid), sec_id)] = opsin

    def _find_cell(self, gid: int):
        for cells in self.cells.values():
            if gid in cells:
                return cells[gid]
        return None

    # -- recording ------------------------------------------------------------

    def _record_spikes(self, population: str) -> Self:
        for gid, cell in self.cells.get(population, {}).items():
            gid = int(gid)
            if gid in self._spike_gids:
                continue
            L.check(self._lib.rcsd_record_spikes(self._sim, cell.index), self._lib)
            self._spike_gids.add(gid)
        return self

    def _record_voltage(
        self, population: str, dt: float, gids=None, sections=None
    ) -> Self:
        previous = self.v_dt.get(population)
        if previous is not None and abs(previous - dt) > 1e-12 and self.v_recs:
            raise ValueError(
                f"{population} voltage is already being recorded at dt="
                f"{previous} ms, so asking for {dt} ms would leave two "
                "resolutions in one channel; call clear() first"
            )
        self.v_dt[population] = dt
        wanted = None if sections is None else {str(s) for s in sections}
        for gid, cell in self.cells.get(population, {}).items():
            if gids is not None and int(gid) not in gids:
                continue
            for sec_id, name in enumerate(cell.section_names):
                if wanted is not None and name not in wanted:
                    continue
                if (int(gid), sec_id) in self.v_recs:
                    continue
                trace = L.check(
                    self._lib.rcsd_record_voltage(
                        self._sim, cell.index, sec_id, float(dt)
                    ),
                    self._lib,
                )
                self.v_recs[(int(gid), sec_id)] = trace
                self.v_sections[(int(gid), sec_id)] = name
        return self

    def _record_membrane_current(self, population: str, dt: float) -> Self:
        cells = self.cells.get(population, {})
        if not cells:
            return self
        spn = self.recording_sections_per_cell(population)
        if not spn:
            return self
        self.i_dt[population] = dt
        for gid, cell in cells.items():
            for sec_id in range(min(spn, len(cell.sections))):
                if (int(gid), sec_id) in self.i_recs:
                    continue
                trace = L.check(
                    self._lib.rcsd_record_current(
                        self._sim, cell.index, sec_id, float(dt)
                    ),
                    self._lib,
                )
                self.i_recs[(int(gid), sec_id)] = trace
        return self

    def clear_recordings(self) -> Self:
        if self._sim:
            L.check(self._lib.rcsd_clear_recordings(self._sim), self._lib)
        return self

    @staticmethod
    def _recorded_dt(dts: dict[str, float], default: float) -> float:
        if not dts:
            return default
        vals = set(dts.values())
        return next(iter(vals)) if len(vals) == 1 else min(vals)

    @property
    def voltage_recording_dt(self) -> float:
        return self._recorded_dt(self.v_dt, 0.1)

    @property
    def membrane_current_recording_dt(self) -> float:
        return self._recorded_dt(self.i_dt, 0.1)

    @property
    def v_init(self) -> float:
        return self._v_init

    @v_init.setter
    def v_init(self, value: float) -> None:
        if self.t != 0.0:
            raise RuntimeError(
                "v_init only takes effect at initialization and cannot be changed "
                "after the simulation has started; call clear() to reset first."
            )
        self._v_init = float(value)

    # -- running ----------------------------------------------------------------

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        if self._sim is None:
            raise RuntimeError("call init() before run()")
        self.duration = duration
        current_time = self.t
        first_run = self.t == 0 or not self._lib.rcsd_initialized(self._sim)
        requested_dt = (
            dt if dt is not None else (self._dt if self._dt is not None else DEFAULT_DT)
        )
        if (
            not first_run
            and self._dt is not None
            and abs(requested_dt - self._dt) > 1e-12
        ):
            raise ValueError("Cannot change dt mid-simulation; call clear() first.")
        if first_run:
            self._dt = requested_dt
            L.check(self._lib.rcsd_set_dt(self._sim, requested_dt), self._lib)
            L.check(self._lib.rcsd_set_v_init(self._sim, self._v_init), self._lib)

        self._prune_streams(current_time)
        if stimulus is not None:
            stimulus = Stimulus.from_arg(stimulus, env=self, duration=duration)
            stimulus = self.model.prepare_stimulus(stimulus)
            if stimulus.input_mode not in SUPPORTED_MODES:
                raise ValueError(
                    f"the native backend has no mechanism for a "
                    f"{stimulus.input_mode!r} stimulus; it delivers "
                    f"{', '.join(SUPPORTED_MODES)}"
                )
            self._install_stimulus(stimulus, current_time)

        self.clear_recordings()
        if first_run:
            L.check(self._lib.rcsd_init(self._sim), self._lib)
            self._refresh_views()

        n_steps = round(duration / self._dt)
        start_step = int(self._lib.rcsd_step(self._sim))
        end_step = start_step + n_steps
        for state in self._stim.values():
            self._feed(state, start_step, end_step, refill=False)

        if self._weight_recording_active and self._weight_rows:
            w_steps = max(1, round(self._weight_rec_dt / self._dt))
            done_total = 0
            while done_total < n_steps:
                chunk = min(w_steps, n_steps - done_total)
                self._advance(chunk, end_step)
                done_total += chunk
                self._sample_weights()
        else:
            self._advance(n_steps, end_step)

        self.t = current_time + duration
        ii, tt, iv, v, sv, im, mp = self._collect(current_time)
        self.duration = None
        return (
            Run(t0=current_time, duration=duration)
            .add_spikes(ii, tt)
            .add_voltage(iv, v, dt=self.voltage_recording_dt, sections=sv)
            .add_current(im, mp, dt=self.membrane_current_recording_dt)
        )

    def _advance(self, n_steps: int, end_step: int) -> None:
        lib, sim = self._lib, self._sim
        remaining = int(n_steps)
        done = L.c_long()
        while remaining > 0:
            code = L.check(lib.rcsd_run(sim, remaining, done), lib)
            remaining -= int(done.value)
            if code == L.NEED_STIMULUS:
                fed = False
                for state in self._stim.values():
                    needed = int(lib.rcsd_stimulus_needed(sim, state.mode))
                    if needed >= 0:
                        self._feed(
                            state, int(lib.rcsd_step(sim)), end_step, refill=True
                        )
                        fed = True
                if not fed:
                    raise RuntimeError(
                        "librcsd asked for a stimulus window it does not need"
                    )
            elif remaining > 0 and int(done.value) == 0:
                raise RuntimeError("librcsd made no progress")

    def _collect(self, current_time: float):
        lib, sim = self._lib, self._sim
        n = lib.rcsd_spike_count(sim)
        cells = L.copy_ints(lib.rcsd_spike_cells(sim), n)
        tt = L.copy_doubles(lib.rcsd_spike_times(sim), n)
        gids = np.asarray(self._index_gid, dtype=np.int64)
        ii = gids[cells].astype(np.uint32) if n else np.zeros(0, dtype=np.uint32)
        if current_time != 0.0 and tt.size > 0:
            tt = tt - current_time
            tt[tt < 0.0] = 0.0

        if self.v_recs:
            iv = np.asarray([gid for (gid, _sec) in self.v_recs], dtype=np.uint32)
            sv = np.asarray([self.v_sections[key] for key in self.v_recs])
            traces = []
            for trace in self.v_recs.values():
                m = lib.rcsd_voltage_record_length(sim, trace)
                traces.append(L.copy_doubles(lib.rcsd_voltage_record(sim, trace), m))
            width = min(len(tr) for tr in traces) if traces else 0
            v = np.array([tr[:width] for tr in traces], dtype=np.float32)
        else:
            iv = v = sv = None

        im = mp = None
        lengths = [
            lib.rcsd_current_record_length(sim, trace) for trace in self.i_recs.values()
        ]
        T = max(lengths, default=0)
        if T:
            im = np.asarray(self.recording_coordinates(simulated_only=True))[
                :, 0
            ].astype(np.int32)
            mp = np.zeros((len(im), T), dtype=np.float32)
            section_of: dict[int, int] = {}
            for row, gid in enumerate(im):
                sec_id = section_of.get(int(gid), 0)
                section_of[int(gid)] = sec_id + 1
                trace = self.i_recs.get((int(gid), sec_id))
                if trace is None:
                    continue
                m = lib.rcsd_current_record_length(sim, trace)
                arr = L.copy_doubles(lib.rcsd_current_record(sim, trace), m) * 1e-3
                k = min(arr.shape[0], T)
                mp[row, :k] = arr[:k]
        return ii, tt, iv, v, sv, im, mp

    def apply_init_ic(self) -> None:
        """Pin each cell's resting current (the library does it at init as well)."""
        if self._sim:
            L.check(self._lib.rcsd_pin_resting(self._sim), self._lib)

    # -- stimulus --------------------------------------------------------------

    def _install_stimulus(self, stimulus: Stimulus, current_time: float) -> None:
        state = self._stim[stimulus.input_mode]
        if stimulus.input_mode == "irradiance":
            wavelength_nm = stimulus.extra.get("wavelength_nm", None)
            if wavelength_nm is None:
                wavelength_nm = self._opsin_config().get("wavelength_nm", 473.0)
            if stimulus.units == "photon_flux":
                scale = 1.0
            else:
                scale = 1.0 / (6.626e-34 * 3e8 / (wavelength_nm * 1e-9) * 1e3)
            installed = state.streams or state.block is not None
            if installed and abs(scale - state.scale) > 1e-9 * max(scale, state.scale):
                raise ValueError(
                    "an irradiance stimulus with a different wavelength is "
                    "already installed; call clear() first"
                )
            state.scale = scale
            if not self._opsin_refs:
                raise ValueError(
                    "Stimulus has input_mode='irradiance' but no opsins are attached; "
                    "configure the model's opsin_config()."
                )

        n_neurons = len(self.active_neuron_coordinates())
        if stimulus.gids is None:
            if stimulus.input_mode == "extracellular":
                if len(stimulus) % max(1, n_neurons):
                    raise ValueError(
                        f"stimulus has {len(stimulus)} channels, which is not a "
                        f"whole number of sections per neuron over {n_neurons} "
                        "neurons, so its gids cannot be inferred. Please pass "
                        "`gids` naming the cell each channel belongs to"
                    )
                stimulus.gids = np.repeat(
                    self.active_gids(), max(1, len(stimulus) // max(1, n_neurons))
                )
            elif len(stimulus) == n_neurons:
                stimulus.gids = self.active_gids()
            elif len(stimulus) % max(1, n_neurons) == 0:
                stimulus.gids = np.repeat(
                    self.active_gids(), len(stimulus) // max(1, n_neurons)
                )
            else:
                raise ValueError(
                    f"stimulus has {len(stimulus)} channels over {n_neurons} neurons, "
                    "so its gids cannot be inferred. Please pass `gids`"
                )

        if state.dt is None:
            state.dt = float(stimulus.dt)
        elif not math.isclose(state.dt, stimulus.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")
        state.units = stimulus.units
        declares = getattr(self.model, "stimulus_bounds", None)
        state.bounds = declares(stimulus.input_mode) if declares else None

        start_step = round(current_time / stimulus.dt)
        rows, columns = [], []
        for column, (gid, section_id) in enumerate(stimulus.columns()):
            cell = self._find_cell(int(gid))
            if cell is None or section_id >= len(cell.sections):
                continue
            if stimulus.input_mode == "irradiance" and (
                (int(gid), int(section_id)) not in self._opsin_refs
            ):
                continue
            rows.append(state.row(cell.index, int(section_id), int(gid)))
            columns.append(column)
        if not rows:
            return
        if stimulus.deferred:
            self._install_stream(state, stimulus, rows, columns, start_step)
        else:
            self._install_block(state, stimulus, rows, columns, start_step)

    def _install_block(self, state, stimulus, rows, columns, start_step) -> None:
        values = np.asarray(stimulus.array)
        check_bounds(values, state.bounds, stimulus.input_mode, stimulus.units)
        dtype = np.promote_types(values.dtype, np.float32)
        end_step = start_step + values.shape[0]
        width = end_step
        previous = state.block
        if previous is not None:
            width = max(width, previous.shape[1])
            dtype = np.promote_types(dtype, previous.dtype)
        block = np.zeros((len(state.rows), width), dtype=dtype)
        if previous is not None:
            block[: previous.shape[0], : previous.shape[1]] = previous
        for row, column in zip(rows, columns, strict=False):
            block[row, start_step:end_step] = values[:, column]
        state.block = block
        state.window = None

    def _install_stream(self, state, stimulus, rows, columns, start_step) -> None:
        itemsize = np.dtype(np.float64).itemsize
        chunk_steps = max(1, int(chunk_bytes() // max(1, stimulus.width * itemsize)))
        state.streams.append(
            {
                "stimulus": stimulus,
                "rows": np.asarray(rows, dtype=np.int64),
                "columns": np.asarray(columns, dtype=np.int64),
                "start_step": int(start_step),
                "n_steps": round(stimulus.duration / stimulus.dt),
                "chunk_steps": chunk_steps,
                "chunk": None,
                "chunk_start": 0,
                "chunk_stop": 0,
            }
        )
        state.window = None

    def _prune_streams(self, current_time: float) -> None:
        for state in self._stim.values():
            if not state.streams or not state.dt:
                continue
            idx = int(current_time / state.dt)
            kept = [
                stream
                for stream in state.streams
                if idx < stream["start_step"] + stream["n_steps"]
            ]
            if len(kept) != len(state.streams):
                state.streams = kept
                state.window = None

    def _render(self, state: _ModeState, first: int, n: int) -> np.ndarray:
        """Samples ``[first, first + n)`` of a mode, laid out ``[n, rows]``."""
        field = np.zeros((n, len(state.rows)), dtype=np.float64)
        block = state.block
        if block is not None:
            lo = max(first, 0)
            hi = min(first + n, block.shape[1])
            if hi > lo:
                field[lo - first : hi - first, : block.shape[0]] = block[:, lo:hi].T
        for stream in state.streams:
            stimulus = stream["stimulus"]
            lo = max(first, stream["start_step"])
            hi = min(first + n, stream["start_step"] + stream["n_steps"])
            if hi <= lo:
                continue
            rel_lo = lo - stream["start_step"]
            rel_hi = hi - stream["start_step"]
            values = np.asarray(
                stimulus.window(rel_lo * stimulus.dt, rel_hi * stimulus.dt)
            )
            if values.shape[0] == 0:
                continue
            check_bounds(values, state.bounds, stimulus.input_mode, stimulus.units)
            k = min(values.shape[0], hi - lo)
            at = lo - first
            field[at : at + k, stream["rows"]] += values[:k][:, stream["columns"]]
        if state.scale != 1.0:
            field *= state.scale
        return field

    def _feed(
        self, state: _ModeState, start_step: int, end_step: int, refill: bool
    ) -> None:
        """Give the library the window of a mode it needs from ``start_step`` on."""
        lib, sim = self._lib, self._sim
        if not state.rows:
            return
        extent = state.extent()
        if extent == 0:
            if state.dirty or state.window is not None:
                L.check(lib.rcsd_clear_stimulus(sim, state.mode), lib)
                state.window = None
                state.dirty = True
            return
        if state.dirty:
            cells = L.as_int_array(state.row_cells)
            sections = L.as_int_array(state.row_sections)
            L.check(
                lib.rcsd_set_stimulus_rows(
                    sim,
                    state.mode,
                    len(state.rows),
                    L.int_ptr(cells),
                    L.int_ptr(sections),
                    state.dt,
                ),
                lib,
            )
            state.dirty = False
            state.window = None
            if state.input_mode == "extracellular":
                self._warn_unbacked_junctions()
        # the first sample this run can need, in the stimulus's own grid
        dt = self._dt or DEFAULT_DT
        first = math.floor(start_step * dt / state.dt)
        needed = int(lib.rcsd_stimulus_needed(sim, state.mode)) if refill else -1
        if needed >= 0:
            first = needed
        last = math.ceil(end_step * dt / state.dt) + 2
        n = max(1, min(state.chunk_steps(), max(0, min(last, extent) - first) + 1))
        if not refill and state.window is not None:
            held_first, held_n = state.window
            if held_first <= first and first + n <= held_first + held_n:
                return
        values = np.ascontiguousarray(self._render(state, first, n))
        L.check(
            lib.rcsd_set_stimulus_window(
                sim, state.mode, first, n, L.double_ptr(values), extent
            ),
            lib,
        )
        state.window = (first, n)

    def _warn_unbacked_junctions(self) -> None:
        driven, dropped = L.c_int(), L.c_int()
        L.check(
            self._lib.rcsd_extracellular_junctions(self._sim, driven, dropped),
            self._lib,
        )
        if dropped.value and not self._stim_unbacked_warned:
            self._stim_unbacked_warned = True
            logger.warning(
                "%d of %d axial junctions are not driven: the stimulus does not "
                "reach both of their sections, and driving them would inject a "
                "field difference against an unsampled 0 mV. Have %s."
                "stimulus_coordinates() name every section that appears in "
                "axial_couplings() to drive them all",
                dropped.value,
                dropped.value + driven.value,
                type(self.model).__name__ if self.model is not None else "the model",
            )

    # the NEURON backend's names for the extracellular state, for tooling
    @property
    def _stim_rows(self) -> dict:
        return self._stim["extracellular"].rows

    @property
    def _stim_block(self):
        return self._stim["extracellular"].block

    @property
    def _stim_streams(self) -> list:
        return self._stim["extracellular"].streams

    @property
    def _stim_dt(self):
        return self._stim["extracellular"].dt

    # -- weights ---------------------------------------------------------------------

    @property
    def weight_names(self) -> list[str]:
        if self.conn is None:
            return []
        pop_of = {code: name for name, code in self._pop_code.items()}
        sec_of = {code: name for name, code in self._sectype_code.items()}
        mech_of = {code: name for name, code in self._mech_code.items()}
        built = {
            (pop_of[po], pop_of[pr], sec_of[ds], mech_of[mi])
            for po, pr, ds, mi in zip(
                self.conn.post_pop.tolist(),
                self.conn.pre_pop.tolist(),
                self.conn.dest_sectype.tolist(),
                self.conn.mech_id.tolist(),
                strict=False,
            )
        }
        mech_names = self.model.neuron_synapse_mechanisms()
        names = []
        for (
            post,
            pre,
            _config_section,
            syn_name,
            _,
        ) in self.system.synapse_projections():
            mech = mech_names.get(syn_name, syn_name)
            for sec_type in sorted(
                {s for (po, pr, s, m) in built if (po, pr, m) == (post, pre, mech)}
            ):
                name = f"{post}_{pre}-{sec_type}-{syn_name}-weight"
                if name not in names:
                    names.append(name)
        return names

    def admissible_params(self, cells: bool = False) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        found["weights"] = list(self.weight_names)

        sites = set()
        if self.syn is not None and self.syn.size:
            post_pop = np.full(self.syn.size, -1, dtype=np.int16)
            if self.conn is not None and self.conn.size:
                post_pop[self.conn.syn_row] = self.conn.post_pop
            pop_of = {code: name for name, code in self._pop_code.items()}
            sec_of = {code: name for name, code in self._sectype_code.items()}
            mech_of = {code: name for name, code in self._mech_code.items()}
            for pp, sec, mech in zip(
                post_pop.tolist(),
                self.syn.dest_sectype.tolist(),
                self.syn.mech_id.tolist(),
                strict=False,
            ):
                if pp < 0 or pp not in pop_of:
                    continue
                sites.add((pop_of[pp], sec_of.get(sec, ""), mech_of.get(mech, "")))

        receptor_of = {
            pp: receptor
            for receptor, pp in self.model.neuron_synapse_mechanisms().items()
        }
        rules = self.model.neuron_synapse_rules()
        mechanisms: list[str] = []
        for post, sec, pp_name in sorted(sites):
            if not sec or not pp_name:
                continue
            receptor = receptor_of.get(pp_name, pp_name)
            for param in (rules.get(pp_name) or {}).get("mech_params") or []:
                name = f"{post}-{sec}-{receptor}-{param}"
                if name not in mechanisms:
                    mechanisms.append(name)
        found["mechanisms"] = mechanisms

        noise: list[str] = []
        if hasattr(self.model, "neuron_noise_configure"):
            skip = {"self", "population", "mechanism", "state"}
            noise = [
                f"noise-{p}"
                for p in inspect.signature(self.model.neuron_noise_configure).parameters
                if p not in skip
            ]
        found["noise"] = noise
        found["cells"] = (
            [f"cells-{name}" for name in self.cells.get_params()] if cells else []
        )
        return found

    def _set_synapse_mech_params(self, params: dict) -> Self:
        from livn.types import SynapticParam

        if self.syn is None or self.syn.size == 0:
            return self
        post_pop = np.full(self.syn.size, -1, dtype=np.int16)
        if self.conn is not None and self.conn.size:
            post_pop[self.conn.syn_row] = self.conn.post_pop

        for key, value in params.items():
            try:
                p = SynapticParam.from_string(key)
            except ValueError:
                continue
            name = p.param_path
            if not isinstance(name, str):
                continue
            if p.source is not None:
                raise ValueError(
                    f"{key!r} names a source, but {name!r} is a property of the "
                    f"point process, which every source targeting that site "
                    f"shares. Drop the source: "
                    f"{p.population}-{p.sec_type}-{p.syn_name}-{name}"
                )
            mask = np.ones(self.syn.size, dtype=bool)
            if p.population is not None and p.population in self._pop_code:
                mask &= post_pop == self._pop_code[p.population]
            if p.syn_name is not None:
                mech = self.model.neuron_synapse_mechanisms().get(
                    p.syn_name, p.syn_name
                )
                if mech in self._mech_code:
                    mask &= self.syn.mech_id == self._mech_code[mech]
                else:
                    mask &= False
            if p.sec_type is not None:
                if p.sec_type in self._sectype_code:
                    mask &= self.syn.dest_sectype == self._sectype_code[p.sec_type]
                else:
                    mask &= False
            column = L.SP.get(name)
            if column is None:
                continue
            rows = np.flatnonzero(mask)
            self._sp[column, rows] = float(value)
            if name in ("tau_rise", "tau_decay", "U", "tau_rec"):
                for row in rows:
                    L.check(
                        self._lib.rcsd_synapse_refresh(self._sim, int(row)), self._lib
                    )
        return self

    def set_weights(self, weights: dict) -> Self:
        from livn.types import SynapticParam

        weights = dict(weights)
        mech = {}
        for key in list(weights):
            try:
                path = SynapticParam.from_string(key).param_path
            except ValueError:
                continue
            if isinstance(path, str) and path != "weight":
                mech[key] = weights.pop(key)
        if mech:
            self._set_synapse_mech_params(mech)

        if self.conn is None or self.conn.size == 0:
            return self
        for key, val in weights.items():
            try:
                p = SynapticParam.from_string(key)
            except ValueError:
                continue
            mask = np.ones(self.conn.size, dtype=bool)
            if p.population is not None and p.population in self._pop_code:
                mask &= self.conn.post_pop == self._pop_code[p.population]
            if p.source is not None and p.source in self._pop_code:
                mask &= self.conn.pre_pop == self._pop_code[p.source]
            if p.syn_name is not None:
                mech_name = self.model.neuron_synapse_mechanisms().get(
                    p.syn_name, p.syn_name
                )
                if mech_name in self._mech_code:
                    mask &= self.conn.mech_id == self._mech_code[mech_name]
                else:
                    mask &= False
            if p.sec_type is not None:
                if p.sec_type in self._sectype_code:
                    mask &= self.conn.dest_sectype == self._sectype_code[p.sec_type]
                else:
                    mask &= False
            idx = np.flatnonzero(mask)
            self.conn.weight[idx] = val
            self._w[idx, self.conn.wslot[idx].astype(np.int64)] = float(val)
        return self

    def _iter_stdp_point_processes(self):
        if self.syn is None:
            return
        for row in self._stdp_syn_rows:
            row = int(row)
            name = self._mech_id_to_name.get(int(self.syn.mech_id[row]))
            yield (
                int(self.syn.post_gid[row]),
                int(self.syn.syn_id[row]),
                name,
                SynapseSite(self, row),
            )

    def _iter_stdp_connections(self):
        if self.conn is None:
            return
        for row in self._stdp_conn_rows:
            row = int(row)
            syn_row = int(self.conn.syn_row[row])
            name = self._mech_id_to_name.get(int(self.conn.mech_id[row]))
            yield (
                int(self.syn.post_gid[syn_row]),
                int(self.syn.syn_id[syn_row]),
                name,
                SynapseSite(self, syn_row),
                ConnectionHandle(self, row),
            )

    def get_weights(self) -> dict:
        weights: dict[tuple, float] = {}
        for gid, syn_id, mech_name, _pp, nc in self._iter_stdp_connections():
            slot = self._wplastic_slot.get(mech_name, 2)
            weights[(int(gid), int(syn_id), mech_name)] = float(nc.weight[slot])
        return weights

    def normalize_weights(self, target: float | None = None) -> Self:
        from livn.weights import normalize_weights

        rows = []
        weight, w_min, w_max, group = [], [], [], []
        for gid, _syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            slot = self._wplastic_slot.get(mech_name, 2)
            rows.append((nc, slot))
            weight.append(float(nc.weight[slot]))
            w_min.append(float(pp.w_min))
            w_max.append(float(pp.w_max))
            group.append(int(gid))
        if not rows:
            return self
        new_w = normalize_weights(
            np.asarray(weight),
            np.asarray(w_min),
            np.asarray(w_max),
            np.asarray(group),
            target=target,
        )
        for (nc, slot), w in zip(rows, new_w, strict=False):
            nc.weight[slot] = float(w)
        return self

    def record_weights(self, dt: float = 0.1) -> Self:
        self._weight_rec_dt = dt
        self._weight_rows = {}
        for gid, syn_id, mech_name, _pp, nc in self._iter_stdp_connections():
            slot = self._wplastic_slot.get(mech_name, 2)
            key = (int(gid), int(syn_id), mech_name)
            self._weight_rows[key] = (nc._row, slot)
            self.w_recs.setdefault(key, [])
        if self._w_rec_times is None:
            self._w_rec_times = []
        self._weight_recording_active = True
        return self

    def _sample_weights(self) -> None:
        if self._w_rec_times is not None:
            self._w_rec_times.append(float(self._lib.rcsd_time(self._sim)))
        for key, (row, slot) in self._weight_rows.items():
            self.w_recs[key].append(float(self._w[row, slot]))

    def enable_plasticity(self, config: dict | None = None) -> Self:
        if config is None:
            config = (
                self.model.neuron_plasticity_defaults()
                if hasattr(self.model, "neuron_plasticity_defaults")
                else {}
            )
        per_population = bool(config) and isinstance(next(iter(config.values())), dict)
        mech_to_group: dict[str, str] = {}
        if per_population and hasattr(self.model, "neuron_plasticity_mechanism_groups"):
            for group, mechs in self.model.neuron_plasticity_mechanism_groups().items():
                for m in mechs:
                    mech_to_group[m] = group
        for _gid, _syn_id, mech_name, pp in self._iter_stdp_point_processes():
            if per_population:
                group = mech_to_group.get(mech_name)
                group_config = config.get(group, {}) if group else {}
            else:
                group_config = config
            for param, value in group_config.items():
                if param in L.SP:
                    setattr(pp, param, value)
            pp.plasticity_on = 1
        self._plasticity_enabled = True
        return self

    def disable_plasticity(self) -> Self:
        for _gid, _syn_id, _mech_name, pp in self._iter_stdp_point_processes():
            pp.plasticity_on = 0
        self._plasticity_enabled = False
        return self

    # -- noise --------------------------------------------------------------------

    def set_noise(self, noise: dict) -> Self:
        if not hasattr(self.model, "neuron_noise_configure"):
            return self
        if self._sim is None:
            raise RuntimeError("call init() before set_noise()")
        self._noise_state.update(noise)
        merged = self._noise_by_population(dict(self._noise_state))
        base = int(self.seed if self.seed is not None else 0)
        lib, sim = self._lib, self._sim
        for population, cells in self.cells.items():
            for gid, cell in cells.items():
                for idx, name in enumerate(cell.section_names):
                    key = f"{gid}-{idx}"
                    found = self._flucts.get(key)
                    if found is None:
                        site = _NoiseSite(f"{type(cell.template).__name__}.{name}")
                        self._flucts[key] = (cell.index, idx, site)
                        fresh = True
                    else:
                        site = found[2]
                        fresh = False
                    self.model.neuron_noise_configure(
                        population, site, None, **merged[population]
                    )
                    L.check(
                        lib.rcsd_set_noise(
                            sim,
                            cell.index,
                            idx,
                            float(site.g_e0),
                            float(site.g_i0),
                            float(site.std_e),
                            float(site.std_i),
                            float(site.tau_e),
                            float(site.tau_i),
                            float(site.E_e),
                            float(site.E_i),
                            float(site.h),
                            int(site.on),
                        ),
                        lib,
                    )
                    if fresh:
                        L.check(
                            lib.rcsd_set_noise_stream(
                                sim,
                                cell.index,
                                idx,
                                int(gid) + 1,
                                int(idx) + 1,
                                base + self._noise_stream * NOISE_STREAM_STRIDE,
                            ),
                            lib,
                        )
        return self

    def _noise_by_population(self, state: dict) -> dict[str, dict]:
        populations = set(self.cells)
        shared, per_population = {}, {p: {} for p in populations}
        for key, value in state.items():
            prefix, _, rest = key.partition("-")
            if not rest:
                shared[key] = value
                continue
            if prefix not in populations:
                raise KeyError(
                    f"{key!r} names population {prefix!r}, which this env does "
                    f"not build (has: {', '.join(sorted(populations))})"
                )
            per_population[prefix][rest] = value
        return {p: {**shared, **per_population[p]} for p in populations}

    def reseed_noise(self, stream: int | None = None) -> Self:
        if stream is None:
            self._noise_stream += 1
            stream = self._noise_stream
        else:
            self._noise_stream = int(stream)
        if not self._flucts or self._sim is None:
            return self
        base = int(self.seed if self.seed is not None else 0)
        for key, (cell, idx, _site) in self._flucts.items():
            gid = int(key.rpartition("-")[0])
            L.check(
                self._lib.rcsd_set_noise_stream(
                    self._sim,
                    cell,
                    idx,
                    gid + 1,
                    idx + 1,
                    base + stream * NOISE_STREAM_STRIDE,
                ),
                self._lib,
            )
        return self

    # -- external inputs -------------------------------------------------------------

    @property
    def input_gids(self) -> list[int]:
        return sorted(self._input_indices.keys())

    def play_input_spikes(self, spikes: dict) -> Self:
        for gid, times in spikes.items():
            index = self._input_indices.get(int(gid))
            if index is None:
                continue
            times = L.as_double_array(times)
            L.check(
                self._lib.rcsd_set_input_spikes(
                    self._sim, index, L.double_ptr(times), len(times)
                ),
                self._lib,
            )
        return self

    def apply_stimulus_from_h5(
        self,
        filepath: str,
        namespace: str,
        attribute: str = "Spike Train",
        onset: float = 0.0,
        io_size: int = 1,
        microcircuit_inputs: bool = True,
        n_trials: int = 1,
        equilibration_duration: float = 250.0,
    ) -> Self:
        local = set(self._input_indices.keys())
        by_pop: dict[str, list[int]] = defaultdict(list)
        ranges = self.system.population_ranges
        for gid in local:
            for pop, (start, count) in ranges.items():
                if start <= gid < start + count:
                    by_pop[pop].append(gid)
                    break
        shift = float(equilibration_duration) + float(onset)
        syn_cfg = self.system.connections_config["synapses"]
        input_pops = sorted({pre for post in syn_cfg for pre in syn_cfg[post]})
        for pop in input_pops:
            gids = by_pop.get(pop, [])
            if not gids:
                continue
            try:
                trains = _read_spike_trains(
                    filepath, namespace, attribute, pop, gids, int(ranges[pop][0])
                )
            except Exception:
                logger.debug(
                    "no spike input for %s in %s", pop, filepath, exc_info=True
                )
                continue
            self.play_input_spikes(
                {gid: train + shift for gid, train in trains.items()}
            )
        return self

    # -- lifecycle --------------------------------------------------------------------

    def clear(self, reseed: bool = True) -> Self:
        if reseed:
            self.reseed_noise()
        self.clear_recordings()
        for samples in self.w_recs.values():
            samples.clear()
        if self._w_rec_times is not None:
            self._w_rec_times.clear()
        self.t = 0.0
        self._dt = None
        for state in self._stim.values():
            state.clear()
        self._stim_unbacked_warned = False
        if self._sim:
            for state in self._stim.values():
                L.check(self._lib.rcsd_clear_stimulus(self._sim, state.mode), self._lib)
        return self

    def close(self) -> Self:
        if self._closed:
            return self
        self._closed = True
        if self._sim:
            self._lib.rcsd_destroy(self._sim)
            self._sim = None
        self._sp = np.zeros((L.SP_N, 0))
        self._ss = np.zeros((L.SS_N, 0))
        self._w = np.zeros((0, L.NWEIGHT))
        self.syn = None
        self.conn = None
        self.cells.clear()
        self._cell_index = {}
        self._index_gid = []
        self._thresholds = []
        self.__dict__.pop("_pop_code_cells", None)
        self._flucts.clear()
        self._opsin_refs.clear()
        self._input_indices = {}
        self.v_recs.clear()
        self.v_sections.clear()
        self.i_recs.clear()
        self._spike_gids.clear()
        self.w_recs.clear()
        self._weight_rows.clear()
        return self


def _open_h5(filepath: str):
    try:
        import h5py

        return h5py.File(filepath, "r")
    except ImportError:
        import pyfive

        return pyfive.File(filepath)


def _read_spike_trains(filepath, namespace, attribute, population, gids, first_gid):
    """``gid -> spike times`` for ``gids`` of one population."""
    fh = _open_h5(filepath)
    try:
        group = fh["Populations"][population][namespace][attribute]
        pointer = np.asarray(group["Attribute Pointer"][:]).astype(np.int64)
        index = np.asarray(group["Cell Index"][:]).astype(np.int64)
        values = group["Attribute Value"]
        chunk = (getattr(values, "chunks", None) or (1,))[0]

        order = np.argsort(index)
        ordered = index[order]
        wanted = np.asarray(sorted(gids), dtype=np.int64) - first_gid
        at = np.searchsorted(ordered, wanted)
        found = (at < ordered.size) & (
            ordered[np.minimum(at, ordered.size - 1)] == wanted
        )
        rows = order[at[found]]
        start, stop = pointer[rows], pointer[rows + 1]
        gid_of = wanted[found] + first_gid

        keep = stop > start
        start, stop, gid_of = start[keep], stop[keep], gid_of[keep]
        if not start.size:
            return {}
        by_offset = np.argsort(start)
        start, stop, gid_of = start[by_offset], stop[by_offset], gid_of[by_offset]
        reach = np.maximum.accumulate(stop)
        cut = np.flatnonzero(start[1:] - reach[:-1] > chunk) + 1
        run_first = np.concatenate(([0], cut))
        run_last = np.concatenate((cut, [start.size]))
        trains: dict[int, np.ndarray] = {}
        for lo, hi in zip(run_first, run_last, strict=True):
            base = int(start[lo])
            block = np.asarray(values[base : int(reach[hi - 1])])
            for i in range(lo, hi):
                trains[int(gid_of[i])] = block[
                    int(start[i]) - base : int(stop[i]) - base
                ].astype(np.float64)
        return trains
    finally:
        close = getattr(fh, "close", None)
        if callable(close):
            close()
