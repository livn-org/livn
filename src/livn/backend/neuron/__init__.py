from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Self, Union

import numpy as np

from livn.backend.neuron import mechanisms
from livn.backend.neuron.cells import CellBuilder, CellHandle
from livn.backend.neuron.synapses import SynapseBuilder
from livn.cells import CellRegistry
from livn.run import Run
from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LIVN_NEURON_LOGGING", "WARNING"))


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str, int],
        model: Union["Model", None] = None,
        io: Union["IO", None] = None,
        seed: int | None = 123,
        comm: "MPI.Intracomm | None" = None,
        subworld_size: int | None = None,
    ):
        from mpi4py import MPI

        from livn.system import resolve

        self.seed = seed
        self._select_spec = None
        self._select_method = "first"
        self._select_bounds = None
        self._selection: dict[str, object] | None = None
        self._selected_gids: set[int] | None = None
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.system = resolve(system, comm=comm)
        self.model = (
            model if model is not None else self.system.default_model(comm=comm)
        )
        self.io = io if io is not None else self.system.default_io(comm=comm)
        self.subworld_size = subworld_size
        self.store_kind = "auto"

        self.encoding = None
        self.decoding = None
        self.duration = None

        # Compile mechanisms: rank 0 builds, the rest wait on it.
        #
        # NB this barrier is collective over ``comm``, so every rank sharing it
        # must reach this constructor. Passing a communicator that only some of
        # its ranks construct an Env on (e.g. COMM_WORLD from one rank while the
        # others hold per-subworld communicators) deadlocks here.
        mech_dir = self.model.neuron_mechanisms_directory()
        if mech_dir is not None:
            if self.comm.Get_rank() == 0:
                mechanisms.compile_mechanisms(mech_dir)
            if self.comm.Get_size() > 1:
                self.comm.Barrier()
        self._h = mechanisms.configure(mech_dir)
        self.pc = self._h.pc
        if subworld_size is not None:
            self.pc.subworlds(subworld_size)
        # Read rank AFTER subworlds() so pc.id()/nhost() are subworld-local.
        # gid registration (set_gid2node) must use the subworld-local rank.
        self.rank = int(self.pc.id())

        # graph state
        self.cells = CellRegistry(self, comm=self.comm)
        self._detectors: dict[int, dict] = {}  # gid -> {filter,in_nc,out_nc}
        self.syn = None
        self.conn = None
        self._input_vecstims: dict[int, object] = {}
        self._input_spike_vecs: dict[int, object] = {}
        self._pop_code: dict[str, int] = {}
        self._mech_code: dict[str, int] = {}
        self._sectype_code: dict[str, int] = {}
        self._mech_id_to_name: dict[int, str] = {}
        self._wplastic_slot: dict[str, int] = {}
        self._stdp_syn_rows = np.empty(0, dtype=np.int64)
        self._stdp_conn_rows = np.empty(0, dtype=np.int64)

        # plasticity / noise / weight-recording state
        self._plasticity_enabled = False
        self._flucts: dict[str, tuple] = {}
        self._noise_state: dict = {}
        self.w_recs: dict[tuple, object] = {}
        self._weight_nc_refs: dict[tuple, object] = {}
        self._weight_rec_dt = 0.1
        self._weight_recording_active = False
        self._w_rec_times = None

        # recording buffers (spike times/ids on this host, per-(gid, sec) traces)
        self.t_vec = self._h.Vector()
        self.id_vec = self._h.Vector()
        self._spike_gids: set[int] = set()
        self.v_recs: dict[tuple[int, int], object] = {}
        self.v_sections: dict[tuple[int, int], str] = {}
        self.v_dt: dict[str, float] = {}
        self.i_recs: dict[tuple[int, int], object] = {}
        self.i_dt: dict[str, float] = {}

        # sim state
        self.t = 0.0
        # membrane potential the simulation initializes from at the next
        # (re)initialization; see the ``v_init`` property
        self._v_init = -75.0
        self._dt: float | None = None
        # dt whose 2*dt floor is currently applied to the NetCon delays (the
        # builder pre-applies DEFAULT_DT; re-applied only when run() uses another)
        from livn.backend.neuron.synapses import DEFAULT_DT

        self._delay_floor_dt: float = DEFAULT_DT
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
        self._h.celsius = self._celsius

        # extracellular stimulus block
        self._stim_segments: list = []
        self._stim_rows: dict[tuple[int, int], int] = {}
        self._stim_block: np.ndarray | None = None
        self._stim_bounds: tuple[float, float] | None = None
        self._stim_streams: list[dict] = []
        self._stim_idle = False
        self._stim_dt: float | None = None
        self._stim_step = 0
        self._stim_registered = False
        self._stim_cb = self._update_extracellular
        self._opsin_cb = self._update_opsin_phi

        # opsin (irradiance) stimulus block
        self._opsin_refs: dict[tuple[int, int], object] = {}  # (gid, sec_id) -> pp
        self._opsin_pps: list = []
        self._opsin_block: np.ndarray | None = None
        self._opsin_dt: float | None = None
        self._opsin_step = 0
        self._opsin_registered = False

        # current-clamp (IClamp) stimulus block with one IClamp on the soma per gid,
        # amplitude (nA) updated per step like the extracellular block
        self._iclamp_pps: list = []
        self._iclamp_block: np.ndarray | None = None
        self._iclamp_dt: float | None = None
        self._iclamp_step = 0
        self._iclamp_registered = False

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

    def init(self) -> Self:
        self.pc.gid_clear()
        builder = CellBuilder(self.system, self.model, self.pc, self.comm)

        ignored = (
            set(self.model.ignored_populations())
            if hasattr(self.model, "ignored_populations")
            else set()
        )
        # only populations with a cell factory are simulated biophysically while
        # any other source population is driven as an external VecStim input
        factories = set(self.model.neuron_cells().keys())
        buildable = [
            p for p in self.system.populations if p not in ignored and p in factories
        ]
        self._resolve_selection(buildable)

        if self._selection is None:
            simulated_pops = set(buildable)
        else:
            simulated_pops = {
                p for p in buildable if len(self._selection.get(p, [])) > 0
            }

        for pop in buildable:
            if pop not in simulated_pops:
                continue
            sel = None
            if self._selection is not None:
                sel = set(int(g) for g in self._selection.get(pop, []))
            cells = builder.build_local(pop, selection=sel)
            self.cells.add(
                pop,
                {gid: CellHandle(self, pop, gid, cell) for gid, cell in cells.items()},
            )
            for gid, cell in cells.items():
                self._register_cell(gid, cell)

        self._h.define_shape()

        sb = SynapseBuilder(
            self.system,
            self.model,
            self.pc,
            self.comm,
            store=self.store_kind,
            selected_gids=self._selected_gids,
            simulated_pops=simulated_pops,
            io_size=int(self.pc.nhost()),
        )
        (
            self.syn,
            self.conn,
            self._pop_code,
            self._mech_code,
            self._sectype_code,
            self._input_vecstims,
        ) = sb.build(self.cells)
        self.store_kind = sb.store_kind  # resolved value when store="auto"
        self._index_plastic_synapses()
        self._insert_opsins()

        self.pc.set_maxstep(10)
        return self

    def _insert_opsins(self) -> None:
        """Insert opsin point processes on the sections named by the model.

        Enabled when the model exposes ``neuron_opsin_config()`` though the opsins sit
        idle (``phi = 0``) until an ``input_mode="irradiance"`` stimulus drives them.
        """
        if not hasattr(self.model, "neuron_opsin_config"):
            return
        cfg = self.model.neuron_opsin_config()
        if not cfg:
            return
        mech_name = cfg.get("mechanism", "RhO3c")
        populations = cfg.get("populations", list(self.cells.keys()))
        target_sections = set(cfg.get("sections", ["soma"]))
        params = cfg.get("params", {})
        mech = getattr(self._h, mech_name, None)
        if mech is None:
            logger.warning("opsin mechanism %s not available", mech_name)
            return

        for pop in populations:
            for gid, cell in self.cells.get(pop, {}).items():
                sections = getattr(cell, "sections", None)
                if sections is None:
                    continue
                for sec_id, sec in enumerate(sections):
                    sec_name = sec.name().split(".")[-1]
                    if sec_name not in target_sections:
                        continue
                    pp = mech(sec(0.5))
                    for pname, value in params.items():
                        setattr(pp, pname, value)
                    self._opsin_refs[(int(gid), sec_id)] = pp

    def _index_plastic_synapses(self) -> None:
        """Precompute which mechanisms and rows are plastic / have a w_plastic slot"""
        rules = self.model.neuron_synapse_rules()
        self._mech_id_to_name = {v: k for k, v in self._mech_code.items()}
        self._wplastic_slot: dict[str, int] = {}
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

    def _register_cell(self, gid: int, cell) -> None:
        """Register the gid with a spike detector.

        When the model requests a refractory period and the ``SpikeFilter``
        mechanism is available, the somatic threshold detector is routed through
        it, otherwise a plain threshold NetCon is used.
        """
        h = self._h
        self.pc.set_gid2node(gid, self.rank)

        soma_seg = cell.spike_source()
        soma_sec = soma_seg.sec

        use_filter = self._refractory_period > 0 and hasattr(h, "SpikeFilter")
        if use_filter:
            spike_filter = h.SpikeFilter()
            spike_filter.tref = float(self._refractory_period)
            in_nc = h.NetCon(soma_sec(0.5)._ref_v, spike_filter, sec=soma_sec)
            in_nc.threshold = float(cell.threshold)
            in_nc.delay = 0.0
            in_nc.weight[0] = 1.0
            out_nc = h.NetCon(spike_filter, None)
            out_nc.delay = max(2.0 * 0.025, 1e-3)
            out_nc.weight[0] = 1.0
            self.pc.cell(gid, out_nc)
            self._detectors[gid] = {
                "filter": spike_filter,
                "in_nc": in_nc,
                "out_nc": out_nc,
            }
        else:
            det = h.NetCon(soma_sec(0.5)._ref_v, None, sec=soma_sec)
            det.threshold = float(cell.threshold)
            self.pc.cell(gid, det)
            self._detectors[gid] = {"out_nc": det}

    def active_populations(self) -> list[str]:
        ignored: set[str] = set()
        if hasattr(self.model, "ignored_populations"):
            ignored = set(self.model.ignored_populations())
        return [p for p in self.system.populations if p not in ignored]

    def _record_spikes(self, population: str) -> Self:
        for gid in self.cells.get(population, {}):
            gid = int(gid)
            if gid in self._spike_gids:
                continue
            self.pc.spike_record(gid, self.t_vec, self.id_vec)
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
            for sec_id, sec in enumerate(cell.sections):
                name = sec.name().split(".")[-1]
                if wanted is not None and name not in wanted:
                    continue
                if (int(gid), sec_id) in self.v_recs:
                    continue  # already recording this compartment
                vec = self._h.Vector()
                vec.record(sec(0.5)._ref_v, dt)
                self.v_recs[(int(gid), sec_id)] = vec
                self.v_sections[(int(gid), sec_id)] = name
        return self

    def _record_membrane_current(self, population: str, dt: float) -> Self:
        cells = self.cells.get(population, {})
        if not cells:
            return self  # enabling fast_imem with no sections asserts in psolve
        self._h.cvode.use_fast_imem(1)
        self.i_dt[population] = dt
        for gid, cell in cells.items():
            for sec_id, sec in enumerate(cell.sections):
                vec = self._h.Vector()
                vec.record(sec(0.5)._ref_i_membrane_, dt)
                self.i_recs[(int(gid), sec_id)] = vec
        return self

    def clear_recordings(self) -> Self:
        self.t_vec.resize(0)
        self.id_vec.resize(0)
        for vec in self.v_recs.values():
            vec.resize(0)
        for vec in self.i_recs.values():
            vec.resize(0)
        return self

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float | None = None,
        **kwargs,
    ):
        h = self._h
        self.duration = duration
        current_time = self.t
        self._prune_stim_streams(current_time)

        if stimulus is not None:
            stimulus = Stimulus.from_arg(stimulus, env=self, duration=duration)
            stimulus = self.model.prepare_stimulus(stimulus)
            if stimulus.input_mode == "irradiance":
                self._setup_opsin_stimulus(stimulus, current_time)
            elif stimulus.input_mode == "current":
                self._setup_iclamp(stimulus, current_time)
            elif stimulus.input_mode == "extracellular":
                self._setup_extracellular(stimulus, current_time)
            else:
                raise ValueError(
                    f"the neuron backend has no mechanism for a "
                    f"{stimulus.input_mode!r} stimulus"
                )

        first_run = self.t == 0
        requested_dt = (
            dt if dt is not None else (self._dt if self._dt is not None else 0.025)
        )
        if (
            not first_run
            and self._dt is not None
            and abs(requested_dt - self._dt) > 1e-12
        ):
            raise ValueError("Cannot change dt mid-simulation; call clear() first.")
        if first_run:
            self._dt = requested_dt
            self._apply_delay_floor(requested_dt)
        self._stim_step = int(round(current_time / self._dt))
        self._opsin_step = int(round(current_time / self._dt))
        self._iclamp_step = int(round(current_time / self._dt))

        self.clear_recordings()

        if first_run:
            h.celsius = self._celsius
            h.v_init = self._v_init
            h.stdinit()
            h.secondorder = 2
            h.dt = requested_dt
            self.pc.timeout(600.0)
            self._apply_init_ic()
            h.finitialize(h.v_init)
            h.finitialize(h.v_init)
        else:
            self.pc.timeout(600.0)

        target_time = self.t + duration
        if self._weight_recording_active and self._weight_nc_refs:
            w_dt = self._weight_rec_dt
            while h.t < target_time - w_dt / 2:
                self.pc.psolve(min(h.t + w_dt, target_time))
                self._sample_weights()
            if h.t < target_time - 1e-6:
                self.pc.psolve(target_time)
        else:
            self.pc.psolve(target_time)
        self.t = target_time

        ii, tt, iv, v, sv, im, mp = self._collect(self.active_gids(), current_time)
        self.duration = None

        return (
            Run(t0=current_time, duration=duration)
            .add_spikes(ii, tt)
            .add_voltage(iv, v, dt=self.voltage_recording_dt, sections=sv)
            .add_current(im, mp, dt=self.membrane_current_recording_dt)
        )

    def _collect(self, active_gids, current_time: float):
        """Assemble recorded buffers into the (it, tt, iv, v, im, mp) format."""
        tt = np.array(self.t_vec.as_numpy(), copy=True)
        ii = np.asarray(self.id_vec.as_numpy(), dtype=np.uint32)
        if current_time != 0.0 and tt.size > 0:
            tt = tt - current_time
            tt[tt < 0.0] = 0.0

        if self.v_recs:
            iv = np.asarray([gid for (gid, _sec) in self.v_recs], dtype=np.uint32)
            sv = np.asarray([self.v_sections[key] for key in self.v_recs])
            v = np.array(
                [rec.as_numpy() for rec in self.v_recs.values()], dtype=np.float32
            )
        else:
            iv = v = sv = None

        im = mp = None
        if self.i_recs and len(active_gids):
            gid_to_index = {int(g): i for i, g in enumerate(active_gids)}
            spn = max((int(sec) for (_gid, sec) in self.i_recs), default=0) + 1
            # i_membrane_ (fast_imem) is absolute nA per segment -> microampere;
            # pack into a [n_neurons*spn, T] matrix in active_gids order
            T = max((len(rec) for rec in self.i_recs.values()), default=0)
            if T:
                rows = len(active_gids) * spn
                mp = np.zeros((rows, T), dtype=np.float32)
                im = np.full(rows, -1, dtype=np.int32)
                for (gid, sec_id), rec in self.i_recs.items():
                    idx = gid_to_index.get(int(gid))
                    if idx is None:
                        continue
                    row = idx * spn + int(sec_id)
                    if row >= rows:
                        continue
                    arr = np.asarray(rec.as_numpy(), dtype=np.float32) * 1e-3
                    n = min(arr.shape[0], T)
                    mp[row, :n] = arr[:n]
                    im[row] = gid

        return ii, tt, iv, v, sv, im, mp

    def _apply_delay_floor(self, dt: float) -> None:
        """Ensure every NetCon delay is >= 2*dt."""
        if self.conn is None or self.conn.size == 0:
            return
        if abs(dt - self._delay_floor_dt) <= 1e-12:
            # builder pre-applies the default floor, so only needed if different dt
            return
        floor = 2.0 * dt
        phys = self.conn.delay  # physical delays (float32[C])
        store = self.conn.store
        for i in range(self.conn.size):
            d = float(phys[i])
            store.get(i).delay = d if d > floor else floor
        self._delay_floor_dt = dt

    def _apply_init_ic(self) -> None:
        """Pin each cell's resting current via its ``init_ic``.

        Each ``init_ic`` calls ``h.finitialize`` internally, which is a
        collective in parallel NEURON. Ranks own different numbers of cells, so
        we pad to the global maximum with balancing ``finitialize`` calls to keep
        the collective count identical across ranks (otherwise psolve deadlocks).
        """
        ic_cells = [
            cell
            for cells in self.cells.values()
            for cell in cells.values()
            if callable(getattr(cell, "init_ic", None))
        ]
        local_n = len(ic_cells)

        n_iter = local_n
        if int(self.pc.nhost()) > 1 and self.comm is not None:
            from mpi4py import MPI

            n_iter = self.comm.allreduce(local_n, op=MPI.MAX)

        for i in range(n_iter):
            if i < local_n:
                ic_cells[i].init_ic()
            else:
                # balancing collective call (state reset only; ic params persist
                # and the final finitialize in run() re-initializes)
                self._h.finitialize(self._h.v_init)

    def _setup_extracellular(self, stimulus: Stimulus, current_time: float) -> None:
        n_neurons = len(self.active_neuron_coordinates())
        if stimulus.gids is None:
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
        if self._stim_dt is None:
            self._stim_dt = stimulus.dt
        elif not math.isclose(self._stim_dt, stimulus.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        declares = getattr(self.model, "stimulus_bounds", None)
        self._stim_bounds = declares(stimulus.input_mode) if declares else None

        start_step = int(round(current_time / stimulus.dt))
        rows, columns = self._stim_rows_for(stimulus)
        if not rows:
            return

        if stimulus.deferred:
            self._install_stim_stream(stimulus, rows, columns, start_step)
        else:
            self._install_stim_block(stimulus, rows, columns, start_step)

        if not self._stim_registered:
            self._h.cvode.extra_scatter_gather(0, self._stim_cb)
            self._stim_registered = True

    def _stim_rows_for(self, stimulus: Stimulus) -> tuple[list, list]:
        """`(block rows, stimulus columns)` for what this rank can drive."""
        rows, columns = [], []
        for column, (gid, section_id) in enumerate(stimulus.columns()):
            if not self.pc.gid_exists(gid):
                continue
            cell = self._find_cell(gid)
            if cell is None or section_id >= len(cell.sections):
                continue
            sec = cell.sections[section_id]
            sec.push()
            has_extracellular = self._h.ismembrane("extracellular")
            self._h.pop_section()
            if not has_extracellular:
                continue

            key = (gid, section_id)
            row = self._stim_rows.get(key)
            if row is None:
                row = len(self._stim_segments)
                self._stim_rows[key] = row
                self._stim_segments.append(sec(0.5))
            rows.append(row)
            columns.append(column)
        return rows, columns

    def _install_stim_block(self, stimulus, rows, columns, start_step) -> None:
        """Hold the whole command, indexed `[section, absolute step]`."""
        from livn.stimulus import check_bounds

        values = np.asarray(stimulus.array)
        check_bounds(values, self._stim_bounds, stimulus.input_mode, stimulus.units)
        dtype = np.promote_types(values.dtype, np.float32)
        end_step = start_step + values.shape[0]

        width = end_step
        previous = self._stim_block
        if previous is not None:
            width = max(width, previous.shape[1])
            dtype = np.promote_types(dtype, previous.dtype)

        block = np.zeros((len(self._stim_segments), width), dtype=dtype)
        if previous is not None:
            block[: previous.shape[0], : previous.shape[1]] = previous
        for row, column in zip(rows, columns):
            block[row, start_step:end_step] = values[:, column]
        self._stim_block = block

    def _install_stim_stream(self, stimulus, rows, columns, start_step) -> None:
        from livn.stimulus import chunk_bytes

        itemsize = np.dtype(np.float64).itemsize
        chunk_steps = max(1, int(chunk_bytes() // max(1, stimulus.width * itemsize)))

        self._stim_streams.append(
            {
                "stimulus": stimulus,
                "rows": np.asarray(rows, dtype=np.int64),
                "columns": np.asarray(columns, dtype=np.int64),
                "start_step": start_step,
                "n_steps": int(round(stimulus.duration / stimulus.dt)),
                "chunk_steps": chunk_steps,
                "chunk": None,
                "chunk_start": 0,
                "chunk_stop": 0,
            }
        )

    def _prune_stim_streams(self, current_time: float) -> None:
        if not self._stim_streams or not self._stim_dt:
            return
        idx = int(current_time / self._stim_dt)
        self._stim_streams = [
            stream
            for stream in self._stim_streams
            if idx < stream["start_step"] + stream["n_steps"]
        ]

    def _update_extracellular(self) -> None:
        if self._closed or not self._stim_segments:
            return
        block = self._stim_block
        streams = self._stim_streams
        if block is None and not streams:
            if not self._stim_idle:
                for seg in self._stim_segments:
                    seg.e_extracellular = 0.0
                self._stim_idle = True
            return
        self._stim_idle = False

        current_time = self._stim_step * (self._dt or 0.025)
        idx = int(current_time / self._stim_dt) if self._stim_dt else 0

        if not streams:
            col = block[:, min(idx, block.shape[1] - 1)]
        else:
            col = np.zeros(len(self._stim_segments), dtype=np.float64)
            if block is not None:
                held = block[:, min(idx, block.shape[1] - 1)]
                col[: len(held)] = held
            for stream in streams:
                self._accumulate_stim_stream(stream, idx, col)

        for i, seg in enumerate(self._stim_segments):
            seg.e_extracellular = float(col[i])
        self._stim_step += 1

    def _accumulate_stim_stream(self, stream, idx: int, col) -> None:
        offset = idx - stream["start_step"]
        if offset >= stream["n_steps"]:
            stream["chunk"] = None
            stream["chunk_start"] = stream["chunk_stop"] = 0
            return
        if offset < 0:
            return

        if not stream["chunk_start"] <= idx < stream["chunk_stop"]:
            self._refill_stim_stream(stream, idx)
            if not stream["chunk_start"] <= idx < stream["chunk_stop"]:
                return
        col[stream["rows"]] += stream["chunk"][:, idx - stream["chunk_start"]]

    def _refill_stim_stream(self, stream, idx: int) -> None:
        from livn.stimulus import check_bounds

        stimulus = stream["stimulus"]
        first = max(idx, stream["start_step"])
        start_ms = (first - stream["start_step"]) * stimulus.dt
        stop_ms = min(start_ms + stream["chunk_steps"] * stimulus.dt, stimulus.duration)

        rendered = np.asarray(stimulus.window(start_ms, stop_ms))
        check_bounds(rendered, self._stim_bounds, stimulus.input_mode, stimulus.units)
        if rendered.shape[0] == 0:
            stream["chunk"] = None
            stream["chunk_start"] = stream["chunk_stop"] = 0
            return

        stream["chunk"] = np.ascontiguousarray(
            rendered[:, stream["columns"]].T, dtype=np.float64
        )
        stream["chunk_start"] = first
        stream["chunk_stop"] = first + rendered.shape[0]

    def _soma_seg(self, cell):
        """Middle segment of the cell's soma section (fallback: first section)."""
        for sec in getattr(cell, "sections", []):
            if sec.name().split(".")[-1] == "soma":
                return sec(0.5)
        return cell.sections[0](0.5)

    def _setup_iclamp(self, stimulus: Stimulus, current_time: float) -> None:
        """Attach an IClamp on each stimulated cell's soma and play its amplitude
        (nA) time series into ``IClamp.amp`` via ``Vector.play``.

        ``Vector.play`` handles the stimulus->sim time mapping natively (re-armed
        by ``finitialize`` at each run start), so the step timing is exact without
        a manual per-step callback."""
        if stimulus.gids is None:
            stimulus.gids = self.active_gids()
        h = self._h

        for gid, _section_id, series in stimulus:
            gid = int(gid)
            if not self.pc.gid_exists(gid):
                continue
            cell = self._find_cell(gid)
            if cell is None:
                continue
            series = np.asarray(series, dtype=np.float64)
            ic = h.IClamp(self._soma_seg(cell))
            ic.delay = 0.0
            ic.dur = 1e9  # amp is driven by the played vector
            amp_vec = h.Vector(series)
            t_vec = h.Vector(current_time + np.arange(series.size) * stimulus.dt)
            amp_vec.play(ic._ref_amp, t_vec, True)  # continuous (interpolated)
            # keep ic + vectors alive for the run
            self._iclamp_pps.append((ic, amp_vec, t_vec))

    def _setup_opsin_stimulus(self, stimulus: Stimulus, current_time: float) -> None:
        if not self._opsin_refs:
            raise ValueError(
                "Stimulus has input_mode='irradiance' but no opsins are attached; "
                "configure the model's neuron_opsin_config()."
            )
        phi_stim = stimulus.convert_to("photon_flux")
        if phi_stim.gids is None:
            phi_stim.gids = self.active_gids()
        if self._opsin_dt is None:
            self._opsin_dt = phi_stim.dt
        elif not math.isclose(self._opsin_dt, phi_stim.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        start_step = int(round(current_time / phi_stim.dt))

        pending: list[tuple[object, np.ndarray]] = []
        for gid, section_id, series in phi_stim:
            gid, section_id = int(gid), int(section_id)
            pp = self._opsin_refs.get((gid, section_id))
            if pp is None:
                continue
            pending.append((pp, np.asarray(series, dtype=np.float64)))

        if not pending:
            return

        end_step = start_step + max(len(v) for _pp, v in pending)
        base = len(self._opsin_pps)
        block = np.zeros((base + len(pending), end_step), dtype=np.float64)
        if self._opsin_block is not None:
            block[:base, : self._opsin_block.shape[1]] = self._opsin_block
        for j, (pp, values) in enumerate(pending):
            self._opsin_pps.append(pp)
            block[base + j, start_step : start_step + len(values)] = values
        self._opsin_block = block

        if not self._opsin_registered:
            self._h.cvode.extra_scatter_gather(0, self._opsin_cb)
            self._opsin_registered = True

    def _update_opsin_phi(self) -> None:
        block = self._opsin_block
        if block is None or self._closed or not self._opsin_pps:
            return
        current_time = self._opsin_step * (self._dt or 0.025)
        idx = int(current_time / self._opsin_dt) if self._opsin_dt else 0
        if idx >= block.shape[1]:
            idx = block.shape[1] - 1
        col = block[:, idx]
        for i, pp in enumerate(self._opsin_pps):
            pp.phi = float(col[i])
        self._opsin_step += 1

    def _find_cell(self, gid: int):
        for cells in self.cells.values():
            if gid in cells:
                return cells[gid]
        return None

    def destination_sections(self) -> dict[str, dict[str, str]]:
        from livn.backend.neuron.cells import CONFIG_SECTION_NAMES, config_section_swc

        declared = CONFIG_SECTION_NAMES | {
            section for _, _, section, _, _ in self.system.synapse_projections()
        }

        sections: dict[str, dict[str, str]] = {}
        for population, factory in self.model.neuron_cells().items():
            cell = factory(morphology=None)
            sections[population] = {
                section: str(cell.dest_sec_type(config_section_swc(section)))
                for section in declared
            }
        return sections

    @property
    def weight_names(self) -> list[str]:
        if self.conn is None:
            return []

        pop_of = {code: name for name, code in self._pop_code.items()}
        sec_of = {code: name for name, code in self._sectype_code.items()}
        mech_of = {code: name for name, code in self._mech_code.items()}

        # (post, pre, section, mechanism) by name
        built = {
            (pop_of[po], pop_of[pr], sec_of[ds], mech_of[mi])
            for po, pr, ds, mi in zip(
                self.conn.post_pop.tolist(),
                self.conn.pre_pop.tolist(),
                self.conn.dest_sectype.tolist(),
                self.conn.mech_id.tolist(),
            )
        }
        if self.comm is not None and self.comm.Get_size() > 1:
            for other in self.comm.allgather(built):
                built |= other

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
        import inspect

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
            ):
                if pp < 0 or pp not in pop_of:
                    continue
                sites.add((pop_of[pp], sec_of.get(sec, ""), mech_of.get(mech, "")))
        if self.comm is not None and self.comm.Get_size() > 1:
            for other in self.comm.allgather(sites):
                sites |= other

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
        if hasattr(self.model, "neuron_noise_mechanism") and hasattr(
            self.model, "neuron_noise_configure"
        ):
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
        """Set mechanism parameters on synaptic point processes."""
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

            for row in np.flatnonzero(mask):
                pp = self.syn.store.get(int(row))
                if hasattr(pp, name):
                    setattr(pp, name, float(value))
        return self

    def set_weights(self, weights: dict) -> Self:
        from livn.types import SynapticParam

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
            if p.sec_type is not None:
                if p.sec_type in self._sectype_code:
                    mask &= self.conn.dest_sectype == self._sectype_code[p.sec_type]
                else:
                    # sec_type names a destination section that this network
                    # has no synapses on -> selects nothing
                    mask &= False
            idx = np.flatnonzero(mask)
            self.conn.weight[idx] = val
            for i in idx:
                nc = self.conn.store.get(int(i))
                nc.weight[int(self.conn.wslot[i])] = val
        return self

    def get_weights(self) -> dict:
        weights: dict[tuple, float] = {}
        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            slot = self._wplastic_slot.get(mech_name, 2)
            weights[(int(gid), int(syn_id), mech_name)] = float(nc.weight[slot])
        return weights

    def normalize_weights(self, target: float | None = None) -> Self:
        from livn.weights import normalize_weights

        rows = []
        weight, w_min, w_max, group = [], [], [], []
        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
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
        for (nc, slot), w in zip(rows, new_w):
            nc.weight[slot] = float(w)
        return self

    def record_weights(self, dt: float = 0.1) -> Self:
        h = self._h
        self._weight_rec_dt = dt
        self._weight_nc_refs = {}
        for gid, syn_id, mech_name, pp, nc in self._iter_stdp_connections():
            slot = self._wplastic_slot.get(mech_name, 2)
            key = (int(gid), int(syn_id), mech_name)
            self._weight_nc_refs[key] = (nc, slot)
            self.w_recs.setdefault(key, h.Vector())
        if self._w_rec_times is None:
            self._w_rec_times = h.Vector()
        self._weight_recording_active = True
        return self

    def _sample_weights(self) -> None:
        if self._w_rec_times is not None:
            self._w_rec_times.append(float(self._h.t))
        for key, (nc, slot) in self._weight_nc_refs.items():
            self.w_recs[key].append(float(nc.weight[slot]))

    def _iter_stdp_point_processes(self):
        if self.syn is None:
            return
        for row in self._stdp_syn_rows:
            row = int(row)
            pp = self.syn.store.get(row)
            name = self._mech_id_to_name.get(int(self.syn.mech_id[row]))
            yield int(self.syn.post_gid[row]), int(self.syn.syn_id[row]), name, pp

    def _iter_stdp_connections(self):
        if self.conn is None:
            return
        for row in self._stdp_conn_rows:
            row = int(row)
            syn_row = int(self.conn.syn_row[row])
            pp = self.syn.store.get(syn_row)
            nc = self.conn.store.get(row)
            name = self._mech_id_to_name.get(int(self.conn.mech_id[row]))
            yield (
                int(self.syn.post_gid[syn_row]),
                int(self.syn.syn_id[syn_row]),
                name,
                pp,
                nc,
            )

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

        for gid, syn_id, mech_name, pp in self._iter_stdp_point_processes():
            if per_population:
                group = mech_to_group.get(mech_name)
                group_config = config.get(group, {}) if group else {}
            else:
                group_config = config
            for param, value in group_config.items():
                if hasattr(pp, param):
                    setattr(pp, param, value)
            pp.plasticity_on = 1

        self._plasticity_enabled = True
        return self

    def disable_plasticity(self) -> Self:
        for gid, syn_id, mech_name, pp in self._iter_stdp_point_processes():
            pp.plasticity_on = 0
        self._plasticity_enabled = False
        return self

    def set_noise(self, noise: dict) -> Self:
        if not hasattr(self.model, "neuron_noise_mechanism"):
            return self
        self._noise_state.update(noise)
        merged = dict(self._noise_state)
        for population, cells in self.cells.items():
            for gid, cell in cells.items():
                for idx, sec in enumerate(cell.sections):
                    sec.push()
                    key = f"{gid}-{idx}"
                    fluct, state = self._flucts.get(key, (None, None))
                    if fluct is None:
                        fluct, state = self.model.neuron_noise_mechanism(
                            sec(0.5), gid=int(gid), index=int(idx), seed=self.seed
                        )
                        self._flucts[key] = (fluct, state)
                    self.model.neuron_noise_configure(
                        population, fluct, state, **merged
                    )
                    self._h.pop_section()
        return self

    @property
    def v_init(self) -> float:
        """Membrane potential (mV) the simulation initializes from."""
        return self._v_init

    @v_init.setter
    def v_init(self, value: float) -> None:
        if self.t != 0.0:
            raise RuntimeError(
                "v_init only takes effect at initialization and cannot be changed "
                "after the simulation has started; call clear() to reset first."
            )
        self._v_init = float(value)

    @staticmethod
    def _recorded_dt(dts: dict[str, float], default: float) -> float:
        """Resolve a single recording dt from a per-population map.

        Returns the common dt if all populations agree, the finest (min) dt if
        they differ, or ``default`` if nothing has been recorded yet.
        """
        if not dts:
            return default
        vals = set(dts.values())
        return next(iter(vals)) if len(vals) == 1 else min(vals)

    @property
    def voltage_recording_dt(self) -> float:
        """Recording dt (ms) for voltage traces.

        Reflects the ``dt`` passed to ``record_voltage`` (0.1 ms until anything
        is recorded), rather than a fixed constant.
        """
        return self._recorded_dt(self.v_dt, 0.1)

    @property
    def membrane_current_recording_dt(self) -> float:
        """Recording dt (ms) for membrane-current traces (see
        ``voltage_recording_dt``)."""
        return self._recorded_dt(self.i_dt, 0.1)

    @property
    def input_gids(self) -> list[int]:
        """Gids of external input sources wired into the local network."""
        return sorted(self._input_vecstims.keys())

    def play_input_spikes(self, spikes: dict) -> Self:
        """Play spike trains into external input sources.

        ``spikes`` maps an input source gid to a sequence of spike times (ms).
        Only gids that project onto local cells (i.e. in ``input_gids``) have a
        VecStim to receive them; others are ignored. Replaces any previously
        played train for that gid.
        """
        for gid, times in spikes.items():
            vs = self._input_vecstims.get(int(gid))
            if vs is None:
                continue
            vec = self._h.Vector(np.asarray(times, dtype=np.float64))
            self._input_spike_vecs[int(gid)] = vec  # keep alive
            vs.play(vec)
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
        """Play spike trains read from an H5 file into external input sources.

        Reads ``attribute`` from ``namespace`` for each input population and
        plays the trains (shifted by ``equilibration_duration + onset``) into the
        matching VecStim sources. Only sources wired into the local network
        (``input_gids``) are populated, so this composes with ``selection()``.
        """
        from neuroh5.io import scatter_read_cell_attribute_selection

        local = set(self._input_vecstims.keys())

        by_pop: dict[str, list[int]] = defaultdict(list)
        ranges = self.system.population_ranges
        for gid in local:
            for pop, (start, count) in ranges.items():
                if start <= gid < start + count:
                    by_pop[pop].append(gid)
                    break

        shift = float(equilibration_duration) + float(onset)
        # Every rank must call the (collective) scatter read for the same set of
        # populations in the same order, even if it owns no gids there.
        syn_cfg = self.system.connections_config["synapses"]
        input_pops = sorted({pre for post in syn_cfg for pre in syn_cfg[post]})
        for pop in input_pops:
            gids = by_pop.get(pop, [])
            try:
                it, info = scatter_read_cell_attribute_selection(
                    filepath,
                    pop,
                    sorted(gids),
                    namespace=namespace,
                    mask={attribute},
                    comm=self.comm,
                    io_size=io_size,
                    return_type="tuple",
                )
            except Exception:
                logger.debug(
                    "no spike input for %s in %s", pop, filepath, exc_info=True
                )
                continue
            attr_idx = info.get(attribute)
            if attr_idx is None:
                continue
            spikes = {}
            for gid, data in it:
                train = np.asarray(data[attr_idx], dtype=np.float64)
                if train.size:
                    spikes[int(gid)] = train + shift
            self.play_input_spikes(spikes)
        return self

    def _unregister_stim_callback(self) -> None:
        """Detach the cvode stimulus callbacks (extracellular + opsin).

        Essential on teardown as a callback left registered keeps firing after
        this env's sections are deleted, which crashes any later env's psolve.
        """
        if self._stim_registered:
            try:
                self._h.cvode.extra_scatter_gather_remove(self._stim_cb)
            except Exception:
                logger.debug("failed to remove extracellular callback", exc_info=True)
            self._stim_registered = False
        if self._opsin_registered:
            try:
                self._h.cvode.extra_scatter_gather_remove(self._opsin_cb)
            except Exception:
                logger.debug("failed to remove opsin callback", exc_info=True)
            self._opsin_registered = False
        # IClamp uses Vector.play (no cvode callback); nothing to unregister

    def clear(self) -> Self:
        self.clear_recordings()
        for vec in self.w_recs.values():
            vec.resize(0)
        if self._w_rec_times is not None:
            self._w_rec_times.resize(0)
        self._unregister_stim_callback()
        self.t = 0.0
        self._dt = None
        self._stim_segments = []
        self._stim_rows = {}
        self._stim_block = None
        self._stim_streams = []
        self._stim_idle = False
        self._stim_dt = None
        self._stim_step = 0
        self._opsin_pps = []
        self._opsin_block = None
        self._opsin_dt = None
        self._opsin_step = 0
        self._iclamp_pps = []
        self._iclamp_block = None
        self._iclamp_dt = None
        self._iclamp_step = 0
        return self

    def close(self) -> Self:
        if self._closed:
            return self
        self._closed = True
        import gc

        # Detach the cvode callbacks before any section they reference is freed.
        self._unregister_stim_callback()
        self._stim_segments = []
        self._stim_rows = {}
        self._stim_block = None
        self._stim_streams = []
        self._opsin_pps = []
        self._opsin_block = None
        self._opsin_refs.clear()

        # Drop every NEURON reference that points at this env's sections, so
        # deleting the cells does not leave dangling recorders / point
        # processes / NetCons that would crash a later env's psolve. A recording
        # Vector detaches from its _ref_ pointer when deallocated, so clearing the
        # dicts (and the GC below) is what stops the recording.
        try:
            self.t_vec.resize(0)
            self.id_vec.resize(0)
        except Exception:
            pass
        self.v_recs.clear()
        self.v_sections.clear()
        self.i_recs.clear()
        if self.syn is not None:
            self.syn.store.clear()
        if self.conn is not None:
            self.conn.store.clear()
        self.syn = None
        self.conn = None
        self.w_recs.clear()
        self._weight_nc_refs.clear()
        self._flucts.clear()
        self._detectors.clear()
        self._input_vecstims.clear()
        self._input_spike_vecs.clear()

        # Drop cells (Python-managed sections delete on GC), then explicitly
        # delete any sections that survive, then release gids.
        self.cells.clear()
        gc.collect()
        try:
            for sec in list(self._h.allsec()):
                self._h.delete_section(sec=sec)
        except Exception:
            logger.debug("section teardown failed", exc_info=True)
        try:
            self.pc.gid_clear()
        except Exception:
            pass
        gc.collect()
        return self
