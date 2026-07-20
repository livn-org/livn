from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Self, Union

import numpy as np

from livn.backend.neuron import mechanisms
from livn.backend.neuron.cells import CellBuilder
from livn.backend.neuron.synapses import SynapseBuilder
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
        system: Union["System", str],
        model: Union["Model", None] = None,
        io: Union["IO", None] = None,
        seed: int | None = 123,
        comm: "MPI.Intracomm | None" = None,
        subworld_size: int | None = None,
    ):
        from mpi4py import MPI

        from livn.system import System

        self.seed = seed
        self._select_spec = None
        self._select_method = "first"
        self._selection: dict[str, object] | None = None
        self._selected_gids: set[int] | None = None
        self.system = (
            system if not isinstance(system, str) else System(system, comm=comm)
        )
        self.model = model if model is not None else self.system.default_model()
        self.io = io if io is not None else self.system.default_io()
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.subworld_size = subworld_size
        self.store_kind = "auto"

        self.encoding = None
        self.decoding = None
        self.duration = None

        # Compile mechanisms
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
        self.cells: dict[str, dict[int, object]] = {}
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
        self.v_dt: dict[str, float] = {}
        self.i_recs: dict[tuple[int, int], object] = {}
        self.i_dt: dict[str, float] = {}

        # sim state
        self.t = 0.0
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

        # extracellular stimulus block
        self._stim_segments: list = []
        self._stim_block: np.ndarray | None = None
        self._stim_dt: float | None = None
        self._stim_step = 0
        self._stim_registered = False

        # opsin (irradiance) stimulus block
        self._opsin_refs: dict[tuple[int, int], object] = {}  # (gid, sec_id) -> pp
        self._opsin_pps: list = []
        self._opsin_block: np.ndarray | None = None
        self._opsin_dt: float | None = None
        self._opsin_step = 0
        self._opsin_registered = False

    def selection(self, select, method: str = "first") -> Self:
        if self.cells:
            raise RuntimeError("selection() must be called before init()")
        self._select_spec = select
        self._select_method = method
        return self

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

        for pop in buildable:
            sel = None
            if self._selection is not None:
                sel = set(int(g) for g in self._selection.get(pop, []))
            cells = builder.build_local(pop, selection=sel)
            self.cells[pop] = cells
            for gid, cell in cells.items():
                self._register_cell(gid, cell)

        self._h.define_shape()

        simulated_pops = set(buildable)
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

    def _record_voltage(self, population: str, dt: float) -> Self:
        self.v_dt[population] = dt
        for gid, cell in self.cells.get(population, {}).items():
            for sec_id, sec in enumerate(cell.sections):
                vec = self._h.Vector()
                vec.record(sec(0.5)._ref_v, dt)
                self.v_recs[(int(gid), sec_id)] = vec
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

        if stimulus is not None:
            if not isinstance(stimulus, Stimulus):
                stimulus = Stimulus.from_arg(stimulus)
            stimulus = self.model.prepare_stimulus(stimulus)
            if stimulus.input_mode == "irradiance":
                self._setup_opsin_stimulus(stimulus, current_time)
            else:
                self._setup_extracellular(stimulus, current_time)

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

        self.clear_recordings()

        if first_run:
            h.v_init = -75.0
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

        result = self._collect(self.active_gids(), current_time)
        self.duration = None
        return result

    def _collect(self, active_gids, current_time: float):
        """Assemble recorded buffers into the (it, tt, iv, v, im, mp) format."""
        tt = np.array(self.t_vec.as_numpy(), copy=True)
        ii = np.asarray(self.id_vec.as_numpy(), dtype=np.uint32)
        if current_time != 0.0 and tt.size > 0:
            tt = tt - current_time
            tt[tt < 0.0] = 0.0

        if self.v_recs:
            iv = np.asarray([gid for (gid, _sec) in self.v_recs], dtype=np.uint32)
            v = np.array(
                [rec.as_numpy() for rec in self.v_recs.values()], dtype=np.float32
            )
        else:
            iv = v = None

        im = mp = None
        if self.i_recs and len(active_gids):
            gid_to_index = {int(g): i for i, g in enumerate(active_gids)}
            spn = max(1, len(self.i_recs) // len(active_gids))  # sections per neuron
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

        return ii, tt, iv, v, im, mp

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
            stimulus.gids = self.active_gids()
        sections_per_neuron = max(1, len(stimulus) // max(1, n_neurons))

        if self._stim_dt is None:
            self._stim_dt = stimulus.dt
        elif not math.isclose(self._stim_dt, stimulus.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        start_step = int(round(current_time / stimulus.dt))

        # Collect (segment, values) for sections that carry an extracellular
        # mechanism, then size the block once.
        pending: list[tuple[object, np.ndarray]] = []
        for count, (gid, series) in enumerate(stimulus):
            gid = int(gid)
            if not self.pc.gid_exists(gid):
                continue
            section_id = count % sections_per_neuron
            cell = self._find_cell(gid)
            if cell is None or section_id >= len(cell.sections):
                continue
            sec = cell.sections[section_id]
            sec.push()
            has_extracellular = self._h.ismembrane("extracellular")
            self._h.pop_section()
            if not has_extracellular:
                continue
            pending.append((sec(0.5), np.asarray(series, dtype=np.float64)))

        if not pending:
            return

        end_step = start_step + max(len(v) for _seg, v in pending)
        base_segs = len(self._stim_segments)
        block = np.zeros((base_segs + len(pending), end_step), dtype=np.float64)
        if self._stim_block is not None:
            block[:base_segs, : self._stim_block.shape[1]] = self._stim_block
        for j, (seg, values) in enumerate(pending):
            self._stim_segments.append(seg)
            block[base_segs + j, start_step : start_step + len(values)] = values
        self._stim_block = block

        if not self._stim_registered:
            self._h.cvode.extra_scatter_gather(0, self._update_extracellular)
            self._stim_registered = True

    def _update_extracellular(self) -> None:
        block = self._stim_block
        if block is None or self._closed or not self._stim_segments:
            return
        # block columns are at stimulus dt; map the current sim step (sim dt)
        # to real time, then to a stimulus-resolution column index.
        current_time = self._stim_step * (self._dt or 0.025)
        idx = int(current_time / self._stim_dt) if self._stim_dt else 0
        if idx >= block.shape[1]:
            idx = block.shape[1] - 1
        col = block[:, idx]
        for i, seg in enumerate(self._stim_segments):
            seg.e_extracellular = float(col[i])
        self._stim_step += 1

    def _setup_opsin_stimulus(self, stimulus: Stimulus, current_time: float) -> None:
        if not self._opsin_refs:
            raise ValueError(
                "Stimulus has input_mode='irradiance' but no opsins are attached; "
                "configure the model's neuron_opsin_config()."
            )
        phi_stim = stimulus.convert_to("photon_flux")
        if phi_stim.gids is None:
            phi_stim.gids = self.active_gids()
        n_neurons = len(self.active_neuron_coordinates())
        sections_per_neuron = max(1, len(phi_stim) // max(1, n_neurons))

        if self._opsin_dt is None:
            self._opsin_dt = phi_stim.dt
        elif not math.isclose(self._opsin_dt, phi_stim.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        start_step = int(round(current_time / phi_stim.dt))

        pending: list[tuple[object, np.ndarray]] = []
        for count, (gid, series) in enumerate(phi_stim):
            gid = int(gid)
            section_id = count % sections_per_neuron
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
            self._h.cvode.extra_scatter_gather(0, self._update_opsin_phi)
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

    def set_weights(self, weights: dict) -> Self:
        from livn.types import SynapticParam

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
                        fluct, state = self.model.neuron_noise_mechanism(sec(0.5))
                        self._flucts[key] = (fluct, state)
                    self.model.neuron_noise_configure(
                        population, fluct, state, **merged
                    )
                    self._h.pop_section()
        return self

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
        ranges = self.system.cells_meta_data.population_ranges
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
                self._h.cvode.extra_scatter_gather_remove(self._update_extracellular)
            except Exception:
                logger.debug("failed to remove extracellular callback", exc_info=True)
            self._stim_registered = False
        if self._opsin_registered:
            try:
                self._h.cvode.extra_scatter_gather_remove(self._update_opsin_phi)
            except Exception:
                logger.debug("failed to remove opsin callback", exc_info=True)
            self._opsin_registered = False

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
        self._stim_block = None
        self._stim_dt = None
        self._stim_step = 0
        self._opsin_pps = []
        self._opsin_block = None
        self._opsin_dt = None
        self._opsin_step = 0
        return self

    def close(self) -> Self:
        if self._closed:
            return self
        self._closed = True
        import gc

        # Detach the cvode callbacks before any section they reference is freed.
        self._unregister_stim_callback()
        self._stim_segments = []
        self._stim_block = None
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
