import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

import numpy as np
from machinable.config import to_dict
from miv_simulator import config
from miv_simulator.network import connect_cells, connect_gjs, make_cells
from miv_simulator.optimization import update_network_params
from miv_simulator.synapses import SynapseManager
from miv_simulator.utils import ExprClosure, from_yaml
from miv_simulator.utils.neuron import configure_hoc
from miv_simulator.network import init_input_cells
from miv_simulator import cells
from mpi4py import MPI
from neuroh5.io import read_projection_names, read_cell_attribute_info
from neuron import h

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.types import SynapticParam
from livn.utils import DotDict, import_object_by_path

if TYPE_CHECKING:
    from livn.io import IO
    from livn.system import System
    from livn.types import Model


if hasattr(h, "nrnmpi_init"):
    h.nrnmpi_init()


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: MPI.Intracomm | None = None,
        subworld_size: int | None = None,
    ):
        from livn.system import System

        if model is None:
            from livn.models.rcsd import ReducedCalciumSomaDendrite

            model = ReducedCalciumSomaDendrite()

        self.seed = seed

        self.system = (
            system if not isinstance(system, str) else System(system, comm=comm)
        )
        if io is None:
            io = self.system.default_io()
        self.model = model
        self.io = io

        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.subworld_size = subworld_size

        # --- Resources

        self.gidset = set()
        self.node_allocation = None  # node rank map

        # --- Statistics

        self.mkcellstime = -0.0
        self.connectgjstime = -0.0
        self.connectcellstime = -0.0
        self.psolvetime = -0.0

        # --- Graph

        self.cells = defaultdict(lambda: dict())
        self.artificial_cells = defaultdict(lambda: dict())
        self.biophys_cells = defaultdict(lambda: dict())
        self.spike_onset_delay = {}
        self.recording_sets = {}
        self.synapse_manager = None
        self.edge_count = defaultdict(dict)
        self.syns_set = defaultdict(set)

        # --- State
        self.cells_meta_data = None
        self.connections_meta_data = None
        self.input_sources = None

        # --- Compat

        self.template_dict = {}

        self._flucts = {}

        # --- Simulator
        self.template_directory = self.model.neuron_template_directory()
        self.mechanisms_directory = self.model.neuron_mechanisms_directory()
        configure_hoc(
            template_directory=self.template_directory,
            mechanisms_directory=self.mechanisms_directory,
        )

        self.pc = h.pc
        self.rank = int(self.pc.id())

        if self.subworld_size is not None:
            self.pc.subworlds(subworld_size)

        # Spike time of all cells on this host
        self.t_vec = h.Vector()
        # Ids of spike times on this host
        self.id_vec = h.Vector()
        # Timestamps of intracellular traces on this host
        self.t_rec = h.Vector()

        self.v_recs = {}
        self.v_recs_dt = {}

        # Membrane current recordings per gid
        self.i_recs = {}
        self.i_recs_dt = {}
        self.i_area = {}  # membrane surface area (cm^2) for absolute current conversion

        self.t = 0

    def _clear(self):
        self.t_vec.resize(0)
        self.id_vec.resize(0)
        self.t_rec.resize(0)
        for v_rec in self.v_recs.values():
            v_rec.resize(0)
        for i_rec in self.i_recs.values():
            i_rec.resize(0)

    def init(self):
        self._load_cells()
        self._load_connections()

        # disable defaultdicts
        self.cells = dict(self.cells)
        self.artificial_cells = dict(self.artificial_cells)
        self.biophys_cells = dict(self.biophys_cells)
        self.edge_count = dict(self.edge_count)
        self.syns_set = dict(self.syns_set)

        self.pc.set_maxstep(10)

        return self

    def _load_cells(self):
        filepath = self.system.files["cells"]
        io_size: int = 0

        if self.rank == 0:
            logger.info("*** Creating cells...")
        st = time.time()

        rank = self.comm.Get_rank()

        population_ranges = self.system.cells_meta_data.population_ranges

        celltypes = to_dict(self.system.synapses_config["cell_types"])

        self.model.neuron_celltypes(celltypes)

        typenames = sorted(celltypes.keys())
        for k in typenames:
            population_range = population_ranges.get(k, None)
            if population_range is not None:
                celltypes[k]["start"] = population_ranges[k][0]
                celltypes[k]["num"] = population_ranges[k][1]

                if "mechanism" in celltypes[k]:
                    mech_dict = celltypes[k]["mechanism"]
                    if isinstance(mech_dict, str):
                        if rank == 0:
                            mech_dict = from_yaml(mech_dict)
                        mech_dict = self.comm.bcast(mech_dict, root=0)
                    celltypes[k]["mech_dict"] = mech_dict
                    celltypes[k]["mech_file_path"] = "$mechanism"

                if "synapses" in celltypes[k]:
                    synapses_dict = celltypes[k]["synapses"]
                    if "weights" in synapses_dict:
                        weights_config = synapses_dict["weights"]
                        if isinstance(weights_config, list):
                            weights_dicts = weights_config
                        else:
                            weights_dicts = [weights_config]
                        for weights_dict in weights_dicts:
                            if "expr" in weights_dict:
                                expr = weights_dict["expr"]
                                parameter = weights_dict["parameter"]
                                const = weights_dict.get("const", {})
                                clos = ExprClosure(parameter, expr, const)
                                weights_dict["closure"] = clos
                        synapses_dict["weights"] = weights_dicts

        self.cells_meta_data = {
            "source": filepath,
            "cell_attribute_info": self.system.cells_meta_data.cell_attribute_info,
            "population_ranges": population_ranges,
            "population_names": self.system.cells_meta_data.population_names,
            "celltypes": celltypes,
        }

        class _binding:
            pass

        this = _binding()
        this.__dict__.update(
            {
                # bound
                "pc": self.pc,
                "data_file_path": filepath,
                "io_size": io_size,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "cells": self.cells,
                "artificial_cells": self.artificial_cells,
                "biophys_cells": self.biophys_cells,
                "spike_onset_delay": self.spike_onset_delay,
                "recording_sets": self.recording_sets,
                "t_vec": self.t_vec,
                "id_vec": self.id_vec,
                "t_rec": self.t_rec,
                # compat
                "gapjunctions_file_path": None,
                "gapjunctions": None,
                "recording_profile": None,
                "dt": 0.025,
                "datasetName": "",
                "gidset": self.gidset,
                "SWC_Types": config.SWCTypesDef.__members__,
                "template_paths": [self.template_directory],
                "dataset_path": None,
                "dataset_prefix": "",
                "template_dict": self.template_dict,
                "cell_attribute_info": self.system.cells_meta_data.cell_attribute_info,
                "celltypes": celltypes,
                "model_config": {
                    "Random Seeds": {"Intracellular Recording Sample": self.seed}
                },
                "coordinates_ns": "Generated Coordinates",
            }
        )

        class _Cell(cells.BiophysCell):
            def position(self, x: float, y: float, z: float) -> None:
                target = self.hoc_cell if self.hoc_cell is not None else self.cell_obj
                if target is None or not hasattr(target, "position"):
                    raise RuntimeError("cell has no position()")
                target.position(x, y, z)

        def make_cell(target):
            if not target.startswith("@"):
                return None

            def _cell(gid, pop_name, env, mech_dict):
                cell = env.biophys_cells[pop_name][gid] = _Cell(
                    gid=gid,
                    population_name=pop_name,
                    cell_obj=import_object_by_path(target[1:])(),
                    mech_dict=mech_dict,
                    env=env,
                )
                return cell

            return _cell

        cells.get_reduced_cell_constructor = make_cell

        make_cells(this)

        # HACK given its initial `None` primitive data type, the
        #  env.node_allocation copy at the end of make_cells will
        #  be lost when the local function stack is freed;
        #  fortunately, gidid is heap-allocated so we can
        #  simply repeat the set operation here
        self.node_allocation = set()
        for gid in self.gidset:
            self.node_allocation.add(gid)

        self.mkcellstime = time.time() - st
        if self.rank == 0:
            logger.info(f"*** Cells created in {self.mkcellstime:.02f} s")
        local_num_cells = sum(len(cells) for cells in self.cells.values())

        logger.info(f"*** Rank {self.rank} created {local_num_cells} cells")

        st = time.time()

        connect_gjs(this)

        self.pc.setup_transfer()
        self.connectgjstime = time.time() - st
        if rank == 0:
            logger.info(f"*** Gap junctions created in {self.connectgjstime:.02f} s")

    def _load_connections(self):
        synapses = self.system.connections_config["synapses"]
        filepath = self.system.files["connections"]
        cell_filepath = self.system.files["cells"]
        io_size: int = 0

        microcircuit_inputs = False
        if hasattr(self.model, "neuron_microcircuit_inputs"):
            microcircuit_inputs = self.model.neuron_microcircuit_inputs()
        if not self.cells_meta_data:
            raise RuntimeError("Please load the cells first using load_cells()")

        st = time.time()
        if self.rank == 0:
            logger.info("*** Creating connections:")

        rank = self.comm.Get_rank()
        if rank == 0:
            color = 1
        else:
            color = 0
        comm0 = self.comm.Split(color, 0)

        projection_dict = None
        if rank == 0:
            projection_dict = defaultdict(list)
            for src, dst in read_projection_names(filepath, comm=comm0):
                projection_dict[dst].append(src)
            projection_dict = dict(projection_dict)
            logger.info(f"projection_dict = {str(projection_dict)}")
        projection_dict = self.comm.bcast(projection_dict, root=0)
        comm0.Free()

        self.input_sources = {
            pop_name: set() for pop_name in self.cells_meta_data["celltypes"].keys()
        }

        class _binding:
            pass

        self.this = this = _binding()
        this.__dict__.update(
            {
                "pc": self.pc,
                "connectivity_file_path": filepath,
                "forest_file_path": cell_filepath,
                "io_size": io_size,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "edge_count": self.edge_count,
                "biophys_cells": self.biophys_cells,
                "gidset": self.gidset,
                "recording_sets": self.recording_sets,
                "microcircuit_inputs": microcircuit_inputs,
                "microcircuit_input_sources": self.input_sources,
                "spike_input_attribute_info": None,
                "cell_selection": None,
                "netclamp_config": None,
                "use_cell_attr_gen": False,
                "cleanup": False,
                "projection_dict": projection_dict,
                "Populations": self.system.connections_config["population_definitions"],
                "layers": self.system.connections_config["layer_definitions"],
                "connection_config": DotDict.create(synapses),
                "connection_velocity": defaultdict(lambda: 250),
                "SWC_Types": config.SWCTypesDef.__members__,
                "celltypes": self.cells_meta_data["celltypes"],
                "cells": self.cells,
                "artificial_cells": self.artificial_cells,
                "dt": 0.025,  # TODO: hoist into run
                "t_vec": self.t_vec,
                "id_vec": self.id_vec,
                "t_rec": self.t_rec,
            }
        )
        self.synapse_manager = SynapseManager(
            this,
            self.model.neuron_synapse_mechanisms(),
            self.model.neuron_synapse_rules(),
        )
        this.__dict__["synapse_manager"] = self.synapse_manager

        connect_cells(this)

        self.pc.set_maxstep(10.0)

        self.connectcellstime = time.time() - st

        if self.rank == 0:
            logger.info(
                f"*** Done creating connections: time = {self.connectcellstime:.02f} s"
            )
        edge_count = int(sum(self.edge_count[dest] for dest in self.edge_count))
        logger.info(f"*** Rank {rank} created {edge_count} connections")

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float = 0.025,
        **kwargs,
    ):
        if stimulus is not None:
            stimulus = Stimulus.from_arg(stimulus)

            if stimulus.gids is None:
                stimulus.gids = self.system.gids

            stim = []  # prevent garbage collection
            sections_per_neuron = len(stimulus) // len(self.system.neuron_coordinates)
            for count, (gid, st) in enumerate(stimulus):
                if not (self.pc.gid_exists(gid)):
                    continue

                section_id = count % sections_per_neuron

                cell = self.pc.gid2cell(gid)
                if "VecStim" in getattr(cell, "hname", lambda: "")():
                    # artificial STIM cell
                    print(
                        f"Warning: Cell {gid} is a VecStim cell; use play() to induce spikes. Skipping stimulus ..."
                    )
                    continue

                stim.append(h.Vector(st))

                sec = cell.sections[section_id]
                sec.push()
                if h.ismembrane("extracellular"):
                    stim[-1].play(sec(0.5)._ref_e_extracellular, stimulus.dt)
                h.pop_section()

        verbose = self.rank == 0 and kwargs.get("verbose", True)

        if verbose:
            logger.info("*** finitialize")
        self._clear()
        h.v_init = -65
        h.stdinit()
        h.secondorder = 2  # crank-nicholson
        h.dt = dt
        self.pc.timeout(600.0)

        h.finitialize(h.v_init)
        h.finitialize(h.v_init)

        if verbose:
            logger.info("*** Completed finitialize")

        if verbose:
            logger.info(f"*** Simulating {duration} ms")

        q = time.time()
        self.pc.psolve(duration)
        self.psolvetime = time.time() - q

        self.t += duration

        if verbose:
            logger.info(f"*** Done simulating within {self.psolvetime:.2f} s")

        # collect spikes
        tt = self.t_vec.as_numpy()
        ii = np.asarray(self.id_vec.as_numpy(), dtype=np.uint32)

        if len(self.v_recs) == 0:
            # collect voltages
            iv = []
            v = []
            for (gid, sec_id), rec in self.v_recs.items():
                iv.append(gid)
                v.append(rec.as_numpy())
            iv = np.asarray(iv, dtype=np.uint32)
            v = np.array(v, dtype=np.float32)
        else:
            iv = None
            v = None

        # collect membrane currents
        if len(self.i_recs) == 0:
            return ii, tt, iv, v, None, None

        gids = self.system.gids
        sections_per_neuron = len(self.i_recs) // len(gids)
        gid_to_index = {int(g): idx for idx, g in enumerate(gids)}
        any_rec = next(iter(self.i_recs.values()))
        T = len(any_rec)
        currents = np.zeros((len(self.i_recs), T), dtype=np.float32)
        im = np.ones([len(self.i_recs)], dtype=np.int32) * -1

        for (gid, sec_id), rec in self.i_recs.items():
            idx = gid_to_index.get(int(gid))
            if idx is None:
                continue
            arr = rec.as_numpy()
            # Convert current density (mA/cm^2) to absolute current (μA):
            # I_μA = (i_membrane_mA_per_cm2) * (area_cm2) * 1000
            area_cm2 = float(self.i_area.get((int(gid), sec_id), 0.0))
            if area_cm2 > 0.0:
                arr = arr * area_cm2 * 1000.0
            if len(arr) != T:
                # pad or truncate to T
                if len(arr) < T:
                    pad = np.zeros(T, dtype=np.float32)
                    pad[: len(arr)] = arr
                    arr = pad
                else:
                    arr = arr[:T]
            currents[idx * sections_per_neuron + sec_id, :] = arr
            im[idx * sections_per_neuron + sec_id] = gid

        return ii, tt, iv, v, im, currents

    def set_weights(self, weights):
        params = []
        for k, v in weights.items():
            try:
                params.append((SynapticParam.from_string(k), v))
            except ValueError:
                pass

        self.this.__dict__.update({"phenotype_dict": {}, "cache_queries": False})
        update_network_params(self.this, params)

        return self

    def set_noise(self, exc: float = 1.0, inh: float = 1.0):
        for population, cells in self.cells.items():
            for gid, cell in cells.items():
                if not (self.pc.gid_exists(gid)):
                    continue
                secs = []
                if hasattr(cell, "soma_list"):
                    secs = cell.soma_list
                elif hasattr(cell, "soma"):
                    secs.append(cell.soma)

                for idx, sec in enumerate(secs):
                    sec.push()
                    fluct, state = self._flucts.get(f"{gid}-{idx}", (None, None))
                    if fluct is None:
                        fluct, state = self.model.neuron_noise_mechanism(sec(0.5))
                        self._flucts[f"{gid}-{idx}"] = (fluct, state)

                    self.model.neuron_noise_configure(
                        population, fluct, state, exc, inh
                    )

                    h.pop_section()

        return self

    def _record_voltage(self, population: str, dt: float) -> "Env":
        if population not in self.cells:
            logger.info(
                f"Rank {self.rank} has no cells; try reducing the number of ranks."
            )
            return

        self.v_recs_dt[population] = dt

        for gid, cell in self.cells[population].items():
            if not (self.pc.gid_exists(gid)):
                continue

            if "VecStim" in getattr(cell, "hname", lambda: "")():
                continue

            for sec_id, sec in enumerate(cell.sections):
                self.v_recs[(int(gid), sec_id)] = h.Vector()
                self.v_recs[(int(gid), sec_id)].record(sec(0.5)._ref_v, dt)

        return self

    def _record_membrane_current(self, population: str, dt: float) -> "Env":
        """Record transmembrane current (i_membrane) at soma midpoint for each cell"""
        if population not in self.cells:
            logger.info(
                f"Rank {self.rank} has no cells; try reducing the number of ranks."
            )
            return

        try:
            h.cvode.use_fast_imem(1)
        except Exception:
            pass

        self.i_recs_dt[population] = dt

        for gid, cell in self.cells[population].items():
            if not (self.pc.gid_exists(gid)):
                continue

            if "VecStim" in getattr(cell, "hname", lambda: "")():
                continue

            for sec_id, sec in enumerate(cell.sections):
                self.i_recs[(int(gid), sec_id)] = h.Vector()
                self.i_recs[(int(gid), sec_id)].record(sec(0.5)._ref_i_membrane, dt)
                area_um2 = h.area(0.5, sec=sec)
                self.i_area[(int(gid), sec_id)] = float(area_um2) * 1e-8

        return self

    def apply_stimulus_from_h5(
        self,
        filepath: str,
        namespace: str,
        attribute: str = "Spike Train",
        onset: int = 0,
        io_size: int = 10,
        microcircuit_inputs: bool = True,
    ):
        rank = self.comm.Get_rank()
        if rank == 0:
            color = 1
        else:
            color = 0
        comm0 = self.comm.Split(color, 0)

        spike_input_attribute_info = None
        if rank == 0:
            spike_input_attribute_info = read_cell_attribute_info(
                filepath,
                sorted(self.system.connections_config["population_definitions"].keys()),
                comm=comm0,
            )

        spike_input_attribute_info = self.comm.bcast(spike_input_attribute_info, root=0)

        comm0.Free()

        celltypes = self.cells_meta_data["celltypes"]
        vecstim_pops: list[str] = []
        for pop_name, pop_config in celltypes.items():
            if pop_config.get("template") != "VecStim":
                continue

            vecstim_pops.append(pop_name)
            spike_cfg = pop_config.setdefault("spike train", {})
            if spike_cfg.get("namespace") != namespace:
                spike_cfg["namespace"] = namespace
            if spike_cfg.get("attribute") != attribute:
                spike_cfg["attribute"] = attribute

        if self.input_sources is None:
            self.input_sources = {}
        else:
            for pop_name in vecstim_pops:
                self.input_sources.pop(pop_name, None)

        has_micro_inputs = any(bool(gid_set) for gid_set in self.input_sources.values())
        has_micro_inputs = self.comm.allreduce(has_micro_inputs, op=MPI.LOR)
        microcircuit_inputs = microcircuit_inputs and has_micro_inputs

        class _binding:
            pass

        this = _binding()
        this.__dict__.update(
            {
                "pc": self.pc,
                "comm": self.comm,
                "node_allocation": self.node_allocation,
                "cells": self.cells,
                "artificial_cells": self.artificial_cells,
                "biophys_cells": self.biophys_cells,
                "stimulus_config": {},
                "stimulus_onset": onset,
                "data_file_path": self.system.files["cells"],
                "celltypes": celltypes,
                "microcircuit_inputs": microcircuit_inputs,
                "microcircuit_input_sources": self.input_sources,
                "cell_selection": None,
                "spike_input_attribute_info": spike_input_attribute_info,
                "spike_input_namespaces": [namespace],
                "spike_input_path": filepath,
                "n_trials": 1,
                "cell_attribute_info": self.system.cells_meta_data.cell_attribute_info,
                "spike_input_attr": attribute,
                "io_size": io_size,
            }
        )

        init_input_cells(this)
