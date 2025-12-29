import gc
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
from miv_simulator import cells
from mpi4py import MPI
from neuroh5.io import (
    read_projection_names,
    read_cell_attribute_info,
    scatter_read_cell_attributes,
    scatter_read_cell_attribute_selection,
)
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
        if io is None and self.system is not None:
            io = self.system.default_io()
        self.model = model
        self.io = io

        self._original_get_reduced_cell_constructor = getattr(
            cells, "get_reduced_cell_constructor", None
        )
        self._closed = False

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
        io_size: int = 1

        if self.rank == 0:
            logger.info("*** Creating cells...")
        st = time.time()

        try:
            self.pc.gid_clear()
        except Exception:
            logger.debug(
                "ParallelContext gid_clear before cell load failed", exc_info=True
            )

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
            if not target or not target.startswith("@"):
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
        io_size: int = 1

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
                "use_cell_attr_gen": True,
                "cell_attr_gen_cache_size": 4,
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

        self.input_sources = this.microcircuit_input_sources
        self.node_allocation = this.node_allocation

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

        self.t_rec.record(h._ref_t)

        self._clear()
        h.v_init = -75.0
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

        # collect voltages
        if len(self.v_recs) > 0:
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
        currents = np.zeros((T, len(self.i_recs)), dtype=np.float32)
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
            currents[:, idx * sections_per_neuron + sec_id] = arr
            im[idx * sections_per_neuron + sec_id] = gid

        return ii, tt, iv, v, im, currents

    def set_weights(self, weights):
        params = []
        for k, v in weights.items():
            try:
                params.append((SynapticParam.from_string(k), v))
            except ValueError:
                pass

        self.this.__dict__.update({"phenotype_dict": {}, "cache_queries": True})
        update_network_params(self.this, params)

        return self

    def set_noise(self, **params):
        if not hasattr(self.model, "neuron_noise_mechanism"):
            if self.rank == 0:
                print(f"Model {self.model} does not support noise setter")
            return self

        for population, pop_cells in self.cells.items():
            for gid, cell in pop_cells.items():
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
                        population, fluct, state, **params
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
        io_size: int = 1,
        microcircuit_inputs: bool = True,
        n_trials: int = 1,
        equilibration_duration: float = 250.0,
    ):
        rank = self.comm.Get_rank()

        if rank == 0:
            logger.info(f"*** Stimulus onset is {onset} ms")

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
        input_file_path = self.system.files["cells"]

        trial_index_attr = "Trial Index"
        trial_dur_attr = "Trial Duration"

        # VecStim populations
        for pop_name in sorted(celltypes.keys()):
            if "spike train" not in celltypes[pop_name]:
                continue

            vecstim_namespace = celltypes[pop_name]["spike train"].get("namespace")
            vecstim_attr = celltypes[pop_name]["spike train"].get("attribute")

            has_vecstim = False
            vecstim_source_loc = []

            if spike_input_attribute_info is not None:
                if pop_name in spike_input_attribute_info:
                    if namespace in spike_input_attribute_info[pop_name]:
                        has_vecstim = True
                        vecstim_source_loc.append((filepath, namespace, attribute))

            if self.system.cells_meta_data.cell_attribute_info is not None:
                if vecstim_namespace is not None:
                    if pop_name in self.system.cells_meta_data.cell_attribute_info:
                        if (
                            vecstim_namespace
                            in self.system.cells_meta_data.cell_attribute_info[pop_name]
                        ):
                            has_vecstim = True
                            vecstim_source_loc.append(
                                (input_file_path, vecstim_namespace, vecstim_attr)
                            )

            if not has_vecstim:
                continue

            for input_path, input_ns, input_attr in vecstim_source_loc:
                if rank == 0:
                    logger.info(
                        f"*** Initializing stimulus population {pop_name} from input path {input_path} namespace {input_ns}"
                    )

                if self.node_allocation is None:
                    cell_vecstim_dict = scatter_read_cell_attributes(
                        input_path,
                        pop_name,
                        namespaces=[input_ns],
                        mask={
                            input_attr,
                            vecstim_attr,
                            trial_index_attr,
                            trial_dur_attr,
                        },
                        comm=self.comm,
                        io_size=io_size,
                        return_type="tuple",
                    )
                    vecstim_iter, vecstim_attr_info = cell_vecstim_dict[input_ns]
                else:
                    selection = list(self.artificial_cells.get(pop_name, {}).keys())
                    (
                        vecstim_iter,
                        vecstim_attr_info,
                    ) = scatter_read_cell_attribute_selection(
                        input_path,
                        pop_name,
                        selection,
                        namespace=input_ns,
                        mask={
                            input_attr,
                            vecstim_attr,
                            trial_index_attr,
                            trial_dur_attr,
                        },
                        comm=self.comm,
                        io_size=io_size,
                        return_type="tuple",
                    )

                vecstim_attr_index = vecstim_attr_info.get(vecstim_attr, None)
                trial_index_attr_index = vecstim_attr_info.get(trial_index_attr, None)
                trial_dur_attr_index = vecstim_attr_info.get(trial_dur_attr, None)

                for gid, vecstim_tuple in vecstim_iter:
                    if not self.pc.gid_exists(gid):
                        continue

                    cell = self.artificial_cells[pop_name][gid]

                    spiketrain = vecstim_tuple[vecstim_attr_index]
                    trial_duration = None
                    trial_index = None
                    if trial_index_attr_index is not None:
                        trial_index = vecstim_tuple[trial_index_attr_index]
                        trial_duration = vecstim_tuple[trial_dur_attr_index]

                    if len(spiketrain) > 0:
                        spiketrain = self._merge_spiketrain_trials(
                            spiketrain,
                            trial_index,
                            trial_duration,
                            n_trials,
                        )
                        spiketrain += equilibration_duration + onset

                        if len(spiketrain) > 0:
                            cell.play(h.Vector(spiketrain.astype(np.float64)))
                            if rank == 0:
                                logger.info(
                                    f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                )

        gc.collect()

        if microcircuit_inputs and self.input_sources is not None:
            for pop_name in sorted(self.input_sources.keys()):
                gid_range = self.input_sources.get(pop_name, set())
                this_gid_range = gid_range

                has_spike_train = False
                spike_input_source_loc = []

                if spike_input_attribute_info is not None:
                    if pop_name in spike_input_attribute_info:
                        if namespace in spike_input_attribute_info[pop_name]:
                            has_spike_train = True
                            spike_input_source_loc.append((filepath, namespace))

                if self.system.cells_meta_data.cell_attribute_info is not None:
                    if pop_name in self.system.cells_meta_data.cell_attribute_info:
                        if (
                            namespace
                            in self.system.cells_meta_data.cell_attribute_info[pop_name]
                        ):
                            has_spike_train = True
                            spike_input_source_loc.append((input_file_path, namespace))

                if rank == 0:
                    logger.info(
                        f"*** Initializing input source {pop_name} from locations {spike_input_source_loc}"
                    )

                if has_spike_train:
                    vecstim_attr_set = {"t", trial_index_attr, trial_dur_attr}
                    if attribute is not None:
                        vecstim_attr_set.add(attribute)
                    if pop_name in celltypes:
                        if "spike train" in celltypes[pop_name]:
                            vecstim_attr_set.add(
                                celltypes[pop_name]["spike train"]["attribute"]
                            )

                    cell_spikes_items = []
                    for input_path, input_ns in spike_input_source_loc:
                        item = scatter_read_cell_attribute_selection(
                            input_path,
                            pop_name,
                            list(this_gid_range),
                            namespace=input_ns,
                            mask=vecstim_attr_set,
                            comm=self.comm,
                            io_size=io_size,
                            return_type="tuple",
                        )
                        cell_spikes_items.append(item)

                    for cell_spikes_iter, cell_spikes_attr_info in cell_spikes_items:
                        if len(cell_spikes_attr_info) == 0:
                            continue

                        trial_index_attr_index = cell_spikes_attr_info.get(
                            trial_index_attr, None
                        )
                        trial_dur_attr_index = cell_spikes_attr_info.get(
                            trial_dur_attr, None
                        )

                        if (attribute is not None) and (
                            attribute in cell_spikes_attr_info
                        ):
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                attribute, None
                            )
                        elif "t" in cell_spikes_attr_info.keys():
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                "t", None
                            )
                        elif "Spike Train" in cell_spikes_attr_info.keys():
                            spike_train_attr_index = cell_spikes_attr_info.get(
                                "Spike Train", None
                            )
                        elif len(this_gid_range) > 0:
                            raise RuntimeError(
                                f"apply_stimulus_from_h5: unable to determine spike train attribute for population {pop_name} in spike input file {filepath}; "
                                f"namespace {namespace}; attr keys {list(cell_spikes_attr_info.keys())}"
                            )
                        else:
                            continue

                        for gid, cell_spikes_tuple in cell_spikes_iter:
                            if not self.pc.gid_exists(gid):
                                continue
                            if gid not in self.artificial_cells[pop_name]:
                                logger.warning(
                                    f"apply_stimulus_from_h5: Rank {rank}: gid {gid} not in artificial_cells[{pop_name}]"
                                )
                                continue

                            input_cell = self.artificial_cells[pop_name][gid]

                            spiketrain = cell_spikes_tuple[spike_train_attr_index]
                            trial_index = None
                            trial_duration = None
                            if trial_index_attr_index is not None:
                                trial_index = cell_spikes_tuple[trial_index_attr_index]
                                trial_duration = cell_spikes_tuple[trial_dur_attr_index]

                            if len(spiketrain) > 0:
                                spiketrain = self._merge_spiketrain_trials(
                                    spiketrain,
                                    trial_index,
                                    trial_duration,
                                    n_trials,
                                )
                                spiketrain += equilibration_duration + onset

                                if len(spiketrain) > 0:
                                    input_cell.play(
                                        h.Vector(spiketrain.astype(np.float64))
                                    )
                                    if rank == 0:
                                        logger.info(
                                            f"*** Spike train for {pop_name} gid {gid} is of length {len(spiketrain)} ({spiketrain[0]} : {spiketrain[-1]} ms)"
                                        )
                else:
                    if rank == 0 and len(this_gid_range) > 0:
                        logger.warning(
                            f"No spike train data found for population {pop_name} in spike input file {filepath}; "
                            f"namespace: {namespace}"
                        )

        gc.collect()

        return self

    @staticmethod
    def _merge_spiketrain_trials(
        spiketrain: np.ndarray,
        trial_index: np.ndarray,
        trial_duration: np.ndarray,
        n_trials: int,
    ) -> np.ndarray:
        if (trial_index is not None) and (trial_duration is not None):
            trial_spiketrains = []
            for trial_i in range(n_trials):
                trial_spiketrain_i = spiketrain[np.where(trial_index == trial_i)[0]]
                trial_spiketrain_i += np.sum(trial_duration[:trial_i])
                trial_spiketrains.append(trial_spiketrain_i)
            spiketrain = np.concatenate(trial_spiketrains)
        spiketrain.sort()
        return spiketrain

    def legacy(self, **kwargs):
        class _binding:
            pass

        this = _binding()

        attrs = {
            "_wrapped": self,
            "pc": self.pc,
            "comm": self.comm,
            "node_allocation": self.node_allocation,
            "cells": self.cells,
            "artificial_cells": self.artificial_cells,
            "biophys_cells": self.biophys_cells,
            "recording_sets": self.recording_sets,
            "t_vec": self.t_vec,
            "id_vec": self.id_vec,
            "t_rec": self.t_rec,
            "gidset": self.gidset,
            "spike_onset_delay": self.spike_onset_delay,
            "template_dict": self.template_dict,
            "SWC_Types": config.SWCTypesDef.__members__,
            "template_paths": [self.template_directory],
            "model_config": {
                "Random Seeds": {"Intracellular Recording Sample": self.seed}
            },
            "coordinates_ns": "Generated Coordinates",
            "gapjunctions_file_path": None,
            "gapjunctions": None,
            "recording_profile": None,
            "datasetName": "",
            "dataset_path": None,
            "dataset_prefix": "",
            "edge_count": self.edge_count,
            "microcircuit_input_sources": self.input_sources,
            "synapse_manager": self.synapse_manager,
            "phenotype_dict": {},
            "max_walltime_hours": 0.5,
            "results_write_time": 0,
            "cache_queries": False,
            "globals": {},
            "optlptbal": None,
            "optldbal": None,
            "profile_memory": False,
            "LFP_config": {},
            "simtime": None,
            "use_coreneuron": False,
            "v_init": -75.0,
            "checkpoint_interval": None,
            "nrn_timeout": 600.0,
            "mkcellstime": self.mkcellstime,
            "connectgjstime": self.connectgjstime,
            "connectcellstime": self.connectcellstime,
            "psolvetime": self.psolvetime,
            "stimulus_config": {
                "Equilibration Duration": 250.0,
                "Temporal Resolution": 50.0,
            },
            "stimulus_onset": 0,
            "spike_input_namespaces": [],
            "spike_input_path": None,
            "n_trials": 1,
            "spike_input_attr": None,
            "io_size": 1,
            "dt": 0.025,
            "spike_input_attribute_info": None,
            "cell_selection": None,
            "analysis_config": {
                "Firing Rate Inference": {
                    "BAKS Beta": 0.4196905892734352,
                    "Temporal Resolution": 10.0,
                    "Pad Duration": 1000.0,
                    "BAKS Alpha": 4.7725100028345535,
                },
                "Mutual Information": {"Spatial Resolution": 5.0},
                "Place Fields": {"Minimum Width": 10.0, "Minimum Rate": 1.0},
            },
            "netclamp_config": None,
            "use_cell_attr_gen": True,
            "cell_attr_gen_cache_size": 4,
            "cleanup": False,
        }

        if self.system:
            if hasattr(self.system, "files"):
                attrs["data_file_path"] = self.system.files.get("cells")
                attrs["connectivity_file_path"] = self.system.files.get("connections")
                attrs["forest_file_path"] = self.system.files.get("cells")

            if hasattr(self.system, "cells_meta_data"):
                attrs["cell_attribute_info"] = (
                    self.system.cells_meta_data.cell_attribute_info
                )

            if hasattr(self.system, "connections_config"):
                attrs["Populations"] = self.system.connections_config.get(
                    "population_definitions"
                )
                attrs["layers"] = self.system.connections_config.get(
                    "layer_definitions"
                )
                if "synapses" in self.system.connections_config:
                    attrs["connection_config"] = DotDict.create(
                        self.system.connections_config["synapses"]
                    )

        if self.cells_meta_data:
            attrs["celltypes"] = self.cells_meta_data.get("celltypes")

        if "microcircuit_inputs" not in kwargs:
            microcircuit_inputs = False
            if hasattr(self.model, "neuron_microcircuit_inputs"):
                microcircuit_inputs = self.model.neuron_microcircuit_inputs()
            attrs["microcircuit_inputs"] = microcircuit_inputs

        if "projection_dict" not in kwargs:
            attrs["projection_dict"] = None

        if "connection_velocity" not in kwargs:
            attrs["connection_velocity"] = defaultdict(lambda: 250)

        attrs.update(kwargs)

        this.__dict__.update(attrs)

        return this

    def _release_cell_container(self, container):
        if not container:
            return
        for pop_name, gid_map in list(container.items()):
            for gid, cell in list(gid_map.items()):
                try:
                    if hasattr(cell, "spike_detector"):
                        cell.spike_detector = None
                    if hasattr(cell, "hoc_cell"):
                        cell.hoc_cell = None
                    if hasattr(cell, "cell_obj"):
                        cell.cell_obj = None
                    if hasattr(cell, "sections"):
                        cell.sections = None
                except Exception:
                    logger.debug(
                        "Failed to release NEURON references for %s gid %s",
                        pop_name,
                        gid,
                        exc_info=True,
                    )
            gid_map.clear()
        container.clear()

    def close(self):
        if getattr(self, "_closed", False):
            return

        self._closed = True

        try:
            syn_manager = getattr(self, "synapse_manager", None)
            if syn_manager is not None and hasattr(syn_manager, "clear"):
                try:
                    syn_manager.clear()
                except Exception:
                    logger.debug("Failed to clear synapse manager", exc_info=True)
            self.synapse_manager = None

            self._release_cell_container(getattr(self, "cells", None))
            self._release_cell_container(getattr(self, "biophys_cells", None))
            self._release_cell_container(getattr(self, "artificial_cells", None))

            for attr_name in (
                "syns_set",
                "edge_count",
                "recording_sets",
                "v_recs",
                "v_recs_dt",
                "i_recs",
                "i_recs_dt",
                "i_area",
            ):
                mapping = getattr(self, attr_name, None)
                if mapping is not None:
                    mapping.clear()

            gidset = getattr(self, "gidset", None)
            if gidset is not None:
                gidset.clear()
            flucts = getattr(self, "_flucts", None)
            if flucts is not None:
                flucts.clear()
            template_dict = getattr(self, "template_dict", None)
            if template_dict is not None:
                template_dict.clear()

            for vec_name in ("t_vec", "id_vec", "t_rec"):
                vec = getattr(self, vec_name, None)
                if vec is not None:
                    try:
                        vec.resize(0)
                    except Exception:
                        logger.debug("Failed to resize %s", vec_name, exc_info=True)

            if hasattr(self, "cells_meta_data"):
                self.cells_meta_data = None
            if hasattr(self, "connections_meta_data"):
                self.connections_meta_data = None
            if hasattr(self, "input_sources"):
                self.input_sources = None
            if hasattr(self, "this"):
                self.this = None
            if hasattr(self, "node_allocation"):
                self.node_allocation = None

            pc = getattr(self, "pc", None)
            if pc is not None:
                try:
                    pc.gid_clear()
                except Exception:
                    logger.debug("ParallelContext gid_clear failure", exc_info=True)
            self.pc = None

            try:
                secs = list(h.allsec())
            except Exception:
                secs = []
            for sec in reversed(secs):
                try:
                    h.delete_section(sec=sec)
                except Exception:
                    sec_name = "<unknown>"
                    try:
                        sec_name = sec.hname()
                    except Exception:
                        pass
                    logger.debug(
                        "Unable to delete section %s during teardown",
                        sec_name,
                        exc_info=True,
                    )

            gc.collect()
        finally:
            original_ctor = getattr(
                self, "_original_get_reduced_cell_constructor", None
            )
            if original_ctor is not None:
                cells.get_reduced_cell_constructor = original_ctor
                self._original_get_reduced_cell_constructor = None

        return self

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
