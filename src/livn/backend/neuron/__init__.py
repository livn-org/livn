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
from mpi4py import MPI
from neuroh5.io import read_projection_names
from neuron import h

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.types import SynapticParam
from livn.utils import DotDict

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

        self.pc.set_maxstep(10)

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

        self.t = 0

    def _clear(self):
        self.t_vec.resize(0)
        self.id_vec.resize(0)
        self.t_rec.resize(0)
        for v_rec in self.v_recs.values():
            v_rec.resize(0)

    def init(self):
        self._load_cells()
        self._load_connections()

        # disable defaultdicts
        self.cells = dict(self.cells)
        self.artificial_cells = dict(self.artificial_cells)
        self.biophys_cells = dict(self.biophys_cells)
        self.edge_count = dict(self.edge_count)
        self.syns_set = dict(self.syns_set)

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
                "coordinates_ns": "Coordinates",
            }
        )

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
                "microcircuit_inputs": False,
                "use_cell_attr_gen": False,  # TODO
                "cleanup": False,
                "projection_dict": projection_dict,
                "Populations": self.system.connections_config["population_definitions"],
                "layers": self.system.connections_config["layer_definitions"],
                "connection_config": DotDict.create(synapses),
                "connection_velocity": defaultdict(lambda: 250),
                "SWC_Types": config.SWCTypesDef.__members__,
                "celltypes": self.cells_meta_data["celltypes"],
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

            stim = []  # prevent garbage collection
            for gid, st in zip(self.system.gids, stimulus.array.T):
                if not (self.pc.gid_exists(gid)):
                    continue

                for p in self.cells.values():
                    if gid in p:
                        cell = p[gid]
                        stim.append(h.Vector(st))
                        secs = []
                        if hasattr(cell, "soma_list"):
                            secs = cell.soma_list
                        else:
                            secs.append(cell.soma)
                        for sec in secs:
                            sec.push()
                            if h.ismembrane("extracellular"):
                                stim[-1].play(
                                    sec(0.5)._ref_e_extracellular, stimulus.dt
                                )
                            h.pop_section()

                        break

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
            logger.info(f"*** Completed finitialize")

        t_start = self.t

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
            return ii, tt, None, None

        iv = []
        v = []
        for gid, rec in self.v_recs.items():
            iv.append(gid)
            v.append(rec.as_numpy())
        iv = np.asarray(iv, dtype=np.uint32)
        v = np.array(v, dtype=np.float32)

        return ii, tt, iv, v

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
        for kind, level in [("EXC", exc), ("INH", inh)]:
            if kind not in self.cells:
                logger.info(
                    f"Rank {self.rank} has no {kind} cells; try reducing the number of ranks."
                )
                continue
            for gid, cell in self.cells[kind].items():
                if not (self.pc.gid_exists(gid)):
                    continue
                secs = []
                if hasattr(cell, "soma_list"):
                    secs = cell.soma_list
                else:
                    secs.append(cell.soma)
                for idx, sec in enumerate(secs):
                    sec.push()
                    fluct, state = self._flucts.get(f"{gid}-{idx}", (None, None))
                    if fluct is None:
                        fluct, state = self.model.neuron_noise_mechanism(sec(0.5))
                        self._flucts[f"{gid}-{idx}"] = (fluct, state)

                    self.model.neuron_noise_configure(kind, fluct, state, level)

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

            self.v_recs[gid] = h.Vector()
            secs = []
            if hasattr(cell, "soma_list"):
                secs = cell.soma_list
            else:
                secs.append(cell.soma)
            for sec in secs:
                self.v_recs[gid].record(sec(0.5)._ref_v, dt)
                break

        return self
