import gc
from typing import TYPE_CHECKING, Optional, Union

import brian2 as b2
import numpy as np
from numpy.random import RandomState

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.types import SynapticParam
from livn.utils import P

b2.prefs.codegen.target = "numpy"

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model


class Env(EnvProtocol):
    def __init__(
        self,
        system: Union["System", str],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ):
        from livn.system import CachedSystem

        if model is None:
            from livn.models.izhikevich import Izhikevich

            model = Izhikevich()

        self.system = (
            system if not isinstance(system, str) else CachedSystem(system, comm=comm)
        )
        if io is None:
            io = self.system.default_io()
        self.model = model
        self.io = io

        self.comm = comm
        self.subworld_size = subworld_size

        self.prng = RandomState(seed)

        self._populations = {}
        self._synapses = {}
        self._spike_monitors = {}
        self._voltage_monitors = {}
        self._voltage_monitors_dt = {}
        self._noise_ops = set()
        self._network = b2.Network()

        self.t = 0

    @property
    def population_ranges(self):
        return self.system.cells_meta_data.population_ranges

    def init(self):
        self._load_cells()
        self._load_connections()
        self._set_delays()

        self.set_noise(0, 0)  # force noise op init

        return self

    def _load_cells(self):
        population_ranges = self.system.cells_meta_data.population_ranges
        for population_name in self.system.cells_meta_data.population_names:
            n = self.system.cells_meta_data.population_count(population_name)
            offset = population_ranges[population_name][0]

            population = self.model.brian2_population_group(
                population_name=population_name,
                n=n,
                offset=offset,
                coordinates=self.system.coordinate_array(population_name),
                prng=self.prng,
            )

            population.add_attribute("kind")
            population.add_attribute("gid_offset")
            population.gid_offset = offset
            population.kind = "excitatory" if population_name == "EXC" else "inhibitory"

            self._network.add(population)

            self._populations[population_name] = population

        return self

    def _load_connections(self):
        for post, v in self.system.connections_config["synapses"].items():
            for pre, synapse in v.items():
                S = self.model.brian2_connection_synapse(
                    self._populations[pre], self._populations[post]
                )
                S.add_attribute("kind")
                S.kind = synapse["type"]

                population_ranges = self.system.cells_meta_data.population_ranges

                all_i = []
                all_j = []
                all_multipliers = []
                all_distances = []

                for post_gid, (pre_gids, projection) in self.system.projection_array(
                    pre, post
                ):
                    distances = projection
                    if isinstance(projection, dict):
                        distances = projection["Connections"][0]

                    # filter autapses
                    autapse = pre_gids == post_gid
                    distances = distances[~autapse]
                    pre_gids = pre_gids[~autapse]

                    if not getattr(self.model, "reduced", True):
                        all_i.append(pre_gids - population_ranges[pre][0])
                        j = post_gid - population_ranges[post][0]
                        all_j.append(np.full_like(pre_gids, j))
                        all_multipliers.append(
                            np.random.uniform(0, 1, size=pre_gids.size).reshape(
                                pre_gids.shape
                            )
                        )
                        all_distances.append(distances)
                    else:
                        # multiplier connectivity
                        q_values = pre_gids - population_ranges[pre][0]
                        unique_q, inverse_indices = np.unique(
                            q_values, return_inverse=True
                        )
                        multiplier = np.bincount(inverse_indices) / 10000.0

                        distances = distances[
                            np.sort(np.unique(inverse_indices, return_index=True)[1])
                        ]

                        all_i.append(unique_q)
                        all_j.append(
                            np.full_like(
                                unique_q, post_gid - population_ranges[post][0]
                            )
                        )
                        all_multipliers.append(multiplier)
                        all_distances.append(distances)

                all_i = np.concatenate(all_i).astype(np.int32)
                all_j = np.concatenate(all_j).astype(np.int32)
                all_multipliers = np.concatenate(all_multipliers)
                all_distances = np.concatenate(all_distances)

                S.connect(i=all_i, j=all_j)
                S.multiplier[:] = all_multipliers
                S.distance[:] = all_distances

                S.prefix = 1.0 if synapse["type"] == "excitatory" else -1.0
                S.w[:] = 0

                # S.delay = 1 * b2.ms

                self._synapses[(post, pre)] = S

                self._network.add(S)

        return self

    def _set_delays(self, velocity=1.0, diffusion=1.0):
        for S in self._synapses.values():
            if velocity == 0:
                S.delay = 0 * b2.ms
                continue
            distances = S.distance * b2.um
            delays = distances / (velocity * b2.metre / b2.second)
            S.delay = delays + diffusion * b2.ms

        return self

    def set_weights(self, weights):
        for k, v in weights.items():
            param = SynapticParam.from_string(k)
            if param.sec_type is not None:
                print(f"Warning: brian2 backend does not support sections ({k})")
            # key: (post, pre)
            self._synapses[(param.population, param.source)].w = v

        return self

    def set_noise(self, exc: float = 1.0, inh: float = 1.0):
        if not self._noise_ops:
            for population in self._populations.values():
                self._noise_ops.add(self.model.brian2_noise_op(population, self.prng))

        for population in self._populations.values():
            self.model.brian2_noise_configure(
                population, level=exc if population.kind == "excitatory" else inh
            )

        return self

    def record_spikes(self, population: str | list | tuple | None = None):
        if population is None:
            population = self.system.populations
        if isinstance(population, (list, tuple)):
            for p in population:
                self.record_spikes(p)
            return self

        self._spike_monitors[population] = monitor = b2.SpikeMonitor(
            self._populations[population]
        )
        self._network.add(monitor)

        return self

    def record_voltage(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ):
        if population is None:
            population = self.system.populations
        if isinstance(population, (list, tuple)):
            for p in population:
                self.record_voltage(p, dt=dt)
            return self

        self._voltage_monitors[population] = monitor = b2.StateMonitor(
            self._populations[population],
            "v",
            record=True,
            dt=dt * b2.ms,
        )

        self._voltage_monitors_dt[population] = dt
        self._network.add(monitor)

        return self

    def run(
        self,
        duration,
        stimulus: Stimulus | None = None,
        dt: float = 0.025,
        **kwargs,
    ):
        if kwargs.get("root_only", True):
            if not P.is_root():
                raise RuntimeError(
                    "The brian2 backend does not support MPI parallelization on multiple ranks."
                )

        b2.defaultclock.dt = dt * b2.ms

        if stimulus is None:
            stimulus = Stimulus()
            stimulus.array = np.zeros(
                [
                    int((self.t + duration) / stimulus.dt),
                    len(self.system.gids),
                ]
            )

        stimulus = Stimulus.from_arg(stimulus)

        if stimulus.array.shape[0] < (self.t + duration) / stimulus.dt:
            # left-pad with zeros
            padding = np.zeros(
                (
                    int((self.t + duration) / stimulus.dt) - stimulus.array.shape[0],
                    stimulus.array.shape[1],
                )
            )
            stimulus.array = np.vstack((padding, stimulus))

        t_start = self.t
        self._network.run(
            duration * b2.ms,
            namespace={
                "stim": b2.TimedArray(
                    stimulus.array * b2.mV,
                    dt=stimulus.dt * b2.ms,
                )
            },
        )
        self.t += duration

        gids = []
        vv = []
        for population, monitor in self._voltage_monitors.items():
            gids.append(
                np.arange(
                    monitor.source.gid_offset,
                    monitor.source.gid_offset + len(monitor.source),
                )
            )
            vv.append(
                monitor.v[:, int(t_start / self._voltage_monitors_dt[population]) :]
                / b2.mV
            )

        def concat(a):
            if len(a) == 1:
                return a[0]

            if len(a) > 1:
                return np.concatenate(a)

            return None

        ii = []
        tt = []
        for population, monitor in self._spike_monitors.items():
            ts = monitor.t / b2.ms
            ii.append(monitor.i[ts >= t_start] + monitor.source.gid_offset)
            tt.append(ts[ts >= t_start] - t_start)

        return concat(ii), concat(tt), concat(gids), concat(vv)

    def clear(self):
        self.t = 0
        self.clear_monitors()

        return self

    def clear_monitors(self):
        spms = list(self._spike_monitors.keys())
        vms = list(self._voltage_monitors.keys())

        self._spike_monitors = {}
        self._voltage_monitors = {}

        gc.collect()

        for p in spms:
            self.record_spikes(p)

        for p in vms:
            self.record_voltage(p, dt=self._voltage_monitors_dt.get(p, 0.1))

    def reinit(self):
        self.clear()

        self._network = None
        gc.collect()

        self._network = b2.Network()
        self._populations = {}
        self._synapses = {}
        self._noise_ops = set()

        gc.collect()

        self.init()
