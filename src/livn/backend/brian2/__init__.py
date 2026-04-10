import gc
from typing import TYPE_CHECKING, Optional, Union

import brian2 as b2
import numpy as np
from numpy.random import RandomState

from livn.stimulus import Stimulus
from livn.types import Env as EnvProtocol
from livn.types import SynapticParam
from livn.utils import P

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

        self.system = (
            system if not isinstance(system, str) else CachedSystem(system, comm=comm)
        )
        if model is None:
            model = self.system.default_model()
        self.model = model
        if io is None:
            io = self.system.default_io()
        self.io = io

        self.comm = comm
        self.subworld_size = subworld_size

        self.encoding = None
        self.decoding = None

        self.prng = RandomState(seed)
        if seed is not None:
            b2.seed(seed)

        self._populations = {}
        self._synapses = {}
        self._spike_monitors = {}
        self._voltage_monitors = {}
        self._voltage_monitors_dt = {}
        self._membrane_monitors = {}
        self._membrane_monitors_dt = {}
        self._noise_ops = set()
        self._network = b2.Network()

        self.t = 0

    @property
    def network(self):
        return self._network

    @property
    def voltage_recording_dt(self) -> float:
        if self._voltage_monitors_dt:
            return next(iter(self._voltage_monitors_dt.values()))
        return super().voltage_recording_dt

    @property
    def membrane_current_recording_dt(self) -> float:
        if self._membrane_monitors_dt:
            return next(iter(self._membrane_monitors_dt.values()))
        return super().membrane_current_recording_dt

    @property
    def population_ranges(self):
        return self.system.cells_meta_data.population_ranges

    def init(self):
        self._load_cells()
        self._load_connections()
        self._set_delays()

        self.set_noise({})  # force noise op init

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

    def set_noise(self, noise: dict):
        if not self._noise_ops:
            for population in self._populations.values():
                op = self.model.brian2_noise_op(population, self.prng)
                if op is not None:
                    self._noise_ops.add(op)

        if noise:
            for population in self._populations.values():
                self.model.brian2_noise_configure(population, **noise)

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

    def _record_membrane_current(self, population: str, dt: float) -> "Env":
        if population not in self._populations:
            return self

        pop = self._populations[population]
        has_compartments = "I_memb_s" in pop.variables and "I_memb_d" in pop.variables

        if has_compartments:
            monitor_s = b2.StateMonitor(pop, "I_memb_s", record=True, dt=dt * b2.ms)
            monitor_d = b2.StateMonitor(pop, "I_memb_d", record=True, dt=dt * b2.ms)
            self._membrane_monitors[population] = (monitor_s, monitor_d)
            self._network.add(monitor_s)
            self._network.add(monitor_d)
        else:
            monitor = b2.StateMonitor(pop, "I", record=True, dt=dt * b2.ms)
            self._membrane_monitors[population] = monitor
            self._network.add(monitor)

        self._membrane_monitors_dt[population] = dt
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

        # Check for stimulus dt consistency across continued runs
        if not hasattr(self, "_stimulus_dt"):
            self._stimulus_dt = stimulus.dt
        elif not np.isclose(self._stimulus_dt, stimulus.dt, rtol=0.0, atol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        # Left-pad stimulus to align with brian2's absolute time reference.
        # Brian2's TimedArray indexes from t=0 (simulation start), so for
        # continued runs (self.t > 0) the stimulus must be padded.
        pad_rows = max(0, int(round(self.t / stimulus.dt)))
        if pad_rows > 0:
            padding = np.zeros((pad_rows, stimulus.array.shape[1]))
            stimulus.array = np.vstack((padding, stimulus.array))

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

        # membrane currents aligned to global gid order if enabled
        if len(self._membrane_monitors) == 0:
            return concat(ii), concat(tt), concat(gids), concat(vv), None, None

        # [T, n_neurons] matrix aligned to system.neuron_coordinates
        coords = getattr(self.system, "neuron_coordinates", None)
        if coords is None or len(coords) == 0:
            return concat(ii), concat(tt), concat(gids), concat(vv), None, None

        all_gids = np.asarray(coords)[:, 0].astype(np.uint32)
        gid_to_index = {int(g): idx for idx, g in enumerate(all_gids)}

        # Check if any population uses per-compartment recording
        has_compartments = any(
            isinstance(m, tuple) for m in self._membrane_monitors.values()
        )
        sections_per_neuron = 2 if has_compartments else 1

        # determine maximum T across monitors after slicing
        lengths = []
        per_pop_data = {}
        for population, monitor in self._membrane_monitors.items():
            start_idx = int(
                t_start / max(self._membrane_monitors_dt.get(population, 0.1), 1e-9)
            )
            if isinstance(monitor, tuple):
                # Two-compartment: (monitor_soma, monitor_dend)
                mon_s, mon_d = monitor
                data_s = np.array(mon_s.I_memb_s[:, start_idx:])
                data_d = np.array(mon_d.I_memb_d[:, start_idx:])
                per_pop_data[population] = (data_s, data_d)
                lengths.append(data_s.shape[1] if data_s.ndim == 2 else 0)
            else:
                data = (monitor.I / b2.uA)[:, start_idx:]
                per_pop_data[population] = data
                lengths.append(data.shape[1] if data.ndim == 2 else 0)

        T = int(max(lengths) if lengths else 0)
        if T == 0:
            return concat(ii), concat(tt), concat(gids), concat(vv), None, None

        n_neurons = int(len(all_gids))
        currents = np.zeros((n_neurons * sections_per_neuron, T), dtype=np.float32)

        for population, data in per_pop_data.items():
            base = int(self._populations[population].gid_offset)

            if isinstance(data, tuple):
                # Two-compartment: interleave soma/dend (soma0, dend0, soma1, dend1, ...)
                data_s, data_d = data
                if data_s.size == 0:
                    continue
                n_pop = data_s.shape[0]
                for k in range(n_pop):
                    gid = base + k
                    idx = gid_to_index.get(int(gid))
                    if idx is None:
                        continue
                    for sec_idx, sec_data in enumerate([data_s[k], data_d[k]]):
                        series = np.array(sec_data, dtype=np.float32)
                        if series.shape[0] < T:
                            pad = np.zeros(T, dtype=np.float32)
                            pad[: series.shape[0]] = series
                            series = pad
                        elif series.shape[0] > T:
                            series = series[:T]
                        currents[idx * sections_per_neuron + sec_idx, :] = series
            else:
                if data.size == 0:
                    continue
                n_pop = data.shape[0]
                for k in range(n_pop):
                    gid = base + k
                    idx = gid_to_index.get(int(gid))
                    if idx is None:
                        continue
                    series = np.array(data[k], dtype=np.float32)
                    if series.shape[0] < T:
                        pad = np.zeros(T, dtype=np.float32)
                        pad[: series.shape[0]] = series
                        series = pad
                    elif series.shape[0] > T:
                        series = series[:T]
                    currents[idx, :] = series

        return concat(ii), concat(tt), concat(gids), concat(vv), all_gids, currents

    def clear_recordings(self):
        self.clear_monitors()
        return self

    def clear(self):
        self.t = 0
        self.clear_monitors()

        return self

    def clear_monitors(self):
        spms = list(self._spike_monitors.keys())
        vms = list(self._voltage_monitors.keys())
        mms = list(self._membrane_monitors.keys())

        self._spike_monitors = {}
        self._voltage_monitors = {}
        self._membrane_monitors = {}

        gc.collect()

        for p in spms:
            self.record_spikes(p)

        for p in vms:
            self.record_voltage(p, dt=self._voltage_monitors_dt.get(p, 0.1))

        for p in mms:
            self._record_membrane_current(p, dt=self._membrane_monitors_dt.get(p, 0.1))

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

    def cleosim(self):
        """Wrap the brian2 network in a cleo CLSimulator.

        Assigns livn neuron coordinates to the brian2 NeuronGroups and
        returns a CLSimulator ready for device injection. Use env.run()
        instead of sim.run() to step the simulation.
        """
        import cleo
        from cleo.coords import assign_coords

        for name, pop in self._populations.items():
            coords_array = self.system.coordinate_array(name)  # [n, 4] = gid, x, y, z
            xyz_um = coords_array[:, 1:4]
            assign_coords(pop, xyz_um * b2.um)

        sim = cleo.CLSimulator(self._network)

        def _run_disabled(*args, **kwargs):
            raise RuntimeError(
                "Use env.run() instead of sim.run() to step the simulation. "
                "The livn environment manages time tracking, stimulus, and monitors"
            )

        sim.run = _run_disabled

        return sim
