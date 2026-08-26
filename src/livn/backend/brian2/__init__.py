import gc
import warnings
from typing import TYPE_CHECKING, Optional, Union

import brian2 as b2
import numpy as np
from numpy.random import RandomState

from livn.cells import CellRegistry
from livn.run import Run
from livn.stimulus import Stimulus
from livn.types import Capability, Cell, SynapticParam
from livn.types import Env as EnvProtocol
from livn.utils import NOISE_STREAM_STRIDE, P

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.system import System
    from livn.types import Model


def _population_parameters(group) -> list[str]:
    from brian2.equations.equations import PARAMETER

    _INTERNAL_PARAMETERS = frozenset({"lastspike", "not_refractory"})

    return sorted(
        name
        for name, equation in group.equations.items()
        if equation.type == PARAMETER and name not in _INTERNAL_PARAMETERS
    )


STIMULUS_TERMS = {
    "extracellular": ("stim_v", b2.mV),
    "current": ("stim_i", b2.nA),
}


def stimulus_namespace(stimulus, n_gids: int, duration: float) -> dict:
    driven, unit = STIMULUS_TERMS[stimulus.input_mode]
    namespace = {driven: b2.TimedArray(stimulus.array * unit, dt=stimulus.dt * b2.ms)}
    for name, other in STIMULUS_TERMS.values():
        if name != driven:
            # one row held past its end, so it costs nothing to carry
            namespace[name] = b2.TimedArray(
                np.zeros((1, n_gids)) * other,
                dt=max(duration, 1.0) * b2.ms,
            )
    return namespace


class CellHandle(Cell):
    """Per-cell parameter handle over one row of a brian2 ``NeuronGroup``.

    Parameters are the group's parameter-typed state variables, read and
    written in brian2's base SI units (volt, second, ...)::

        env.cells[3].set_params({"E_K": -0.077})
    """

    def __init__(self, env: "Env", population: str, gid: int, index: int):
        super().__init__(env, population, gid)
        self._index = int(index)

    @property
    def index(self) -> int:
        """Row of the cell within its ``NeuronGroup``"""
        return self._index

    @property
    def group(self):
        """The ``NeuronGroup`` holding the cell"""
        return self._env._populations[self._population]

    def get_params(self) -> dict[str, float]:
        group = self.group
        return {
            name: float(group.state(name, use_units=False)[self._index])
            for name in _population_parameters(group)
        }

    def set_params(self, params: dict[str, float]) -> "Env":
        group = self.group
        known = set(_population_parameters(group))
        for name, value in params.items():
            if name not in known:
                raise self.unknown_param(name, known)
            group.state(name, use_units=False)[self._index] = float(value)

        return self._env


class Env(EnvProtocol):
    capabilities = frozenset(
        {
            Capability.SIMULATION,
            Capability.PER_GID_VOLTAGE,
            Capability.NOISE,
            Capability.PLASTICITY,
            Capability.EXTRACELLULAR_STIMULUS,
        }
    )

    def __init__(
        self,
        system: Union["System", str, int],
        model: Union["Model", None] = None,
        io: Union["IO"] = None,
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ):
        from livn.system import resolve

        self.system = resolve(system, comm=comm)
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

        self._select_spec = None
        self._select_method = "first"
        self._select_bounds = None
        self._selection: dict[str, object] | None = None
        self._selected_gids: set[int] | None = None

        self.seed = seed
        self.prng = RandomState(seed)
        self._noise_stream = 0
        if seed is not None:
            b2.seed(seed)

        self._populations = {}
        self.cells = CellRegistry(self)
        self._synapses = {}
        self._spike_monitors = {}
        self._voltage_monitors = {}
        self._voltage_monitors_dt = {}
        self._membrane_monitors = {}
        self._membrane_monitors_dt = {}
        self._noise_ops = set()
        self._network = b2.Network()
        self._weight_monitors = {}
        self._plasticity_enabled = False

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
        return self.system.population_ranges

    def selection(self, select, method: str = "first", bounds=None) -> "Env":
        if self._populations:
            raise RuntimeError("selection() must be called before init()")

        self._select_spec = select
        self._select_method = method
        self._select_bounds = bounds
        self._resolve_selection()
        return self

    @property
    def selection_name(self) -> str | None:
        spec = self._select_spec
        return spec if isinstance(spec, str) else None

    def _resolve_selection(self) -> None:
        self._selection = self.system.selection(
            self._select_spec,
            populations=self.active_populations(),
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

    def init(self):
        import logging

        # Suppress expected dt mismatch warning: neuron groups use a finer dt
        # (e.g. 0.005ms for Euler stability) while synapses use 0.025ms
        # matching the NEURON backend's timestep
        logging.getLogger("brian2.synapses.synapses").setLevel(logging.ERROR)

        self._load_cells()
        self._load_connections()
        self._set_delays()

        logging.getLogger("brian2.synapses.synapses").setLevel(logging.WARNING)

        self.set_noise({})  # force noise op init

        return self

    def _load_cells(self):
        ignored = set()
        if hasattr(self.model, "ignored_populations"):
            ignored = set(self.model.ignored_populations())
        population_ranges = self.system.population_ranges
        self.cells.clear()
        stimulus_column = {int(g): i for i, g in enumerate(self.system.gids)}

        for population_name in self.system.populations:
            if population_name in ignored:
                continue
            n = self.system.population_count(population_name)
            offset = population_ranges[population_name][0]
            coordinates = self.system.coordinate_array(population_name)

            if coordinates is not None and len(coordinates) == n:
                gids = np.asarray(coordinates[:, 0], dtype=np.int64)
            else:
                gids = np.arange(offset, offset + n, dtype=np.int64)

            if self._selected_gids is not None:
                keep = np.isin(gids, np.fromiter(self._selected_gids, dtype=np.int64))
                gids = gids[keep]
                if coordinates is not None and len(coordinates) == n:
                    coordinates = coordinates[keep]
                n = len(gids)
                if n == 0:
                    continue

            rows = np.asarray([stimulus_column[int(g)] for g in gids], dtype=np.int64)

            population = self.model.brian2_population_group(
                population_name=population_name,
                n=n,
                offset=offset,
                coordinates=coordinates,
                prng=self.prng,
                rows=rows,
            )

            if "stim_index" in population.variables:
                population.stim_index = rows

            population.add_attribute("kind")
            population.add_attribute("gid_offset")
            population.gid_offset = offset
            population.add_attribute("gids")
            population.gids = gids
            population.kind = "excitatory" if population_name == "EXC" else "inhibitory"

            self._network.add(population)

            self._populations[population_name] = population
            self.cells.add(
                population_name,
                {
                    int(gid): CellHandle(self, population_name, int(gid), index)
                    for index, gid in enumerate(gids)
                },
            )

        return self

    def _group_index(self, population: str) -> dict[int, int]:
        """``{gid: position within the brian2 group}`` for one population."""
        cached = self.__dict__.setdefault("_group_indices", {})
        index = cached.get(population)
        if index is None:
            gids = np.asarray(self._populations[population].gids)
            index = {int(g): i for i, g in enumerate(gids)}
            cached[population] = index
        return index

    def _load_connections(self):
        ignored = set()
        if hasattr(self.model, "ignored_populations"):
            ignored = set(self.model.ignored_populations())
        for post, v in self.system.connections_config["synapses"].items():
            if post in ignored:
                continue
            for pre, synapse in v.items():
                if pre in ignored:
                    continue

                if pre not in self._populations or post not in self._populations:
                    # a selection can empty a population out entirely
                    continue

                # gid -> index within the group
                pre_index = self._group_index(pre)
                post_index = self._group_index(post)

                # Build connectivity arrays
                all_i = []
                all_j = []
                all_multipliers = []
                all_distances = []

                for post_gid, (pre_gids, projection) in self.system.projection_array(
                    pre, post
                ):
                    j = post_index.get(int(post_gid))
                    if j is None:
                        continue  # post cell was not selected

                    distances = projection
                    if isinstance(projection, dict):
                        from livn.system import projection_attribute

                        distances = projection_attribute(
                            projection.get("Connections"), "distance"
                        )

                    pre_gids = np.asarray(pre_gids)
                    distances = np.asarray(distances)

                    rows = np.array(
                        [pre_index.get(int(g), -1) for g in pre_gids], dtype=np.int64
                    )
                    reached = rows >= 0
                    if not reached.any():
                        continue

                    all_i.append(rows[reached])
                    all_j.append(np.full(int(reached.sum()), j, dtype=np.int64))
                    all_multipliers.append(np.ones(int(reached.sum())))
                    all_distances.append(distances[reached])

                if not all_i:
                    continue

                all_i = np.concatenate(all_i).astype(np.int32)
                all_j = np.concatenate(all_j).astype(np.int32)
                all_multipliers = np.concatenate(all_multipliers)
                all_distances = np.concatenate(all_distances)

                # Iterate over mechanisms for this connection
                mechanisms = synapse.get("mechanisms", {}).get("default", {})
                if not mechanisms:
                    # Fallback: use legacy single synapse
                    S = self.model.brian2_connection_synapse(
                        self._populations[pre], self._populations[post]
                    )
                    S.add_attribute("kind")
                    S.kind = synapse["type"]
                    S.connect(i=all_i, j=all_j)
                    S.multiplier[:] = all_multipliers
                    S.distance[:] = all_distances
                    S.prefix = 1.0 if synapse["type"] == "excitatory" else -1.0
                    S.w[:] = 0
                    self._synapses[(post, pre)] = S
                    self._network.add(S)
                    continue

                post_group = self._populations[post]
                sections = synapse.get("sections") or ["soma"]
                proportions = synapse.get("proportions") or [1.0] * len(sections)

                for section, proportion in zip(sections, proportions, strict=False):
                    compartment, sec_name = self._synapse_site(post, section)

                    suffix = "_d" if compartment == "dend" else ""
                    if f"A_ampa{suffix}" not in post_group.variables:
                        suffix = ""
                    for mech_name, mech_params in mechanisms.items():
                        mech_lower = mech_name.lower()
                        if mech_params.get("tau_rise") is not None:
                            setattr(
                                post_group,
                                f"tau_rise_{mech_lower}{suffix}",
                                mech_params["tau_rise"],
                            )
                        if mech_params.get("tau_decay") is not None:
                            setattr(
                                post_group,
                                f"tau_decay_{mech_lower}{suffix}",
                                mech_params["tau_decay"],
                            )
                        if mech_params.get("e") is not None:
                            setattr(
                                post_group, f"e_{mech_lower}{suffix}", mech_params["e"]
                            )
                        # Set NMDA Mg block parameters if present
                        if mech_name == "NMDA":
                            for p in ("mg", "Kd", "gamma", "vshift"):
                                if p in mech_params and mech_params[p] is not None:
                                    setattr(
                                        post_group,
                                        f"{p}_nmda{suffix}",
                                        mech_params[p],
                                    )

                    for mech_name, mech_params in mechanisms.items():
                        if mech_params.get("tau_decay") is None:
                            continue

                        S = self.model.brian2_mechanism_synapse(
                            self._populations[pre],
                            self._populations[post],
                            mech_name,
                            mech_params,
                            synapse["type"],
                            compartment=compartment,
                        )
                        S.add_attribute("kind")
                        S.kind = synapse["type"]

                        S.connect(i=all_i, j=all_j)
                        S.multiplier[:] = all_multipliers * float(proportion)
                        S.distance[:] = all_distances
                        S.w[:] = mech_params.get("weight", 1.0)

                        if hasattr(S, "w_plastic"):
                            S.w_plastic[:] = 1.0
                            S.last_int[:] = 0.0
                            S.w_min[:] = 0.0
                            S.w_max[:] = 10.0

                        self._synapses[(post, pre, sec_name, mech_name)] = S
                        self._network.add(S)

        return self

    def _synapse_site(self, population: str, section: str) -> tuple[str, str]:
        site = getattr(self.model, "brian2_synapse_site", None)
        if site is None:
            return "soma", section
        return site(population, section)

    def _set_delays(self, velocity=250.0, dt=0.025):
        """Compute delays matching miv-simulator: max(distance/velocity, 2*dt).

        Args:
            velocity: Connection velocity in um/ms (default 250, matching NEURON backend).
            dt: Simulation timestep in ms (default 0.025).
        """
        for S in self._synapses.values():
            if velocity == 0:
                S.delay = 0 * b2.ms
                continue
            distances = np.array(S.distance)  # in um
            delays_ms = np.maximum(distances / velocity, 2.0 * dt)
            S.delay = delays_ms * b2.ms

        return self

    @property
    def weight_names(self) -> list[str]:
        names = []
        for key in self._synapses:
            if len(key) != 4:
                continue
            post, pre, sec_name, mech = key
            name = f"{post}_{pre}-{sec_name}-{mech}-weight"
            if name not in names:
                names.append(name)
        return names or list(self.system.weight_names)

    def set_weights(self, weights):
        for k, v in weights.items():
            param = SynapticParam.from_string(k)
            if isinstance(param.param_path, str) and param.param_path != "weight":
                raise NotImplementedError(
                    f"brian2 backend cannot set the synapse mechanism parameter "
                    f"{param.param_path!r} ({k})"
                )
            post = param.population
            pre = param.source
            mech_name = param.syn_name  # e.g., "AMPA", "NMDA", "GABA_A"
            sec_name = None
            if param.sec_type is not None:
                _, sec_name = self._synapse_site(post, param.sec_type)

            matched = False
            for key, S in self._synapses.items():
                if len(key) != 4:
                    continue
                if (key[0], key[1]) != (post, pre):
                    continue
                if mech_name is not None and key[3] != mech_name:
                    continue
                if sec_name is not None and key[2] != sec_name:
                    continue
                S.w = v
                matched = True

            if not matched and (post, pre) in self._synapses:
                # legacy projection without per-mechanism synapses
                self._synapses[(post, pre)].w = v
                matched = True

            if not matched:
                warnings.warn(
                    f"{k} names no synapse in this network, so it was dropped; "
                    f"`weight_names` lists the keys it accepts",
                    stacklevel=2,
                )

        return self

    def set_noise(self, noise: dict):
        if not self._noise_ops:
            for population in self._populations.values():
                op = self.model.brian2_noise_op(population, self.prng)
                if op is not None:
                    self._noise_ops.add(op)
                    self._network.add(op)

        if noise:
            for population in self._populations.values():
                self.model.brian2_noise_configure(population, **noise)

        return self

    def _record_spikes(self, population: str) -> "Env":
        if population not in self._populations:
            return self

        self._spike_monitors[population] = monitor = b2.SpikeMonitor(
            self._populations[population]
        )
        self._network.add(monitor)

        return self

    def _record_voltage(
        self, population: str, dt: float, gids=None, sections=None
    ) -> "Env":
        if population not in self._populations:
            return self

        if sections is not None and "soma" not in {str(s) for s in sections}:
            raise NotImplementedError(
                f"a brian2 cell is a single compartment, recorded as 'soma', so "
                f"{sorted(sections)} names nothing it has"
            )
        group = self._populations[population]
        record = True
        if gids is not None:
            index = self._group_index(population)
            record = sorted(index[int(g)] for g in gids if int(g) in index)

        self._voltage_monitors[population] = monitor = b2.StateMonitor(
            group,
            "v",
            record=record,
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
        if kwargs.get("root_only", True) and not P.is_root():
            raise RuntimeError(
                "The brian2 backend does not support MPI parallelization on multiple ranks."
            )

        b2.defaultclock.dt = dt * b2.ms

        if stimulus is not None:
            stimulus = Stimulus.from_arg(stimulus, env=self, duration=duration)
            stimulus = self.model.prepare_stimulus(stimulus)
            if stimulus.input_mode not in STIMULUS_TERMS:
                raise ValueError(
                    f"the brian2 backend has no term for a "
                    f"{stimulus.input_mode!r} stimulus; it delivers "
                    f"{sorted(STIMULUS_TERMS)}. The neuron backend gives this "
                    "mode its own mechanism, and reading it as one of these "
                    "would be a different experiment"
                )

        if stimulus is None:
            stimulus = Stimulus(
                array=np.zeros(
                    [
                        int((self.t + duration) / 1.0),
                        len(self.system.gids),
                    ]
                ),
                dt=1.0,
            )
        else:
            n_gids = len(self.system.gids)
            if stimulus.array.shape[-1] != n_gids and stimulus.gids is not None:
                gid_to_idx = {int(g): i for i, g in enumerate(self.system.gids)}
                expanded = np.zeros(
                    (stimulus.array.shape[0], n_gids), dtype=stimulus.array.dtype
                )
                for col_idx, gid in enumerate(stimulus.gids):
                    sys_idx = gid_to_idx.get(int(gid))
                    if sys_idx is not None:
                        expanded[:, sys_idx] += stimulus.array[:, col_idx]
                stimulus = Stimulus(
                    array=expanded,
                    dt=stimulus.dt,
                    input_mode=stimulus.input_mode,
                    units=stimulus.units,
                )

        # Check for stimulus dt consistency across continued runs
        if not hasattr(self, "_stimulus_dt"):
            self._stimulus_dt = stimulus.dt
        elif not np.isclose(self._stimulus_dt, stimulus.dt, rtol=0.0, atol=1e-12):
            raise ValueError("Stimulus dt mismatch; call clear() before rerunning")

        # Left-pad stimulus to align with brian2's absolute time reference.
        # Brian2's TimedArray indexes from t=0 (simulation start), so for
        # continued runs (self.t > 0) the stimulus must be padded.
        pad_rows = max(0, round(self.t / stimulus.dt))
        if pad_rows > 0:
            padding = np.zeros((pad_rows, stimulus.array.shape[1]))
            stimulus = Stimulus(
                array=np.vstack((padding, stimulus.array)),
                dt=stimulus.dt,
                input_mode=stimulus.input_mode,
                units=stimulus.units,
            )

        t_start = self.t
        self._network.run(
            duration * b2.ms,
            namespace=stimulus_namespace(stimulus, stimulus.array.shape[-1], duration),
        )
        self.t += duration

        gids = []
        vv = []
        for population, monitor in self._voltage_monitors.items():
            gids.append(np.asarray(monitor.source.gids)[np.asarray(monitor.record)])
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
        for monitor in self._spike_monitors.values():
            ts = monitor.t / b2.ms
            ii.append(np.asarray(monitor.source.gids)[monitor.i[ts >= t_start]])
            tt.append(ts[ts >= t_start] - t_start)

        run = (
            Run(t0=t_start, duration=duration)
            .add_spikes(concat(ii), concat(tt))
            .add_voltage(concat(gids), concat(vv), dt=self.voltage_recording_dt)
        )

        # membrane currents aligned to global gid order if enabled
        if len(self._membrane_monitors) == 0:
            return run

        # [T, n_neurons] matrix aligned to active neuron coordinates
        coords = self.active_neuron_coordinates()
        if coords is None or len(coords) == 0:
            return run

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
                # I_memb_s/I_memb_d are current densities (mA/cm^2)
                # in the brian2 equations; convert to absolute current per
                # section in microamperes (I[uA] = density * area_cm2 * 1e3)
                # to follow the convention
                pop = mon_s.source
                area_s_cm2 = float(getattr(pop, "area_soma_cm2", 0.0))
                area_d_cm2 = float(getattr(pop, "area_dend_cm2", 0.0))
                if area_s_cm2 <= 0.0 or area_d_cm2 <= 0.0:
                    raise RuntimeError(
                        f"brian2 population {population!r} is missing "
                        "area_soma_cm2/area_dend_cm2 attributes required "
                        "to convert I_memb_* (mA/cm^2) to uA"
                    )
                data_s = np.array(mon_s.I_memb_s[:, start_idx:]) * (area_s_cm2 * 1000.0)
                data_d = np.array(mon_d.I_memb_d[:, start_idx:]) * (area_d_cm2 * 1000.0)
                per_pop_data[population] = (data_s, data_d)
                lengths.append(data_s.shape[1] if data_s.ndim == 2 else 0)
            else:
                data = (monitor.I / b2.uA)[:, start_idx:]
                per_pop_data[population] = data
                lengths.append(data.shape[1] if data.ndim == 2 else 0)

        T = int(max(lengths) if lengths else 0)
        if T == 0:
            return run

        n_neurons = len(all_gids)
        currents = np.zeros((n_neurons * sections_per_neuron, T), dtype=np.float32)

        for population, data in per_pop_data.items():
            pop_gids = np.asarray(self._populations[population].gids)

            if isinstance(data, tuple):
                # Two-compartment: interleave soma/dend (soma0, dend0, soma1, dend1, ...)
                data_s, data_d = data
                if data_s.size == 0:
                    continue
                n_pop = data_s.shape[0]
                for k in range(n_pop):
                    gid = pop_gids[k]
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
                    gid = pop_gids[k]
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

        return run.add_current(
            all_gids, currents, dt=self.membrane_current_recording_dt
        )

    def _iter_stdp_synapses(self):
        for key, S in self._synapses.items():
            if hasattr(S, "_has_stdp") and S._has_stdp:
                yield key, S

    def enable_plasticity(self, config=None):
        if config is None:
            config = {}

        for pop in self._populations.values():
            pop.plasticity_on = 1
            for param, value in config.items():
                if hasattr(pop, param):
                    setattr(pop, param, value)

        for key, S in self._iter_stdp_synapses():
            if len(S) > 0:
                post_pop = self._populations[key[0]]
                suffix = "_d" if getattr(S, "_compartment", "soma") == "dend" else ""
                learn_int_var = (
                    f"learn_int_exc{suffix}"
                    if S.kind == "excitatory"
                    else f"learn_int_inh{suffix}"
                )
                S.last_int[:] = getattr(post_pop, learn_int_var)[S.j]

        self._plasticity_enabled = True
        return self

    def disable_plasticity(self):
        for pop in self._populations.values():
            pop.plasticity_on = 0

        self._plasticity_enabled = False
        return self

    def get_weights(self):
        """Returns current synaptic weights of all plastic synapses

        (post, pre, mechanism, pre_idx, post_idx) -> w_plastic value
        """
        weights = {}
        for key, S in self._iter_stdp_synapses():
            if len(S) > 0:
                post, pre, mech = key[0], key[1], key[-1]
                w_plastic = np.array(S.w_plastic[:])
                i_arr = np.array(S.i[:])
                j_arr = np.array(S.j[:])
                for idx in range(len(S)):
                    weights[(post, pre, mech, int(i_arr[idx]), int(j_arr[idx]))] = (
                        float(w_plastic[idx])
                    )
        return weights

    def normalize_weights(self, target=None):
        from livn.weights import normalize_weights

        weight, w_min, w_max, group = [], [], [], []
        refs = []  # (S, syn_idx)
        group_ids: dict[tuple, int] = {}
        for key, S in self._iter_stdp_synapses():
            if len(S) == 0:
                continue
            pop = key[0]
            w_arr = np.array(S.w_plastic[:])
            j_arr = np.array(S.j[:])
            wmin_arr = np.array(S.w_min[:])
            wmax_arr = np.array(S.w_max[:])
            for idx in range(len(S)):
                gid = group_ids.setdefault((pop, int(j_arr[idx])), len(group_ids))
                weight.append(float(w_arr[idx]))
                w_min.append(float(wmin_arr[idx]))
                w_max.append(float(wmax_arr[idx]))
                group.append(gid)
                refs.append((S, idx))

        if not refs:
            return self

        new_w = normalize_weights(
            np.asarray(weight),
            np.asarray(w_min),
            np.asarray(w_max),
            np.asarray(group),
            target=target,
        )
        for (S, idx), w in zip(refs, new_w, strict=False):
            S.w_plastic[idx] = float(w)

        return self

    def record_weights(self, dt=0.1):
        for key, S in self._iter_stdp_synapses():
            if len(S) > 0 and key not in self._weight_monitors:
                monitor = b2.StateMonitor(S, "w_plastic", record=True, dt=dt * b2.ms)
                self._weight_monitors[key] = monitor
                self._network.add(monitor)

        return self

    def clear_recordings(self):
        self.clear_monitors()
        return self

    def clear(self, reseed: bool = True):
        if reseed:
            self.reseed_noise()

        self.t = 0
        self.clear_monitors()

        if hasattr(self, "_stimulus_dt"):
            del self._stimulus_dt

        return self

    def reseed_noise(self, stream: int | None = None):
        if self.seed is None:
            return self

        if stream is None:
            self._noise_stream += 1
            stream = self._noise_stream
        else:
            self._noise_stream = int(stream)

        b2.seed(int(self.seed) + stream * NOISE_STREAM_STRIDE)
        return self

    def clear_monitors(self):
        for monitor in self._spike_monitors.values():
            self._network.remove(monitor)
        for monitor in self._voltage_monitors.values():
            self._network.remove(monitor)
        for monitor in self._membrane_monitors.values():
            if isinstance(monitor, tuple):
                for m in monitor:
                    self._network.remove(m)
            else:
                self._network.remove(monitor)
        for monitor in self._weight_monitors.values():
            self._network.remove(monitor)

        spms = list(self._spike_monitors.keys())
        vms = list(self._voltage_monitors.keys())
        mms = list(self._membrane_monitors.keys())

        self._spike_monitors = {}
        self._voltage_monitors = {}
        self._membrane_monitors = {}
        self._weight_monitors = {}

        gc.collect()

        for p in spms:
            self._record_spikes(p)

        for p in vms:
            self._record_voltage(p, dt=self._voltage_monitors_dt.get(p, 0.1))

        for p in mms:
            self._record_membrane_current(p, dt=self._membrane_monitors_dt.get(p, 0.1))

    def reinit(self):
        self.clear()

        self._network = None
        gc.collect()

        self._network = b2.Network()
        self._populations = {}
        self.__dict__.pop("_group_indices", None)
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
