"""IO

i = neuron_gid
c = channel_id
x = x coordinate
y = y coordinate
z = z coordinate
p = payload value (distance, induction strength etc.)
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections import defaultdict
from typing import TYPE_CHECKING

import gymnasium
import numpy as _np

from livn.backend import backend
from livn.utils import (
    Jsonable,
    class_path,
    import_instance,
    is_traced,
    lnp,
    with_fields,
)

if TYPE_CHECKING:
    from livn.stimulus import Stimulus
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import equinox as eqx
    import jax.numpy as np

    _USES_JAX = True
    Module = eqx.Module

    def static_field(default):
        return eqx.field(default=default, static=True)

else:
    import numpy as np

    Module = object

    def static_field(default):
        return dataclasses.field(default=default)


def _plain(value):
    """A JSON-ready copy of a field value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, _np.floating)):
        return float(value)
    return _np.asarray(value, dtype=float).tolist()


class Component(Module):
    if not _USES_JAX:

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            dataclasses.dataclass(cls)

    def serialize(self) -> dict:
        document = {"cls": class_path(self)}
        for f in dataclasses.fields(self):
            document[f.name] = _plain(getattr(self, f.name))
        return document

    def set_params(self, params: dict):
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(k for k in params if k not in known)
        if unknown:
            raise KeyError(
                f"{unknown} are not parameters of {type(self).__name__}; "
                f"its parameters are {sorted(known)}"
            )

        def coerce(old, new):
            if old is None or (
                isinstance(old, (float, int)) and not isinstance(old, bool)
            ):
                return None if new is None else float(new)
            if isinstance(old, bool):
                return bool(new)
            return lnp().asarray(new, dtype=getattr(old, "dtype", None))

        return with_fields(
            self, **{k: coerce(getattr(self, k), v) for k, v in params.items()}
        )


DEFAULT_MAX_STIMULUS_GB = 5.0
MAX_STIMULUS_GB_ENV = "LIVN_MAX_STIMULUS_GB"


def _max_stimulus_bytes() -> float:
    """The ceiling in bytes; `0` or `inf` in the environment removes it."""
    given = os.environ.get(MAX_STIMULUS_GB_ENV)
    if given is None:
        return DEFAULT_MAX_STIMULUS_GB * 2**30
    try:
        limit = float(given)
    except ValueError:
        raise ValueError(
            f"{MAX_STIMULUS_GB_ENV}={given!r} is not a number of gigabytes"
        ) from None
    return float("inf") if limit <= 0 else limit * 2**30


def _check_stimulus_size(n_timesteps: int, n_gids: int, itemsize: int, n_reached=None):
    limit = _max_stimulus_bytes()
    size = float(n_timesteps) * float(n_gids) * float(itemsize)
    if size <= limit:
        return

    reach = ""
    if n_reached is not None and n_reached < n_gids:
        reach = f", of which {n_reached:,} are within reach of the driven channels"
    from livn.stimulus import STIMULUS_CHUNK_MB_ENV

    raise MemoryError(
        f"a [{n_timesteps:,} x {n_gids:,}] stimulus of "
        f"{itemsize}-byte values is {size / 2**30:.1f} GiB, over the "
        f"{limit / 2**30:.1f} GiB ceiling.\n"
        f"A stimulus is dense in time and cells: {n_timesteps:,} timesteps over "
        f"{n_gids:,} cells{reach}. Pass a `Policy` to `run` rather than an "
        "array as a policy is expanded onto cells one "
        f"window at a time and never held whole ({STIMULUS_CHUNK_MB_ENV} sizes "
        f"the window). Raise {MAX_STIMULUS_GB_ENV} (in GiB) to override."
    )


def section_labels(neuron_coordinates):
    """`(gids, sections)` for every coordinate row."""
    import numpy as _np

    gids = _np.asarray(neuron_coordinates)[:, 0].astype(_np.int64)
    seen: dict[int, int] = {}
    sections = _np.empty(len(gids), dtype=_np.int64)
    for row, gid in enumerate(gids):
        gid = int(gid)
        sections[row] = seen.get(gid, 0)
        seen[gid] = sections[row] + 1
    return gids, sections


def coupled_sections(neuron_coordinates, cell_induction, dense: bool = False):
    """`(gids, sections, keep)` for non-zero columns."""
    import numpy as _np

    induction = _np.asarray(cell_induction)
    n_rows = len(_np.asarray(neuron_coordinates))

    if dense:
        reached = _np.any(induction != 0.0, axis=0)
    else:
        columns = _np.arange(len(induction)) % max(1, n_rows)
        reached = _np.zeros(n_rows, dtype=bool)
        _np.logical_or.at(reached, columns, induction[:, 2] != 0.0)

    keep = _np.flatnonzero(reached)
    gids, sections = section_labels(neuron_coordinates)
    return gids[keep], sections[keep], keep


def _empty_array():
    return np.array([])


class IO(Jsonable):
    """
    IO

    partial-like utility to maintain state
    associated with an IO transformation
    """

    @classmethod
    def from_directory(cls, directory: str, comm=None):
        return cls.from_json(
            os.path.join(directory, cls.__name__.lower() + ".json"), comm=comm
        )

    def serialize(self) -> dict:
        return {}

    def parameter_children(self) -> dict:
        return {}

    def set_params(self, params: dict) -> IO:
        children = self.parameter_children()
        own: dict = {}
        nested: dict = {}
        for key, value in params.items():
            head, _, rest = key.partition("-")
            if rest and head in children:
                nested.setdefault(head, {})[rest] = value
            else:
                own[key] = value

        for name, values in nested.items():
            setattr(self, name, children[name].set_params(values))

        for name, value in own.items():
            if not hasattr(self, name):
                raise KeyError(
                    f"{name!r} is not a parameter of {type(self).__name__}"
                    + (f"; its children are {sorted(children)}" if children else "")
                )
            setattr(self, name, float(value))

        self.invalidate()
        return self

    def invalidate(self) -> None:
        if getattr(self, "_cell_induction", None) is not None:
            self._cell_induction = None
        self._induction_cache = None
        if getattr(self, "cell_measurement", None) is not None:
            self.cell_measurement = None
        for child in self.parameter_children().values():
            if hasattr(child, "invalidate"):
                child.invalidate()

    @property
    def num_channels(self) -> int:
        raise NotImplementedError("Please specify an IO")

    @property
    def input_units(self) -> str:
        """What the `channel_inputs` of `cell_stimulus` are measured in."""
        raise NotImplementedError("Please specify an IO")

    @property
    def channel_ids(self) -> Int[Array, " n_channel_ids"]:
        raise NotImplementedError("Please specify an IO")

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        """Field induced per unit command at each coordinate row, per channel."""
        raise NotImplementedError("Please specify an IO")

    def _get_input_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(self.num_channels,))

    @property
    def input_space(self) -> gymnasium.Space:
        if not hasattr(self, "_input_space"):
            self._input_space = self._get_input_space()
        return self._input_space

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
        dt: float = 1.0,
    ) -> Stimulus:
        """Transforms channel inputs into neural inputs."""
        raise NotImplementedError("Please specify an IO")

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        ii: Float[Array, " i"],
        *recordings: Float[Array, " _"],
    ) -> tuple[dict[int, Array], ...]:
        """Transforms neural recordings identified by their gids into per channel recordings"""
        raise NotImplementedError("Please specify an IO")

    def source_gain(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_channels n_recording_coords"]:
        """Compute the per-recording-coordinate gain matrix

        Parameters
        - distances: triplets [channel_id, gid, distance_um] as returned by `self.distances(...)`
        """
        raise NotImplementedError("Please specify an IO")

    @property
    def needs_dt(self) -> bool:
        """Whether `potential_recording` needs the sampling interval."""
        return False

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "n_neurons timestep"],
        dt: float | None = None,
        key=None,
    ) -> Float[Array, "n_channels timestep"]:
        """Estimate the voltage recorded from the channels

        Parameters
        - distances: triplets [channel_id, gid, distance_um] as returned by `self.distances(...)`.
        - membrane_currents: current per neuron index (aligned with the coordinate order
            used to compute `distances`). Shape must be (n_neurons, timestep).
        - dt: sampling interval in ms; required when the medium or a transfer filters in time.
        - key: random key for stochastic transfers (`Noise`); without one they pass the signal on.
        """
        raise NotImplementedError("Please specify an IO")


class _Static:
    def __init__(self, value):
        self.value = value

    def _members(self):
        return self.value if isinstance(self.value, tuple) else (self.value,)

    def __eq__(self, other):
        if not isinstance(other, _Static):
            return False
        a, b = self._members(), other._members()
        return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))

    def __hash__(self):
        return hash(tuple(id(x) for x in self._members()))


def _unique(*devices):
    seen: list = []
    for device in devices:
        if not any(device is other for other in seen):
            seen.append(device)
    return seen


def _bound(device, medium, transfers):
    bound = copy.copy(device)
    if hasattr(bound, "medium"):
        bound.medium = medium
    bound._stack = transfers
    return bound


def _device(spec):
    device = import_instance(spec)
    if not isinstance(device, IO) or isinstance(device, ComposedIO):
        raise TypeError(
            f"a device is a single IO such as MEA or LightArray, not {spec!r}"
        )
    return device


class ComposedIO(IO):
    """A stimulating device and a recording device in one preparation."""

    def __init__(self, inputs=None, outputs=None, medium=None):
        from livn.io.medium import Medium
        from livn.io.transfer import Transfer

        if isinstance(inputs, list | tuple):
            raise NotImplementedError(
                "a list of input stages ([*transfers, device]) is reserved and not "
                "implemented; pass the stimulating device"
            )
        if outputs is None:
            raise ValueError(
                "ComposedIO needs outputs: a device or [device, *transfers]"
            )

        stages = list(outputs) if isinstance(outputs, list | tuple) else [outputs]
        out_device = _device(stages[0])
        transfers = tuple(import_instance(t) for t in stages[1:])
        for k, t in enumerate(transfers, start=1):
            if not isinstance(t, Transfer):
                raise TypeError(
                    f"outputs[{k}] is {type(t).__name__}, but every element after "
                    "the device must be a Transfer (channel to channel); only the "
                    "first element is a device"
                )
        in_device = out_device if inputs is None else _device(inputs)

        if medium is None:
            media = [
                d.medium for d in _unique(in_device, out_device) if hasattr(d, "medium")
            ]
            default = Medium().serialize()
            chosen = [m for m in media if m.serialize() != default]
            if any(m.serialize() != chosen[0].serialize() for m in chosen[1:]):
                raise ValueError(
                    "the inputs and outputs devices carry different media; pass the "
                    "one `medium=` the composition uses"
                )
            medium = chosen[0] if chosen else Medium()
        else:
            medium = import_instance(medium)

        self._medium = medium
        self.transfers = transfers
        self.outputs = _bound(out_device, medium, transfers)
        self.inputs = (
            self.outputs
            if in_device is out_device
            else _bound(in_device, medium, transfers)
        )

    @property
    def medium(self):
        return self._medium

    @medium.setter
    def medium(self, medium):
        self._medium = medium
        for device in _unique(self.inputs, self.outputs):
            if hasattr(device, "medium"):
                device.medium = medium

    def _set_transfers(self, transfers):
        self.transfers = tuple(transfers)
        for device in _unique(self.inputs, self.outputs):
            device._stack = self.transfers

    def __getattr__(self, name):
        # a single MEA that stimulates and records is still addressed as `io.<attr>`
        if name in ("inputs", "outputs", "transfers", "_medium") or name.startswith(
            "__"
        ):
            raise AttributeError(name)
        for device in (self.outputs, self.inputs):
            if hasattr(device, name):
                return getattr(device, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def parameter_children(self) -> dict:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "medium": self.medium,
            **{f"transfer{k}": t for k, t in enumerate(self.transfers)},
        }

    def set_params(self, params: dict) -> ComposedIO:
        children = self.parameter_children()
        nested: dict = {}
        for key, value in params.items():
            head, _, rest = key.partition("-")
            if not rest or head not in children:
                raise KeyError(
                    f"{key!r} is not a parameter of {type(self).__name__}; "
                    f"its children are {sorted(children)}"
                )
            if head in ("inputs", "outputs") and rest.partition("-")[0] == "medium":
                raise KeyError(f"{key!r}: the medium is shared, address it as '{rest}'")
            nested.setdefault(head, {})[rest] = value

        transfers = list(self.transfers)
        for head, values in nested.items():
            updated = children[head].set_params(values)
            if head == "medium":
                self.medium = updated
            elif head.startswith("transfer"):
                transfers[int(head[len("transfer") :])] = updated
        self._set_transfers(transfers)
        self.invalidate()
        return self

    def invalidate(self) -> None:
        for device in _unique(self.inputs, self.outputs):
            device.invalidate()

    def serialize(self) -> dict:
        def document(device):
            doc = {"cls": class_path(device), **device.serialize()}
            doc.pop("medium", None)
            return doc

        stages = [document(self.outputs), *(t.serialize() for t in self.transfers)]
        return {
            "inputs": None if self.inputs is self.outputs else document(self.inputs),
            "outputs": stages if self.transfers else stages[0],
            "medium": self.medium.serialize(),
        }

    @property
    def needs_dt(self) -> bool:
        return self.outputs.needs_dt or any(t.needs_dt for t in self.transfers)

    @property
    def num_channels(self) -> int:
        return self.outputs.num_channels

    @property
    def input_units(self) -> str:
        return self.inputs.input_units

    @property
    def channel_ids(self) -> Int[Array, " n_channel_ids"]:
        return self.outputs.channel_ids

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
        **kwargs,
    ):
        return self.inputs.cell_stimulus(neuron_coordinates, channel_inputs, **kwargs)

    def reach(self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]):
        return self.inputs.reach(neuron_coordinates)

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"] | None,
        ii: Float[Array, " i"],
        *recordings: Float[Array, " _"],
    ) -> tuple[dict[int, Array], ...]:
        return self.outputs.channel_recording(neuron_coordinates, ii, *recordings)

    def source_gain(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_channels n_recording_coords"]:
        return self.outputs.source_gain(distances)

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "n_neurons timestep"],
        dt: float | None = None,
        key=None,
    ) -> Float[Array, "n_channels timestep"]:
        from livn.io.transfer import stage_key

        if dt is None and self.needs_dt:
            raise ValueError("this IO filters in time, so potential_recording needs dt")
        signal = self.outputs.potential_recording(distances, membrane_currents, dt=dt)
        for k, transfer in enumerate(self.transfers):
            signal = transfer(signal, dt=dt, key=stage_key(key, k))
        return signal

    def distances(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_sources*n_coords cip=3"]:
        return self.outputs.distances(neuron_coordinates)


if _USES_JAX:
    import jax

    def _composed_flatten(io):
        origin = io.__dict__.get("_origin")
        if (
            origin is not None
            and io.inputs is origin[1][0]
            and io.outputs is origin[1][1]
        ):
            return (io._medium, io.transfers), origin[0]
        return (io._medium, io.transfers), _Static((io.inputs, io.outputs))

    def _composed_unflatten(aux, children):
        medium, transfers = children
        inputs, outputs = aux.value
        io = ComposedIO.__new__(ComposedIO)
        io._medium = medium
        io.transfers = tuple(transfers)
        io.outputs = _bound(outputs, medium, io.transfers)
        io.inputs = (
            io.outputs if inputs is outputs else _bound(inputs, medium, io.transfers)
        )
        io._origin = (aux, (io.inputs, io.outputs))
        return io

    jax.tree_util.register_pytree_node(
        ComposedIO, _composed_flatten, _composed_unflatten
    )


def calculate_distances(
    source: Float[Array, "n_sources cxyz=4"],
    coords: Float[Array, "n_coords cxyz=4"],
) -> Float[Array, "n_sources*n_coords cip=3"]:
    """
    Calculate the Euclidean distances between each electrode and each coordinate.
    """
    xp = _np if not is_traced(source, coords) else np
    source = xp.asarray(source)
    coords = xp.asarray(coords)

    ex, ey, ez = source[:, 1], source[:, 2], source[:, 3]
    cx, cy, cz = coords[:, 1], coords[:, 2], coords[:, 3]

    ex = ex[:, xp.newaxis]
    ey = ey[:, xp.newaxis]
    ez = ez[:, xp.newaxis]

    distances = xp.sqrt((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2)

    channel_ids = xp.repeat(source[:, 0], len(coords))
    gids = xp.tile(coords[:, 0], len(source))

    return xp.column_stack((channel_ids, gids, distances.flatten()))


def channel_recording(
    ci: Float[Array, "n ci_"], ii: Float[Array, " i"], *recordings: Float[Array, " _"]
) -> tuple[dict[int, Array], ...]:
    """Given channel-neuron mapping, converts recording into per-channel recording"""
    r = tuple(defaultdict(_empty_array) for _ in range(len(recordings) + 1))
    ii = np.array(ii)
    for channel in np.unique(ci[:, 0]):
        gids = ci[ci[:, 0] == channel, 1].astype(int)
        channel_mask = np.isin(ii, gids)
        r[0][int(channel)] = ii[channel_mask]
        for q in range(len(recordings)):
            r[q + 1][int(channel)] = np.array(recordings[q])[channel_mask]

    if len(recordings) == 0:
        return dict(r[0])

    return tuple(dict(x) for x in r)


if _USES_JAX:

    def with_amplitude(distances, amplitudes):
        """`distances` with its last column replaced."""
        if not is_traced(distances, amplitudes):
            out = _np.array(distances, copy=True)
            out[:, -1] = amplitudes
            return out
        return np.asarray(distances).at[:, -1].set(amplitudes)

    def relative_distance(
        distances: Float[Array, "n_distances cip=3"],
        boundary: float,
        filter_out_of_bounds: bool = False,
    ) -> Float[Array, "n_distances cip=3"]:
        """
        Normalizes the distances within a boundary radius to [0, 1]
        and optionally discards out-of-bounds distances.
        """
        if not is_traced(distances):
            distances = _np.array(distances, copy=True)
            if filter_out_of_bounds:
                distances = distances[distances[:, -1] <= boundary]
            distances[:, -1] = distances[:, -1] / boundary
            return distances
        distances = np.array(distances)
        if filter_out_of_bounds:
            distances = distances[distances[:, -1] <= boundary]
        return distances.at[:, -1].set(distances[:, -1] / boundary)

    def induction_matrix(
        cell_induction: Float[Array, "n_inductions cip=3"],
        n_gids: int | None = None,
        *,
        n_channels: int | None = None,
        keep: Int[Array, " n_keep"] | None = None,
        dtype=None,
    ):
        per_coordinate = n_gids is not None
        induction_rows = _np.asarray(cell_induction)
        if n_gids is None:
            n_gids = len(np.unique(cell_induction[:, 1]))
        if n_channels is None:
            n_channels = len(_np.unique(induction_rows[:, 0]))

        channel_ids = induction_rows[:, 0].astype(int)
        amplitudes = cell_induction[:, 2]
        gids = induction_rows[:, 1].astype(int)

        matrix = np.zeros(
            (n_channels, n_gids), dtype=np.float32 if dtype is None else dtype
        )
        if per_coordinate:
            columns = _np.arange(induction_rows.shape[0]) % n_gids
        else:
            _unique, columns = np.unique(gids, return_inverse=True, size=n_gids)
        matrix = matrix.at[channel_ids, columns].set(amplitudes)
        n_reached = int(_np.count_nonzero(induction_rows[:, 2])) or None
        if keep is not None:
            matrix = matrix[:, keep]
        return matrix, n_reached

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        c_induction: Float[Array, "n_inductions cip=3"] | None = None,
        n_gids: int | None = None,
        keep: Int[Array, " n_keep"] | None = None,
        matrix=None,
        n_reached: int | None = None,
    ) -> Float[Array, "batch timestep n_gids"]:
        stimulus = np.asarray(electrode_stimulus)
        _batch_size, n_timesteps, n_channels = electrode_stimulus.shape
        dtype = np.promote_types(stimulus.dtype, np.float32)

        if matrix is None:
            matrix, n_reached = induction_matrix(
                c_induction, n_gids, n_channels=n_channels, keep=keep, dtype=dtype
            )
        _check_stimulus_size(
            n_timesteps,
            matrix.shape[1],
            dtype.itemsize,
            n_reached=n_reached,
        )

        # reduce over gids -> [batch, timestep, n_gids]
        return np.einsum("btn,ng->btg", stimulus, matrix)


else:

    def with_amplitude(distances, amplitudes):
        """`distances` with its last column replaced."""
        out = np.array(distances, copy=True)
        out[:, -1] = amplitudes
        return out

    def relative_distance(
        distances: Float[Array, "n_distances cip=3"],
        boundary: float,
        filter_out_of_bounds: bool = False,
    ) -> Float[Array, "n_distances cip=3"]:
        """
        Normalizes the distances within a boundary radius to [0, 1]
        and optionally discards out-of-bounds distances.
        """
        if filter_out_of_bounds:
            distances = distances[distances[:, -1] <= boundary]
        distances[:, -1] = distances[:, -1] / boundary
        return distances

    def induction_matrix(
        cell_induction: Float[Array, "n_inductions cip=3"],
        n_gids: int | None = None,
        *,
        n_channels: int | None = None,
        keep: Int[Array, " n_keep"] | None = None,
        dtype=None,
    ):
        cell_induction = np.asarray(cell_induction)
        dtype = np.float32 if dtype is None else dtype

        channel_ids = cell_induction[:, 0].astype(int)
        gids = cell_induction[:, 1].astype(int)
        amplitudes = cell_induction[:, 2]

        if n_gids is None:
            n_gids = len(np.unique(gids))
            _, columns = np.unique(gids, return_inverse=True)
        else:
            if len(cell_induction) % n_gids:
                raise ValueError(
                    f"{len(cell_induction)} induction rows is not a whole "
                    f"number of channels over {n_gids} coordinates"
                )
            columns = np.arange(len(cell_induction)) % n_gids

        if n_channels is None:
            n_channels = len(np.unique(channel_ids))
        matrix = np.zeros((n_channels, n_gids), dtype=dtype)

        _, channel_indices = np.unique(channel_ids, return_inverse=True)
        matrix[channel_indices, columns] = amplitudes
        n_reached = int(np.count_nonzero(amplitudes)) or None
        if keep is not None:
            matrix = matrix[:, keep]
        return matrix, n_reached

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        cell_induction: Float[Array, "n_inductions cip=3"] | None = None,
        n_gids: int | None = None,
        *args,
        keep: Int[Array, " n_keep"] | None = None,
        matrix=None,
        n_reached: int | None = None,
        **kwargs,
    ) -> Float[Array, "batch timestep n_gids"]:
        electrode_stimulus = np.asarray(electrode_stimulus)
        _batch_size, n_timesteps, _n_channels = electrode_stimulus.shape
        dtype = np.promote_types(electrode_stimulus.dtype, np.float32)

        if matrix is None:
            matrix, n_reached = induction_matrix(
                cell_induction,
                n_gids,
                n_channels=electrode_stimulus.shape[-1],
                keep=keep,
                dtype=dtype,
            )
        _check_stimulus_size(
            n_timesteps,
            matrix.shape[1],
            dtype.itemsize,
            n_reached=n_reached,
        )

        # reduce over gids -> [batch, timestep, n_gids]
        return np.matmul(electrode_stimulus, matrix)
