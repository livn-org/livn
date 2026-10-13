"""IO

i = neuron_gid
c = channel_id
x = x coordinate
y = y coordinate
z = z coordinate
p = payload value (distance, induction strength etc.)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

import gymnasium
import numpy as _np

from livn.backend import backend
from livn.utils import Jsonable, is_traced

if TYPE_CHECKING:
    from livn.stimulus import Stimulus
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


MIN_DISTANCE_UM = 5.0
ELECTRODE_RADIUS_UM = 15.0
CULTURE_HEIGHT_UM = 50.0
_PI = 3.141592653589793
_H_EFF_UM = (ELECTRODE_RADIUS_UM**2 + CULTURE_HEIGHT_UM**2) ** 0.5

DEFAULT_RECORDING_GAIN = 1000.0 / (4.0 * _PI * 0.0003 * MIN_DISTANCE_UM)
DEFAULT_STIMULATION_GAIN = 1000.0 * 3.5 / (4.0 * _PI * _H_EFF_UM)

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

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "timestep n_neurons"],
    ) -> Float[Array, "timestep n_channels"]:
        """Estimate the voltage recorded from the channels

        Parameters
        - distances: triplets [channel_id, gid, distance_um] as returned by `self.distances(...)`.
        - membrane_currents: current per neuron index (aligned with the coordinate order
            used to compute `distances`). Shape must be (timestep, n_neurons).
        """
        raise NotImplementedError("Please specify an IO")


class ComposedIO(IO):
    def __init__(self, inputs: IO, outputs: IO):
        self.inputs = inputs
        self.outputs = outputs

    def parameter_children(self) -> dict:
        return {"inputs": self.inputs, "outputs": self.outputs}

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
        membrane_currents: Float[Array, "timestep n_neurons"],
    ) -> Float[Array, "timestep n_channels"]:
        return self.outputs.potential_recording(distances, membrane_currents)

    def distances(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_sources*n_coords cip=3"]:
        return self.outputs.distances(neuron_coordinates)


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
