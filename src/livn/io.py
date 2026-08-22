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
from typing import TYPE_CHECKING
from collections import defaultdict

import gymnasium

from livn.backend import backend
from livn.utils import Jsonable

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax
    import jax.numpy as np
    import equinox as eqx

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


def _induction_shape(distances_um, electrode_radius_um, culture_height_um):
    """1 at the electrode surface, falling as 1/sqrt(r^2 + h^2)."""
    h_eff = (electrode_radius_um**2 + culture_height_um**2) ** 0.5
    return h_eff / np.sqrt(distances_um**2 + h_eff**2)


def _source_shape(distances_um, min_distance_um):
    """1 at the electrode surface, falling as 1/r with a floor on r."""
    return min_distance_um / (distances_um + min_distance_um)


if _USES_JAX:

    class PointSourceModel(eqx.Module):
        stimulation_gain: float = eqx.field(
            default=DEFAULT_STIMULATION_GAIN, static=True
        )
        recording_gain: float = eqx.field(default=DEFAULT_RECORDING_GAIN, static=True)
        min_distance_um: float = eqx.field(default=MIN_DISTANCE_UM, static=True)
        electrode_radius_um: float = eqx.field(default=ELECTRODE_RADIUS_UM, static=True)
        culture_height_um: float = eqx.field(default=CULTURE_HEIGHT_UM, static=True)

        def source_gain(self, distances_um):
            return self.recording_gain * _source_shape(
                distances_um, self.min_distance_um
            )

        def induction_gain(self, distances_um):
            return self.stimulation_gain * _induction_shape(
                distances_um, self.electrode_radius_um, self.culture_height_um
            )

        def potential_recording(self, gain, membrane_currents):
            return np.matmul(gain, membrane_currents)

        def set_params(self, params: dict) -> "PointSourceModel":
            unknown = sorted(k for k in params if not hasattr(self, k))
            if unknown:
                raise KeyError(f"{unknown} are not parameters of {type(self).__name__}")
            import dataclasses

            return dataclasses.replace(self, **{k: float(v) for k, v in params.items()})

        def serialize(self) -> dict:
            return {
                "stimulation_gain": float(self.stimulation_gain),
                "recording_gain": float(self.recording_gain),
                "min_distance_um": float(self.min_distance_um),
                "electrode_radius_um": float(self.electrode_radius_um),
                "culture_height_um": float(self.culture_height_um),
            }

else:

    class PointSourceModel:
        def __init__(
            self,
            stimulation_gain=DEFAULT_STIMULATION_GAIN,
            recording_gain=DEFAULT_RECORDING_GAIN,
            min_distance_um=MIN_DISTANCE_UM,
            electrode_radius_um=ELECTRODE_RADIUS_UM,
            culture_height_um=CULTURE_HEIGHT_UM,
        ):
            self.stimulation_gain = stimulation_gain
            self.recording_gain = recording_gain
            self.min_distance_um = min_distance_um
            self.electrode_radius_um = electrode_radius_um
            self.culture_height_um = culture_height_um

        def source_gain(self, distances_um):
            return self.recording_gain * _source_shape(
                distances_um, self.min_distance_um
            )

        def induction_gain(self, distances_um):
            return self.stimulation_gain * _induction_shape(
                distances_um, self.electrode_radius_um, self.culture_height_um
            )

        def potential_recording(self, gain, membrane_currents):
            return np.matmul(gain, membrane_currents)

        def set_params(self, params: dict) -> "PointSourceModel":
            unknown = sorted(k for k in params if not hasattr(self, k))
            if unknown:
                raise KeyError(f"{unknown} are not parameters of {type(self).__name__}")
            for name, value in params.items():
                setattr(self, name, float(value))
            return self

        def serialize(self) -> dict:
            return {
                "stimulation_gain": float(self.stimulation_gain),
                "recording_gain": float(self.recording_gain),
                "min_distance_um": float(self.min_distance_um),
                "electrode_radius_um": float(self.electrode_radius_um),
                "culture_height_um": float(self.culture_height_um),
            }


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

    def parameter_children(self) -> dict:
        return {}

    def set_params(self, params: dict) -> "IO":
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
    def channel_ids(self) -> Int[Array, "n_channel_ids"]:
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
        return getattr(self, "_input_space")

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
        dt: float = 1.0,
    ) -> "Stimulus":
        """Transforms channel inputs into neural inputs."""
        raise NotImplementedError("Please specify an IO")

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
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


class MEA(IO):
    """
    A multi-electrode array

    # Arguments

    electrode_coordinates
        Array of shape (n_electrodes, 4) containing the coordinates of electrodes.
        Each row represents [id, x, y, z] for an electrode.
    input_radius
        Radius within which an electrode can stimulate neurons, in micrometers.
        Default is 250.
    output_radius
        Radius within which an electrode can record from neurons, in micrometers.
        Default is 250.

    # Computed attributes

    cell_measurement
        Array containing the relative measurement weights for each neuron-electrode pair,
        filtered based on output_radius.
    """

    def __init__(
        self,
        electrode_coordinates: Float[Array, "n_electrodes ixyz=4"] | None = None,
        input_radius=250,
        output_radius=250,
        volume_conductor=None,
    ):
        if electrode_coordinates is None:
            electrode_coordinates = self.default_electrode_coordinates()
        self.electrode_coordinates = np.array(electrode_coordinates)
        self.input_radius = input_radius
        self.output_radius = output_radius
        if volume_conductor is None:
            volume_conductor = PointSourceModel()
        elif isinstance(volume_conductor, dict):
            volume_conductor = PointSourceModel(**volume_conductor)
        self.volume_conductor = volume_conductor

        self.cell_measurement = None
        self._cell_induction = None

    def parameter_children(self) -> dict:
        return {"volume_conductor": self.volume_conductor}

    @property
    def num_channels(self) -> int:
        return len(self.electrode_coordinates)

    @property
    def input_units(self) -> str:
        return "uA"

    @property
    def channel_ids(self) -> Int[Array, "n_channel_ids"]:
        return self.electrode_coordinates[:, 0].astype(np.int32)

    def serialize(self) -> dict:
        return {
            "electrode_coordinates": self.electrode_coordinates,
            "input_radius": self.input_radius,
            "output_radius": self.output_radius,
            "volume_conductor": self.volume_conductor.serialize()
            if hasattr(self.volume_conductor, "serialize")
            else None,
        }

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
        dt: float = 1.0,
    ) -> "Stimulus":
        """Map per-channel inputs (uA) to extracellular voltages (mV)."""
        from livn.stimulus import Stimulus

        autobatch = False
        if len(channel_inputs.shape) == 2:
            autobatch = True
            channel_inputs = channel_inputs[np.newaxis, ...]

        if self._cell_induction is None:
            distances = self.distances(neuron_coordinates)

            self._cell_induction = self.cell_induction(distances)

        gids, sections, keep = coupled_sections(
            neuron_coordinates, self._cell_induction
        )
        stimulus = calculate_cell_stimulus(
            channel_inputs,
            self._cell_induction,
            n_gids=len(neuron_coordinates),
            keep=keep,
        )

        return Stimulus(
            stimulus[0] if autobatch else stimulus,
            dt=dt,
            gids=gids,
            sections=sections,
            input_mode="extracellular",
            units="mV",
        )

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"] | None,
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        if ii is None:
            ii = np.unique(neuron_coordinates[:, 0])

        if self.cell_measurement is None:
            self.cell_measurement = relative_distance(
                self.distances(neuron_coordinates),
                self.output_radius,
                filter_out_of_bounds=True,
            )

        return channel_recording(self.cell_measurement, ii, *recordings)

    def source_gain(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_channels n_recording_coords"]:
        """
        Compute per-recording-coordinate gain for electrode potential.

        Uses the point-source volume conductor formula:

            g = (1 / (4*pi*sigma_S_per_mm)) * (1 / r_mm)

        Only recording coordinates within output_radius get non-zero gain.

        Arguments
        - distances: as returned by `MEA.distances(neuron_coordinates)`
        """
        n_electrodes = int(self.num_channels)
        d = distances[:, -1]
        if d.size % n_electrodes != 0:
            raise ValueError(
                f"distances size mismatch: expected a multiple of {n_electrodes}, got {d.size}"
            )

        n_recording_coords = d.size // n_electrodes
        d = d.reshape(n_electrodes, n_recording_coords)

        gain = self.volume_conductor.source_gain(d)
        mask = d <= float(self.output_radius)
        return np.where(mask, gain, 0.0)

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "n_neurons timestep"],
    ) -> Float[Array, "n_channels timestep"]:
        """
        Estimate electrode potentials from membrane currents using
        the point-source volume conductor formula:

            V [uV] = (1 / (4*pi*sigma_S_per_mm)) * sum_i( I_i[uA] / r_i[mm] )

        Only neurons within output_radius contribute.

        Arguments
        - distances: as returned by `MEA.distances(neuron_coordinates)`
        - membrane_currents: in uA, shape [n_neurons, timestep]

        Returns: microvolts array (uV), shape [n_channels, timestep]
        """
        gain = self.source_gain(distances)
        return self.volume_conductor.potential_recording(gain, membrane_currents)

    def distances(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_sources*n_coords cip=3"]:
        return calculate_distances(self.electrode_coordinates, neuron_coordinates)

    def default_electrode_coordinates(self):
        return electrode_array_coordinates()

    def cell_induction(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_amplitudes cip=3"]:
        return with_amplitude(
            distances, self.volume_conductor.induction_gain(distances[:, -1])
        )

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        """mV induced per uA at each coordinate row, per electrode."""
        induction = np.asarray(self.cell_induction(self.distances(neuron_coordinates)))
        return induction[:, -1].reshape(len(self.channel_ids), -1)

    def channel_contribution(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_id: int,
    ):
        import pandas as pd

        distances = self.distances(neuron_coordinates)
        full_gain = self.source_gain(distances)  # (n_channels, n_recording_coords)

        ch_idx = int(np.where(self.channel_ids == channel_id)[0][0])
        ch_gain = full_gain[ch_idx]

        mask = distances[:, 0] == channel_id
        ch = distances[mask]

        within = ch_gain > 0
        ch = ch[within]
        ch_gain = ch_gain[within]

        order = ch[:, 2].argsort()
        ch = ch[order]
        ch_gain = ch_gain[order]

        total = ch_gain.sum()
        contribution_pct = ch_gain / total * 100 if total > 0 else ch_gain * 0

        return pd.DataFrame(
            {
                "gid": ch[:, 1].astype(int),
                "distance_um": ch[:, 2],
                "gain": ch_gain,
                "contribution_pct": contribution_pct,
            }
        )


if _USES_JAX:

    def _mea_tree_flatten(mea):
        flat_vc = jax.tree_util.tree_leaves(mea.volume_conductor)
        vc_is_pytree = not (len(flat_vc) == 1 and flat_vc[0] is mea.volume_conductor)
        children = (mea.volume_conductor if vc_is_pytree else None,)
        aux = {
            "electrode_coordinates": mea.electrode_coordinates,
            "input_radius": mea.input_radius,
            "output_radius": mea.output_radius,
            "volume_conductor": None if vc_is_pytree else mea.volume_conductor,
            "vc_is_pytree": vc_is_pytree,
            "cell_measurement": mea.cell_measurement,
            "_cell_induction": mea._cell_induction,
        }
        return children, aux

    def _mea_tree_unflatten(aux, children):
        (vc_child,) = children
        vc = vc_child if aux["vc_is_pytree"] else aux["volume_conductor"]
        mea = MEA.__new__(MEA)
        mea.electrode_coordinates = aux["electrode_coordinates"]
        mea.input_radius = aux["input_radius"]
        mea.output_radius = aux["output_radius"]
        mea.volume_conductor = vc
        mea.cell_measurement = aux["cell_measurement"]
        mea._cell_induction = aux["_cell_induction"]
        return mea

    jax.tree_util.register_pytree_node(MEA, _mea_tree_flatten, _mea_tree_unflatten)


class LightArray(IO):
    """Fiber optic array for optical stimulation

    Maps per-fiber light power (mW) to per-neuron irradiance (mW/mm^2)
    using a Kubelka-Munk scattering propagation model
    """

    def __init__(
        self,
        fiber_coordinates: Float[Array, "n_fibers ixyz=4"],
        numerical_aperture: float = 0.37,
        fiber_radius_um: float = 100.0,
        wavelength_nm: float = 473.0,
        scattering_coefficient: float = 11.2,
    ):
        self.fiber_coordinates = np.array(fiber_coordinates)
        self.numerical_aperture = numerical_aperture
        self.fiber_radius_um = fiber_radius_um
        self.wavelength_nm = wavelength_nm
        self.scattering_coefficient = scattering_coefficient
        self._cell_induction = None

    def serialize(self) -> dict:
        return {
            "fiber_coordinates": self.fiber_coordinates.tolist(),
            "numerical_aperture": self.numerical_aperture,
            "fiber_radius_um": self.fiber_radius_um,
            "wavelength_nm": self.wavelength_nm,
            "scattering_coefficient": self.scattering_coefficient,
        }

    @property
    def num_channels(self) -> int:
        return len(self.fiber_coordinates)

    @property
    def input_units(self) -> str:
        return "mW"

    @property
    def channel_ids(self) -> Int[Array, "n_channel_ids"]:
        return self.fiber_coordinates[:, 0].astype(np.int32)

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_fibers"],
        dt: float = 0.1,
    ):
        from livn.stimulus import Stimulus

        autobatch = False
        if len(channel_inputs.shape) == 2:
            autobatch = True
            channel_inputs = channel_inputs[np.newaxis, ...]

        if self._cell_induction is None:
            self._cell_induction = self.cell_induction(neuron_coordinates)

        gids, sections, keep = coupled_sections(
            neuron_coordinates, self._cell_induction, dense=True
        )
        _check_stimulus_size(
            int(channel_inputs.shape[-2]),
            len(keep),
            np.promote_types(channel_inputs.dtype, np.float32).itemsize,
            n_reached=len(keep),
        )
        irradiance = np.einsum(
            "...tf,fg->...tg", channel_inputs, self._cell_induction[:, keep]
        )

        if autobatch:
            irradiance = irradiance[0]

        return Stimulus.from_irradiance(irradiance, dt=dt, gids=gids, sections=sections)

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"] | None,
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        raise NotImplementedError("LightArray does not support recording")

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents: Float[Array, "timestep n_neurons"],
    ) -> Float[Array, "timestep n_channels"]:
        raise NotImplementedError("LightArray does not support recording")

    def cell_induction(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
    ) -> Float[Array, "n_fibers n_gids"]:
        """Compute irradiance at each neuron per unit source power

        Kubelka-Munk model (Aravanis et al. 2007):
            I(r,z) = (NA^2) / (n^2 * (z + z0)^2) * exp(-mu_s * sqrt(r^2 + z^2))
        """
        fiber_xyz = self.fiber_coordinates[:, 1:4]
        neuron_xyz = neuron_coordinates[:, 1:4]

        n_fibers = len(fiber_xyz)
        n_gids = len(neuron_xyz)

        NA = self.numerical_aperture
        n_tissue = 1.36
        mu_s = self.scattering_coefficient  # mm^-1
        fiber_r_mm = self.fiber_radius_um / 1000.0

        # z0: distance from fiber tip where cone area equals fiber area
        z0 = fiber_r_mm * n_tissue / NA

        T = np.zeros((n_fibers, n_gids))

        for fi in range(n_fibers):
            diff = neuron_xyz - fiber_xyz[fi]
            diff_mm = diff / 1000.0  # um to mm

            r = np.sqrt(diff_mm[:, 0] ** 2 + diff_mm[:, 1] ** 2)
            z = np.abs(diff_mm[:, 2])

            dist = np.sqrt(r**2 + z**2)
            denom = (z + z0) ** 2
            denom = np.maximum(denom, 1e-12)

            T[fi, :] = (
                (NA**2 / (n_tissue**2 * denom)) * np.exp(-mu_s * dist) / (4 * np.pi)
            )

        return T

    def reach(
        self, neuron_coordinates: "Float[Array, 'n_coords ixyz=4']"
    ) -> "Float[Array, 'n_channels n_coords']":
        """Irradiance per unit source power at each coordinate row, per fiber."""
        return np.asarray(self.cell_induction(neuron_coordinates))


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
    def channel_ids(self) -> "Int[Array, 'n_channel_ids']":
        return self.outputs.channel_ids

    def cell_stimulus(
        self,
        neuron_coordinates: "Float[Array, 'n_coords ixyz=4']",
        channel_inputs: "Float[Array, 'batch timestep n_channels']",
        **kwargs,
    ):
        return self.inputs.cell_stimulus(neuron_coordinates, channel_inputs, **kwargs)

    def reach(self, neuron_coordinates: "Float[Array, 'n_coords ixyz=4']"):
        return self.inputs.reach(neuron_coordinates)

    def channel_recording(
        self,
        neuron_coordinates: "Float[Array, 'n_coords ixyz=4']" | None,
        ii: "Float[Array, 'i']",
        *recordings: "Float[Array, '_']",
    ) -> "tuple[dict[int, Array], ...]":
        return self.outputs.channel_recording(neuron_coordinates, ii, *recordings)

    def source_gain(
        self,
        distances: "Float[Array, 'n_distances cip=3']",
    ) -> "Float[Array, 'n_channels n_recording_coords']":
        return self.outputs.source_gain(distances)

    def potential_recording(
        self,
        distances: "Float[Array, 'n_distances cip=3']",
        membrane_currents: "Float[Array, 'timestep n_neurons']",
    ) -> "Float[Array, 'timestep n_channels']":
        return self.outputs.potential_recording(distances, membrane_currents)

    def distances(
        self, neuron_coordinates: "Float[Array, 'n_coords ixyz=4']"
    ) -> "Float[Array, 'n_sources*n_coords cip=3']":
        return self.outputs.distances(neuron_coordinates)


def calculate_distances(
    source: Float[Array, "n_sources cxyz=4"],
    coords: Float[Array, "n_coords cxyz=4"],
) -> Float[Array, "n_sources*n_coords cip=3"]:
    """
    Calculate the Euclidean distances between each electrode and each coordinate.
    """
    source = np.asarray(source)
    coords = np.asarray(coords)

    ex, ey, ez = source[:, 1], source[:, 2], source[:, 3]
    cx, cy, cz = coords[:, 1], coords[:, 2], coords[:, 3]

    ex = ex[:, np.newaxis]
    ey = ey[:, np.newaxis]
    ez = ez[:, np.newaxis]

    distances = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2)

    channel_ids = np.repeat(source[:, 0], len(coords))
    gids = np.tile(coords[:, 0], len(source))

    return np.column_stack((channel_ids, gids, distances.flatten()))


def channel_recording(
    ci: Float[Array, "n ci_"], ii: Float[Array, "i"], *recordings: Float[Array, "_"]
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


def electrode_array_coordinates(
    pitch: float = 1000, xs: int = 4, ys: int = 4, xoffset=500, yoffset=500, z=350 / 2
) -> Float[Array, "n_electrodes ixyz=4"]:
    """
    Generate regular electrode array coordinates.
    """
    coordinates = np.array(
        [
            [xoffset + x * pitch, yoffset + y * pitch, z]
            for y in range(ys)
            for x in range(xs)
        ]
    )

    coordinates = np.hstack(
        (np.arange(coordinates.shape[0]).reshape(-1, 1), coordinates)
    )

    return coordinates


def electrode_array_coordinates_for_area(
    pitch: float, area: tuple[tuple[float, float], tuple[float, float]], z=0
) -> Float[Array, "n_electrodes ixyz=4"]:
    """
    Generate electrode array coordinates that fit within a given area

    The number of electrodes in each dimension is the largest centered power-of-2
    that fits within the area given the pitch

    Arguments:
        pitch: spacing between electrodes
        area: ((x_min, y_min), (x_max, y_max))
        z: z coordinate for all electrodes
    """
    (x_min, y_min), (x_max, y_max) = area
    width = x_max - x_min
    height = y_max - y_min

    max_xs = int(width // pitch) + 1
    max_ys = int(height // pitch) + 1

    xs = 1 << (max_xs.bit_length() - 1) if max_xs >= 1 else 1
    ys = 1 << (max_ys.bit_length() - 1) if max_ys >= 1 else 1

    grid_width = (xs - 1) * pitch if xs > 1 else 0
    grid_height = (ys - 1) * pitch if ys > 1 else 0

    xoffset = x_min + (width - grid_width) / 2
    yoffset = y_min + (height - grid_height) / 2

    return electrode_array_coordinates(
        pitch=pitch, xs=xs, ys=ys, xoffset=xoffset, yoffset=yoffset, z=z
    )


if _USES_JAX:

    def with_amplitude(distances, amplitudes):
        """`distances` with its last column replaced."""
        return distances.at[:, -1].set(amplitudes)

    def relative_distance(
        distances: Float[Array, "n_distances cip=3"],
        boundary: float,
        filter_out_of_bounds: bool = False,
    ) -> Float[Array, "n_distances cip=3"]:
        """
        Normalizes the distances within a boundary radius to [0, 1]
        and optionally discards out-of-bounds distances.
        """
        distances = np.array(distances)
        if filter_out_of_bounds:
            distances = distances[distances[:, -1] <= boundary]
        distances = distances.at[:, -1].set(distances[:, -1] / boundary)
        return distances

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        c_induction: Float[Array, "n_inductions cip=3"],
        n_gids: int | None = None,
        keep: "Int[Array, 'n_keep'] | None" = None,
    ) -> Float[Array, "batch timestep n_gids"]:
        """
        Calculate the stimulus strength for each cell gid and each timestep
        by multiplying cell induction and electrode stimulus.
        """
        stimulus = np.asarray(electrode_stimulus)
        c_induction = np.asarray(c_induction)

        batch_size, n_timesteps, n_channels = electrode_stimulus.shape
        per_coordinate = n_gids is not None
        if n_gids is None:
            # no-jit
            n_gids = len(np.unique(c_induction[:, 1]))

        # sparse matrix for cell induction
        channel_ids = c_induction[:, 0].astype(int)
        amplitudes = c_induction[:, 2]

        gids = c_induction[:, 1].astype(int)
        _check_stimulus_size(
            n_timesteps,
            n_gids if keep is None else len(keep),
            np.promote_types(stimulus.dtype, np.float32).itemsize,
            n_reached=int(np.count_nonzero(c_induction[:, 2])) or None,
        )
        induction_matrix = np.zeros(
            (n_channels, n_gids), dtype=np.promote_types(stimulus.dtype, np.float32)
        )
        if per_coordinate:
            columns = np.arange(c_induction.shape[0]) % n_gids
        else:
            _unique, columns = np.unique(gids, return_inverse=True, size=n_gids)
        induction = induction_matrix.at[channel_ids, columns].set(amplitudes)
        if keep is not None:
            induction = induction[:, keep]

        # reduce over gids
        # Result shape: [batch, timestep, n_gids]
        cell_stimulus = np.einsum("btn,ng->btg", stimulus, induction)

        return cell_stimulus

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

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        cell_induction: Float[Array, "n_inductions cip=3"],
        n_gids: int | None = None,
        *args,
        keep: "Int[Array, 'n_keep'] | None" = None,
        **kwargs,
    ) -> Float[Array, "batch timestep n_gids"]:
        """
        Calculate the stimulus strength for each cell gid and each timestep
        by multiplying cell induction and electrode stimulus.
        """
        electrode_stimulus = np.asarray(electrode_stimulus)
        cell_induction = np.asarray(cell_induction)

        batch_size, n_timesteps, n_channels = electrode_stimulus.shape

        # sparse matrix for cell induction
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

        dtype = np.promote_types(electrode_stimulus.dtype, np.float32)
        _check_stimulus_size(
            n_timesteps,
            n_gids if keep is None else len(keep),
            dtype.itemsize,
            n_reached=int(np.count_nonzero(cell_induction[:, 2])) or None,
        )

        induction_matrix = np.zeros((n_channels, n_gids), dtype=dtype)
        # handle case where channel ids do not start at 0
        _, channel_indices = np.unique(channel_ids, return_inverse=True)
        induction_matrix[channel_indices, columns] = amplitudes
        if keep is not None:
            induction_matrix = induction_matrix[:, keep]

        # reduce over gids
        # Result shape: [batch, timestep, n_gids]
        cell_stimulus = np.matmul(electrode_stimulus, induction_matrix)

        return cell_stimulus
