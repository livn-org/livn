"""IO

i = neuron_gid
c = channel_id
x = x coordinate
y = y coordinate
z = z coordinate
p = payload value (distance, induction strength etc.)
"""

import os
from collections import defaultdict

import gymnasium

from livn.backend import backend
from livn.types import Array, Float, Int
from livn.utils import Jsonable

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


def _empty_array():
    return np.array([])


class IO(Jsonable):
    """
    IO

    partial-like utility to maintain state
    associated with an IO transformation
    """

    @classmethod
    def from_directory(cls, directory: str):
        return cls.from_json(os.path.join(directory, cls.__name__.lower() + ".json"))

    @property
    def num_channels(self) -> int:
        raise NotImplementedError("Please specify an IO")

    @property
    def channel_ids(self) -> Int[Array, "n_channel_ids"]:
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
    ) -> Float[Array, "batch timestep n_gids"]:
        """Transforms channel inputs into neural inputs"""
        raise NotImplementedError("Please specify an IO")

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        """Transforms neural recordings identified by their gids into per channel recordings"""
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
    cell_induction
        Array containing the stimulation amplitudes for each neuron-electrode pair,
        calculated based on distances.
    """

    def __init__(
        self,
        electrode_coordinates: Float[Array, "n_electrodes ixyz=4"] | None = None,
        input_radius=250,
        output_radius=250,
    ):
        if electrode_coordinates is None:
            electrode_coordinates = self.default_electrode_coordinates()
        self.electrode_coordinates = np.array(electrode_coordinates)
        self.input_radius = input_radius
        self.output_radius = output_radius

        self.cell_measurement = None
        self.cell_induction = None

    @property
    def num_channels(self) -> int:
        return len(self.electrode_coordinates)

    @property
    def channel_ids(self) -> Int[Array, "n_channel_ids"]:
        return self.electrode_coordinates[:, 0].astype(np.int32)

    def serialize(self) -> dict:
        return {
            "electrode_coordinates": self.electrode_coordinates,
            "input_radius": self.input_radius,
            "output_radius": self.output_radius,
        }

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
    ) -> Float[Array, "batch timestep n_gids"]:
        """Map per-channel current commands (µA) to extracellular voltages (mV)."""
        autobatch = False
        if len(channel_inputs.shape) == 2:
            autobatch = True
            channel_inputs = channel_inputs[np.newaxis, ...]

        if self.cell_induction is None:
            distances = self.distances(neuron_coordinates)

            self.cell_induction = self.amplitude_from_distance(distances)

        stimulus = calculate_cell_stimulus(
            channel_inputs, self.cell_induction, n_gids=len(neuron_coordinates)
        )

        if autobatch:
            return stimulus[0]

        return stimulus

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
        n_electrodes = int(self.num_channels)
        d = distances[:, -1]
        if d.size % n_electrodes != 0:
            raise ValueError(
                f"distances size mismatch: expected a multiple of {n_electrodes}, got {d.size}"
            )

        n_neurons_expected = d.size // n_electrodes

        d = d.reshape(n_electrodes, n_neurons_expected)

        weights = electrode_potential_point_source_weights(d)
        mask = d <= float(self.output_radius)
        weights = np.where(mask, weights, 0.0)

        # compute potentials [E, t] = [E, I] @ [I, t]
        return np.matmul(weights, membrane_currents)

    def distances(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_sources*n_coords cip=3"]:
        return np.copy(
            calculate_distances(self.electrode_coordinates, neuron_coordinates)
        )

    def default_electrode_coordinates(self):
        return electrode_array_coordinates()

    def amplitude_from_distance(
        self,
        distances: Float[Array, "n_distances cip=3"],
    ) -> Float[Array, "n_amplitudes cip=3"]:
        return disk_electrode_model(distances)


def calculate_distances(
    source: Float[Array, "n_sources cxyz=4"],
    coords: Float[Array, "n_coords cxyz=4"],
) -> Float[Array, "n_sources*n_coords cip=3"]:
    """
    Calculate the Euclidean distances between each electrode and each coordinate.
    """
    source = np.array(source)
    coords = np.array(coords)

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


def electrode_potential_point_source_weights(
    distances_um: Float[Array, "E I"],
    *,
    sigma_S_per_mm: float = 0.0003,
    min_distance_um: float = 5.0,
) -> Float[Array, "E I"]:
    """
    Pre-scale weights for electrode potential accumulation.
    Returns: (1 / (4*pi*sigma_S_per_mm)) * (1 / r_mm)
    """
    r_mm = (np.asarray(distances_um) + float(min_distance_um)) / 1000.0
    factor = 1.0 / (4.0 * np.pi * float(sigma_S_per_mm))
    return factor * (1.0 / r_mm)


if _USES_JAX:

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

    def point_source_model(
        distances: Float[Array, "n_distances cip=3"],
        tissue_resistivity_ohm_m: float = 3.5,
        per_unit_current_uA: float = 1.0,
        min_distance_um: float = 5.0,
    ) -> Float[Array, "n_amplitudes cip=3"]:
        """
        Compute the per-microamp amplitude using V = (pI)/(4πr).
        """
        current_A = per_unit_current_uA * 1e-6
        r_values = distances[:, -1]
        r = (r_values + min_distance_um) * 1e-6  # cap small distances
        eV = (tissue_resistivity_ohm_m * current_A) / (4 * np.pi * r)
        amplitudes = distances.at[:, -1].set(eV * 1000)  # mV per uA

        return amplitudes

    def disk_electrode_model(
        distances: Float[Array, "n_distances cip=3"],
        tissue_resistivity_ohm_m: float = 3.5,
        per_unit_current_uA: float = 1.0,
        electrode_radius_um: float = 15.0,
        culture_height_um: float = 50.0,
    ) -> Float[Array, "n_amplitudes cip=3"]:
        """
        Compute extracellular potential for in vitro planar MEA setup

        V ~ (pI)/(4π) * 1/sqrt(r^2 + h^2)

        where h is an effective height parameter that accounts for
        the disk electrode geometry and finite medium
        """
        current_A = per_unit_current_uA * 1e-6

        # effective distance
        r_um = distances[:, -1]
        h_eff_um = np.sqrt(electrode_radius_um**2 + culture_height_um**2)
        r_eff = np.sqrt(r_um**2 + h_eff_um**2) * 1e-6  # meters

        eV = (tissue_resistivity_ohm_m * current_A) / (4 * np.pi * r_eff)
        amplitudes = distances.at[:, -1].set(eV * 1000)  # mV per uA

        return amplitudes

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        c_induction: Float[Array, "n_inductions cip=3"],
        n_gids: int | None = None,
    ) -> Float[Array, "batch timestep n_gids"]:
        """
        Calculate the stimulus strength for each cell gid and each timestep
        by multiplying cell induction and electrode stimulus.
        """
        stimulus = np.asarray(electrode_stimulus)
        c_induction = np.asarray(c_induction)

        batch_size, n_timesteps, n_channels = electrode_stimulus.shape
        if n_gids is None:
            # no-jit
            n_gids = len(np.unique(c_induction[:, 1]))

        # sparse matrix for cell induction
        channel_ids = c_induction[:, 0].astype(int)
        gids = c_induction[:, 1].astype(int)
        amplitudes = c_induction[:, 2]

        # induction matrix [n_channels, n_gids]
        induction_matrix = np.zeros((n_channels, n_gids))
        unique_gids, gids_indices = np.unique(gids, return_inverse=True, size=n_gids)
        induction = induction_matrix.at[channel_ids, gids_indices].set(amplitudes)

        # reduce over gids
        # Result shape: [batch, timestep, n_gids]
        cell_stimulus = np.einsum("btn,ng->btg", stimulus, induction)

        return cell_stimulus

else:

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

    def point_source_model(
        distances: Float[Array, "n_distances cip=3"],
        tissue_resistivity_ohm_m: float = 3.5,
        per_unit_current_uA: float = 1.0,
        min_distance_um: float = 5.0,
    ) -> Float[Array, "n_amplitudes cip=3"]:
        """
        Compute the per-microamp amplitude using V = (pI)/(4πr).
        """
        distances = np.asarray(distances)
        current_A = per_unit_current_uA * 1e-6
        r = (distances[:, -1] + min_distance_um) * 1e-6
        eV = (tissue_resistivity_ohm_m * current_A) / (4 * np.pi * r)
        distances[:, -1] = eV * 1000  # mV per uA

        return distances

    def disk_electrode_model(
        distances: Float[Array, "n_distances cip=3"],
        tissue_resistivity_ohm_m: float = 3.5,
        per_unit_current_uA: float = 1.0,
        electrode_radius_um: float = 15.0,
        culture_height_um: float = 50.0,
    ) -> Float[Array, "n_amplitudes cip=3"]:
        """
        Compute extracellular potential for in vitro planar MEA setup

        V ~ (pI)/(4π) * 1/sqrt(r^2 + h^2)

        where h is an effective height parameter that accounts for
        the disk electrode geometry and finite medium
        """
        current_A = per_unit_current_uA * 1e-6

        # effective distance
        r_um = distances[:, -1]
        h_eff_um = np.sqrt(electrode_radius_um**2 + culture_height_um**2)
        r_eff = np.sqrt(r_um**2 + h_eff_um**2) * 1e-6  # meters

        eV = (tissue_resistivity_ohm_m * current_A) / (4 * np.pi * r_eff)
        distances[:, -1] = eV * 1000  # mV per uA

        return distances

    def calculate_cell_stimulus(
        electrode_stimulus: Float[Array, "batch timestep n_channels"],
        cell_induction: Float[Array, "n_inductions cip=3"],
        *args,
        **kwargs,
    ) -> Float[Array, "batch timestep n_gids"]:
        """
        Calculate the stimulus strength for each cell gid and each timestep
        by multiplying cell induction and electrode stimulus.
        """
        electrode_stimulus = np.asarray(electrode_stimulus)
        cell_induction = np.asarray(cell_induction)

        batch_size, n_timesteps, n_channels = electrode_stimulus.shape
        n_gids = len(np.unique(cell_induction[:, 1]))

        # sparse matrix for cell induction
        channel_ids = cell_induction[:, 0].astype(int)
        gids = cell_induction[:, 1].astype(int)
        amplitudes = cell_induction[:, 2]

        # induction matrix [n_channels, n_gids]
        induction_matrix = np.zeros((n_channels, n_gids))
        unique_gids, gids_indices = np.unique(gids, return_inverse=True)
        induction_matrix[channel_ids, gids_indices] = amplitudes

        # reduce over gids
        # Result shape: [batch, timestep, n_gids]
        cell_stimulus = np.matmul(electrode_stimulus, induction_matrix)

        return cell_stimulus
