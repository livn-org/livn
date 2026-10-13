from __future__ import annotations

from typing import TYPE_CHECKING

from livn.backend import backend
from livn.io.base import IO, _check_stimulus_size, coupled_sections

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


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
    def channel_ids(self) -> Int[Array, " n_channel_ids"]:
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
        ii: Float[Array, " i"],
        *recordings: Float[Array, " _"],
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
        neuron_xyz = np.asarray(neuron_coordinates)[:, 1:4]

        NA = self.numerical_aperture
        n_tissue = 1.36
        mu_s = self.scattering_coefficient  # mm^-1
        fiber_r_mm = self.fiber_radius_um / 1000.0

        # z0: distance from fiber tip where cone area equals fiber area
        z0 = fiber_r_mm * n_tissue / NA

        diff_mm = (neuron_xyz[np.newaxis, :, :] - fiber_xyz[:, np.newaxis, :]) / 1000.0

        r = np.sqrt(diff_mm[..., 0] ** 2 + diff_mm[..., 1] ** 2)
        z = np.abs(diff_mm[..., 2])

        dist = np.sqrt(r**2 + z**2)
        denom = np.maximum((z + z0) ** 2, 1e-12)

        return (NA**2 / (n_tissue**2 * denom)) * np.exp(-mu_s * dist) / (4 * np.pi)

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        """Irradiance per unit source power at each coordinate row, per fiber."""
        return np.asarray(self.cell_induction(neuron_coordinates))
