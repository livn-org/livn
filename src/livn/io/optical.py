from __future__ import annotations

from typing import TYPE_CHECKING

from livn.backend import backend
from livn.io._component import resolve_component
from livn.io.base import IO, _check_stimulus_size, coupled_sections
from livn.io.medium import Medium

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class LightArray(IO):
    """Fiber optic array for optical stimulation

    Maps per-fiber light power (mW) to per-neuron irradiance (mW/mm^2)
    through the medium's optical part (by default Kubelka-Munk scattering).
    """

    def __init__(
        self,
        fiber_coordinates: Float[Array, "n_fibers ixyz=4"],
        numerical_aperture: float = 0.37,
        fiber_radius_um: float = 100.0,
        wavelength_nm: float = 473.0,
        medium=None,
    ):
        self.fiber_coordinates = np.array(fiber_coordinates)
        self.numerical_aperture = numerical_aperture
        self.fiber_radius_um = fiber_radius_um
        self.wavelength_nm = wavelength_nm
        self.medium = Medium() if medium is None else resolve_component(medium)
        self._cell_induction = None
        self._stack = ()

    def parameter_children(self) -> dict:
        return {"medium": self.medium}

    @property
    def optical(self):
        return self.medium.require("optical")

    def serialize(self) -> dict:
        return {
            "fiber_coordinates": self.fiber_coordinates.tolist(),
            "numerical_aperture": self.numerical_aperture,
            "fiber_radius_um": self.fiber_radius_um,
            "wavelength_nm": self.wavelength_nm,
            "medium": self.medium.serialize(),
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
        membrane_currents: Float[Array, "n_neurons timestep"],
        dt: float | None = None,
        key=None,
    ) -> Float[Array, "n_channels timestep"]:
        raise NotImplementedError("LightArray does not support recording")

    def cell_induction(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
    ) -> Float[Array, "n_fibers n_gids"]:
        """Compute irradiance at each neuron per unit source power, through
        the medium's optical part."""
        fiber_xyz = self.fiber_coordinates[:, 1:4]
        neuron_xyz = np.asarray(neuron_coordinates)[:, 1:4]
        offsets_um = neuron_xyz[np.newaxis, :, :] - fiber_xyz[:, np.newaxis, :]

        return self.optical.irradiance_gain(
            offsets_um, self.numerical_aperture, self.fiber_radius_um
        )

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        """Irradiance per unit source power at each coordinate row, per fiber."""
        return np.asarray(self.cell_induction(neuron_coordinates))


if _USES_JAX:

    def _light_flatten(array):
        aux = {k: v for k, v in array.__dict__.items() if k != "medium"}
        return (array.medium,), aux

    def _light_unflatten(aux, children):
        (medium,) = children
        array = LightArray.__new__(LightArray)
        array.__dict__.update(aux)
        array.medium = medium
        return array

    jax.tree_util.register_pytree_node(LightArray, _light_flatten, _light_unflatten)
