from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as _np

from livn.backend import backend
from livn.io.base import IO, _check_stimulus_size, coupled_sections
from livn.io.medium import GaussianBlur, Medium
from livn.utils import import_instance

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

_USES_JAX = False

if "ax" in backend():
    import jax
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class _Light(IO):
    medium: Medium
    wavelength: float
    _cell_induction: object

    @property
    def optical(self):
        return self.medium.require("optical")

    def parameter_children(self) -> dict:
        return {"medium": self.medium}

    def cell_induction(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        raise NotImplementedError

    def cell_stimulus(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
        channel_inputs: Float[Array, "batch timestep n_channels"],
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

        return Stimulus.from_irradiance(
            irradiance,
            dt=dt,
            gids=gids,
            sections=sections,
            wavelength=float(self.wavelength),
        )

    def channel_recording(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"] | None,
        ii: Float[Array, " i"],
        *recordings: Float[Array, " _"],
    ) -> tuple[dict[int, Array], ...]:
        raise NotImplementedError(f"{type(self).__name__} does not support recording")

    def potential_recording(
        self,
        distances: Float[Array, "n_distances cip=3"],
        membrane_currents_uA: Float[Array, "n_neurons timestep"],
        dt: float | None = None,
        key=None,
    ) -> Float[Array, "n_channels timestep"]:
        raise NotImplementedError(f"{type(self).__name__} does not support recording")

    def reach(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_channels n_coords"]:
        return np.asarray(self.cell_induction(neuron_coordinates))


class LightArray(_Light):
    """Fiber optic array for optical stimulation."""

    def __init__(
        self,
        fiber_coordinates: Float[Array, "n_fibers ixyz=4"],
        numerical_aperture: float = 0.37,
        fiber_radius: float = 100.0,
        wavelength: float = 473.0,
        medium=None,
    ):
        self.fiber_coordinates = np.array(fiber_coordinates)
        self.numerical_aperture = numerical_aperture
        self.fiber_radius = fiber_radius
        self.wavelength = wavelength
        self.medium = Medium() if medium is None else import_instance(medium)
        self._cell_induction = None
        self._stack = ()

    def serialize(self) -> dict:
        return {
            "fiber_coordinates": self.fiber_coordinates.tolist(),
            "numerical_aperture": self.numerical_aperture,
            "fiber_radius": self.fiber_radius,
            "wavelength": self.wavelength,
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

    def cell_induction(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
    ) -> Float[Array, "n_fibers n_gids"]:
        """Compute irradiance at each neuron per unit source power, through
        the medium's optical part."""
        fiber_xyz = self.fiber_coordinates[:, 1:4]
        neuron_xyz = np.asarray(neuron_coordinates)[:, 1:4]
        offsets = neuron_xyz[np.newaxis, :, :] - fiber_xyz[:, np.newaxis, :]

        return self.optical.irradiance_gain(
            offsets, self.numerical_aperture, self.fiber_radius
        )


class PatternedLight(_Light):
    """Spots of light projected onto a monolayer."""

    def __init__(
        self,
        spots: Float[Array, "n_spots ixyr=4"],
        irradiance: float = 1.0,
        wavelength: float = 470.0,
        medium=None,
        labels: Sequence[str] | None = None,
    ):
        spots = _np.asarray(spots, dtype=float).reshape(-1, 4)
        if _np.any(spots[:, 3] <= 0.0):
            raise ValueError("a spot is a disc: every radius must be positive")
        if len(_np.unique(spots[:, 0])) != len(spots):
            raise ValueError("every spot needs its own id")
        if labels is not None and len(labels) != len(spots):
            raise ValueError(f"{len(labels)} labels for {len(spots)} spots")
        self.spots = np.array(spots)
        self.irradiance = float(irradiance)
        self.wavelength = float(wavelength)
        self.labels = None if labels is None else [str(label) for label in labels]
        self.medium = (
            Medium(optical=GaussianBlur())
            if medium is None
            else import_instance(medium)
        )
        self._cell_induction = None
        self._stack = ()

    @classmethod
    def from_channels(cls, channels, **kwargs) -> PatternedLight:
        """The spots of a list of light channels, in sample um."""
        channels = list(channels)
        if not channels:
            raise ValueError("there are no light channels")
        spots, ids = [], []
        for c in channels:
            shape = c.get("shape") or {}
            kind = shape.get("kind")
            if kind != "disc":
                what = "has no shape" if kind is None else f"is a {kind!r} shape"
                raise ValueError(
                    f"channel {c.get('id')!r} {what}; a PatternedLight "
                    "models discs, {'kind': 'disc', 'x', 'y', 'radius'}"
                )
            missing = [k for k in ("x", "y", "radius") if shape.get(k) is None]
            if missing:
                raise ValueError(f"channel {c.get('id')!r} has no {missing}")
            ids.append(str(c["id"]))
            spots.append([float(shape["x"]), float(shape["y"]), float(shape["radius"])])
        numeric = all(_is_int(i) for i in ids)
        spots = [
            [int(i) if numeric else k, *spot]
            for k, (i, spot) in enumerate(zip(ids, spots, strict=True))
        ]
        if any("label" in c for c in channels):
            kwargs.setdefault(
                "labels",
                [str(c.get("label", i)) for c, i in zip(channels, ids, strict=True)],
            )
        elif not numeric:
            kwargs.setdefault("labels", ids)
        return cls(spots, **kwargs)

    def serialize(self) -> dict:
        return {
            "spots": _np.asarray(self.spots).tolist(),
            "irradiance": self.irradiance,
            "wavelength": self.wavelength,
            "medium": self.medium.serialize(),
            "labels": self.labels,
        }

    @property
    def num_channels(self) -> int:
        return len(self.spots)

    @property
    def input_units(self) -> str:
        return "normalized"

    @property
    def channel_ids(self) -> Int[Array, " n_channel_ids"]:
        return self.spots[:, 0].astype(np.int32)

    def _irradiance_at(self, xy):
        spot_xy = self.spots[:, 1:3]
        offsets = np.asarray(xy)[np.newaxis, :, :] - spot_xy[:, np.newaxis, :]
        return self.irradiance * self.optical.spot_gain(offsets, self.spots[:, 3])

    def cell_induction(
        self, neuron_coordinates: Float[Array, "n_coords ixyz=4"]
    ) -> Float[Array, "n_spots n_coords"]:
        """Irradiance (mW/mm^2) at each coordinate row per unit command, per spot."""
        return self._irradiance_at(np.asarray(neuron_coordinates)[:, 1:3])

    def electrode_irradiance(
        self, electrode_xy: Float[Array, "n_electrodes xy=2"]
    ) -> Float[Array, "n_spots n_electrodes"]:
        """Irradiance (mW/mm^2) on each electrode per unit command, per spot."""
        return self._irradiance_at(electrode_xy)


def _is_int(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return False
    return True


if _USES_JAX:

    def _light_flatten(device):
        aux = {k: v for k, v in device.__dict__.items() if k != "medium"}
        return (device.medium,), (type(device), aux)

    def _light_unflatten(aux, children):
        cls, fields = aux
        (medium,) = children
        device = cls.__new__(cls)
        device.__dict__.update(fields)
        device.medium = medium
        return device

    for _cls in (LightArray, PatternedLight):
        jax.tree_util.register_pytree_node(_cls, _light_flatten, _light_unflatten)
