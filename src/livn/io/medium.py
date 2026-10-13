from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as _np

from livn.backend import backend
from livn.io.base import (
    CULTURE_HEIGHT_UM,
    DEFAULT_RECORDING_GAIN,
    DEFAULT_STIMULATION_GAIN,
    ELECTRODE_RADIUS_UM,
    MIN_DISTANCE_UM,
)
from livn.utils import is_traced

if TYPE_CHECKING:
    pass

_USES_JAX = False

if "ax" in backend():
    import equinox as eqx
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


def _induction_shape(distances_um, electrode_radius_um, culture_height_um):
    """1 at the electrode surface, falling as 1/sqrt(r^2 + h^2)."""
    xp = _np if not is_traced(distances_um) else np
    h_eff = (electrode_radius_um**2 + culture_height_um**2) ** 0.5
    return h_eff / xp.sqrt(distances_um**2 + h_eff**2)


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

        def set_params(self, params: dict) -> PointSourceModel:
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

        def set_params(self, params: dict) -> PointSourceModel:
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
