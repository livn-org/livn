from __future__ import annotations

import dataclasses
import numbers
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as _np

if TYPE_CHECKING:
    from livn.types import Array, Float, Int

__all__ = ["Conditions", "static"]


def static(default=None):
    return dataclasses.field(default=default, metadata={"static": True})


# classes defined before jax was imported
_pending: list[type] = []


def _jax_ready() -> bool:
    return "jax" in sys.modules or "ax" in os.environ.get("LIVN_BACKEND", "")


def _register(cls) -> None:
    if not _jax_ready():
        _pending.append(cls)
        return
    import jax

    fields = dataclasses.fields(cls)
    jax.tree_util.register_dataclass(
        cls,
        data_fields=[f.name for f in fields if not f.metadata.get("static")],
        meta_fields=[f.name for f in fields if f.metadata.get("static")],
    )


def _register_pending() -> None:
    while _pending and _jax_ready():
        _register(_pending.pop(0))


@dataclasses.dataclass(frozen=True, eq=False)
class Conditions:
    """Everything a recording composition may depend on in one call."""

    # run
    membrane_currents_uA: Float[Array, "n_neurons timestep"] | None = None
    dt: float | None = static()  # of membrane_currents_uA and the signal
    sections: tuple[str, ...] | None = static()
    # inputs
    inputs: Float[Array, "timestep n_input_channels"] | None = None
    inputs_dt: float | None = static()
    stim_markers: Float[Array, " n_markers"] | None = None  # on the window clock
    # env
    neuron_coordinates: Float[Array, "n_neurons ixyz=4"] | None = None
    gids: Int[Array, " n_neurons"] | None = None
    # data window
    t0: float | None = None  # since the session's frame 0
    segment: int | None = None
    start: int | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True, eq=False)(cls)
        _register(cls)

    def __post_init__(self):
        if self.sections is not None and not isinstance(self.sections, tuple):
            names = tuple(str(n) for n in _np.asarray(self.sections).ravel())
            object.__setattr__(self, "sections", names)
        for f in dataclasses.fields(self):
            if f.metadata.get("static"):
                value = getattr(self, f.name)
                if isinstance(value, numbers.Real) and not isinstance(value, bool):
                    object.__setattr__(self, f.name, float(value))
        if _pending:
            _register_pending()

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The names a block of a composition may ask for."""
        return tuple(f.name for f in dataclasses.fields(cls))

    def replace(self, **fields) -> Any:
        """A copy with `fields` set (`None` unsets one)."""
        unknown = sorted(set(fields) - set(self.field_names()))
        if unknown:
            raise TypeError(
                f"{unknown} are not fields of {type(self).__name__}; subclass it "
                "to add fields"
            )
        return dataclasses.replace(self, **fields)

    def as_dict(self) -> dict[str, Any]:
        """The fields that are set, by name."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }


_register(Conditions)
