from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Self,
    Tuple,
    Union,
    runtime_checkable,
)

import gymnasium
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from jaxtyping import Array as JaxArray
    from mpi4py import MPI
    from numpy import ndarray
    from tensorflow import TfTensor
    from torch import TorchTensor

    from livn.io import IO
    from livn.stimulus import Stimulus
    from livn.system import System
    from livn.types import Model

    Array = Union[TorchTensor, ndarray, JaxArray, TfTensor]
else:
    arrays = []
    try:
        from torch import Tensor as TorchTensor
    except Exception:
        pass
    else:
        arrays.append(TorchTensor)
    try:
        from numpy import ndarray
    except Exception:
        pass
    else:
        arrays.append(ndarray)
    try:
        from jaxtyping import Array as JaxArray
    except Exception:
        pass
    else:
        arrays.append(JaxArray)
    try:
        from tensorflow import Tensor as TfTensor
    except Exception:
        pass
    else:
        arrays.append(TfTensor)
    Array = Union[tuple(arrays)]

from jaxtyping import Float, Int

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName


class SynapticParam(BaseModel):
    population: Optional[str] = None
    source: Optional[str] = None
    sec_type: Optional[str] = None
    syn_name: Optional[str] = None
    param_path: Optional[Union[str, Tuple[str, ...]]] = None
    param_range: Optional[str] = None
    phenotype: Optional[str] = None

    @field_validator("param_path")
    @classmethod
    def parse_path(
        cls, v: Optional[Union[str, Tuple[str, ...]]]
    ) -> Optional[Tuple[str, ...]]:
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, str):
            if "/" not in v:
                return v

            return tuple(v.split("/"))
        raise ValueError(f"Invalid param_path type: {type(v)}")

    @classmethod
    def from_string(cls, string: str) -> "SynapticParam":
        "'population_source-sec_type-syn_name-param_path-param_range-phenotype'"
        try:
            pop_rest = string.split("_", 1)
            if len(pop_rest) != 2:
                raise ValueError("String must contain exactly one underscore")

            population, rest = pop_rest

            parts = rest.split("-")

            data = {"population": population}

            data["source"] = parts[0]

            optional_fields = [
                "sec_type",
                "syn_name",
                "param_path",
                "param_range",
                "phenotype",
            ]
            for i, field in enumerate(optional_fields):
                if len(parts) > i + 1:
                    data[field] = parts[i + 1]
                else:
                    data[field] = None

            return cls(**data)

        except Exception as e:
            raise ValueError(f"Failed to parse string '{string}': {str(e)}")


@runtime_checkable
class Env(Protocol):
    """Protocol defining the interface for livn environments"""

    def __init__(
        self,
        system: Union["System", str],
        model: "Model",
        io: "IO",
        seed: int | None = 123,
        comm: Optional["MPI.Intracomm"] = None,
        subworld_size: int | None = None,
    ): ...

    def apply_model_defaults(self, weights: bool = True, noise: bool = True) -> Self:
        self.model.apply_defaults(self, weights=weights, noise=noise)

        return self

    def cell_stimulus(
        self,
        channel_inputs: Float[Array, "batch timestep n_channels"],
    ) -> Float[Array, "batch timestep n_gids"]:
        """Transforms channel inputs into neural inputs"""
        return self.io.cell_stimulus(self.system.neuron_coordinates, channel_inputs)

    def channel_recording(
        self,
        ii: Float[Array, "i"],
        *recordings: Float[Array, "_"],
    ) -> tuple[dict[int, Array], ...]:
        """Transforms neural recordings identified by their gids into per channel recordings"""
        return self.io.channel_recording(
            self.system.neuron_coordinates, ii, *recordings
        )

    def init(self) -> Self:
        """Initialize the environment."""
        ...
        return self

    def set_weights(self, weights: dict) -> Self:
        """Set the synaptic weights"""
        ...
        return self

    def set_noise(self, exc: float = 1.0, inh: float = 1.0) -> Self:
        """Set noise"""
        ...
        return self

    def record_spikes(self, population: str | list | tuple | None = None) -> Self:
        """Enable spike recording for population"""
        if population is None:
            population = self.system.populations
        if isinstance(population, (list, tuple)):
            for p in population:
                self.record_spikes(p)
            return self

        self._record_spikes(population)

        return self

    def _record_spikes(self, population: str) -> Self: ...

    def record_voltage(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        """Enable voltage recording for population"""
        if population is None:
            population = self.system.populations
        if isinstance(population, (list, tuple)):
            for p in population:
                self.record_voltage(p, dt=dt)
            return self

        self._record_voltage(population, dt)

        return self

    def _record_voltage(self, population: str, dt: float) -> Self: ...

    def run(
        self,
        duration,
        stimulus: Optional["Stimulus"] = None,
        dt: float = 0.025,
        **kwargs,
    ) -> Tuple[
        Int[Array, "n_spiking_neuron_ids"],
        Float[Array, "n_spiking_neuron_times"],
        Int[Array, "n_voltage_neuron_ids"],
        Float[Array, "neuron_ids voltages"],
    ]:
        """Run the simulation"""
        ...

    def clear(self) -> Self:
        """Discard the simulation and reset to t=0"""
        ...

        return self


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for livn models"""

    def apply_defaults(self, env, weights: bool = True, noise: bool = True):
        if weights:
            env.set_weights(self.default_weights(env.system.name, default={}))

        if noise:
            env.set_noise(**self.default_noise(env.system.name, default={}))

    def default_noise(self, system: str, backend: str | None = None, default=None):
        from livn.backend import backend as current_backend

        if backend is None:
            backend = current_backend()

        try:
            return getattr(self, f"{backend}_default_noise")(system)
        except (AttributeError, KeyError):
            if default is None:
                raise
            return default

    def default_weights(self, system: str, backend: str | None = None, default=None):
        from livn.backend import backend as current_backend

        if backend is None:
            backend = current_backend()

        try:
            return getattr(self, f"{backend}_default_weights")(system)
        except (AttributeError, KeyError):
            if default is None:
                raise
            return default


@runtime_checkable
class Encoding(Protocol):
    def __call__(
        self, features: Any, *args, **kwargs
    ) -> Tuple[Float[Array, "batch timestep n_channels"], float]: ...

    @property
    def feature_space(self) -> gymnasium.Space:
        if not hasattr(self, "_feature_space"):
            self._feature_space = self._get_feature_space()
        return getattr(self, "_feature_space")

    def _get_feature_space(self) -> gymnasium.Space:
        raise NotImplementedError


@runtime_checkable
class Decoding(Protocol):
    def __call__(
        self,
        duration: float,
        it: Int[Array, "n_spiking_neuron_ids"],
        tt: Float[Array, "n_spiking_neuron_times"],
        iv: Int[Array, "n_voltage_neuron_ids"],
        vv: Float[Array, "neuron_ids voltages"],
    ) -> Any: ...

    @property
    def output_space(self) -> gymnasium.Space:
        raise NotImplementedError
