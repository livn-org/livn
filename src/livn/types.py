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

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    import gymnasium
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
    from numpy import ndarray

    Array = ndarray

from jaxtyping import Float, Int

PopulationName = str
PostSynapticPopulationName = PopulationName
PreSynapticPopulationName = PopulationName

# list | dict | tuple | Stimulus | Float[Array, "batch timestep n_channels"] | None
StimulusLike = Any


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
        return self.io.cell_stimulus(
            self.model.stimulus_coordinates(self.system.neuron_coordinates),
            channel_inputs,
        )

    def channel_recording(
        self,
        ii: Float[Array, "i"] | None,
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

    def record_membrane_current(
        self, population: str | list | tuple | None = None, dt: float = 0.1
    ) -> Self:
        """Enable membrane current recording for population"""
        if population is None:
            population = self.system.populations
        if isinstance(population, (list, tuple)):
            for p in population:
                self.record_membrane_current(p, dt=dt)
            return self

        self._record_membrane_current(population, dt)

        return self

    def _record_membrane_current(self, population: str, dt: float) -> Self: ...

    def run(
        self,
        duration,
        stimulus: Optional["Stimulus"] = None,
        dt: float = 0.025,
        **kwargs,
    ) -> Tuple[
        Int[Array, "n_spiking_neuron_ids"] | None,
        Float[Array, "n_spiking_neuron_times"] | None,
        Int[Array, "n_voltage_neuron_ids"] | None,
        Float[Array, "neuron_ids voltages"] | None,
        Int[Array, "n_membrane_current_neuron_ids"] | None,
        Float[Array, "neuron_ids membrane_currents"] | None,
    ]:
        """Run the simulation"""
        ...

    def __call__(
        self,
        decoding: Union["Decoding", int],
        inputs: StimulusLike = None,
        encoding: Optional["Encoding"] = None,
        **kwargs,
    ) -> Any:
        if isinstance(decoding, int):
            duration = decoding
        else:
            duration = decoding.duration
            decoding.setup(self)

        if duration <= 0:
            raise ValueError(f"Encoding duration must be > 0, not {duration}.")

        stimulus = inputs
        if encoding is not None:
            stimulus = encoding(self, duration, inputs)

        response = self.run(duration, stimulus, **kwargs)

        if isinstance(decoding, int):
            return response

        return decoding(self, *response)

    def potential_recording(
        self, membrane_currents: Float[Array, "timestep n_neurons"] | None
    ) -> Float[Array, "timestep n_channels"]:
        distances = self.io.distances(
            self.model.recording_coordinates(self.system.neuron_coordinates),
        )
        return self.io.potential_recording(distances, membrane_currents)

    def clear(self) -> Self:
        """Discard the simulation and reset to t=0"""
        ...

        return self


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for livn models"""

    def stimulus_coordinates(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
    ) -> Float[Array, "n_stim_coords ixyz=4"]:
        return neuron_coordinates

    def recording_coordinates(
        self,
        neuron_coordinates: Float[Array, "n_coords ixyz=4"],
    ) -> Float[Array, "n_stim_coords ixyz=4"]:
        return neuron_coordinates

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


class Encoding(BaseModel):
    def __call__(self, env: "Env", t_end: int, inputs: Any) -> StimulusLike: ...

    @property
    def input_space(self) -> "gymnasium.Space":
        raise NotImplementedError


class Decoding(BaseModel):
    duration: int

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"duration must be > 0, not {v}.")
        return v

    def setup(self, env: "Env"):
        """Optional setup"""

    def __call__(
        self,
        env: "Env",
        it: Int[Array, "n_spiking_neuron_ids"] | None,
        tt: Float[Array, "n_spiking_neuron_times"] | None,
        iv: Int[Array, "n_voltage_neuron_ids"] | None,
        vv: Float[Array, "neuron_ids voltages"] | None,
        im: Int[Array, "n_membrane_current_neuron_ids"] | None,
        m: Float[Array, "neuron_ids membrane_currents"] | None,
    ) -> Any:
        return it, tt, iv, vv, im, m

    @property
    def output_space(self) -> "gymnasium.Space":
        raise NotImplementedError
