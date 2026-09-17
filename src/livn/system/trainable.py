from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from livn import types
from livn.backend import backend

if TYPE_CHECKING:
    from livn.io import IO

_USES_JAX = "ax" in backend()

if _USES_JAX:
    import equinox as eqx
    import jax
    import jax.numpy as np

    class PositionParameterization(eqx.Module):
        def __call__(self):
            raise NotImplementedError

        @classmethod
        def from_cartesian(cls, offsets, **kwargs):
            raise NotImplementedError

        @classmethod
        def from_lateral_depth(cls, lateral_xy, depth=50.0, **kwargs):
            """Construct from 2D lateral offsets and a depth value

            Convenience alternative to ``from_cartesian`` for the common case
            where neurons are initialised from their peak-channel (x, y) offset
            from the probe origin and a uniform starting depth

            Args:
                lateral_xy: [n_neurons, 2] or [n_pop, n_neurons, 2] lateral (x, y)
                    offsets from the population origin
                depth: Scalar or array of z-depths in um (positive = into tissue)
                **kwargs: Passed to ``from_cartesian`` (e.g. ``r_min``, ``r_max``)
            """
            lateral_xy = np.array(lateral_xy)
            if lateral_xy.ndim == 2:
                n_neurons = lateral_xy.shape[0]
                z = (
                    np.full((n_neurons, 1), float(depth))
                    if np.isscalar(depth)
                    else np.array(depth).reshape(n_neurons, 1)
                )
                offsets = np.concatenate([lateral_xy, z], axis=-1)[
                    None
                ]  # [1, n_neurons, 3]
            else:
                n_pop, n_neurons = lateral_xy.shape[:2]
                z = (
                    np.full((n_pop, n_neurons, 1), float(depth))
                    if np.isscalar(depth)
                    else np.array(depth).reshape(n_pop, n_neurons, 1)
                )
                offsets = np.concatenate(
                    [lateral_xy, z], axis=-1
                )  # [n_pop, n_neurons, 3]
            return cls.from_cartesian(offsets, **kwargs)

    class CartesianParameterization(PositionParameterization):
        offsets: Any

        def __call__(self):
            return self.offsets

        @classmethod
        def from_cartesian(cls, offsets, **kwargs):
            return cls(offsets=np.array(offsets))

    class LogRadialParameterization(PositionParameterization):
        """Log-radial (log r, unnormalized direction)

        Args:
            r_min: Minimum radial distance in um
            r_max: Maximum radial distance in um
        """

        log_r: Any
        dir_raw: Any
        r_min: float = eqx.field(static=True)
        r_max: float = eqx.field(static=True)

        def __init__(self, log_r, dir_raw, r_min: float = 5.0, r_max: float = 500.0):
            self.log_r = log_r
            self.dir_raw = dir_raw
            self.r_min = r_min
            self.r_max = r_max

        def __call__(self):
            r = np.exp(np.clip(self.log_r, np.log(self.r_min), np.log(self.r_max)))
            # Force z > 0
            dir_z_pos = np.abs(self.dir_raw[..., 2:3])
            dir_clipped = np.concatenate([self.dir_raw[..., :2], dir_z_pos], axis=-1)
            direction = dir_clipped / np.maximum(
                np.linalg.norm(dir_clipped, axis=-1, keepdims=True), 1e-6
            )
            return r[..., None] * direction  # [n_pop, n_neurons, 3]

        @classmethod
        def from_cartesian(cls, offsets, r_min: float = 5.0, r_max: float = 500.0):
            offsets = np.array(offsets)
            r = np.maximum(np.linalg.norm(offsets, axis=-1), r_min)
            log_r = np.log(r)
            dir_raw = offsets / r[..., None]
            return cls(log_r=log_r, dir_raw=dir_raw, r_min=r_min, r_max=r_max)

    class SphericalParameterization(PositionParameterization):
        """Explicit spherical coordinates (log r, theta_raw, phi_raw).

        Args:
            r_min: Minimum radial distance in um.
            r_max: Maximum radial distance in um.
        """

        log_r: Any
        theta_raw: Any
        phi_raw: Any
        r_min: float = eqx.field(static=True)
        r_max: float = eqx.field(static=True)

        def __init__(
            self,
            log_r,
            theta_raw,
            phi_raw,
            r_min: float = 5.0,
            r_max: float = 500.0,
        ):
            self.log_r = log_r
            self.theta_raw = theta_raw
            self.phi_raw = phi_raw
            self.r_min = r_min
            self.r_max = r_max

        def __call__(self):
            r = np.exp(np.clip(self.log_r, np.log(self.r_min), np.log(self.r_max)))
            theta = (np.pi / 2) * (1.0 / (1.0 + np.exp(-self.theta_raw)))
            x = r * np.sin(theta) * np.cos(self.phi_raw)
            y = r * np.sin(theta) * np.sin(self.phi_raw)
            z = r * np.cos(theta)
            return np.stack([x, y, z], axis=-1)  # [n_pop, n_neurons, 3]

        @classmethod
        def from_cartesian(cls, offsets, r_min: float = 5.0, r_max: float = 500.0):
            offsets = np.array(offsets)
            r = np.maximum(np.linalg.norm(offsets, axis=-1), r_min)
            log_r = np.log(r)
            z_clipped = np.clip(offsets[..., 2], 0.0, None)
            theta = np.arccos(np.clip(z_clipped / r, 0.0, 1.0))  # [0, pi/2]
            t = np.clip(theta / (np.pi / 2), 1e-6, 1.0 - 1e-6)
            theta_raw = np.log(t / (1.0 - t))
            phi_raw = np.arctan2(offsets[..., 1], offsets[..., 0])
            return cls(
                log_r=log_r,
                theta_raw=theta_raw,
                phi_raw=phi_raw,
                r_min=r_min,
                r_max=r_max,
            )

    class TrainableSystem:
        """System with trainable neuron positions

        Neuron absolute coordinates = origins + parameterization()

        Arguments:
            n_neurons: Number of neurons per population
            n_populations: Number of populations
            parameterization: A PositionParameterization instance
            origins: Reference origins [n_populations, xyz=3] for channel locations;
                defaults to vec{0} for all populations
            uri: Optional URI for loading IO configuration
        """

        def __init__(
            self,
            parameterization: PositionParameterization,
            n_neurons: int,
            n_populations: int = 1,
            origins: types.Float[types.Array, "n_populations xyz=3"] | None = None,
            uri: str | None = None,
        ):
            self.n_neurons = n_neurons
            self.n_populations = n_populations
            self.name = "TrainableSystem"
            self.populations = [str(i) for i in range(n_populations)]
            self.uri = uri

            if origins is None:
                origins = np.tile(np.array([0.0, 0.0, 0.0]), (n_populations, 1))
            self.origins = np.array(origins)

            self.parameterization = parameterization

        @property
        def params(self):
            return self.parameterization()

        def default_io(self) -> "IO":
            from livn.io import MEA

            if hasattr(self, "uri") and self.uri is not None:
                try:
                    return MEA.from_directory(self.uri)
                except (FileNotFoundError, AttributeError):
                    pass

            return MEA()

        @property
        def num_neurons(self):
            return self.n_populations * self.n_neurons

        @property
        def neuron_coordinates(
            self,
        ) -> types.Float[types.Array, "n_total_neurons ixyz=4"]:
            absolute_coords = (
                self.origins[:, None, :] + self.params
            )  # [n_pop, n_neurons, 3]

            xyz_flat = absolute_coords.reshape(-1, 3)  # [n_total_neurons, 3]

            # [n_total_neurons, 1]
            n_total = self.n_populations * self.n_neurons
            gids = np.arange(n_total, dtype=xyz_flat.dtype).reshape(-1, 1)

            # [n_total_neurons, ixyz=4]
            return np.concatenate([gids, xyz_flat], axis=1)

        def coordinate_array(
            self, population: str, all: bool = True
        ) -> types.Float[types.Array, "n_coords cxyz=4"]:
            pop_idx = self.populations.index(population)
            absolute_coords = (
                self.origins[pop_idx] + self.params[pop_idx]
            )  # [n_neurons, 3]
            gid_offset = pop_idx * self.n_neurons
            gids = np.arange(
                gid_offset, gid_offset + self.n_neurons, dtype=absolute_coords.dtype
            ).reshape(-1, 1)
            return np.concatenate([gids, absolute_coords], axis=1)

        def transform_coordinates(
            self,
            transform: Callable,
            populations: list[str] | None = None,
            all: bool = True,
        ) -> types.Float[types.Array, "n_coords ixyz=4"]:
            if populations is None:
                populations = self.populations
            return np.vstack(
                [
                    transform(self.coordinate_array(p, all=all), population=p)
                    for p in populations
                ]
            )

        @property
        def gids(self) -> types.Int[types.Array, "n_total_neurons"]:
            return np.arange(self.n_populations * self.n_neurons, dtype=np.int32)

        @property
        def bounding_box(self) -> types.Float[types.Array, "2 xyz=3"]:
            coords = self.neuron_coordinates[:, 1:4]
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)

            padding = 100.0
            min_coords = min_coords - padding
            max_coords = max_coords + padding

            return np.stack([min_coords, max_coords])

        @property
        def center_point(self) -> types.Float[types.Array, "xyz=3"]:
            bb = self.bounding_box
            return (bb[0] + bb[1]) / 2.0

    def _trainable_system_flatten(system):
        children = (system.parameterization, system.origins)
        aux = (
            system.n_neurons,
            system.n_populations,
            system.name,
            tuple(system.populations),
            system.uri,
        )
        return children, aux

    def _trainable_system_unflatten(aux, children):
        parameterization, origins = children
        n_neurons, n_populations, name, populations, uri = aux

        system = object.__new__(TrainableSystem)
        system.n_neurons = n_neurons
        system.n_populations = n_populations
        system.origins = origins
        system.name = name
        system.populations = list(populations)
        system.uri = uri
        system.parameterization = parameterization
        return system

    jax.tree_util.register_pytree_node(
        TrainableSystem, _trainable_system_flatten, _trainable_system_unflatten
    )
