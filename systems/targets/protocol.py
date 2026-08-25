import math
from typing import Any

# Memory a wired NetCon costs
NETCON_BYTES = 1686
RANK_FLOOR_BYTES = 257 * 1024**2

_NETCON_CACHE: dict[tuple[str, str | None], float] = {}


def _active_mechanisms(projection: dict) -> int:
    blocks = projection.get("mechanisms") or {}
    return max(
        (
            sum(1 for m in mechanisms.values() if m.get("tau_decay") is not None)
            for mechanisms in blocks.values()
            if isinstance(mechanisms, dict)
        ),
        default=0,
    )


def wiring_profile(system) -> dict[str, tuple[float, int]]:
    import h5py

    from livn.system import resolve

    resolved = resolve(system)
    connections = resolved.connections_config["synapses"]

    profile: dict[str, tuple[float, int]] = {}
    with h5py.File(resolved.files["connections"], "r") as fh:
        projections = fh.get("Projections")
        if projections is None:
            return profile
        for post, sources in connections.items():
            if post not in projections:
                continue
            netcons, cells = 0, 0
            for pre in sources:
                if pre not in projections[post]:
                    continue
                edges = projections[post][pre]["Edges"]
                netcons += edges["Source Index"].shape[0] * _active_mechanisms(
                    connections[post][pre]
                )
                cells = max(cells, edges["Destination Pointer"].shape[0] - 1)
            if cells:
                profile[post] = (netcons / cells, cells)
    return profile


def estimated_netcons(system, selection: str | None = None) -> float:
    from livn.system import resolve

    key = (repr(system), selection)
    if key in _NETCON_CACHE:  # sizing asks repeatedly while it searches a layout
        return _NETCON_CACHE[key]

    profile = wiring_profile(system)
    if selection is None:
        counts = {pop: cells for pop, (_, cells) in profile.items()}
    else:
        document = resolve(system).selection_document(selection)
        counts = {pop: len(gids) for pop, gids in document["gids"].items()}

    _NETCON_CACHE[key] = sum(
        counts.get(pop, 0) * per_cell for pop, (per_cell, _) in profile.items()
    )
    return _NETCON_CACHE[key]


class TuningTargets:
    MIN_RANKS_PER_WORKER = 1

    def worker_memory(
        self, system, ranks: int = 1, selection: str | None = None
    ) -> float:
        if selection is None:
            selection = getattr(self, "selection_name", None)
        return ranks * RANK_FLOOR_BYTES + NETCON_BYTES * estimated_netcons(
            system, selection
        )

    def _space_metadata(self, model=None) -> tuple:
        raw_space = {
            **self._weight_space(model),
            **self._noise_space(model),
            **self._protocol_space(model),
        }

        transforms = {}
        search_space = {}

        for name, bounds in raw_space.items():
            if len(bounds) == 2:
                low, high = bounds
                transform_fn = self.transform_identity
            elif len(bounds) == 3:
                low, high, transform_fn = bounds
                if not callable(transform_fn):
                    raise ValueError(
                        f"Third element of bounds must be callable, got {type(transform_fn)}"
                    )
            else:
                raise ValueError(
                    f"Bounds must be [min, max] or [min, max, transform_fn], got {bounds}"
                )

            transforms[name] = transform_fn
            search_space[name] = [transform_fn(low), transform_fn(high)]

        return transforms, search_space

    @staticmethod
    def transform_identity(x: float, inverse: bool = False) -> float:
        return x

    @staticmethod
    def transform_log10(x: float, inverse: bool = False) -> float:
        """Log base 10 transformation for positive values.

        Useful for parameters spanning multiple orders of magnitude.
        """
        if inverse:
            return 10**x
        return math.log10(x)

    @staticmethod
    def transform_log1p(x: float, inverse: bool = False) -> float:
        """Log base 10 of (1 + value) transformation.

        Useful for parameters that can be zero or very small.
        Maps [0, inf) -> [0, inf) with log10(1 + x).
        """
        if inverse:
            return max(0.0, 10**x - 1.0)
        return math.log10(x + 1.0)

    def _weight_space(self, model) -> dict[str, list]:
        """Define weight parameter bounds.

        Override in subclasses to define weight parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn] where transform_fn is a bound method.
        """
        return {}

    def _noise_space(self, model) -> dict[str, list]:
        """Define noise parameter bounds.

        Override in subclasses to define noise parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn].
        """
        return {}

    def _protocol_space(self, model) -> dict[str, list]:
        """Define protocol-specific parameter bounds.

        Override in subclasses for protocol-specific parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn].
        """
        return {}

    def init(self, env):
        return env

    def search_space(self, model=None) -> dict[str, list[float]]:
        """Return the transformed search space for optimization.

        Computes the search space with forward transforms applied to bounds.

        Returns:
            Dictionary mapping parameter names to [transformed_min, transformed_max].
        """
        _, search_space = self._space_metadata(model)
        return search_space

    def decode_params(self, params: dict[str, Any], model=None) -> dict[str, Any]:
        """Decode parameters from optimization space to the natural domain.

        Args:
            params: Dictionary of parameter values in optimization space.

        Returns:
            Dictionary of decoded parameters in the natural domain.
        """
        transforms, _ = self._space_metadata(model)

        decoded = {}
        for name, value in params.items():
            if name in transforms:
                decoded[name] = transforms[name](float(value), inverse=True)
            else:
                decoded[name] = value

        return decoded

    def transform_params(self, params: dict[str, Any], model=None) -> dict[str, Any]:
        """Transform parameters from optimization space and apply set_params.

        Decodes parameters using inverse transforms, then passes them through
        set_params() to consume protocol-specific parameters.

        Args:
            params: Dictionary of parameter values in optimization space.

        Returns:
            Dictionary of remaining parameters in natural domain.
        """
        # Pass through set_params to consume protocol-specific parameters
        return self.set_params(self.decode_params(params, model=model))

    def set_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set protocol parameters from a dictionary.

        Override in subclasses to consume protocol-specific parameters.
        Should return remaining parameters not consumed by the protocol.

        Args:
            params: Dictionary of decoded parameters.

        Returns:
            Dictionary of remaining parameters.
        """
        return params.copy()
