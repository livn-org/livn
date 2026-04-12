import math
from typing import Any, Callable


class TuningTargets:
    _transforms: dict[str, Callable[[float, bool], float]]
    _search_space: dict[str, list[float]]

    def __init__(self):
        self._transforms = {}
        self._search_space = {}
        self._prepare_search_space()

    def _prepare_search_space(self) -> None:
        raw_space = {
            **self._weight_space(),
            **self._noise_space(),
            **self._protocol_space(),
        }

        self._transforms = {}
        self._search_space = {}

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

            self._transforms[name] = transform_fn
            self._search_space[name] = [transform_fn(low), transform_fn(high)]

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

    def _weight_space(self, populations: list[str] | None = None) -> dict[str, list]:
        """Define weight parameter bounds.

        Override in subclasses to define weight parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn] where transform_fn is a bound method.
        """
        return {}

    def _noise_space(self) -> dict[str, list]:
        """Define noise parameter bounds.

        Override in subclasses to define noise parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn].
        """
        return {}

    def _protocol_space(self) -> dict[str, list]:
        """Define protocol-specific parameter bounds.

        Override in subclasses for protocol-specific parameters.

        Returns:
            Dictionary mapping parameter names to [min, max] or
            [min, max, transform_fn].
        """
        return {}

    def search_space(self) -> dict[str, list[float]]:
        """Return the transformed search space for optimization.

        Returns the precomputed search space with forward transforms applied to bounds.

        Returns:
            Dictionary mapping parameter names to [transformed_min, transformed_max].
        """
        return self._search_space

    def transform_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform parameters from optimization space and apply set_params.

        Decodes parameters using inverse transforms, then passes them through
        set_params() to consume protocol-specific parameters.

        Args:
            params: Dictionary of parameter values in optimization space.

        Returns:
            Dictionary of remaining parameters in natural domain.
        """
        # Apply inverse transforms
        decoded = {}
        for name, value in params.items():
            if name in self._transforms:
                decoded[name] = self._transforms[name](float(value), inverse=True)
            else:
                decoded[name] = value

        # Pass through set_params to consume protocol-specific parameters
        return self.set_params(decoded)

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
