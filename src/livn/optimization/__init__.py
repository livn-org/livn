try:
    import equinox  # noqa: F401
    import jax  # noqa: F401
    import optax  # noqa: F401
    import optimistix  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"livn.optimization needs jax and its libraries ({e.name!r} is missing): "
        "`pip install livn[optimization]`"
    ) from e

from . import fit, losses, metrics, train, transforms

__all__ = ["fit", "losses", "metrics", "train", "transforms"]
