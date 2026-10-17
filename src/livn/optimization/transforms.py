from typing import NamedTuple

import jax.numpy as jnp

__all__ = [
    "BIJECTORS",
    "Bijector",
    "bijector_for",
    "bounded",
    "bounds_for",
    "pack",
    "unpack",
]


class Bijector(NamedTuple):
    """A pair ``(constrained -> unconstrained, unconstrained -> constrained)``."""

    fwd: object
    inv: object


def _identity_fwd(x):
    return x


def _identity_inv(z):
    return z


def _log_fwd(x):
    return jnp.log(x)


def _log_inv(z):
    return jnp.exp(z)


def _logit_fwd(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x) - jnp.log1p(-x)


def _logit_inv(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def _bounded_fwd(x, lo, hi):
    u = jnp.clip((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return jnp.log(u) - jnp.log1p(-u)


def _bounded_inv(z, lo, hi):
    return lo + (hi - lo) / (1.0 + jnp.exp(-z))


#: name -> (constrained -> unconstrained, unconstrained -> constrained)
BIJECTORS = {
    "identity": (_identity_fwd, _identity_inv),
    "log": (_log_fwd, _log_inv),
    "logit": (_logit_fwd, _logit_inv),
    "bounded": (_bounded_fwd, _bounded_inv),
}


def bounded(lo, hi) -> Bijector:
    """A logit over the box ``(lo, hi)``, as a bijector that carries its own box."""
    return Bijector(
        lambda x: _bounded_fwd(x, lo, hi),
        lambda z: _bounded_inv(z, lo, hi),
    )


def bijector_for(name, spec) -> str | Bijector:
    """The bijector ``spec`` assigns to ``name``; ``"identity"`` when it says nothing."""
    return spec.get(name, "identity")


def bounds_for(name, bounds):
    """The ``(lo, hi)`` box ``bounds`` gives ``name``, or ``None``."""
    return bounds.get(name)


def _apply(direction, name, value, spec, bounds):
    kind = bijector_for(name, spec)
    if isinstance(kind, Bijector):
        return kind[direction](jnp.asarray(value))
    fn = BIJECTORS[kind][direction]
    if kind != "bounded":
        return fn(jnp.asarray(value))
    box = bounds_for(name, bounds or {})
    if box is None:
        return jnp.asarray(value)
    return fn(jnp.asarray(value), *box)


def pack(params, spec, bounds=None) -> dict:
    """``{name: constrained}`` -> ``{name: unconstrained}``.

    Args:
        params: ``{name: value}``
        spec: ``{name: "identity"|"log"|"logit"|"bounded"|Bijector}``; a name it omits is identity
        bounds: ``{name: (lo, hi)}`` for the names ``spec`` marks ``"bounded"``; a bounded name
            without a box is left unconstrained
    """
    return {
        name: _apply(0, name, value, spec, bounds) for name, value in params.items()
    }


def unpack(z, spec, bounds=None) -> dict:
    """Inverse of :func:`pack`."""
    return {name: _apply(1, name, value, spec, bounds) for name, value in z.items()}
