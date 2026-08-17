# Optimization

`optimization` is livn's differentiable ML layer containing metrics, losses, and fitting machinary used to estimate model parameters from data.

::: tip
Naturally, this section assumes a [differentiable environment](/guide/backends) requiring a `JAX` backend.
:::

## Installation

`optimization` ships with the repository as a workspace member:

```sh
uv sync --package optimization
```

It depends on `livn[diffrax]`, so this also installs JAX, diffrax, equinox, and optax.

## Hard and soft

Metrics come in two flavours:

- Hard metrics bin spike times onto a grid before comparing them. They are the numbers to report but the binning has no gradient, so they cannot be optimized against.
- Soft metrics are differentiable approximations over a continuous rate or raster. They are usable as fitting terms, and they are not expected to reproduce their hard counterpart's value.

