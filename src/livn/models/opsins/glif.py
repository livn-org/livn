from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp

from livn.models.glif import GlifMechanism
from livn.models.opsins.chr2 import (
    CHR2_DEFAULTS,
    CHR2_INITIAL,
    chr2_current,
    chr2_derivatives,
)


class ChR2H134R(GlifMechanism):
    """ChR2(H134R) in a GLIF cell, driven by an ``irradiance`` stimulus (mW/mm2).

    ``g_nS`` is the maximal conductance in nS, per cell or shared (a point cell has no
    membrane area, so the opsin's conductance is the cell's, not a density); a cell with
    ``g_nS = 0`` carries the states but feels no current. ``params`` overrides entries of
    the kinetics' table (:data:`livn.models.opsins.chr2.CHR2_DEFAULTS`, its ``gmax`` aside),
    ``celsius`` the temperature the rates are scaled to (the table's own 22 by default).
    """

    g_nS: object
    params: tuple = eqx.field(static=True)
    celsius: float = eqx.field(static=True)
    states: tuple[str, ...] = eqx.field(
        static=True, default=("O1", "O2", "C1", "p"), kw_only=True
    )
    inputs: tuple[str, ...] = eqx.field(
        static=True, default=("irradiance",), kw_only=True
    )

    def __init__(
        self, g_nS=1.0, params: Mapping[str, float] | None = None, celsius=None
    ):
        table = {**CHR2_DEFAULTS, **(params or {})}
        table.pop("gmax")
        self.g_nS = jnp.asarray(g_nS, dtype=jnp.result_type(float))
        self.params = tuple(table.items())
        self.celsius = float(table["temp"] if celsius is None else celsius)

    def initial(self, n_cells: int):
        return jnp.broadcast_to(jnp.asarray(CHR2_INITIAL), (n_cells, len(CHR2_INITIAL)))

    def drift(self, t, v, states, inputs):
        P = dict(self.params)
        kinetics = states.T  # (O1, O2, C1, p) rows, as the shared kinetics take them
        current = -chr2_current(kinetics, v, self.g_nS, P)  # inward positive, pA
        dstates = chr2_derivatives(kinetics, v, inputs["irradiance"], P, self.celsius)
        return current, dstates.T
