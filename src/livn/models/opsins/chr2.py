from collections.abc import Mapping

import jax.numpy as jnp

# ChR2(H134R), Williams et al. (2013)
CHR2_DEFAULTS: dict[str, float] = {
    "gmax": 4e-4,
    "EChR2": 0.0,
    "gamma": 0.1,
    "A": 10.6408,
    "B": -14.6408,
    "C": -42.7671,
    "wavelength": 470.0,
    "hc_J_m": 1.98645e-25,
    "wloss": 1.3,
    "sigma_retinal_m2": 12e-20,
    "tauChR2": 1.3,
    "temp": 22.0,
    "Q10_Gd1": 1.97,
    "Q10_Gd2": 1.77,
    "Q10_Gr": 2.56,
    "Q10_e12dark": 1.1,
    "Q10_e21dark": 1.95,
    "Q10_epsilon1": 1.46,
    "Q10_epsilon2": 2.77,
}

CHR2_INITIAL = (0.0, 0.0, 1.0, 0.0)  # O1, O2, C1, p


def chr2_derivatives(states, V, irradiance, P: Mapping[str, float], celsius: float):
    """d(O1, O2, C1, p)/dt of ChR2H134R at membrane potential V (mV)."""
    O1, O2, C1, p = states
    q = (celsius - P["temp"]) / 10.0
    e12dark = 0.011 * P["Q10_e12dark"] ** q
    e21dark = 0.008 * P["Q10_e21dark"] ** q
    epsilon1 = 0.8535 * P["Q10_epsilon1"] ** q
    epsilon2 = 0.14 * P["Q10_epsilon2"] ** q
    Gd1 = (0.075 + 0.043 * jnp.tanh(-(V + 20.0) / 20.0)) * P["Q10_Gd1"] ** q
    Gd2 = 0.05 * P["Q10_Gd2"] ** q
    Gr = 0.0000434587 * jnp.exp(-0.0211539274 * V) * P["Q10_Gr"] ** q
    # log(1 + I/0.024) for I > 0 and 0 otherwise, without a branch
    logphi0 = jnp.log(1.0 + jnp.maximum(irradiance, 0.0) / 0.024)
    e12 = e12dark + 0.005 * logphi0
    e21 = e21dark + 0.004 * logphi0
    S0 = 0.5 * (1.0 + jnp.tanh(120.0 * (100.0 * irradiance - 0.1)))
    Ephoton = 1e9 * P["hc_J_m"] / P["wavelength"]
    F = 1000.0 * irradiance / Ephoton * P["sigma_retinal_m2"] / (P["wloss"] * 1000.0)
    C2 = 1.0 - C1 - O1 - O2
    dO1 = -(Gd1 + e12) * O1 + e21 * O2 + epsilon1 * F * p * C1
    dO2 = -(Gd2 + e21) * O2 + e12 * O1 + epsilon2 * F * p * C2
    dC1 = -epsilon1 * F * p * C1 + Gd1 * O1 + Gr * C2
    dp = (S0 - p) / P["tauChR2"]
    return jnp.stack([dO1, dO2, dC1, dp])


def chr2_current(states, V, gmax, P: Mapping[str, float]):
    """ChR2H134R current, outward positive, in ``gmax``'s unit times mV
    (S/cm2: mA/cm2; nS: pA)."""
    O1, O2 = states[0], states[1]
    Vs = jnp.where(jnp.abs(V) < 1e-6, jnp.where(V < 0, -1e-6, 1e-6), V)
    rectification = (P["A"] + P["B"] * jnp.exp(V / P["C"])) * (1.0 - P["EChR2"] / Vs)
    return gmax * rectification * (O1 + P["gamma"] * O2)
