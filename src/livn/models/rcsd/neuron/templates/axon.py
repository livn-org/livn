from __future__ import annotations

import math

DEFAULTS = {
    "axon_segments": 0,
    "axon_segment_um": 30.0,
    "axon_diam": 2.0,
    "axon_cm": 0.2,
    "axon_Ra": 70.0,
    "axon_gmax_Na": None,
    "axon_gmax_Na_ratio": 0.4,
    "axon_gmax_K": 0.3,
    "axon_g_pas": 1.0e-5,
    "axon_e_pas": None,
}
"""`axon_segments = 0` means no axon"""

AXON_KEYS = tuple(DEFAULTS)


def axon_parameters(params: dict | None) -> dict:
    given = params or {}
    return {k: given.get(k, v) for k, v in DEFAULTS.items()}


def balanced_e_pas(
    params: dict, v_rest: float, soma_gmax_na: float, celsius: float = 36.0
) -> float:
    """Leak reversal that holds the axon at `v_rest`, in mV"""
    rt_f = 8.3145 * (celsius + 273.15) / 96485.0 * 1e3  # mV
    ena = rt_f * math.log(145.0 / 15.0)  # Na_conc nao0 / nai0
    ek = rt_f * math.log(5.0 / 145.0)  # K_conc  ko0  / ki0

    v = float(v_rest)
    minf = 1.0 / (1.0 + math.exp(-(v + 35.0) / 7.8))
    hinf = 1.0 / (1.0 + math.exp((v + 55.0) / 7.0))
    ninf = 1.0 / (math.exp(-(v + 28.0) / 15.0) + 1.0)

    g_na = sodium_density(params, soma_gmax_na) * minf**3 * hinf
    g_k = float(params["axon_gmax_K"]) * ninf**4
    g_pas = float(params["axon_g_pas"])
    if g_pas <= 0.0:
        return v
    return v + (g_na * (v - ena) + g_k * (v - ek)) / g_pas


def sodium_density(params: dict, soma_gmax_na: float) -> float:
    """Axon sodium density, as a multiple of the soma's unless given outright."""
    given = params.get("axon_gmax_Na")
    if given is not None:
        return float(given)
    ratio = float(params.get("axon_gmax_Na_ratio", DEFAULTS["axon_gmax_Na_ratio"]))
    return ratio * float(soma_gmax_na)


def attach(template, soma, params: dict, e_pas: float, soma_gmax_na: float) -> list:
    """A chain of axon sections at the soma's 0 end, opposite the dendrite."""
    from neuron import h

    n = int(params.get("axon_segments", 0) or 0)
    if n <= 0:
        return []

    sections = []
    parent, end = soma, 0.0
    for index in range(n):
        sec = h.Section(name=f"ais{index}", cell=template)
        sec.connect(parent(end), 0)
        sec.nseg = 1
        for mech in ("pas", "Na_conc", "K_conc", "Nas", "Kdr", "constant"):
            sec.insert(mech)
        sections.append(sec)
        parent, end = sec, 1.0
    configure(sections, params, e_pas, soma_gmax_na)
    return sections


def configure(sections, params: dict, e_pas: float, soma_gmax_na: float) -> None:
    """Apply geometry and conductances to an existing axon."""
    for sec in sections:
        sec.L = float(params["axon_segment_um"])
        sec.diam = float(params["axon_diam"])
        sec.Ra = float(params["axon_Ra"])
        sec.cm = float(params["axon_cm"])
        sec.gmax_Nas = sodium_density(params, soma_gmax_na)
        sec.gmax_Kdr = float(params["axon_gmax_K"])
        sec.g_pas = float(params["axon_g_pas"])
        own = params.get("axon_e_pas")
        sec.e_pas = (
            float(own)
            if own is not None
            else balanced_e_pas(params, e_pas, soma_gmax_na)
        )


def sampling_offsets(params: dict) -> list[float]:
    """Distance from the soma at which each axon link samples the field, in um."""
    n = int(params.get("axon_segments", 0) or 0)
    spacing = float(params.get("axon_segment_um", DEFAULTS["axon_segment_um"]))
    return [spacing * (i + 0.5) for i in range(n)]
