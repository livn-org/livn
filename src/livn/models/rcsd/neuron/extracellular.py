from __future__ import annotations

import numpy as np

AMPLITUDE_RANGE_UM = 60.0
STORED_RANGE_UM = 150.0


def measure(
    population: str = "EXC",
    gap_um: float = 5.0,
    sigma_s_per_m: float = 0.3,
    band_hz: tuple[float, float] = (300.0, 3000.0),
    max_distance_um: float = 400.0,
    step_um: float = 2.5,
    directions: int = 72,
) -> dict:
    from scipy.signal import butter, sosfiltfilt

    from livn.backend.neuron.mechanisms import load_mechanisms
    from livn.models.rcsd import ReducedCalciumSomaDendrite, hold_potential

    model = ReducedCalciumSomaDendrite()
    load_mechanisms(model.neuron_mechanisms_directory())
    from neuron import h

    from livn.models.rcsd.neuron.templates.BRK import BRK
    from livn.models.rcsd.neuron.templates.V1In import V1In

    h.load_file("stdrun.hoc")
    h.celsius = model.neuron_celsius()
    if population == "EXC":
        params = dict(model.params(model._exc_params_name()))
        cell = BRK({"BoothRinzelKiehn": params})
    elif population == "INH":
        params = dict(model.params(model._inh_params_name()))
        cell = V1In(params)
    else:
        raise ValueError(f"population is EXC or INH, not {population!r}")
    h.cvode.use_fast_imem(1)

    dend = getattr(cell, "dend", None)
    positions = [seg.x * cell.soma.L for seg in cell.soma]
    if dend is not None:
        positions.extend(cell.soma.L + seg.x * dend.L for seg in dend)
    start = 0.0
    for sec in cell.axon:
        positions.extend(start - seg.x * sec.L for seg in sec)
        start -= sec.L
    sections = (cell.soma, *(() if dend is None else (dend,)), *cell.axon)
    records = [
        h.Vector().record(seg._ref_i_membrane_) for sec in sections for seg in sec
    ]
    time = h.Vector().record(h._ref_t)
    pulse = h.IClamp(cell.soma(0.5))
    pulse.delay, pulse.dur, pulse.amp = 50.0, 2.0, 0.5
    h.dt = 0.025
    v_hold = float(hold_potential(params))
    cell.init_ic(v_hold)
    h.finitialize(v_hold)
    h.continuerun(80.0)

    currents = np.array([np.asarray(r) for r in records])  # nA
    t = np.asarray(time)
    x = np.asarray(positions)
    sos = butter(4, list(band_hz), btype="bandpass", fs=1000.0 / h.dt, output="sos")
    window = (t > 45.0) & (t < 75.0)
    filtered = sosfiltfilt(sos, currents, axis=1)[:, window]
    gain = 1000.0 / (4.0 * np.pi * sigma_s_per_m)  # uV per (nA / um)
    distance = np.arange(0.0, max_distance_um + step_um / 2, step_um)
    angles = np.linspace(0.0, 2.0 * np.pi, int(directions), endpoint=False)
    px = (distance[:, None] * np.cos(angles)[None, :]).ravel()
    py = (distance[:, None] * np.sin(angles)[None, :]).ravel()
    d = np.sqrt((x[None, :] - px[:, None]) ** 2 + py[:, None] ** 2 + float(gap_um) ** 2)
    potential = gain * (1.0 / np.maximum(d, 5.0)) @ filtered  # points x samples
    trough = (-potential.min(axis=1)).reshape(len(distance), len(angles)).mean(axis=1)
    return {
        "distance_um": distance.round(3).tolist(),
        "trough_uv": trough.round(4).tolist(),
        "gap_um": float(gap_um),
        "sigma_s_per_m": float(sigma_s_per_m),
        "band_hz": list(band_hz),
        "celsius": float(h.celsius),
        "population": population,
    }


def amplitude_ratio(own: dict, reference: dict) -> float:
    distance = np.asarray(reference["distance_um"], dtype=float)
    near = distance <= AMPLITUDE_RANGE_UM
    ratio = np.asarray(own["trough_uv"], dtype=float)[near] / np.maximum(
        np.asarray(reference["trough_uv"], dtype=float)[near], 1e-9
    )
    return float(np.median(ratio))


def constants(exc: dict, inh: dict) -> tuple[dict, dict]:
    distance = np.asarray(exc["distance_um"], dtype=float)
    keep = distance <= STORED_RANGE_UM
    profile = {
        "distance_um": tuple(float(v) for v in distance[keep]),
        "trough_uv": tuple(float(v) for v in np.asarray(exc["trough_uv"])[keep]),
    }
    return profile, {"EXC": 1.0, "INH": amplitude_ratio(inh, exc)}


if __name__ == "__main__":
    # each population in its own process as NEURON keeps one cell's mechanisms
    import json
    import subprocess
    import sys

    measured = {
        population: json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json; from livn.models.rcsd.neuron.extracellular "
                    f"import measure; print(json.dumps(measure({population!r})))",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[-1]
        )
        for population in ("EXC", "INH")
    }
    profile, amplitude = constants(measured["EXC"], measured["INH"])
    print(f"    EXTRACELLULAR_PROFILE: ClassVar[dict] = {profile!r}")
    print(f"    EXTRACELLULAR_AMPLITUDE: ClassVar[dict] = {amplitude!r}")
