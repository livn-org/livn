import os
import re

from systems.targets.protocol import note
from systems.targets.schema import FREE_RUNNING, read_target

GATES = {
    "BRANCHING_RATIO_BAND": ("branching_ratio", "band"),
    "POP_TAU_BAND_MS": ("pop_autocorr_tau", "band"),
    "SYNCHRONY_BAND": ("mean_channel_correlation", "band"),
    "MAX_MEAN_RATE_HZ": ("mfr", "max"),
    "MIN_MEAN_RATE_HZ": ("mfr", "min"),
    "MAX_NEURON_RATE_HZ": ("max_neuron_firing_rate", "max"),
    "MAX_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "max"),
    "MIN_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "min"),
    "MAX_BURST_RATE_HZ": ("burst_rate", "max"),
    "MIN_BURST_RATE_HZ": ("burst_rate", "min"),
    "MAX_SYNC_PEAK": ("max_synchronous_peak", "max"),
    "MIN_SYNC_PEAK": ("max_synchronous_peak", "min"),
}


def free_running_block(document: dict) -> str:
    blocks = document.get("conditions") or {}
    for name in FREE_RUNNING:  # preference order
        if blocks.get(name):
            return name
    raise ValueError(
        f"this document holds no free-running block "
        f"({' or '.join(FREE_RUNNING)}), so there is nothing to fit the resting "
        f"features to; it has {sorted(blocks)}"
    )


def logged(document: dict) -> set[tuple[str, str | None]]:
    experiments = (document.get("metadata") or {}).get("experiments") or {}
    found = set()
    for name, recording in (document.get("recordings") or {}).items():
        channels = [c["channel"] for c in (recording.get("channels") or ())]
        date = re.match(r"\d{4}-(\d{2}-\d{2})_", name)
        if not channels or date is None:
            continue
        array = f"MEA_{chr(ord('A') + min(channels) // 128)}"
        entry = ((experiments.get(date.group(1)) or {}).get("samples") or {}).get(
            array
        ) or {}
        if entry.get("sample"):
            found.add((entry["sample"], entry.get("composition")))
    return found


def sample_of(document: dict) -> str:
    named = {sample for sample, _ in logged(document)}
    if len(named) != 1:
        raise ValueError(
            f"this document reads as {sorted(named) or 'no'} sample(s) of the "
            "experiment log, so a network cannot be drawn from it: the spec "
            "needs one culture's density, geometry and composition. Its "
            "`metadata.experiments` and the channels each recording was "
            "extracted on have to agree on one."
        )
    return named.pop()


def composition_of(document: dict) -> str | None:
    seeded = {composition for _, composition in logged(document) if composition}
    return seeded.pop() if len(seeded) == 1 else None


def widen_to_resolution(band, bin_ms=10.0, bins=4):
    lo, hi = float(band[0]), float(band[1])
    floor = bin_ms * bins
    if hi - lo >= floor:
        return [max(0.0, lo), hi]
    centre = 0.5 * (lo + hi)
    return [max(0.0, centre - floor / 2.0), centre + floor / 2.0]


def widen_to_blocks(spec, blocks: int):
    lo, hi = spec.q_lo, spec.q_hi
    if lo is None or hi is None:
        return None
    if blocks >= 2 or spec.min is None or spec.max is None:
        return float(lo), float(hi)
    return float(min(lo, spec.min)), float(max(hi, spec.max))


def coverage_shifted(band, reachable):
    if reachable is None or reachable >= 1.0:
        return None
    lo, hi = float(band[0]), float(band[1])
    if lo <= reachable:
        return None
    shift = 1.0 - reachable
    return [max(0.0, lo - shift), min(reachable, hi - shift)]


def synchrony_band(measured, floor: float = 0.01):
    lo, hi = measured
    if lo <= 0.0 or hi < floor:
        return [-float(floor), float(floor)]
    return None


def corrected_bands(bands: dict, *, reachable=None, drop=()) -> dict:
    out = {}
    for name, band in bands.items():
        if name in drop:
            continue
        if name == "active_fraction":
            shifted = coverage_shifted(band, reachable)
            if shifted is not None:
                out[name] = shifted
                continue
        if name == "pop_autocorr_tau":
            out[name] = widen_to_resolution(band)
            continue
        out[name] = band
    return out


def electrode_coverage(system, mea, selection=None) -> float | None:
    import numpy as np

    electrodes = np.asarray(mea.get("electrode_coordinates") or (), dtype=float)
    if not len(electrodes):
        return None

    rows = [
        system.coordinate_array(p)
        for p in system.populations
        if system.population_count(p)
    ]
    if not rows:
        return None
    coordinates = np.vstack(rows)

    if selection:
        wanted = {int(g) for v in system.selection(selection).values() for g in v}
        keep = np.array([int(g) in wanted for g in coordinates[:, 0]], dtype=bool)
        coordinates = coordinates[keep]
        if not len(coordinates):
            return None

    xyz = coordinates[:, 1:4].astype(float)
    reached = np.zeros(len(electrodes), dtype=bool)
    radius = float(mea["output_radius"])
    for i, electrode in enumerate(electrodes):  # one row at a time; the cross
        d = np.linalg.norm(xyz - electrode[1:4], axis=1)  # product does not fit
        reached[i] = bool((d <= radius).any())
    return float(reached.mean())


def measured_options(
    observation: str,
    condition: str,
    mea: dict | None,
    system,
    readout: str = "channels",
    skip_constraints: list | None = None,
) -> dict:
    if condition not in FREE_RUNNING:
        raise ValueError(
            f"condition {condition!r} measures windows containing a "
            f"stimulus. Fit on {' or '.join(FREE_RUNNING)} instead."
        )

    block = read_target(observation, condition)
    measured = block.ei_targets
    features = block.summary.features

    if mea is None and readout == "channels":
        raise ValueError(
            "readout='channels' needs the window's electrodes; the spec "
            "carries them as `mea`"
        )

    blocks = len(block.summary.recordings)
    spreads = {}
    for name, spec in features.items():
        if spec is None:
            continue
        spread = widen_to_blocks(spec, blocks)
        if spread is not None:
            spreads[name] = spread
    if blocks < 2:
        note(
            f"{os.path.basename(observation)} pools one recording block, so "
            "its quantiles never saw the drift between blocks; bands taken "
            "from the extremes of that block instead"
        )

    gates = dict(measured)
    for key, (feature, kind) in GATES.items():
        spec = features.get(feature)
        if not spec:
            continue
        spread = spreads.get(feature)
        if spread is None:
            continue
        lo, hi = spread
        slack = (hi - lo) / 2.0  # the quantile spread again, each side
        if kind == "band":
            gates[key] = [float(lo - slack), float(hi + slack)]
        elif kind == "max":
            gates[key] = float(hi + slack)
        else:
            gates[key] = float(max(0.0, lo - slack))

    reachable = None
    floor = gates.get("MIN_ACTIVE_FRACTION")
    if mea is not None and floor is not None:
        reachable = electrode_coverage(system, mea)
        if reachable is not None and reachable < 1.0:
            gates["MIN_ACTIVE_FRACTION"] = max(0.0, reachable - (1.0 - float(floor)))
            note(
                f"{1.0 - reachable:.1%} of the array reaches no cell within "
                f"{mea['output_radius']:g} um, so active_fraction cannot "
                f"exceed {reachable:.4f}; lowered MIN_ACTIVE_FRACTION "
                f"{float(floor):.4f} -> {gates['MIN_ACTIVE_FRACTION']:.4f}"
            )

    options = dict(gates)
    options["targets"] = dict(measured["targets"])

    skip_constraints = list(skip_constraints or ())

    sync_lo, sync_hi = measured.get("SYNCHRONY_BAND") or (0.0, 1.0)
    band = synchrony_band((sync_lo, sync_hi))
    if band is not None:
        options["SYNCHRONY_BAND"] = band
    else:
        options["targets"]["mean_channel_correlation"] = float(
            (sync_lo + sync_hi) / 2.0
        )

    for feature, key in (("pop_autocorr_tau", "POP_TAU_BAND_MS"),):
        spec = features.get(feature)
        if (
            spec
            and spec.q_lo is not None
            and spec.min is not None
            and abs(spec.q_lo - spec.min) <= 1e-9
        ):
            band = options.get(key) or measured.get(key)
            if band is not None:
                options[key] = [0.0, float(band[1])]
        band = options.get(key) or measured.get(key)
        if band is not None:
            options[key] = widen_to_resolution(band)

    options["feature_bands"] = corrected_bands(
        spreads,
        reachable=reachable,
        drop=skip_constraints,
    )
    options["readout"] = readout
    options["mea"] = mea
    options["skip_constraints"] = skip_constraints
    return options
