from __future__ import annotations

import zlib


def compositions(metadata: dict) -> dict[str, str]:
    out = {}
    for entry in (metadata.get("experiments") or {}).values():
        for block in (entry.get("samples") or {}).values():
            if block.get("sample"):
                out[block["sample"]] = block.get("composition")
    return dict(sorted(out.items()))


DEFAULT_CELLS = 2600
CELLS_BY_COMPOSITION: dict[str, int] = {}
COMPOSITIONS = {
    "E": 1.0,
    "83/17": 5.0 / 6.0,
    "75/25": 0.75,
    "50/50": 0.5,
}

DEGREE_REFERENCE = {
    "EXC->EXC": {"pre": "EXC", "share": 1.0, "composition": "E"},
    "INH->EXC": {"pre": "INH", "share": 0.5, "composition": "50/50"},
    "EXC->INH": {"pre": "EXC", "share": 0.5, "composition": "50/50"},
}

INHIBITORY_DEGREE = {"INH->EXC": 40.0, "EXC->INH": 4.0}


def fractions(composition: str) -> tuple[float, float]:
    """`(excitatory, inhibitory)` share for a composition label."""
    if composition in COMPOSITIONS:
        excitatory = COMPOSITIONS[composition]
    else:
        try:
            left, right = (float(part) for part in composition.split("/"))
        except ValueError:
            raise ValueError(
                f"unknown composition {composition!r} (have: {', '.join(COMPOSITIONS)})"
            ) from None
        excitatory = left / (left + right)
    return excitatory, 1.0 - excitatory


def degrees_for(
    composition: str,
    excitatory_degree: float = 20.0,
    rule: str = "fixed_probability",
) -> dict[str, float]:
    if rule not in ("fixed_probability", "fixed_degree"):
        raise ValueError(
            f"unknown degree_rule {rule!r} (have: fixed_probability, fixed_degree)"
        )
    excitatory, inhibitory = fractions(composition)
    share = {"EXC": excitatory, "INH": inhibitory}
    measured = {"EXC->EXC": float(excitatory_degree)} | INHIBITORY_DEGREE

    degrees = {"INH->INH": 0.0, "default": 0.0}
    for projection, value in measured.items():
        pre, post = projection.split("->")
        # a projection into or out of an absent population is absent, whatever
        # the rule says the degree would be
        if share[pre] <= 0.0 or share[post] <= 0.0:
            degrees[projection] = 0.0
            continue
        reference = DEGREE_REFERENCE[projection]
        degrees[projection] = (
            value if rule == "fixed_degree" else value * share[pre] / reference["share"]
        )
    return degrees


MOTONEURON = {
    "synapse_type": "excitatory",
    "transmitter": "cholinergic",
    "soma_only": False,
}
RENSHAW = {
    "synapse_type": "inhibitory",
    "transmitter": "glycinergic",
    "soma_only": True,
}
RECURRENT_NMDA = {
    "EXC->EXC": {
        "NMDA": {
            "e": 0,
            "g_unit": 0.0005,
            "tau_decay": 80.0,
            "tau_rise": 0.5,
            "weight": 0.0,
        }
    }
}


def geometry(metadata: dict, scale: float = 1.0, guard: float | None = None) -> dict:
    pos = metadata["geometry"]["pos"]
    xs = [p[1] for p in pos]
    ys = [p[2] for p in pos]
    pitch = float(metadata["geometry"]["pitch"][0])

    recorded = pitch / 2.0
    margin = recorded if guard is None else float(guard)

    def _box(m, s):
        a0, a1 = min(xs) - m, max(xs) + m
        b0, b1 = min(ys) - m, max(ys) + m
        if s != 1.0:
            cx, cy = (a0 + a1) / 2.0, (b0 + b1) / 2.0
            hw, hh = (a1 - a0) * s / 2.0, (b1 - b0) * s / 2.0
            a0, a1, b0, b1 = cx - hw, cx + hw, cy - hh, cy + hh
        return (a0, b0), (a1, b1)

    (x0, y0), (x1, y1) = _box(margin, scale)
    (ix0, iy0), (ix1, iy1) = _box(recorded, scale)

    electrodes = [
        [int(i), float(x), float(y)]
        for i, x, y in pos
        if ix0 <= x <= ix1 and iy0 <= y <= iy1
    ]
    return {
        "area": ((x0, y0), (x1, y1)),
        "interior": ((ix0, iy0), (ix1, iy1)),
        "electrodes": electrodes,
        "pitch": pitch,
        "guard": float(margin),
        "margin": float(margin - recorded) * float(scale),
        "scale": float(scale),
    }


def spec(
    metadata: dict,
    sample: str,
    cells: int | None = None,
    excitatory_degree: float = 20.0,
    scale: float = 1.0,
    composition: str | None = None,
    inhibitory_fraction: float | None = None,
    seed_key: str | None = None,
    seed: int | None = None,
    sigma: float | None = None,
    boundary: float | None = 0.0,
    degree_rule: str = "fixed_probability",
    mea: bool = True,
) -> dict:
    """The `Monolayer` spec for one sample, as `{"cls": ..., "kwargs": ...}`.

    Args:
        excitatory_degree: `EXC->EXC` in-degree at the reference
            composition, which is `E`. Under `fixed_probability` the value a
            given composition ends up with is this scaled by its excitatory
            share.
        degree_rule: `fixed_probability` (default) carries a measured degree to
            another composition through the presynaptic share; `fixed_degree`
            holds it. See `degrees_for`.
        cells: Total cells at full scale; must leave enough inhibitory cells
            for the `INH->EXC` convergence of 40 to be realisable.
        scale: Linear size of a reduced replica.
        composition: The log's seeding label. It is *metadata*: it sets the
            initial inhibitory fraction and the inhibitory population's
            template, and nothing else reads it. A fit infers the fraction.
        inhibitory_fraction: Overrides the composition's share directly, which
            is what a search over composition varies.
        sigma: Kernel length constant, um.
        boundary: How the kernel treats the edge.
            `0.0` (default) is an open edge at the recorded region
        mea: Carry the recording's electrodes on the system, so a promoted
            `env.json` reads the same channels the target measured.
    """
    known = compositions(metadata)
    composition = composition or known.get(sample)
    if composition is None:
        raise ValueError(
            f"unknown sample {sample!r}; this recording holds "
            f"{', '.join(known) or 'none'}"
        )

    pitch = float(metadata["geometry"]["pitch"][0])
    guard = pitch / 2.0 + (0.0 if boundary is None else float(boundary))
    geo = geometry(metadata, scale=scale, guard=guard)
    (x0, y0), (x1, y1) = geo["area"]
    (ix0, iy0), (ix1, iy1) = geo["interior"]
    widening = ((x1 - x0) * (y1 - y0)) / ((ix1 - ix0) * (iy1 - iy0))
    total = max(
        1,
        round(
            (cells or CELLS_BY_COMPOSITION.get(composition, DEFAULT_CELLS))
            * scale**2
            * widening
        ),
    )

    excitatory, inhibitory_share = fractions(composition)
    if inhibitory_fraction is not None:
        inhibitory_share = float(inhibitory_fraction)
        excitatory = 1.0 - inhibitory_share

    populations = {
        "EXC": MOTONEURON | {"ratio": excitatory},
        "INH": RENSHAW | {"ratio": inhibitory_share},
    }
    degrees = degrees_for(composition, excitatory_degree, degree_rule)
    if inhibitory_fraction is not None:
        share = {"EXC": excitatory, "INH": inhibitory_share}
        for projection, value in (
            {"EXC->EXC": float(excitatory_degree)} | INHIBITORY_DEGREE
        ).items():
            pre, post = projection.split("->")
            if share[pre] <= 0.0 or share[post] <= 0.0:
                degrees[projection] = 0.0
            elif degree_rule == "fixed_probability":
                degrees[projection] = (
                    value * share[pre] / DEGREE_REFERENCE[projection]["share"]
                )

    inhibitory = int(total * inhibitory_share)
    if degrees.get("INH->EXC", 0) > max(inhibitory, 0) > 0:
        raise ValueError(
            f"INH->EXC={degrees['INH->EXC']:.0f} needs at least that many "
            f"inhibitory cells, but {total} cells at ratio "
            f"{inhibitory_share:g} gives {inhibitory}"
        )

    connectivity = {"mean_degree": degrees}
    if sigma is not None:
        connectivity["sigma"] = float(sigma)

    kwargs = {
        "total_cells": total,
        "populations": populations,
        "connectivity": connectivity,
        "area": "rectangle",
        "area_kwargs": {"x_range": [x0, x1], "y_range": [y0, y1]},
        "boundary": None if boundary is None else geo["margin"],
        "synapse_overrides": RECURRENT_NMDA,
        "seed": (
            int(seed)
            if seed is not None
            else zlib.crc32((seed_key or sample).encode()) % 100_000
        ),
        "name": f"{sample}@{metadata['name']}",
    }
    if mea:
        kwargs["mea"] = mea_spec(metadata, scale=scale)

    return {"cls": "livn.system.Monolayer", "kwargs": kwargs}


DEFAULT_INPUT_RADIUS = 50.0
DEFAULT_OUTPUT_RADIUS = 50.0
DEFAULT_Z_RANGE = (0.0, 10.0)


def mea_spec(
    metadata: dict,
    scale: float = 1.0,
    input_radius: float = DEFAULT_INPUT_RADIUS,
    output_radius: float = DEFAULT_OUTPUT_RADIUS,
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
) -> dict:
    z = z_range[0] + (z_range[1] - z_range[0]) / 2.0
    geo = geometry(metadata, scale=scale)
    return {
        "electrode_coordinates": [
            [float(i), float(x), float(y), float(z)] for i, x, y in geo["electrodes"]
        ],
        "input_radius": input_radius,
        "output_radius": output_radius,
    }
