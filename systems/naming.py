from __future__ import annotations

import re

_NAME = re.compile(r"^E(?P<ratio>[0-9]*\.?[0-9]+)?(?P<inhibitory>I)?$")


def base(name: str) -> str:
    """E3I_b -> E3I"""
    return str(name).split("/")[-1].split("_")[0]


def composition_of(name: str) -> tuple[float, float]:
    """The `(excitatory, inhibitory)` ratio a name states."""
    match = _NAME.match(base(name))
    if match is None:
        raise ValueError(
            f"{name!r} is not a composition name; expected `E`, `EI`, or "
            "`E<ratio>I` such as `E3I` (the ratio follows E, and I is always 1)"
        )
    if match["ratio"] is not None and not match["inhibitory"]:
        raise ValueError(
            f"{name!r} gives a ratio without an inhibitory population; write "
            f"`E{match['ratio']}I`, or `E` for excitatory only"
        )
    if not match["inhibitory"]:
        return 1.0, 0.0
    return float(match["ratio"] or 1.0), 1.0


def ratios(name: str) -> dict[str, float]:
    """`{"EXC": fraction, "INH": fraction}`, summing to 1."""
    excitatory, inhibitory = composition_of(name)
    total = excitatory + inhibitory
    return {"EXC": excitatory / total, "INH": inhibitory / total}


def name_for(excitatory: float, inhibitory: float = 1.0) -> str:
    """The name for a ratio: `(3, 1)` -> `E3I`, `(1, 0)` -> `E`."""
    if inhibitory == 0:
        return "E"
    scaled = excitatory / inhibitory
    if scaled == 1:
        return "EI"
    text = f"{scaled:g}"
    return f"E{text}I"
