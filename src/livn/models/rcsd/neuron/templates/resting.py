from __future__ import annotations

CURRENTS = ("ina", "ik", "ica", "i_pas")


def insert_constant(sections) -> None:
    for sec in sections:
        if not _has_constant(sec):
            sec.insert("constant")


def pin(sections) -> None:
    for sec in sections:
        if not _has_constant(sec):
            continue
        for seg in sec:
            seg.ic_constant = -sum(float(getattr(seg, name, 0.0)) for name in CURRENTS)


def _has_constant(sec) -> bool:
    try:
        return bool(sec.has_membrane("constant"))
    except AttributeError:  # older NEURON without Section.has_membrane
        return hasattr(sec(0.5), "ic_constant")
