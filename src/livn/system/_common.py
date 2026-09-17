from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from livn import types

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CellsMetaData(BaseModel):
    """Cells metadata"""

    population_names: list[types.PopulationName]
    population_ranges: dict[types.PopulationName, tuple[int, int]]
    cell_attribute_info: dict[types.PopulationName, dict[str, list[str]]]

    def has(self, population: types.PopulationName, attribute: str) -> bool:
        return attribute in self.cell_attribute_info.get(population, {})

    def population_count(self, population: types.PopulationName) -> int:
        return self.population_ranges[population][1]

    def cell_count(self) -> int:
        """Return the total number of cells across all populations."""
        return sum(
            self.population_count(population) for population in self.population_names
        )


class Tree(BaseModel):
    """Tree"""


class Projection(BaseModel):
    """Projection"""


class Element(BaseModel):
    uuid: str = str | None
    kind: str = "Element"
    module: str | None = None
    version: list[str | dict] = []
    config: dict | None = None
    predicate: dict | None = None
    context: dict | None = None
    lineage: tuple[str, ...] = ()


def resolve_selection(
    system,
    spec,
    populations: Sequence[str] | None = None,
    seed: int | None = 123,
    method: str = "first",
    bounds=None,
) -> dict[str, Any] | None:
    if isinstance(spec, str):
        if bounds is not None or method not in ("first", None):
            raise ValueError(
                f"selection({spec!r}) names a stored selection, which already "
                f"fixes which cells are built; method={method!r}/bounds= would "
                "contradict it"
            )
        gids = system.selection_document(spec).get("gids")
        if not isinstance(gids, dict) or not gids:
            raise ValueError(
                f"selection {spec!r} has no `gids` block; a stored selection "
                "holds the resolved gids per population"
            )
        spec = {p: sorted(int(g) for g in v) for p, v in gids.items()}

    names = system.populations if populations is None else populations
    ranges = system.cells_meta_data.population_ranges

    coordinates = None
    if method == "patch":
        coordinates = {p: system.coordinate_array(p) for p in names if p in ranges}

    return selection_from_ranges(
        ranges,
        spec,
        populations=names,
        seed=seed,
        method=method,
        coordinates=coordinates,
        bounds=bounds,
    )


def selection_from_ranges(
    ranges: dict[types.PopulationName, tuple[int, int]],
    spec,
    populations: Sequence[str] | None = None,
    seed: int | None = 123,
    method: str = "first",
    coordinates: dict[str, Any] | None = None,
    bounds=None,
) -> dict[str, Any] | None:
    """Resolve a subselection spec against ``{population: (start, count)}`` ranges."""
    if spec is None and bounds is None:
        return None

    # deliberately real numpy since selections index host-side gid arrays and must
    # not depend on whichever array library the active backend pulled in
    import numpy as npn

    if populations is None:
        populations = list(ranges.keys())
    pops = [p for p in populations if p in ranges]

    if bounds is not None and method != "patch":
        raise ValueError(
            f"bounds= is only meaningful for method='patch', got {method!r}"
        )

    if method == "patch" and isinstance(spec, dict):
        offenders = [
            p for p, v in spec.items() if not isinstance(v, (list, tuple, npn.ndarray))
        ]
        if offenders:
            raise ValueError(
                "method='patch' resolves one box for every population, so a "
                f"per-population count is ambiguous (got {offenders}); pass a "
                "float area fraction, an int cell budget, or bounds="
            )

    elif method == "patch":
        if coordinates is None:
            raise ValueError(
                "method='patch' needs cell coordinates; use System.selection or "
                "ParallelSystem.selection, which supply them"
            )

        table: dict[str, Any] = {}
        for p in pops:
            c = coordinates.get(p)
            if c is None:
                continue
            c = npn.asarray(c, dtype=npn.float64)
            if c.ndim == 2 and len(c):
                table[p] = c
        if not table:
            return {}

        stacked = npn.vstack(list(table.values()))
        lo, hi = stacked[:, 1:3].min(axis=0), stacked[:, 1:3].max(axis=0)
        centre = (lo + hi) / 2.0

        if bounds is not None:
            (x0, y0), (x1, y1) = bounds
            box_lo = npn.array([min(x0, x1), min(y0, y1)], dtype=npn.float64)
            box_hi = npn.array([max(x0, x1), max(y0, y1)], dtype=npn.float64)
        elif isinstance(spec, float):
            if spec <= 0:
                return {}
            if spec >= 1:
                box_lo, box_hi = lo, hi
            else:
                half = (hi - lo) * npn.sqrt(spec) / 2.0
                box_lo, box_hi = centre - half, centre + half
        else:
            k = min(int(spec), len(stacked))
            if k <= 0:
                return {}
            d2 = ((stacked[:, 1:3] - centre) ** 2).sum(axis=1)
            keep = stacked[npn.lexsort((stacked[:, 0], d2))[:k], 0].astype(npn.int64)
            budgeted: dict[str, Any] = {}
            for p, c in table.items():
                gids = c[:, 0].astype(npn.int64)
                inside = gids[npn.isin(gids, keep)]
                if len(inside):
                    budgeted[p] = npn.sort(inside)
            return budgeted

        boxed: dict[str, Any] = {}
        for p, c in table.items():
            xy = c[:, 1:3]
            inside = npn.all((xy >= box_lo) & (xy <= box_hi), axis=1)
            if inside.any():
                boxed[p] = npn.sort(c[inside, 0].astype(npn.int64))
        return boxed

    counts: dict[str, int] = {}
    explicit: dict[str, npn.ndarray] = {}

    def _frac_count(f: float, size: int) -> int:
        if f <= 0:
            return 0
        if f >= 1:
            return size
        return max(1, round(f * size))

    if isinstance(spec, dict):
        for p in pops:
            if p not in spec:
                continue
            v = spec[p]
            if isinstance(v, (list, tuple, npn.ndarray)):
                explicit[p] = npn.asarray(sorted(int(g) for g in v), dtype=npn.int64)
            elif isinstance(v, float):
                counts[p] = _frac_count(v, ranges[p][1])
            else:
                counts[p] = min(int(v), ranges[p][1])
    elif isinstance(spec, float):
        for p in pops:
            counts[p] = _frac_count(spec, ranges[p][1])
    elif isinstance(spec, int):
        total_size = sum(ranges[p][1] for p in pops)
        if total_size == 0:
            return {}
        for p in pops:
            counts[p] = min(ranges[p][1], round(spec * ranges[p][1] / total_size))
    else:
        raise TypeError(f"unsupported selection spec: {type(spec).__name__}")

    rng = npn.random.default_rng(seed)
    out: dict[str, npn.ndarray] = {}
    for p in pops:
        if p in explicit:
            out[p] = explicit[p]
            continue
        k = counts.get(p, 0)
        if k <= 0:
            continue
        start, count = ranges[p]
        k = min(k, count)
        if method == "random":
            gids = npn.sort(
                rng.choice(count, size=k, replace=False).astype(npn.int64) + start
            )
        else:  # "first" -> contiguous block
            gids = npn.arange(start, start + k, dtype=npn.int64)
        out[p] = gids
    return out


def projection_attribute(namespace, name: str, index: int = 0, default=None):
    if namespace is None:
        return default
    if isinstance(namespace, dict):
        return namespace.get(name, default)
    if isinstance(namespace, list | tuple):
        return namespace[index] if len(namespace) > index else default
    return namespace


def _placement_rows(syn_ids, swc_types, syn_locs):
    import numpy as npn

    syn_ids = npn.asarray(syn_ids).astype(npn.int64, copy=False)
    swc_types = npn.asarray(swc_types).astype(npn.int64, copy=False)
    syn_locs = npn.asarray(syn_locs).astype(npn.float64, copy=False)

    if syn_ids.size == 0:
        return syn_ids, swc_types, syn_locs

    order = npn.argsort(syn_ids, kind="stable")
    ids = syn_ids[order]
    last = npn.append(ids[1:] != ids[:-1], True)
    keep = order[last]
    return ids[last], swc_types[keep], syn_locs[keep]
