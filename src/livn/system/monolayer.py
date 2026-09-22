from __future__ import annotations

import contextlib
import hashlib
import math
from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy
from pydantic import BaseModel, Field, model_validator

from livn import types
from livn.backend import backend
from livn.system._common import (
    CellsMetaData,
    Projection,
    resolve_selection,
    stack_coordinates,
)
from livn.utils import Jsonable, import_object_by_path, sentinel

if TYPE_CHECKING:
    from mpi4py import MPI

    from livn.io import IO
    from livn.types import Model

if "ax" in backend():
    import jax.numpy as np
else:
    import numpy as np


SYN_TYPE = {"excitatory": 0, "inhibitory": 1}
INHIBITORY = ("gabaergic", "glycinergic")

# how far a chunked kernel reaches in one block of postsynaptic columns
CONNECTIVITY_CHUNK = 2048

GRAPH_FORMAT_VERSION = 1


class SWCType(IntEnum):
    """SWC section-type codes (Cannon et al., matching what NeuroH5 records)."""

    soma = 1
    axon = 2
    basal = 3
    apical = 4
    trunk = 5
    tuft = 6
    ais = 7
    hillock = 8


class StableRandom:
    """A Philox stream addressed by position rather than by draw order."""

    __slots__ = ("_key", "_pos")

    def __init__(self, seed: int, *tags: str):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(int(seed)).encode())
        for tag in tags:
            digest.update(b"\x00" + tag.encode())
        self._key = numpy.frombuffer(digest.digest(), dtype="<u8").copy()
        self._pos = 0

    def _raw(self, n: int, at: int | None) -> numpy.ndarray:
        counter = self._pos if at is None else int(at)
        raw = numpy.random.Philox(key=self._key, counter=counter).random_raw(n)
        if at is None:
            self._pos += n
        return raw

    def random(self, n: int, at: int | None = None) -> numpy.ndarray:
        return (self._raw(n, at) >> numpy.uint64(40)).astype(
            numpy.float32
        ) * numpy.float32(2.0**-24)

    def uniform(self, low: float, high: float, size: int) -> numpy.ndarray:
        return (
            numpy.float32(low)
            + (numpy.float32(high) - numpy.float32(low)) * self.random(size)
        ).astype(numpy.float32)


def rectangle(
    count: int,
    rng: StableRandom,
    *,
    margin: float = 0.0,
    x_range: tuple[float, float] = (0.0, 4000.0),
    y_range: tuple[float, float] = (0.0, 4000.0),
):
    xmin, xmax = float(x_range[0]), float(x_range[1])
    ymin, ymax = float(y_range[0]), float(y_range[1])
    if xmin + margin > xmax - margin or ymin + margin > ymax - margin:
        raise ValueError(
            f"a margin of {margin:g} um leaves no interior in a "
            f"{xmax - xmin:g} x {ymax - ymin:g} um rectangle"
        )

    xs = rng.uniform(xmin, xmax, size=count)
    ys = rng.uniform(ymin, ymax, size=count)
    interior = (
        (xs >= xmin + margin)
        & (xs <= xmax - margin)
        & (ys >= ymin + margin)
        & (ys <= ymax - margin)
    )
    return xs, ys, interior, (xmin, ymin, xmax, ymax)


def disk(
    count: int,
    rng: StableRandom,
    *,
    margin: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 500.0,
    inner_radius: float = 0.0,
):
    cx, cy = float(center[0]), float(center[1])
    r_in, r_out = float(inner_radius), float(radius)
    outer, inner = r_out - margin, (r_in + margin if r_in > 0.0 else 0.0)
    if outer < inner:
        raise ValueError(
            f"a margin of {margin:g} um leaves no interior in a disk of radius "
            f"{r_out:g} um"
        )

    u = rng.random(count)
    r = numpy.sqrt(u * (r_out**2 - r_in**2) + r_in**2)
    theta = rng.uniform(0, 2 * numpy.pi, size=count)
    xs = (cx + r * numpy.cos(theta)).astype(numpy.float32)
    ys = (cy + r * numpy.sin(theta)).astype(numpy.float32)
    return (
        xs,
        ys,
        (r <= outer) & (r >= inner),
        (cx - r_out, cy - r_out, cx + r_out, cy + r_out),
    )


AREAS = {
    "rectangle": rectangle,
    "disk": disk,
}


def _area_function(name: str):
    found = AREAS.get(name)
    if found is not None:
        return found
    if "." in name:
        return import_object_by_path(name)
    raise ValueError(
        f"unknown area {name!r}; expected one of {sorted(AREAS)} or a path"
    )


def _released(transmitter: str | None, synapse_type: str) -> str:
    if transmitter:
        return transmitter
    return "glutamatergic" if synapse_type == "excitatory" else "gabaergic"


def _single_compartment(soma_only: bool | None, synapse_type: str) -> bool:
    if soma_only is not None:
        return bool(soma_only)
    return synapse_type == "inhibitory"


class PopulationSpec(BaseModel):
    ratio: float | None = Field(default=None, ge=0.0)
    count: int | None = Field(default=None, ge=0)
    synapse_type: str = "excitatory"
    transmitter: str | None = Field(
        default=None,
        description=(
            "What this population releases: glutamatergic, cholinergic, "
            "gabaergic or glycinergic. Defaults from synapse_type."
        ),
    )
    soma_only: bool | None = Field(
        default=None,
        description=(
            "Single-compartment cells, which can only receive on the soma. "
            "Defaults true for inhibitory populations."
        ),
    )

    @property
    def released(self) -> str:
        return _released(self.transmitter, self.synapse_type)

    @property
    def single_compartment(self) -> bool:
        return _single_compartment(self.soma_only, self.synapse_type)

    @model_validator(mode="after")
    def _validate(self) -> PopulationSpec:
        if self.ratio is None and self.count is None:
            raise ValueError("a population needs either 'ratio' or 'count'")
        if self.synapse_type not in SYN_TYPE:
            raise ValueError(
                f"synapse_type must be one of {list(SYN_TYPE)}, got "
                f"{self.synapse_type!r}"
            )
        expected = "inhibitory" if self.released in INHIBITORY else "excitatory"
        if expected != self.synapse_type:
            raise ValueError(
                f"a population releasing {self.released!r} is {expected}, but "
                f"synapse_type is {self.synapse_type!r}; the transmitter sets "
                "the mechanism and synapse_type is what `syn_types` records, "
                "so the two must agree"
            )
        return self


class ConnectivitySpec(BaseModel):
    """Distance-dependent connectivity.

    ``kernel`` is the shape of the distance dependence: ``"exponential"``
    (``exp(-d / sigma)``, heavy-tailed, matching the long reach of free-growing
    2D-culture axons) or ``"gaussian"`` (``exp(-d^2 / 2 sigma^2)``, tissue-like).

    ``sigma`` is the length constant in um. ``mean_degree`` fixes the expected
    in-degree, so sigma controls only the spatial spread of a cell's inputs, not
    how many it has.
    """

    kernel: Literal["exponential", "gaussian"] = "exponential"
    sigma: float = Field(600.0, gt=0.0)
    mean_degree: float | dict[str, float] = Field(
        default=100.0,
        description="Expected inputs per cell, overall or per `pre->post`",
    )
    cutoff: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_self_connections: bool = False
    velocity: float | None = Field(default=None, gt=0.0)
    degree_rule: Literal["fixed_probability", "fixed_degree"] | None = None
    degree_reference: dict[str, float] = Field(
        default_factory=dict,
        description=("Per projection, presynaptic share its calibrated `mean_degree`"),
    )
    floor: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per projection, the fewest inputs a postsynaptic cell may end up "
            "with. `EXC->INH: 1` because a mean of 4 is a mean, and a Renshaw "
            "cell contacted by zero motoneurons is not a member of the "
            "population that measurement describes."
        ),
    )


class Monolayer(Jsonable):
    """A 2D culture defined by its specification.

    Arguments:
        total_cells: Cells across all populations, split by each population's
            `ratio` (explicit `count`s are taken out first).
        populations: `{name: PopulationSpec}`.
        connectivity: :class:`ConnectivitySpec`.
        area: `"rectangle"`, `"disk"`, or an importable path to a callable with
            the same signature. `area_kwargs` is passed through to it.
        boundary: How the kernel treats the edge of the area.

            - `None` (default) takes distance on a torus. In-degree and
              out-degree are then constant everywhere.
            - A width in um draws that much guard beyond the readout window:
              cells simulated so the ones the array reads have partners on
              every side, but not themselves read. `0.0` is an open edge.

        normalization: `"per_post"` (default) gives every cell its target
            in-degree wherever it sits, from its own kernel mass so it has no
            collective, and no edge deficit. `"global"` normalizes once over
            the whole projection and means "in-degree at the centre of a large draw".
        seed: Everything drawn here derives from it.
    """

    TRANSMITTERS: ClassVar[dict] = {
        "glutamatergic": (
            {
                "AMPA": {"e": 0, "g_unit": 0.0005, "tau_decay": 3.0, "tau_rise": 0.5},
                "NMDA": {"e": 0, "g_unit": 0.0005, "tau_decay": 80.0, "tau_rise": 0.5},
            },
            1,
        ),
        "cholinergic": (
            {"AMPA": {"e": 0, "g_unit": 0.0005, "tau_decay": 7.0, "tau_rise": 0.5}},
            7,
        ),
        "glycinergic": (
            {
                "GABA_A": {
                    "e": -70,
                    "g_unit": 0.00025,
                    "tau_decay": 5.0,
                    "tau_rise": 0.3,
                }
            },
            5,
        ),
        "gabaergic": (
            {"GABA_A": {"e": -60, "g_unit": 0.001, "tau_decay": 6.0, "tau_rise": 0.3}},
            1,
        ),
    }

    # a periodic box narrower than this many sigma wraps onto itself
    MIN_EXTENT_IN_SIGMA: ClassVar[float] = 4.0

    def __init__(self, *args, **kwargs):
        self._configure(*args, **kwargs)
        self._draw_cells()
        self._resolve_projections()

    @classmethod
    def canonical(cls, *args, **kwargs) -> dict:
        blank = cls.__new__(cls)
        blank._configure(*args, **kwargs)
        return blank.serialize()

    @classmethod
    def spec_uuid(cls, *args, **kwargs) -> str:
        blank = cls.__new__(cls)
        blank._configure(*args, **kwargs)
        return blank.uuid

    def _configure(
        self,
        total_cells: int | None = None,
        populations: dict | None = None,
        connectivity: dict | ConnectivitySpec | None = None,
        area: str = "rectangle",
        area_kwargs: dict | None = None,
        boundary: float | None = None,
        normalization: Literal["per_post", "global"] = "per_post",
        z_range: tuple[float, float] = (0.0, 10.0),
        synapse_overrides: dict | None = None,
        population_definitions: dict[str, int] | None = None,
        mea: dict | None = None,
        seed: int = 123,
        name: str = "Monolayer",
        comm: MPI.Intracomm | None = None,
    ) -> None:
        if populations is None:
            populations = {
                "EXC": {"ratio": 0.8, "synapse_type": "excitatory"},
                "INH": {"ratio": 0.2, "synapse_type": "inhibitory"},
            }
        self.populations_spec = {
            p: v if isinstance(v, PopulationSpec) else PopulationSpec(**v)
            for p, v in populations.items()
        }
        if connectivity is None:
            connectivity = {}
        self.connectivity = (
            connectivity
            if isinstance(connectivity, ConnectivitySpec)
            else ConnectivitySpec(**connectivity)
        )
        if boundary is not None and float(boundary) < 0.0:
            raise ValueError(
                f"boundary is a guard width in um, or None for periodic; "
                f"got {boundary!r}"
            )
        if normalization not in ("per_post", "global"):
            raise ValueError(
                f"normalization must be 'per_post' or 'global', got {normalization!r}"
            )

        self.total_cells = total_cells
        self.area = area
        self.area_kwargs = dict(area_kwargs or {})
        self.boundary = None if boundary is None else float(boundary)
        self.normalization = normalization
        self.z_range = (float(z_range[0]), float(z_range[1]))
        if self.z_range[0] > self.z_range[1]:
            raise ValueError("z_range must satisfy zmin <= zmax")
        self.synapse_overrides = dict(synapse_overrides or {})
        self.population_definitions = dict(
            population_definitions
            or {p: 10 + i for i, p in enumerate(self.populations_spec)}
        )
        self.mea_spec = dict(mea) if mea else None
        self.seed = int(seed)
        self.name = name
        self.comm = comm
        self.uri = None
        self.files: dict[str, str] = {}
        self.io_size = 1

        missing = set(self.populations_spec) - set(self.population_definitions)
        if missing:
            raise ValueError(
                f"population(s) {sorted(missing)} are not in population_definitions"
            )

    # -- the spec ------------------------------------------------------------

    def serialize(self) -> dict:
        return {
            "total_cells": self.total_cells,
            "populations": {
                p: v.model_dump(exclude_none=True)
                for p, v in self.populations_spec.items()
            },
            "connectivity": self.connectivity.model_dump(exclude_none=True),
            "area": self.area,
            "area_kwargs": dict(self.area_kwargs),
            "boundary": self.boundary,
            "normalization": self.normalization,
            "z_range": list(self.z_range),
            "synapse_overrides": dict(self.synapse_overrides),
            "population_definitions": dict(self.population_definitions),
            "mea": self.mea_spec,
            "seed": self.seed,
            "name": self.name,
        }

    @property
    def uuid(self) -> str:
        """Content hash of the spec.

        The identity of a generated system is what it was drawn from, so an
        identical spec is an identical system -- which a random uuid stamped at
        generation could not express.
        """
        return hashlib.blake2b(
            self.as_json(sort_keys=True).encode(), digest_size=16
        ).hexdigest()

    def __repr__(self) -> str:
        counts = ", ".join(f"{p}={c}" for p, c in self.population_counts.items())
        return (
            f"Monolayer({counts}, sigma={self.connectivity.sigma:g}, "
            f"boundary={self.boundary!r})"
        )

    # -- cells ---------------------------------------------------------------

    def _resolve_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        ratios: dict[str, float] = {}
        fixed = 0
        for p, spec in self.populations_spec.items():
            if spec.count is not None:
                counts[p] = spec.count
                fixed += spec.count
            else:
                ratios[p] = spec.ratio

        total = self.total_cells if self.total_cells is not None else fixed
        if ratios:
            total_ratio = sum(ratios.values())
            if total_ratio <= 0:
                raise ValueError("population ratios must sum to a positive value")
            residual = total - fixed
            if residual < 0:
                raise ValueError(
                    f"explicit population counts total {fixed}, more than "
                    f"total_cells={total}"
                )
            remainders = []
            allocated = 0
            for p, ratio in ratios.items():
                exact = residual * ratio / total_ratio
                counts[p] = math.floor(exact)
                allocated += counts[p]
                remainders.append((p, exact - counts[p]))
            # hand the rounding residue to the largest fractional parts, so the
            # counts sum to `total` exactly rather than approximately
            left = residual - allocated
            if left > 0:
                remainders.sort(key=lambda item: item[1], reverse=True)
                for p, _ in remainders[:left]:
                    counts[p] += 1

        if sum(counts.values()) < 1:
            raise ValueError("a Monolayer needs at least one cell")
        return {p: counts[p] for p in self.populations_spec}

    def _draw_cells(self) -> None:
        counts = self._resolve_counts()
        self.population_counts = counts

        ranges: dict[str, tuple[int, int]] = {}
        start = 0
        for p, count in counts.items():
            ranges[p] = (start, count)
            start += count
        self.num_neurons = start

        self.cells_meta_data = CellsMetaData(
            population_names=list(counts),
            population_ranges=ranges,
            cell_attribute_info={p: {} for p in counts},
        )

        area_fn = _area_function(self.area)
        zmin, zmax = self.z_range

        xs_all, ys_all, zs_all, interior_all = [], [], [], []
        bounds = None
        for p, count in counts.items():
            rng = StableRandom(self.seed, "coordinates", p)
            xs, ys, interior, box = area_fn(
                count, rng, margin=self.guard, **self.area_kwargs
            )
            zs = (
                rng.uniform(zmin, zmax, size=count)
                if zmax > zmin
                else numpy.full(count, zmin, dtype=numpy.float32)
            )
            xs_all.append(numpy.asarray(xs, dtype=numpy.float32))
            ys_all.append(numpy.asarray(ys, dtype=numpy.float32))
            zs_all.append(numpy.asarray(zs, dtype=numpy.float32))
            interior_all.append(numpy.asarray(interior, dtype=bool))
            bounds = box if bounds is None else bounds

        x = numpy.concatenate(xs_all) if xs_all else numpy.zeros(0, numpy.float32)
        y = numpy.concatenate(ys_all) if ys_all else numpy.zeros(0, numpy.float32)
        z = numpy.concatenate(zs_all) if zs_all else numpy.zeros(0, numpy.float32)
        self.interior = (
            numpy.concatenate(interior_all) if interior_all else numpy.zeros(0, bool)
        )

        # the area the spec asked for, not the bounding box of what was drawn:
        # a periodic kernel wraps on the box, so it has to be the same box at
        # every cell count or the graph would depend on the draw
        self._box = tuple(float(v) for v in bounds)
        # A guard's cells are simulated -- that is what they are for, to give the
        # cells the array reads a full set of presynaptic partners -- but they
        # are outside the recording, so the electrodes cover the interior only.
        x0, y0, x1, y1 = self._box
        self._readout_box = (
            (x0 + self.guard, y0 + self.guard, x1 - self.guard, y1 - self.guard)
            if self.guard
            else self._box
        )
        self._xy = numpy.column_stack([x, y]).astype(numpy.float32)
        self._neuron_coordinates = numpy.column_stack(
            [numpy.arange(self.num_neurons, dtype=numpy.float64), x, y, z]
        )

        if self.boundary is None:
            self._check_periodic_extent()

    def _check_periodic_extent(self) -> None:
        x0, y0, x1, y1 = self._box
        sigma = self.connectivity.sigma
        smallest = min(x1 - x0, y1 - y0)
        floor = self.MIN_EXTENT_IN_SIGMA * sigma
        if smallest < floor:
            raise ValueError(
                f"a periodic culture {x1 - x0:g} x {y1 - y0:g} um across is only "
                f"{smallest / sigma:.1f} sigma at sigma={sigma:g}, below the "
                f"{self.MIN_EXTENT_IN_SIGMA:g} sigma this needs: the kernel wraps "
                "onto itself and every cell is correlated with its own image. "
                f"Widen the area past {floor:g} um, lower sigma, or give a "
                "guard width as `boundary=`"
            )

    @property
    def guard(self) -> float:
        """Width of the generated-but-unread ring, 0 when there is no edge."""
        return 0.0 if self.boundary is None else self.boundary

    @property
    def readout_box(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` the electrodes span -- the interior of a guard."""
        return self._readout_box

    @property
    def period(self) -> tuple[float, float] | None:
        """``(width, height)`` the kernel wraps on, or None when it does not."""
        if self.boundary is not None:
            return None
        x0, y0, x1, y1 = self._box
        return (x1 - x0, y1 - y0)

    # -- projections ---------------------------------------------------------

    @property
    def conduction_velocity(self) -> float | None:
        return self.connectivity.velocity

    def _target_degree(self, pre: str, post: str) -> float:
        configured = self.connectivity.mean_degree
        if not isinstance(configured, dict):
            return float(configured)
        key = f"{pre}->{post}"
        if key in configured:
            return float(configured[key])
        return float(configured.get("default", 0.0))

    def _projection_synapse(self, pre: str, post: str):
        pre_spec = self.populations_spec[pre]
        post_spec = self.populations_spec[post]

        released = pre_spec.released
        receives_on = (
            "soma"
            if (post_spec.single_compartment or released in INHIBITORY)
            else "dend"
        )
        if released not in self.TRANSMITTERS:
            raise ValueError(
                f"population {pre!r} releases {released!r}; known transmitters: "
                f"{', '.join(sorted(self.TRANSMITTERS))}"
            )

        template, contacts = self.TRANSMITTERS[released]
        mechanisms = {
            name: {**params, "weight": float(contacts)}
            for name, params in template.items()
        }
        for name, params in (
            self.synapse_overrides.get(f"{pre}->{post}") or {}
        ).items():
            base = mechanisms.get(name, {"weight": float(contacts)})
            mechanisms[name] = {**base, **params}
        return mechanisms, receives_on, contacts

    def _resolve_projections(self) -> None:
        """Work out which projections exist and where their synapses land."""
        valid = {
            f"{pre}->{post}"
            for pre in self.populations_spec
            for post in self.populations_spec
        }
        degrees = self.connectivity.mean_degree
        if isinstance(degrees, dict):
            unknown = sorted(set(degrees) - valid - {"default"})
            if unknown:
                raise ValueError(
                    f"mean_degree names {unknown}, which are not projections of "
                    f"this system; expected 'default' or one of {sorted(valid)}"
                )
        unknown = sorted(set(self.connectivity.floor) - valid)
        if unknown:
            raise ValueError(f"floor names {unknown}, not projections of this system")
        unknown = sorted(set(self.synapse_overrides) - valid)
        if unknown:
            raise ValueError(
                f"synapse_overrides names {unknown}, not projections of this system"
            )

        synapses: dict[str, dict[str, dict]] = {p: {} for p in self.populations_spec}
        # (pre, post) -> (target_sec, target_swc, syn_type_index, degree)
        self._sites: dict[tuple[str, str], tuple[int, int, int, float]] = {}

        for post in self.populations_spec:
            for pre in self.populations_spec:
                degree = self._target_degree(pre, post)
                if degree <= 0.0:
                    continue
                if not self.population_counts[pre] or not self.population_counts[post]:
                    continue
                mechanisms, section, contacts = self._projection_synapse(pre, post)
                synapses[post][pre] = {
                    "type": self.populations_spec[pre].synapse_type,
                    "contacts": contacts,
                    "layers": ["2d"],
                    "sections": [section],
                    "proportions": [1.0],
                    "mechanisms": {"default": mechanisms},
                    "kernel": {
                        "kernel": self.connectivity.kernel,
                        "sigma": float(self.connectivity.sigma),
                        "mean_degree": float(degree),
                        "normalization": self.normalization,
                        "boundary": self.boundary,
                        "allow_self_connections": bool(
                            self.connectivity.allow_self_connections
                        ),
                        **(
                            {"cutoff": float(self.connectivity.cutoff)}
                            if self.connectivity.cutoff is not None
                            else {}
                        ),
                    },
                }
                if section == "soma":
                    target_sec, target_swc = 0, int(SWCType.soma)
                else:
                    target_sec, target_swc = 1, int(SWCType.apical)
                self._sites[(pre, post)] = (
                    target_sec,
                    target_swc,
                    SYN_TYPE[self.populations_spec[pre].synapse_type],
                    degree,
                )

        self.connections_config = {"synapses": synapses}
        # the order `syn_id`s are handed out in, and the only thing that makes
        # them reproducible: a post cell numbers its inputs by walking its
        # projections in this order, so it must not depend on who is asking
        self._projection_order = {
            post: [pre for pre in self.populations_spec if (pre, post) in self._sites]
            for post in self.populations_spec
        }
        self._amplitudes: dict[tuple[str, str], float] = {}

    # -- the draw ------------------------------------------------------------

    def _population_slice(self, population: str) -> slice:
        try:
            start, count = self.cells_meta_data.population_ranges[population]
        except KeyError:
            raise KeyError(
                f"{self!r} has no population {population!r}; expected one of "
                f"{list(self.population_counts)}"
            ) from None
        return slice(start, start + count)

    def _distances(self, pre_xy, post_xy):
        """Pairwise distance, on a torus when the culture is periodic."""
        d = pre_xy[:, None, :] - post_xy[None, :, :]
        period = self.period
        if period is not None:
            wrap = numpy.asarray(period, dtype=numpy.float32)
            # nearest image: a separation past half the box is shorter the
            # other way round
            d = d - wrap * numpy.round(d / wrap)
        return numpy.linalg.norm(d, axis=2).astype(numpy.float32)

    def _kernel(self, distances):
        sigma = self.connectivity.sigma
        if self.connectivity.kernel == "gaussian":
            return numpy.exp(-(distances**2) / (2.0 * sigma**2))
        return numpy.exp(-distances / sigma)

    def _amplitude(self, pre: str, post: str) -> float:
        """Global normalization constant, computed once per projection.

        Only `normalization="global"` needs it, and it is the one thing here
        that costs a pass over every pair.
        """
        key = (pre, post)
        if key in self._amplitudes:
            return self._amplitudes[key]

        degree = self._sites[key][3]
        pre_xy = self._xy[self._population_slice(pre)]
        post_xy = self._xy[self._population_slice(post)]
        n_post = len(post_xy)

        total = 0.0
        peak = 0.0
        for lo in range(0, n_post, CONNECTIVITY_CHUNK):
            hi = min(lo + CONNECTIVITY_CHUNK, n_post)
            w = self._kernel(self._distances(pre_xy, post_xy[lo:hi]))
            if pre == post and not self.connectivity.allow_self_connections:
                self._zero_diagonal(w, lo, hi)
            total += float(w.sum())
            peak = max(peak, float(w.max(initial=0.0)))

        amplitude = (degree * n_post / total) if total > 0 else 0.0
        if amplitude * peak > 1.0:
            reachable = total / (n_post * peak) if peak else 0.0
            raise ValueError(
                f"{pre}->{post} asks for a mean in-degree of {degree:g}, which "
                f"needs a connection probability of {amplitude * peak:.3g} at the "
                f"closest pair. With {len(pre_xy)} presynaptic cells at "
                f"sigma={self.connectivity.sigma:g} over this area the most this "
                f"kernel can deliver is {reachable:.1f}. Raise sigma, add "
                "presynaptic cells, shrink the area, or lower the degree"
            )
        self._amplitudes[key] = amplitude
        return amplitude

    @staticmethod
    def _zero_diagonal(w, lo: int, hi: int) -> None:
        rows = numpy.arange(lo, hi)
        inside = (rows >= 0) & (rows < w.shape[0])
        w[rows[inside], numpy.arange(hi - lo)[inside]] = 0.0

    def _edge_columns(self, pre: str, post: str, post_indices):
        """``(pre_index, distance)`` per selected edge, for each post column.

        `post_indices` are population-local indices. Each column is drawn at its
        own counter, so this returns exactly what a whole-system draw would put
        in those columns.
        """
        *_placement, degree = self._sites[(pre, post)]
        pre_xy = self._xy[self._population_slice(pre)]
        post_xy = self._xy[self._population_slice(post)]
        n_pre = len(pre_xy)
        draws = StableRandom(self.seed, "connectivity", pre, post)
        cutoff = self.connectivity.cutoff
        amplitude = (
            self._amplitude(pre, post) if self.normalization == "global" else None
        )

        out: dict[int, tuple[numpy.ndarray, numpy.ndarray]] = {}
        for lo in range(0, len(post_indices), CONNECTIVITY_CHUNK):
            block = post_indices[lo : lo + CONNECTIVITY_CHUNK]
            distances = self._distances(pre_xy, post_xy[block])
            w = self._kernel(distances)
            if pre == post and not self.connectivity.allow_self_connections:
                for column, index in enumerate(block):
                    if 0 <= index < n_pre:
                        w[index, column] = 0.0

            if amplitude is None:  # per-post
                mass = w.sum(axis=0)
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    probability = numpy.where(mass > 0, degree * w / mass, 0.0)
            else:
                probability = amplitude * w
            if cutoff is not None:
                probability = numpy.where(probability >= cutoff, probability, 0.0)

            for column, index in enumerate(block):
                u = draws.random(n_pre, at=int(index) * n_pre)
                selected = numpy.flatnonzero(u < probability[:, column])
                out[int(index)] = (selected, distances[selected, column])

        floor = int(self.connectivity.floor.get(f"{pre}->{post}", 0))
        if floor:
            self._apply_floor(out, pre, post, floor)
        return out

    def _apply_floor(self, columns, pre: str, post: str, floor: int) -> None:
        """Give a postsynaptic cell left short its likeliest partners back."""
        pre_xy = self._xy[self._population_slice(pre)]
        post_xy = self._xy[self._population_slice(post)]
        n_pre = len(pre_xy)
        for index, (selected, _) in list(columns.items()):
            if len(selected) >= floor:
                continue
            distances = self._distances(pre_xy, post_xy[index : index + 1])[:, 0]
            w = self._kernel(distances)
            if (
                pre == post
                and not self.connectivity.allow_self_connections
                and 0 <= index < n_pre
            ):
                w[index] = 0.0
            order = numpy.argsort(-w, kind="stable")
            keep = [i for i in order.tolist() if i not in set(selected.tolist())]
            need = floor - len(selected)
            if not keep:
                continue  # nothing to give back
            added = numpy.asarray(keep[:need], dtype=numpy.int64)
            merged = numpy.sort(numpy.concatenate([selected, added]))
            columns[index] = (merged, distances[merged])

    # -- the System protocol -------------------------------------------------

    def placement(self, population: types.PopulationName, gids) -> dict[int, tuple]:
        """Synapse sites on the local cells, numbered exactly as `edges` says."""
        start, count = self.cells_meta_data.population_ranges[population]
        wanted = sorted({int(g) for g in gids if start <= int(g) < start + count})
        if not wanted:
            return {}

        indices = numpy.asarray([g - start for g in wanted], dtype=numpy.int64)
        per_gid: dict[int, list[tuple[int, int]]] = {g: [] for g in wanted}
        for pre in self._projection_order[population]:
            target_sec, target_swc, _syn_type, _degree = self._sites[(pre, population)]
            columns = self._edge_columns(pre, population, indices)
            for gid, index in zip(wanted, indices.tolist(), strict=True):
                selected, _ = columns[index]
                per_gid[gid].extend([(target_swc, target_sec)] * len(selected))

        out = {}
        for gid, sites in per_gid.items():
            n = len(sites)
            out[gid] = (
                numpy.arange(n, dtype=numpy.int64),
                numpy.asarray([s[0] for s in sites], dtype=numpy.int64),
                # mid-section, never 0 or 1: those ends are zero-area nodes in
                # NEURON and a point process there has no membrane to act on
                numpy.full(n, 0.5, dtype=numpy.float64),
            )
        return out

    def edges(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        gids,
    ) -> Iterator[tuple[int, tuple[Any, Projection]]]:
        if (pre, post) not in self._sites:
            return
        pre_start = self.cells_meta_data.population_ranges[pre][0]
        post_start, post_count = self.cells_meta_data.population_ranges[post]
        wanted = sorted(
            {int(g) for g in gids if post_start <= int(g) < post_start + post_count}
        )
        if not wanted:
            return

        indices = numpy.asarray([g - post_start for g in wanted], dtype=numpy.int64)
        # a post cell numbers its synapses across all of its projections, so the
        # offset of this one depends on the projections before it
        order = self._projection_order[post]
        before = order[: order.index(pre)]
        offsets = dict.fromkeys(wanted, 0)
        for earlier in before:
            columns = self._edge_columns(earlier, post, indices)
            for gid, index in zip(wanted, indices.tolist(), strict=True):
                offsets[gid] += len(columns[index][0])

        columns = self._edge_columns(pre, post, indices)
        for gid, index in zip(wanted, indices.tolist(), strict=True):
            selected, distances = columns[index]
            if not len(selected):
                continue
            first = offsets[gid]
            yield (
                gid,
                (
                    (selected + pre_start).astype(numpy.uint32),
                    {
                        "Connections": {"distance": distances.astype(numpy.float32)},
                        "Synapses": {
                            "syn_id": numpy.arange(
                                first, first + len(selected), dtype=numpy.uint32
                            )
                        },
                    },
                ),
            )

    def projections(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
    ):
        start, count = self.cells_meta_data.population_ranges[post]
        yield from self.edges(pre, post, range(start, start + count))

    def projection_array(
        self,
        pre: types.PreSynapticPopulationName,
        post: types.PostSynapticPopulationName,
        all: bool = True,
    ) -> list:
        return [[gid, payload] for gid, payload in self.projections(pre, post)]

    def synapse_projections(self) -> list[tuple[str, str, str, str, str]]:
        found = []
        for post, sources in self.connections_config["synapses"].items():
            for pre, spec in sources.items():
                mechanisms = spec["mechanisms"]["default"]
                for section in spec["sections"]:
                    found.extend(
                        (post, pre, section, mechanism, spec["type"])
                        for mechanism in mechanisms
                    )
        return sorted(set(found))

    @property
    def weight_names(self) -> list[str]:
        namer = self._weight_section_name()
        names = []
        for post, pre, section, mechanism, _ in self.synapse_projections():
            name = f"{post}_{pre}-{namer(post, section)}-{mechanism}-weight"
            if name not in names:
                names.append(name)
        return names

    def _weight_section_name(self):
        cached = getattr(self, "_weight_section_namer", None)
        if cached is not None:
            return cached

        def identity(_population, section):
            return section

        namer = identity
        with contextlib.suppress(AttributeError, ImportError, TypeError, ValueError):
            namer = self.default_model().section_name
        self._weight_section_namer = namer
        return namer

    @property
    def populations(self) -> list[types.PopulationName]:
        return self.cells_meta_data.population_names

    @property
    def population_ranges(self) -> dict[str, tuple[int, int]]:
        return self.cells_meta_data.population_ranges

    def population_count(self, population: types.PopulationName) -> int:
        return self.cells_meta_data.population_count(population)

    @property
    def neuron_coordinates(self):
        return np.asarray(self._neuron_coordinates)

    @property
    def gids(self):
        return np.asarray(self._neuron_coordinates[:, 0], dtype=int)

    def coordinates(self, population: types.PopulationName):
        for row in self._neuron_coordinates[self._population_slice(population)]:
            yield int(row[0]), (float(row[1]), float(row[2]), float(row[3]))

    def coordinate_array(self, population: types.PopulationName):
        # numpy, not the backend's array module, as `NeuroH5System` and
        # `ParallelSystem` also return: the geometry is static, and staging it
        # through `jnp.asarray` inside a `jit` makes it a tracer that a model's
        # `stimulus_coordinates` cannot read -- it indexes gids on the host to
        # draw each cell's dendrite angle
        return self._neuron_coordinates[self._population_slice(population)].copy()

    def transform_coordinates(self, transform, populations=None):
        if populations is None:
            populations = self.populations
        return stack_coordinates(
            [transform(self.coordinate_array(p), population=p) for p in populations]
        )

    @property
    def bounding_box(self):
        x0, y0, x1, y1 = self._box
        zmin, zmax = self.z_range
        return np.asarray([[x0, y0, zmin], [x1, y1, zmax]], dtype=float)

    @property
    def center_point(self):
        box = self.bounding_box
        return (box[0] + box[1]) / 2.0

    def selection(
        self,
        spec,
        populations: Sequence[str] | None = None,
        seed: int | None = 123,
        method: str = "first",
        bounds=None,
    ):
        if isinstance(spec, str):
            raise ValueError(
                f"{self!r} stores no selections: a Monolayer is defined by its "
                f"spec, so a smaller system is a smaller spec, not a named "
                f"subset of this one. A subselection would also keep only the "
                f"edges with both ends selected, which collapses in-degree. "
                f"Build the system you want instead of cutting {spec!r} out of "
                "this one"
            )
        return resolve_selection(self, spec, populations, seed, method, bounds)

    def selections(self, comm=None) -> list[str]:
        return []

    def selection_document(self, name: str, comm=None) -> dict:
        raise FileNotFoundError(f"{self!r} stores no selections")

    def load_file(self, filepath, default: Any = sentinel, **kwargs):
        if default is sentinel:
            raise FileNotFoundError(f"{self!r} has no files ({filepath})")
        return default

    def default_io(self, comm=None) -> IO:
        from livn.io import MEA, electrode_array_coordinates_for_area

        spec = dict(self.mea_spec or {})
        coordinates = spec.pop("electrode_coordinates", None)
        if coordinates is None:
            x0, y0, x1, y1 = self._readout_box
            zmin, zmax = self.z_range
            coordinates = electrode_array_coordinates_for_area(
                pitch=float(spec.pop("pitch", 200.0)),
                area=((x0, y0), (x1, y1)),
                z=zmin + (zmax - zmin) / 2,
            )
        else:
            spec.pop("pitch", None)
        return MEA(electrode_coordinates=coordinates, **spec)

    def default_model(self, comm=None) -> Model:
        from livn.models.rcsd import ReducedCalciumSomaDendrite

        return ReducedCalciumSomaDendrite()

    def connectivity_matrix(self, weights: dict | None = None, seed=123, gids=None):
        import random as _random

        prng = _random.Random(seed)
        if weights is None:
            weights = {}

        w = numpy.zeros([self.num_neurons, self.num_neurons], dtype=numpy.float32)
        for post, sources in self.connections_config["synapses"].items():
            for pre, spec in sources.items():
                prefix = -1.0 if spec["type"] == "inhibitory" else 1.0
                weight = weights.get(f"{post}_{pre}", 1.0)
                for post_gid, (pre_gids, _) in self.projections(pre, post):
                    for pre_gid in pre_gids:
                        w[int(pre_gid), int(post_gid)] = prefix * prng.random() * weight

        if gids is None:
            return w
        index = numpy.asarray(gids, dtype=int)
        return w[numpy.ix_(index, index)]

    def summary(self) -> dict[str, int | dict[str, int]]:
        projections = 0
        for post, sources in self.connections_config["synapses"].items():
            for pre in sources:
                projections += sum(
                    len(pre_gids) for _, (pre_gids, _) in self.projections(pre, post)
                )
        return {
            "num_neurons": self.num_neurons,
            "num_projections": projections,
            "population_counts": dict(self.population_counts),
        }

    # -- export --------------------------------------------------------------

    def graph_document(self) -> dict:
        """The `graph.json` this spec renders to.

        A view for export and for anything that still reads the file format --
        the contract is this object, not the document.
        """
        x0, y0, x1, y1 = self._box
        zmin, zmax = self.z_range
        distributions = {p: {"2d": int(c)} for p, c in self.population_counts.items()}
        layer_extents = {"2D": [[x0, y0, zmin], [x1, y1, zmax]]}
        synapses = {
            post: {
                pre: {
                    **spec,
                    "kernel": {
                        **spec["kernel"],
                        "amplitude": (
                            self._amplitude(pre, post)
                            if self.normalization == "global"
                            else None
                        ),
                    },
                }
                for pre, spec in sources.items()
            }
            for post, sources in self.connections_config["synapses"].items()
        }
        return {
            "version": GRAPH_FORMAT_VERSION,
            "architecture": {
                "uuid": self.uuid,
                "config": {
                    "coordinate_namespace": "Generated Coordinates",
                    "area": [[x0, y0], [x1, y1]],
                    "area_shape": self.area,
                    "area_kwargs": dict(self.area_kwargs),
                    "boundary": self.boundary,
                    "z_range": list(self.z_range),
                    "cell_distributions": distributions,
                    "layer_extents": layer_extents,
                    "cell_counts": {
                        p: int(c) for p, c in self.population_counts.items()
                    },
                    "cells_filepath": "./cells.h5",
                },
            },
            "distances": {
                "culture2d": {
                    "uuid": self.uuid,
                    "config": {
                        "coordinate_namespace": "Generated Coordinates",
                        "cell_distributions": distributions,
                        "layer_extents": layer_extents,
                    },
                }
            },
            "synapse_forest": {},
            "connections": {
                "culture2d": {
                    "uuid": self.uuid,
                    "config": {
                        "coordinates_namespace": "Generated Coordinates",
                        "connectivity_namespace": "Connections",
                        "distances_namespace": "Connections",
                        "population_definitions": dict(self.population_definitions),
                        "layer_definitions": {"2d": 0},
                        "synapses_namespace": "Synapse Attributes",
                        "value_chunk_size": 1000,
                        "synapses": synapses,
                        "connections_filepath": "./connections.h5",
                    },
                }
            },
        }
