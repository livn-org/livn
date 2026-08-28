import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from typing import ClassVar

import h5py
import matplotlib.pyplot as plt
import numpy as np
from machinable import Interface
from machinable.config import to_dict
from machinable.utils import save_file
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from mpi4py import MPI
from neuroh5.io import (
    NeuroH5ProjectionGen,
    read_cell_attributes,
    read_population_names,
    read_population_ranges,
    write_cell_attributes,
    write_graph,
)
from pydantic import BaseModel, Field, model_validator

from livn.io import LightArray, electrode_array_coordinates_for_area
from livn.utils import import_object_by_path

_SYN_TYPE_LOOKUP = {"excitatory": 0, "inhibitory": 1}
_INHIBITORY = ("gabaergic", "glycinergic")


def _released(transmitter: str | None, synapse_type: str) -> str:
    if transmitter:
        return transmitter
    return "glutamatergic" if synapse_type == "excitatory" else "gabaergic"


def _single_compartment(soma_only: bool | None, synapse_type: str) -> bool:
    if soma_only is not None:
        return bool(soma_only)
    return synapse_type == "inhibitory"


CONNECTIVITY_CHUNK = 2048
GRAPH_FORMAT_VERSION = 1
WEIGHTS_NAMESPACE = "Weights"
REGION_NAMESPACE = "Region"


class StableRandom:
    __slots__ = ("_key", "_pos")

    def __init__(self, seed: int, *tags: str):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(int(seed)).encode())
        for tag in tags:
            digest.update(b"\x00" + tag.encode())
        self._key = np.frombuffer(digest.digest(), dtype="<u8").copy()
        self._pos = 0

    def _raw(self, n: int, at: int | None) -> np.ndarray:
        counter = self._pos if at is None else int(at)
        raw = np.random.Philox(key=self._key, counter=counter).random_raw(n)
        if at is None:
            self._pos += n
        return raw

    def random(self, n: int, at: int | None = None) -> np.ndarray:
        return (self._raw(n, at) >> np.uint64(40)).astype(np.float32) * np.float32(
            2.0**-24
        )

    def uniform(self, low: float, high: float, size: int) -> np.ndarray:
        return (
            np.float32(low) + (np.float32(high) - np.float32(low)) * self.random(size)
        ).astype(np.float32)


class SWCTypesDef(IntEnum):
    """SWC section-type codes (Cannon et al. convention, matching neuroh5)."""

    soma = 1
    axon = 2
    basal = 3
    apical = 4
    trunk = 5
    tuft = 6
    ais = 7
    hillock = 8


_grp_h5types = "H5Types"
_grp_populations = "Populations"
_grp_valid_population_projections = "Valid population projections"
_path_population_labels = f"/{_grp_h5types}/Population labels"
_path_population_range = f"/{_grp_h5types}/Population range"
_path_population_projections = f"/{_grp_h5types}/Population projections"


def _h5_get_group(h, groupname):
    if groupname in h:
        return h[groupname]
    return h.create_group(groupname)


def _h5_get_dataset(g, dsetname, **kwargs):
    if dsetname in g:
        return g[dsetname]
    return g.create_dataset(dsetname, (0,), **kwargs)


def create_neural_h5(
    output_filepath: str,
    cell_distributions: dict[str, dict[str, int]],
    synapses: dict[str, dict[str, object]],
    population_definitions: dict[str, int],
    gap_junctions: dict | None = None,
) -> None:
    """Write the NeuroH5 ``H5Types`` group (populations and projections).

    Args:
        output_filepath: Target ``.h5`` file (opened in append mode).
        cell_distributions: ``{population: {layer: count}}``.
        synapses: ``{post: {pre: ...}}`` used to enumerate projections
            (ignored when ``gap_junctions`` is provided).
        population_definitions: ``{population: enum_index}``.
        gap_junctions: Optional ``{(post, pre): ...}`` used for projections
            instead of ``synapses``.
    """
    populations = []
    for pop_name, pop_idx in population_definitions.items():
        if pop_name not in cell_distributions:
            raise ValueError(
                f"Definitions contain a population '{pop_name}' that is not "
                f"specified in the cell distribution populations "
                f"({', '.join(list(cell_distributions))})"
            )
        pop_count = sum(cell_distributions[pop_name].values())
        populations.append((pop_name, pop_idx, pop_count))
    populations.sort(key=lambda x: x[1])

    projections = []
    if gap_junctions:
        for post, pre in gap_junctions:
            projections.append(
                (population_definitions[pre], population_definitions[post])
            )
    else:
        for post, connection_dict in synapses.items():
            projections.extend(
                (population_definitions[pre], population_definitions[post])
                for pre in connection_dict
            )

    # HDF5 enumerated type for the population label
    mapping = dict(population_definitions.items())
    dt_population_labels = h5py.special_dtype(enum=(np.uint16, mapping))

    with h5py.File(output_filepath, "a") as h5:
        h5[_path_population_labels] = dt_population_labels

        dt_populations = np.dtype(
            [
                ("Start", np.uint64),
                ("Count", np.uint32),
                ("Population", h5[_path_population_labels].dtype),
            ]
        )
        h5[_path_population_range] = dt_populations
        dt = h5[_path_population_range].dtype

        g = _h5_get_group(h5, _grp_h5types)

        dset = _h5_get_dataset(
            g, _grp_populations, maxshape=(len(populations),), dtype=dt
        )
        dset.resize((len(populations),))
        a = np.zeros(len(populations), dtype=dt)
        start = 0
        for enum_id, (_name, idx, count) in enumerate(populations):
            a[enum_id]["Start"] = start
            a[enum_id]["Count"] = count
            a[enum_id]["Population"] = idx
            start += count
        dset[:] = a

        dt_projections = np.dtype(
            [
                ("Source", h5[_path_population_labels].dtype),
                ("Destination", h5[_path_population_labels].dtype),
            ]
        )
        h5[_path_population_projections] = dt_projections
        dt = h5[_path_population_projections]

        dset = _h5_get_dataset(
            g,
            _grp_valid_population_projections,
            maxshape=(len(projections),),
            dtype=dt,
        )
        dset.resize((len(projections),))
        a = np.zeros(len(projections), dtype=dt)
        for i, (src, dst) in enumerate(projections):
            a[i]["Source"] = int(src)
            a[i]["Destination"] = int(dst)
        dset[:] = a


def bounding_box(xs, ys) -> tuple:  # ((xmin, ymin), (xmax, ymax))
    return (
        (float(np.min(xs)), float(np.min(ys))),
        (float(np.max(xs)), float(np.max(ys))),
    )


def rectangle(
    count: int,
    rng: np.random.Generator,
    *,
    margin: float = 0.0,
    x_range: tuple[float, float] = (0.0, 4000.0),
    y_range: tuple[float, float] = (0.0, 4000.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = float(x_range[0]), float(x_range[1])
    ymin, ymax = float(y_range[0]), float(y_range[1])
    if xmin + margin > xmax - margin or ymin + margin > ymax - margin:
        raise ValueError(
            f"a margin of {margin:g} um leaves no interior in a "
            f"{xmax - xmin:g} x {ymax - ymin:g} um rectangle"
        )

    xs = rng.uniform(xmin, xmax, size=count).astype(np.float32)
    ys = rng.uniform(ymin, ymax, size=count).astype(np.float32)
    interior = (
        (xs >= xmin + margin)
        & (xs <= xmax - margin)
        & (ys >= ymin + margin)
        & (ys <= ymax - margin)
    )
    return xs, ys, interior


def disk(
    count: int,
    rng: np.random.Generator,
    *,
    margin: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float = 500.0,
    inner_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy = float(center[0]), float(center[1])
    r_in, r_out = float(inner_radius), float(radius)
    outer, inner = r_out - margin, (r_in + margin if r_in > 0.0 else 0.0)
    if outer < inner:
        raise ValueError(
            f"a margin of {margin:g} um leaves no interior in a disk of radius "
            f"{r_out:g} um"
        )

    u = rng.random(size=count)
    r = np.sqrt(u * (r_out**2 - r_in**2) + r_in**2)
    theta = rng.uniform(0, 2 * np.pi, size=count)
    return (
        (cx + r * np.cos(theta)).astype(np.float32),
        (cy + r * np.sin(theta)).astype(np.float32),
        (r <= outer) & (r >= inner),
    )


class PopulationConfig(BaseModel):
    """Specification for a population in the 2D culture."""

    ratio: float | None = Field(
        default=None, ge=0.0, description="proportion of the global cell count"
    )
    count: int | None = Field(default=None, ge=0)
    synapse_type: str = Field("excitatory")
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
            "Defaults true for inhibitory populations. "
        ),
    )

    @property
    def released(self) -> str:
        return _released(self.transmitter, self.synapse_type)

    @property
    def single_compartment(self) -> bool:
        return _single_compartment(self.soma_only, self.synapse_type)

    @model_validator(mode="after")
    def _validate(self) -> "PopulationConfig":
        if self.ratio is None and self.count is None:
            raise ValueError(
                "Either 'ratio' or 'count' must be provided for a population"
            )
        if self.synapse_type not in _SYN_TYPE_LOOKUP:
            raise ValueError(
                f"synapse_type must be one of {list(_SYN_TYPE_LOOKUP)}, got '{self.synapse_type}'"
            )
        expected = "inhibitory" if self.released in _INHIBITORY else "excitatory"
        if expected != self.synapse_type:
            raise ValueError(
                f"a population releasing {self.released!r} is {expected}, but "
                f"synapse_type is {self.synapse_type!r}; the transmitter sets "
                "the mechanism and synapse_type is what `syn_types` records, "
                "so the two must agree"
            )
        return self


class ConnectivityConfig(BaseModel):
    """Distance-dependent connectivity specification.

    The ``kernel`` selects the shape of the distance dependence:

    * ``"exponential"``: ``exp(-d / sigma)``. Heavy-tailed matching the long
      reach of free-growing 2D-culture axons in organoid systems.
    * ``"gaussian"``: ``exp(-d^2 / (2 sigma^2))``. Tissue-like local wiring.

    ``sigma`` is the length constant in space units (Gaussian width or, for the
    exponential kernel, the decay constant lambda). ``mean_degree`` fixes the
    expected in-degree via amplitude normalisation, so ``sigma`` controls only
    the spatial spread of each neuron's connections, not their number.
    """

    kernel: str = Field(
        default="exponential",
        description="Distance-kernel shape: 'exponential' or 'gaussian'",
    )
    sigma: float = Field(
        ..., gt=0.0, description="Length constant lambda/sigma (space units)"
    )
    mean_degree: float | dict[str, float] = Field(
        default=100.0,
        description="Target average number of incoming connections per neuron",
    )
    cutoff: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional probability threshold below which edges are discarded",
    )
    allow_self_connections: bool = False

    @model_validator(mode="after")
    def _validate_kernel(self) -> "ConnectivityConfig":
        if self.kernel not in {"exponential", "gaussian"}:
            raise ValueError(
                f"kernel must be 'exponential' or 'gaussian', got '{self.kernel}'"
            )
        return self


class SynapseRecord(BaseModel):
    syn_ids: list[int] = Field(default_factory=list)
    syn_types: list[int] = Field(default_factory=list)
    syn_cdists: list[float] = Field(default_factory=list)
    syn_secs: list[int] = Field(default_factory=list)
    swc_types: list[int] = Field(default_factory=list)

    def add(self, syn_type: int, distance: float, sec: int, swc_type: int) -> int:
        syn_id = len(self.syn_ids)
        self.syn_ids.append(syn_id)
        self.syn_types.append(syn_type)
        self.syn_cdists.append(distance)
        self.syn_secs.append(sec)
        self.swc_types.append(swc_type)
        return syn_id


class Generate2DSystem(Interface):
    class Config(BaseModel):
        area: str = Field(default="systems.generate_2d.rectangle")
        area_kwargs: dict = Field(default_factory=dict)
        margin: float = Field(
            default=0.0,
            ge=0.0,
            description=("Outer margin to avoid boundary effects"),
        )
        z_range: tuple[float, float] = Field(default=(0.0, 10.0))
        total_cells: int | None = Field(default=None, ge=1)
        populations: dict[str, PopulationConfig] = Field(
            default={
                "EXC": {"ratio": 0.8, "synapse_type": "excitatory"},
                "INH": {"ratio": 0.2, "synapse_type": "inhibitory"},
            }
        )
        connectivity: ConnectivityConfig = Field(
            default={
                "kernel": "exponential",
                "sigma": 600.0,
                "mean_degree": {"default": 0.0},
                "cutoff": None,
                "allow_self_connections": False,
            }
        )
        population_definitions: dict[str, int] = Field(default={"EXC": 10, "INH": 11})
        random_seed: int = 123
        output_directory: str | None = None

        @model_validator(mode="after")
        def _validate(self):
            if not self.populations:
                raise ValueError("At least one population must be defined")
            for pop in self.populations:
                if pop not in self.population_definitions:
                    raise ValueError(
                        f"Population '{pop}' is not declared in population_definitions"
                    )
            zmin, zmax = self.z_range
            if zmin > zmax:
                raise ValueError("z_range must satisfy zmin <= zmax")

            if self.margin > 0.0:
                import_object_by_path(self.area)(
                    0,
                    np.random.default_rng(0),
                    margin=self.margin,
                    **(self.area_kwargs or {}),
                )

            connectivity = self.connectivity
            degrees = (
                connectivity.get("mean_degree")
                if isinstance(connectivity, dict)
                else connectivity.mean_degree
            )
            if isinstance(degrees, dict):
                valid = {
                    f"{pre}->{post}"
                    for pre in self.populations
                    for post in self.populations
                }
                unknown = sorted(set(degrees) - valid - {"default"})
                if unknown:
                    raise ValueError(
                        f"mean_degree names {unknown}, which are not projections "
                        f"of this system. Expected 'default' or one of "
                        f"{sorted(valid)}"
                    )
            return self

    @property
    def cells_filepath(self) -> str:
        if self.config.output_directory is not None:
            return os.path.join(self.config.output_directory, "cells.h5")
        return self.local_directory("cells.h5")

    @property
    def connections_filepath(self) -> str:
        if self.config.output_directory is not None:
            return os.path.join(self.config.output_directory, "connections.h5")
        return self.local_directory("connections.h5")

    @property
    def graph_filepath(self) -> str:
        if self.config.output_directory is not None:
            return os.path.join(self.config.output_directory, "graph.json")
        return self.local_directory("graph.json")

    def mea(
        self,
        pitch: float = 50,
        overwrite: bool = True,
        coordinates: list | None = None,
        input_radius: float = 50,
        output_radius: float = 80,
    ):
        fn = os.path.join(self.config.output_directory, "mea.json")
        if not overwrite and os.path.isfile(fn):
            raise FileExistsError("mea.json already exists")
        z_min, z_max = self.config.z_range
        z = z_min + (z_max - z_min) / 2

        if coordinates is not None:
            coords = np.asarray(
                [[c[0], c[1], c[2], c[3] if len(c) > 3 else z] for c in coordinates],
                dtype=float,
            )
        else:
            with open(self.graph_filepath) as f:
                graph = json.load(f)
            area = graph["architecture"]["config"]["area"]
            bounds = (tuple(area[0]), tuple(area[1]))
            coords = electrode_array_coordinates_for_area(pitch=pitch, area=bounds, z=z)

        data = {
            "electrode_coordinates": coords.tolist(),
            "input_radius": input_radius,
            "output_radius": output_radius,
        }

        with open(fn, "w") as f:
            json.dump(data, f)

        return data

    def lightarray(
        self,
        fiber_coordinates: list[list[float]] | None = None,
        fiber_height_um: float = 100.0,
        wavelength_nm: float = 473.0,
        overwrite: bool = False,
    ):
        """Write a ``lightarray.json`` for use with ``LightArray.from_directory``.

        Args:
            fiber_coordinates: Explicit ``[[id, x, y, z], ...]`` fibre positions.
                If *None*, a single fibre is placed at the centre of the culture
                area at ``-fiber_height_um`` above the surface.
        """
        fn = os.path.join(self.config.output_directory, "lightarray.json")
        if not overwrite and os.path.isfile(fn):
            raise FileExistsError("lightarray.json already exists.")

        if fiber_coordinates is None:
            with open(self.graph_filepath) as f:
                graph = json.load(f)
            area = graph["architecture"]["config"]["area"]
            bounds = (tuple(area[0]), tuple(area[1]))
            (xmin, ymin), (xmax, ymax) = bounds
            fiber_coordinates = [
                [0, (xmin + xmax) / 2, (ymin + ymax) / 2, -fiber_height_um]
            ]

        la = LightArray(
            fiber_coordinates=np.array(fiber_coordinates, dtype=np.float32),
            wavelength_nm=wavelength_nm,
        )

        with open(fn, "w") as f:
            f.write(la.as_json(indent=2))

        return la

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

    def _projection_synapse(self, pre: str, post: str):
        pre_cfg = self.config.populations[pre]
        post_cfg = self.config.populations[post]

        released = _released(pre_cfg.transmitter, pre_cfg.synapse_type)
        soma_only = _single_compartment(post_cfg.soma_only, post_cfg.synapse_type)
        receives_on = "soma" if (soma_only or released in _INHIBITORY) else "dend"
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
        return mechanisms, [receives_on], contacts

    def __call__(self):
        if self.config.output_directory:
            os.makedirs(self.config.output_directory, exist_ok=True)
        for path in (self.cells_filepath, self.connections_filepath):
            if os.path.isfile(path):
                os.remove(path)
        counts: dict[str, int] = {}
        ratios: dict[str, float] = {}
        syn_types = {}
        total_from_counts = 0
        for pop, spec in self.config.populations.items():
            if spec.count is not None:
                counts[pop] = spec.count
                total_from_counts += spec.count
            elif spec.ratio is not None:
                ratios[pop] = spec.ratio
            syn_types[pop] = _SYN_TYPE_LOOKUP[spec.synapse_type]

        total_cells = self.config.total_cells or total_from_counts

        # normalise ratios if needed
        if ratios:
            ratio_sum = sum(ratios.values())
            if ratio_sum <= 0:
                raise ValueError("Population ratios must sum to a positive value")
            residual = total_cells - total_from_counts
            if residual < 0:
                raise ValueError(
                    "Sum of explicit population counts exceeds total_cells"
                )
            remainders: list[tuple[str, float]] = []
            allocated = 0
            for pop, ratio in ratios.items():
                proportional = residual * ratio / ratio_sum
                count = math.floor(proportional)
                counts[pop] = count
                allocated += count
                remainders.append((pop, proportional - count))

            remainder_cells = residual - allocated
            if remainder_cells > 0:
                remainders.sort(key=lambda item: item[1], reverse=True)
                for pop, _ in remainders[:remainder_cells]:
                    counts[pop] += 1

        missing = set(self.config.populations) - set(counts)
        if missing:
            raise RuntimeError(
                f"Missing counts for populations: {', '.join(sorted(missing))}"
            )

        populations = list(self.config.populations.keys())
        cell_distributions = {pop: {"2d": counts[pop]} for pop in populations}
        synapse_flags: dict[str, dict[str, bool]] = {post: {} for post in populations}
        target_degrees = {}
        mean_degree_cfg = self.config.connectivity.mean_degree
        is_mapping = not isinstance(mean_degree_cfg, (int, float))
        for post in populations:
            for pre in populations:
                key = f"{pre}->{post}"
                degree = 0.0

                if is_mapping:
                    if key in mean_degree_cfg:
                        degree = float(mean_degree_cfg[key])
                    elif "default" in mean_degree_cfg:
                        degree = float(mean_degree_cfg["default"])
                else:
                    degree = float(mean_degree_cfg)

                if degree > 0.0:
                    synapse_flags[post][pre] = True
                target_degrees[(pre, post)] = degree

        create_neural_h5(
            self.cells_filepath,
            cell_distributions,
            synapse_flags,
            self.config.population_definitions,
        )
        create_neural_h5(
            self.connections_filepath,
            cell_distributions,
            synapse_flags,
            self.config.population_definitions,
        )

        population_ranges = read_population_ranges(str(self.cells_filepath))[0]
        seed = self.config.random_seed

        # generate coordinates
        area_fn = import_object_by_path(self.config.area)
        zmin, zmax = self.config.z_range

        coords = {}
        all_xs = []
        all_ys = []

        for pop in populations:
            start, count = population_ranges[pop]
            gids = np.arange(start, start + count, dtype=np.uint32)
            rng = StableRandom(seed, "coordinates", pop)
            xs, ys, interior = area_fn(
                count, rng, margin=self.config.margin, **self.config.area_kwargs
            )
            all_xs.append(xs)
            all_ys.append(ys)
            if zmax > zmin:
                zs = rng.uniform(zmin, zmax, size=count).astype(np.float32)
            else:
                zs = np.full(count, zmin, dtype=np.float32)

            coord_dict = {
                int(gid): {
                    "X Coordinate": np.asarray([xs[i]], dtype=np.float32),
                    "Y Coordinate": np.asarray([ys[i]], dtype=np.float32),
                    "Z Coordinate": np.asarray([zs[i]], dtype=np.float32),
                    "U Coordinate": np.asarray([xs[i]], dtype=np.float32),
                    "V Coordinate": np.asarray([ys[i]], dtype=np.float32),
                    "L Coordinate": np.asarray([zs[i]], dtype=np.float32),
                }
                for i, gid in enumerate(gids)
            }

            write_cell_attributes(
                self.cells_filepath,
                pop,
                coord_dict,
                namespace="Generated Coordinates",
                comm=MPI.COMM_WORLD,
            )

            if self.config.margin > 0.0:
                write_cell_attributes(
                    self.cells_filepath,
                    pop,
                    {
                        int(gid): {
                            "interior": np.asarray([interior[i]], dtype=np.uint8)
                        }
                        for i, gid in enumerate(gids)
                    },
                    namespace=REGION_NAMESPACE,
                    comm=MPI.COMM_WORLD,
                )

            coords[pop] = {
                "gids": gids,
                "xy": np.column_stack((xs, ys)),
            }

        area_bounds = bounding_box(np.concatenate(all_xs), np.concatenate(all_ys))
        (xmin, ymin), (xmax, ymax) = area_bounds
        layer_extents = {
            "2D": [
                [float(xmin), float(ymin), float(zmin)],
                [float(xmax), float(ymax), float(zmax)],
            ]
        }

        # generate synapses
        synapses: dict[str, dict[int, SynapseRecord]] = {}
        for pop in populations:
            start, count = population_ranges[pop]
            synapses[pop] = {
                int(gid): SynapseRecord() for gid in range(start, start + count)
            }

        synapse_config = {}
        for post in populations:
            synapse_config[post] = {}
            for pre in populations:
                target_degree = target_degrees[(pre, post)]
                if target_degree <= 0.0:
                    continue

                pre_info = coords[pre]
                post_info = coords[post]
                pre_gids = pre_info["gids"]
                post_gids = post_info["gids"]

                if pre_gids.size == 0 or post_gids.size == 0:
                    continue

                length_constant = self.config.connectivity.sigma
                allow_self = self.config.connectivity.allow_self_connections

                def _kernel(
                    lo: int,
                    hi: int,
                    pre_info=pre_info,
                    post_info=post_info,
                    length_constant=length_constant,
                    allow_self=allow_self,
                    pre=pre,
                    post=post,
                ):
                    diffs = pre_info["xy"][:, None, :] - post_info["xy"][None, lo:hi, :]
                    d = np.linalg.norm(diffs, axis=2).astype(np.float32)
                    if self.config.connectivity.kernel == "gaussian":
                        w = np.exp(-(d**2) / (2.0 * length_constant**2))
                    else:  # exponential
                        w = np.exp(-d / length_constant)
                    if not allow_self and pre == post:
                        # the diagonal of this chunk, in pre-index coordinates
                        rows = np.arange(lo, hi)
                        inside = (rows >= 0) & (rows < w.shape[0])
                        w[rows[inside], np.arange(hi - lo)[inside]] = 0.0
                    return d, w

                n_post = len(post_gids)
                chunk = max(1, min(n_post, CONNECTIVITY_CHUNK))
                weight_sum = 0.0
                weight_max = 0.0
                for lo in range(0, n_post, chunk):
                    block = _kernel(lo, min(lo + chunk, n_post))[1]
                    weight_sum += float(block.sum())
                    weight_max = max(weight_max, float(block.max(initial=0.0)))

                if weight_sum > 0:
                    amp = (target_degree * len(post_gids)) / weight_sum
                else:
                    amp = 0.0

                if amp * weight_max > 1.0:
                    reachable = weight_sum / (len(post_gids) * weight_max)
                    raise ValueError(
                        f"{pre}->{post} asks for a mean in-degree of "
                        f"{target_degree:g}, which needs a connection "
                        f"probability of {amp * weight_max:.3g} at the closest "
                        f"pair. With {len(pre_gids)} presynaptic cells at "
                        f"sigma={self.config.connectivity.sigma:g} over this area "
                        f"the most this kernel can deliver is {reachable:.1f}. "
                        "Raise sigma, add presynaptic cells, shrink the area, or "
                        "lower the degree"
                    )

                kernel = {
                    "amplitude": float(amp),
                    "kernel": str(self.config.connectivity.kernel),
                    "sigma": float(self.config.connectivity.sigma),
                    "allow_self_connections": bool(
                        self.config.connectivity.allow_self_connections
                    ),
                }
                if self.config.connectivity.cutoff is not None:
                    kernel["cutoff"] = float(self.config.connectivity.cutoff)

                pre_type = self.config.populations[pre].synapse_type
                mechanisms, target_sections, contacts = self._projection_synapse(
                    pre, post
                )

                synapse_config[post][pre] = {
                    "type": pre_type,
                    "contacts": contacts,
                    "layers": ["2d"],
                    "sections": target_sections,
                    "proportions": [1.0],
                    "mechanisms": {"default": mechanisms},
                    "kernel": kernel,
                }

                if target_sections[0] == "soma":
                    target_sec = 0
                    target_swc = int(np.uint8(SWCTypesDef.soma))
                else:
                    target_sec = 1
                    target_swc = int(np.uint8(SWCTypesDef.apical))

                pair_edges: dict[
                    int, tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]
                ] = {}
                syn_type_index = syn_types[pre]
                total_edges = 0
                best = (0.0, -1, -1)  # (probability, pre index, post index)

                draws = StableRandom(seed, "connectivity", pre, post)

                for lo in range(0, n_post, chunk):
                    hi = min(lo + chunk, n_post)
                    distances, raw_weights = _kernel(lo, hi)
                    raw_probs = amp * raw_weights
                    probs = raw_probs
                    if self.config.connectivity.cutoff is not None:
                        probs = np.where(
                            probs >= self.config.connectivity.cutoff, probs, 0.0
                        )
                    block = np.empty(probs.shape, dtype=np.float32)
                    n_pre_rows = probs.shape[0]
                    for offset in range(hi - lo):
                        block[:, offset] = draws.random(
                            n_pre_rows, at=(lo + offset) * n_pre_rows
                        )
                    mask = block < probs

                    if raw_probs.size:
                        flat = int(np.argmax(raw_probs))
                        value = float(raw_probs.reshape(-1)[flat])
                        if value > best[0]:
                            r, c = divmod(flat, raw_probs.shape[1])
                            best = (value, r, lo + c)

                    for offset, post_gid in enumerate(post_gids[lo:hi]):
                        post_idx = offset
                        selected = np.where(mask[:, post_idx])[0]
                        if selected.size == 0:
                            pair_edges[int(post_gid)] = (
                                np.zeros(0, dtype=np.uint32),
                                {
                                    "Connections": {
                                        "distance": np.zeros(0, dtype=np.float32)
                                    },
                                    "Synapses": {
                                        "syn_id": np.zeros(0, dtype=np.uint32)
                                    },
                                },
                            )
                            continue

                        selected_pre_gids = pre_gids[selected].astype(np.uint32)
                        selected_distances = distances[selected, post_idx].astype(
                            np.float32
                        )
                        record = synapses[post][int(post_gid)]
                        syn_ids = [
                            record.add(
                                syn_type_index, float(dist), target_sec, target_swc
                            )
                            for dist in selected_distances
                        ]

                        pair_edges[int(post_gid)] = (
                            selected_pre_gids,
                            {
                                "Connections": {"distance": selected_distances},
                                "Synapses": {
                                    "syn_id": np.asarray(syn_ids, dtype=np.uint32)
                                },
                            },
                        )
                        total_edges += selected_pre_gids.size

                if total_edges == 0 and best[1] >= 0:
                    pre_idx, post_idx = best[1], best[2]
                    post_gid = int(post_gids[post_idx])
                    d, _ = _kernel(post_idx, post_idx + 1)
                    dist = float(d[pre_idx, 0])
                    syn_id = synapses[post][post_gid].add(
                        syn_type_index, dist, target_sec, target_swc
                    )
                    pair_edges[post_gid] = (
                        np.asarray([pre_gids[pre_idx]], dtype=np.uint32),
                        {
                            "Connections": {
                                "distance": np.asarray([dist], dtype=np.float32)
                            },
                            "Synapses": {
                                "syn_id": np.asarray([syn_id], dtype=np.uint32)
                            },
                        },
                    )
                    total_edges = 1

                if total_edges == 0:
                    continue
                write_graph(
                    self.connections_filepath,
                    src_pop_name=pre,
                    dst_pop_name=post,
                    edges=pair_edges,
                    comm=MPI.COMM_WORLD,
                )

        # write synapse attributes
        layer_index = 0
        for pop in populations:
            cell_dict = {}
            for gid, record in synapses[pop].items():
                if record.syn_ids:
                    syn_ids = np.asarray(record.syn_ids, dtype=np.uint32)
                    syn_types = np.asarray(record.syn_types, dtype=np.uint8)
                    syn_cdists = np.asarray(record.syn_cdists, dtype=np.float32)
                    # Mid-section, NOT 0.0. In NEURON the 0 and 1 ends of a
                    # section are zero-area nodes for which a point process
                    # has no membrane area to absorb its current
                    syn_locs = np.full_like(syn_ids, 0.5, dtype=np.float32)
                    syn_secs = np.asarray(record.syn_secs, dtype=np.int16)
                    syn_layers = np.full(syn_ids.shape, layer_index, dtype=np.uint8)
                    swc_types = np.asarray(record.swc_types, dtype=np.uint8)
                else:
                    syn_ids = np.zeros(0, dtype=np.uint32)
                    syn_types = np.zeros(0, dtype=np.uint8)
                    syn_cdists = np.zeros(0, dtype=np.float32)
                    syn_locs = np.zeros(0, dtype=np.float32)
                    syn_secs = np.zeros(0, dtype=np.int16)
                    syn_layers = np.zeros(0, dtype=np.uint8)
                    swc_types = np.zeros(0, dtype=np.uint8)

                cell_dict[gid] = {
                    "syn_ids": syn_ids,
                    "syn_types": syn_types,
                    "syn_cdists": syn_cdists,
                    "syn_locs": syn_locs,
                    "syn_secs": syn_secs,
                    "syn_layers": syn_layers,
                    "swc_types": swc_types,
                }
            write_cell_attributes(
                self.cells_filepath,
                pop,
                cell_dict,
                namespace="Synapse Attributes",
                comm=MPI.COMM_WORLD,
            )

        save_file(
            self.graph_filepath,
            {
                "version": GRAPH_FORMAT_VERSION,
                "architecture": {
                    "uuid": str(uuid.uuid4()),
                    "config": {
                        "coordinate_namespace": "Generated Coordinates",
                        "area": list(area_bounds),
                        "area_shape": str(self.config.area),
                        "area_kwargs": dict(self.config.area_kwargs),
                        "margin": float(self.config.margin),
                        "z_range": self.config.z_range,
                        "cell_distributions": {
                            pop: dict(distribution)
                            for pop, distribution in cell_distributions.items()
                        },
                        "layer_extents": layer_extents,
                        "cell_counts": {
                            pop: int(count) for pop, count in counts.items()
                        },
                        "cells_filepath": "./cells.h5",
                    },
                },
                "distances": {
                    "culture2d": {
                        "uuid": str(uuid.uuid4()),
                        "config": {
                            "coordinate_namespace": "Generated Coordinates",
                            "cell_distributions": {
                                pop: dict(distribution)
                                for pop, distribution in cell_distributions.items()
                            },
                            "layer_extents": layer_extents,
                        },
                    }
                },
                "synapse_forest": {},
                "connections": {
                    "culture2d": {
                        "uuid": str(uuid.uuid4()),
                        "config": {
                            "coordinates_namespace": "Generated Coordinates",
                            "connectivity_namespace": "Connections",
                            "distances_namespace": "Connections",
                            "population_definitions": dict(
                                self.config.population_definitions
                            ),
                            "layer_definitions": {"2d": 0},
                            "synapses_namespace": "Synapse Attributes",
                            "value_chunk_size": 1000,
                            "synapses": synapse_config,
                            "connections_filepath": "./connections.h5",
                        },
                    }
                },
            },
        )

        self._write_provenance()

    def _write_provenance(self):
        if not self.config.output_directory:
            return

        from livn import __version__ as livn_version

        document = {
            "generator": f"{type(self).__module__}.{type(self).__name__}",
            "version": GRAPH_FORMAT_VERSION,
            "livn": livn_version,
            "numpy": np.__version__,
            "connectivity_chunk": CONNECTIVITY_CHUNK,
            "config": to_dict(self.config),
            "assumptions": {
                "EXC->EXC": (
                    "mean_degree 20 is an assumption, not literature. Moore, "
                    "Bhumbra, Foster & Beato (2015) constrains EXC->INH (4) and "
                    "INH->EXC (40) but reports nothing on motoneuron-to-"
                    "motoneuron connectivity. It is kept because the "
                    "excitatory-only recordings being fitted are correlated and "
                    "bursting, which requires some excitatory coupling."
                ),
                "EXC->EXC synapse parameters": (
                    "the cholinergic template (7 release sites, tau_decay 7.0 "
                    "ms) is the MN->RC measurement, applied here to MN->MN."
                ),
            },
        }
        with open(
            os.path.join(self.config.output_directory, "provenance.json"), "w"
        ) as f:
            json.dump(document, f, indent=2, sort_keys=True, default=str)

    def plot(
        self,
        sample: int | None = None,
        max_edges: int | None = 2500,
        edge_alpha: float = 0.08,
        edge_width: float = 0.3,
        figsize: tuple[float, float] = (7.0, 7.0),
        mea: bool = True,
        filename: str | Path = "system.png",
        show: bool = False,
    ) -> str:
        cells_path = Path(self.cells_filepath)
        connections_path = Path(self.connections_filepath)

        populations = sorted(str(pop) for pop in read_population_names(str(cells_path)))
        sample_limit = None if sample is None or sample <= 0 else sample
        max_edges_limit = None if max_edges is None or max_edges <= 0 else max_edges

        scatter_coords = {}
        population_sizes = {}
        gid_to_xy = {}

        for population in populations:
            xs_plot: list[float] = []
            ys_plot: list[float] = []
            count = 0
            for gid, attrs in read_cell_attributes(
                str(cells_path), population, namespace="Generated Coordinates"
            ):
                gid_int = int(gid)
                x_coord = float(attrs["X Coordinate"][0])
                y_coord = float(attrs["Y Coordinate"][0])
                gid_to_xy[gid_int] = (x_coord, y_coord)
                if sample_limit is None or count < sample_limit:
                    xs_plot.append(x_coord)
                    ys_plot.append(y_coord)
                count += 1

            scatter_coords[population] = (xs_plot, ys_plot)
            population_sizes[population] = count

        segments_by_source = defaultdict(list)
        drawn_edges = 0
        total_edges = 0

        if connections_path.exists():
            for source_population in populations:
                for target_population in populations:
                    try:
                        generator = NeuroH5ProjectionGen(
                            str(connections_path),
                            source_population,
                            target_population,
                            cache_size=32,
                        )
                    except (KeyError, RuntimeError):
                        continue

                    try:
                        for target_gid, payload in generator:
                            if payload is None:
                                continue
                            source_gids, _ = payload
                            target_xy = gid_to_xy.get(int(target_gid))
                            if target_xy is None:
                                continue
                            for source_gid in source_gids:
                                source_xy = gid_to_xy.get(int(source_gid))
                                if source_xy is None:
                                    continue
                                total_edges += 1
                                if not bool(
                                    max_edges_limit is None
                                    or drawn_edges < max_edges_limit
                                ):
                                    continue
                                segments_by_source[source_population].append(
                                    (source_xy, target_xy)
                                )
                                drawn_edges += 1
                    finally:
                        close_fn = getattr(generator, "close", None)
                        if callable(close_fn):
                            close_fn()
        else:
            connections_path = None

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("tab10")
        population_colors = {
            pop: cmap(idx % cmap.N) for idx, pop in enumerate(populations)
        }

        for source_population, segments in segments_by_source.items():
            if not segments:
                continue
            color = population_colors.get(source_population, cmap(0))
            edge_color = to_rgba(color, edge_alpha)
            collection = LineCollection(
                segments,
                colors=[edge_color],
                linewidths=edge_width,
                zorder=1,
            )
            ax.add_collection(collection)

        for population in populations:
            color = population_colors[population]
            xs, ys = scatter_coords.get(population, ([], []))
            total_cells = population_sizes.get(population, len(xs))
            label = f"{population} ({total_cells} neurons)"
            if sample_limit is not None and len(xs) < total_cells:
                label = f"{population} ({total_cells} neurons, showing {len(xs)})"
            if xs and ys:
                ax.scatter(
                    xs,
                    ys,
                    s=5,
                    alpha=0.7,
                    label=label,
                    color=color,
                    zorder=2,
                )

        ax.set_xlabel("X coordinate (um)")
        ax.set_ylabel("Y coordinate (um)")
        subtitle = ""
        if total_edges:
            if max_edges_limit is not None and drawn_edges < total_edges:
                subtitle = f" ({total_edges} connections, showing {drawn_edges})"
            else:
                subtitle = f" ({total_edges} connections)"
        ax.set_title(
            f"{Path(self.config.output_directory or cells_path.parent).name}{subtitle}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(False)

        if mea:
            mea_path = cells_path.parent / "mea.json"
            if mea_path.exists():
                with open(mea_path) as f:
                    mea = json.load(f)
                coords = np.asarray(mea.get("electrode_coordinates", []), dtype=float)
                print(len(coords), " channels")
                if coords.size:
                    ax.scatter(
                        coords[:, 1],
                        coords[:, 2],
                        s=10,
                        c="green",
                        edgecolors="black",
                        linewidths=0.4,
                        zorder=3,
                    )
                    for electrode_id, x, y, _ in coords:
                        if electrode_id % (len(coords) // 20) > 0:
                            continue

                        ax.annotate(
                            str(int(electrode_id)),
                            (x, y),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=6,
                        )
                        ax.add_patch(
                            plt.Circle(
                                (x, y),
                                mea.get("output_radius", 100),
                                fill=False,
                                edgecolor="black",
                                linewidth=0.4,
                                zorder=3,
                            )
                        )

        output_dir = cells_path.parent
        save_path = output_dir / Path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return str(save_path)
