import math
import os
import uuid
import json

from machinable import Component
from machinable.utils import save_file
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional
import numpy as np
from miv_simulator.utils.io import create_neural_h5
from miv_simulator.config import SWCTypesDef
from neuroh5.io import read_population_ranges, write_cell_attributes, write_graph
from mpi4py import MPI
from livn.io import electrode_array_coordinates_for_area


_SYN_TYPE_LOOKUP = {"excitatory": 0, "inhibitory": 1}


class PopulationConfig(BaseModel):
    """Specification for a population in the 2D culture."""

    ratio: Optional[float] = Field(
        default=None, ge=0.0, description="proportion of the global cell count"
    )
    count: Optional[int] = Field(default=None, ge=0)
    synapse_type: str = Field("excitatory")

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
        return self


class ConnectivityConfig(BaseModel):
    """Gaussian distance-dependent connectivity specification"""

    sigma: float = Field(
        ..., gt=0.0, description="Gaussian width parameter (space units)"
    )
    amplitude: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Map of connection probability amplitudes. Keys can be 'default' or "
            "population-specific pairs in the form 'PRE->POST'"
        ),
    )
    cutoff: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional probability threshold below which edges are discarded",
    )
    allow_self_connections: bool = False


class SynapseRecord(BaseModel):
    syn_ids: list[int] = Field(default_factory=list)
    syn_types: list[int] = Field(default_factory=list)
    syn_cdists: list[float] = Field(default_factory=list)

    def add(self, syn_type: int, distance: float) -> int:
        syn_id = len(self.syn_ids)
        self.syn_ids.append(syn_id)
        self.syn_types.append(syn_type)
        self.syn_cdists.append(distance)
        return syn_id


class Generate2DSystem(Component):
    class Config(BaseModel):
        area: tuple[tuple[float, float], tuple[float, float]] = Field(
            default=((0.0, 0.0), (4000.0, 4000.0))
        )
        z_range: tuple[float, float] = Field(default=(0.0, 10.0))
        total_cells: Optional[int] = Field(default=None, ge=1)
        populations: Dict[str, PopulationConfig] = Field(
            default={"EXC": {"ratio": 1.0, "synapse_type": "excitatory"}}
            # default={
            #     "EXC": {"ratio": 0.8, "synapse_type": "excitatory"},
            #     "INH": {"ratio": 0.2, "synapse_type": "inhibitory"},
            # }
        )
        connectivity: ConnectivityConfig = Field(
            default={
                "sigma": 100,
                "amplitude": {"default": 0.8, "INH->EXC": 0.9},
                "cutoff": 0.1,
                "allow_self_connections": False,
            }
        )
        population_definitions: Dict[str, int] = Field(
            default={"EXC": 10}
        )  # , "INH": 11})
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
            (xmin, ymin), (xmax, ymax) = self.area
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("Culture area must define a valid rectangle")
            zmin, zmax = self.z_range
            if zmin > zmax:
                raise ValueError("z_range must satisfy zmin <= zmax")
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

    def mea(self, pitch: float = 500, overwrite: bool = False):
        fn = os.path.join(self.config.output_directory, "mea.json")
        if not overwrite and os.path.isfile(fn):
            raise FileExistsError("mea.json already exists.")
        z_min, z_max = self.config.z_range
        coords = electrode_array_coordinates_for_area(
            pitch=pitch, area=self.config.area, z=z_min + (z_max - z_min) / 2
        )

        data = {
            "electrode_coordinates": coords.tolist(),
            "input_radius": 200,
            "output_radius": 100,
        }

        with open(fn, "w") as f:
            json.dump(data, f)

        return data

    def __call__(self):
        counts: Dict[str, int] = {}
        ratios: Dict[str, float] = {}
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
                count = int(math.floor(proportional))
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
        synapse_flags: Dict[str, Dict[str, bool]] = {post: {} for post in populations}
        amplitudes = {}
        for post in populations:
            for pre in populations:
                key = f"{pre}->{post}"
                amplitude = 0.0
                if key in self.config.connectivity.amplitude:
                    amplitude = max(
                        0.0, min(1.0, self.config.connectivity.amplitude[key])
                    )
                elif "default" in self.config.connectivity.amplitude:
                    amplitude = max(
                        0.0, min(1.0, self.config.connectivity.amplitude["default"])
                    )
                synapse_flags[post][pre] = amplitude > 0.0
                amplitudes[(pre, post)] = amplitude

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
        rng = np.random.default_rng(self.config.random_seed)

        # generate coordinates
        (xmin, ymin), (xmax, ymax) = self.config.area
        zmin, zmax = self.config.z_range
        layer_extents = {
            "2D": [
                [float(xmin), float(ymin), float(zmin)],
                [float(xmax), float(ymax), float(zmax)],
            ]
        }

        coords: dict[str, dict[str, np.ndarray]] = {}

        for pop in populations:
            start, count = population_ranges[pop]
            gids = np.arange(start, start + count, dtype=np.uint32)
            xs = rng.uniform(xmin, xmax, size=count).astype(np.float32)
            ys = rng.uniform(ymin, ymax, size=count).astype(np.float32)
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

            coords[pop] = {
                "gids": gids,
                "xy": np.column_stack((xs, ys)),
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
                amp = amplitudes[(pre, post)]
                if amp <= 0.0:
                    continue

                kernel = {
                    "amplitude": float(amp),
                    "sigma": float(self.config.connectivity.sigma),
                    "allow_self_connections": bool(
                        self.config.connectivity.allow_self_connections
                    ),
                }
                if self.config.connectivity.cutoff is not None:
                    kernel["cutoff"] = float(self.config.connectivity.cutoff)

                syn_type = self.config.populations[pre].synapse_type
                target_sections = ["soma"]
                if syn_type == "excitatory":
                    target_sections = ["dend"]
                    mechanisms = {
                        "AMPA": {
                            "e": 0,
                            "g_unit": 0.0005,
                            "tau_decay": 3.0,
                            "tau_rise": 0.5,
                            "weight": 1.0,
                        }
                    }
                    if pre == post:
                        mechanisms.update(
                            {
                                "NMDA": {
                                    "e": 0,
                                    "g_unit": 0.0005,
                                    "tau_decay": 80.0,
                                    "tau_rise": 0.5,
                                    "weight": 1.0,
                                }
                            }
                        )
                elif syn_type == "inhibitory":
                    mechanisms = {
                        "GABA_A": {
                            "e": -60,
                            "g_unit": 0.001,
                            "tau_decay": 6.0,
                            "tau_rise": 0.3,
                            "weight": 1.0,
                        }
                    }

                synapse_config[post][pre] = {
                    "type": self.config.populations[pre].synapse_type,
                    "contacts": 1,
                    "layers": ["2d"],
                    "sections": target_sections,
                    "proportions": [1.0],
                    "mechanisms": {"default": mechanisms},
                    "kernel": kernel,
                }

                pre_info = coords[pre]
                post_info = coords[post]
                pre_gids = pre_info["gids"]
                post_gids = post_info["gids"]

                if pre_gids.size == 0 or post_gids.size == 0:
                    continue

                diffs = pre_info["xy"][:, None, :] - post_info["xy"][None, :, :]
                distances = np.linalg.norm(diffs, axis=2).astype(np.float32)

                sigma_sq = self.config.connectivity.sigma**2
                raw_probs = amp * np.exp(-(distances**2) / (2.0 * sigma_sq))
                probs = raw_probs.copy()

                if self.config.connectivity.cutoff is not None:
                    probs = np.where(
                        probs >= self.config.connectivity.cutoff, probs, 0.0
                    )

                mask = rng.random(size=probs.shape, dtype=np.float32) < probs

                if not self.config.connectivity.allow_self_connections and pre == post:
                    diag = np.eye(mask.shape[0], mask.shape[1], dtype=bool)
                    mask = np.logical_and(mask, ~diag)

                if not mask.any():
                    fallback_mask = np.zeros_like(mask, dtype=bool)
                    fallback_probs = raw_probs.copy()
                    if (
                        not self.config.connectivity.allow_self_connections
                        and pre == post
                    ):
                        diag_indices = np.diag_indices(fallback_probs.shape[0])
                        fallback_probs[diag_indices] = -np.inf
                    flat_index = int(np.argmax(fallback_probs))
                    fallback_value = fallback_probs.reshape(-1)[flat_index]
                    if np.isfinite(fallback_value) and fallback_value > 0.0:
                        fallback_mask.reshape(-1)[flat_index] = True
                        mask = fallback_mask

                pair_edges: dict[
                    int, tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]
                ] = {}
                syn_type_index = syn_types[pre]
                total_edges = 0

                for post_idx, post_gid in enumerate(post_gids):
                    selected = np.where(mask[:, post_idx])[0]
                    if selected.size == 0:
                        pair_edges[int(post_gid)] = (
                            np.zeros(0, dtype=np.uint32),
                            {
                                "Connections": {
                                    "distance": np.zeros(0, dtype=np.float32)
                                },
                                "Synapses": {"syn_id": np.zeros(0, dtype=np.uint32)},
                            },
                        )
                        continue

                    selected_pre_gids = pre_gids[selected].astype(np.uint32)
                    selected_distances = distances[selected, post_idx].astype(
                        np.float32
                    )
                    syn_ids = []
                    record = synapses[post][int(post_gid)]
                    for dist in selected_distances:
                        syn_ids.append(record.add(syn_type_index, float(dist)))

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
        swc_type_value = np.uint8(SWCTypesDef.soma)
        for pop in populations:
            cell_dict = {}
            for gid, record in synapses[pop].items():
                if record.syn_ids:
                    syn_ids = np.asarray(record.syn_ids, dtype=np.uint32)
                    syn_types = np.asarray(record.syn_types, dtype=np.uint8)
                    syn_cdists = np.asarray(record.syn_cdists, dtype=np.float32)
                    syn_locs = np.zeros_like(syn_ids, dtype=np.float32)
                    syn_secs = np.zeros_like(syn_ids, dtype=np.int16)
                    syn_layers = np.full(syn_ids.shape, layer_index, dtype=np.uint8)
                    swc_types = np.full(syn_ids.shape, swc_type_value, dtype=np.uint8)

                    # map excitatory synapses (type 0) to dendrite (section 1, swc type 4)
                    is_exc = syn_types == 0
                    syn_secs[is_exc] = 1
                    swc_types[is_exc] = np.uint8(SWCTypesDef.apical)
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
                "architecture": {
                    "uuid": str(uuid.uuid4()),
                    "config": {
                        "coordinate_namespace": "Generated Coordinates",
                        "area": self.config.area,
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
                "synapses": {
                    "culture2d": {
                        "uuid": str(uuid.uuid4()),
                        "config": {
                            "cell_types": {
                                pop: {
                                    "mechanism": None,
                                    "synapses": {},
                                    "synapse_type": self.config.populations[
                                        pop
                                    ].synapse_type,
                                }
                                for pop in populations
                            },
                        },
                    }
                },
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
