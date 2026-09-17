from __future__ import annotations

import os

import numpy as np
from generate_2d import REGION_NAMESPACE, create_neural_h5
from machinable.utils import save_file
from mpi4py import MPI
from neuroh5.io import write_cell_attributes, write_graph

LAYER_INDEX = 0


def write_neuroh5(system, directory: str, mea: dict | None = None) -> str:
    """Write ``system`` to ``directory`` as ``cells.h5``/``connections.h5``/``graph.json``."""
    os.makedirs(directory, exist_ok=True)
    cells_filepath = os.path.join(directory, "cells.h5")
    connections_filepath = os.path.join(directory, "connections.h5")
    for path in (cells_filepath, connections_filepath):
        if os.path.isfile(path):
            os.remove(path)

    populations = list(system.populations)
    counts = {p: system.population_count(p) for p in populations}
    distributions = {p: {"2d": int(counts[p])} for p in populations}
    definitions = dict(getattr(system, "population_definitions", None) or {})
    if not definitions:
        definitions = {p: 10 + i for i, p in enumerate(populations)}

    synapses = system.connections_config["synapses"]
    flags = {post: dict.fromkeys(sources, True) for post, sources in synapses.items()}
    for path in (cells_filepath, connections_filepath):
        create_neural_h5(path, distributions, flags, definitions)

    _write_coordinates(system, cells_filepath, populations)
    _write_edges(system, connections_filepath, synapses)
    _write_synapse_attributes(system, cells_filepath, populations)
    save_file(os.path.join(directory, "graph.json"), system.graph_document())

    if mea is not None:
        save_file(os.path.join(directory, "mea.json"), mea)
    elif hasattr(system, "default_io"):
        io = system.default_io()
        if hasattr(io, "electrode_coordinates"):
            save_file(
                os.path.join(directory, "mea.json"),
                {
                    "electrode_coordinates": np.asarray(
                        io.electrode_coordinates
                    ).tolist(),
                    "input_radius": float(io.input_radius),
                    "output_radius": float(io.output_radius),
                },
            )

    return directory


def _write_coordinates(system, filepath: str, populations: list[str]) -> None:
    interior = getattr(system, "interior", None)
    for population in populations:
        coordinates = np.asarray(system.coordinate_array(population))
        if not len(coordinates):
            continue
        write_cell_attributes(
            filepath,
            population,
            {
                int(row[0]): {
                    "X Coordinate": np.asarray([row[1]], np.float32),
                    "Y Coordinate": np.asarray([row[2]], np.float32),
                    "Z Coordinate": np.asarray([row[3]], np.float32),
                    "U Coordinate": np.asarray([row[1]], np.float32),
                    "V Coordinate": np.asarray([row[2]], np.float32),
                    "L Coordinate": np.asarray([row[3]], np.float32),
                }
                for row in coordinates
            },
            namespace="Generated Coordinates",
            comm=MPI.COMM_WORLD,
        )
        if interior is not None and not bool(np.all(interior)):
            # which cells the array reads, for a guarded system
            gids = coordinates[:, 0].astype(int)
            write_cell_attributes(
                filepath,
                population,
                {
                    int(gid): {"interior": np.asarray([int(interior[gid])], np.uint8)}
                    for gid in gids
                },
                namespace=REGION_NAMESPACE,
                comm=MPI.COMM_WORLD,
            )


def _write_edges(system, filepath: str, synapses: dict) -> None:
    for post, sources in synapses.items():
        start, count = system.population_ranges[post]
        destinations = range(start, start + count)
        for pre in sources:
            edges = {}
            for post_gid, (pre_gids, payload) in system.edges(pre, post, destinations):
                edges[int(post_gid)] = (
                    np.asarray(pre_gids, dtype=np.uint32),
                    {
                        "Connections": {
                            "distance": np.asarray(
                                payload["Connections"]["distance"], dtype=np.float32
                            )
                        },
                        "Synapses": {
                            "syn_id": np.asarray(
                                payload["Synapses"]["syn_id"], dtype=np.uint32
                            )
                        },
                    },
                )
            if not edges:
                continue
            write_graph(
                filepath,
                src_pop_name=pre,
                dst_pop_name=post,
                edges=edges,
                comm=MPI.COMM_WORLD,
            )


def _write_synapse_attributes(system, filepath: str, populations: list[str]) -> None:
    for population in populations:
        start, count = system.population_ranges[population]
        gids = list(range(start, start + count))
        placement = system.placement(population, gids)

        kinds: dict[int, dict[int, int]] = {gid: {} for gid in gids}
        for pre, spec in (
            system.connections_config["synapses"].get(population, {}).items()
        ):
            excitatory = int(spec.get("type", "excitatory") != "excitatory")
            for post_gid, (_pre_gids, payload) in system.edges(pre, population, gids):
                for syn_id in np.asarray(payload["Synapses"]["syn_id"]).tolist():
                    kinds[int(post_gid)][int(syn_id)] = excitatory

        cells = {}
        for gid in gids:
            syn_ids, swc_types, syn_locs = placement.get(
                gid,
                (
                    np.zeros(0, np.int64),
                    np.zeros(0, np.int64),
                    np.zeros(0, np.float64),
                ),
            )
            ids = np.asarray(syn_ids, dtype=np.uint32)
            cells[gid] = {
                "syn_ids": ids,
                "syn_types": np.asarray(
                    [kinds[gid].get(int(i), 0) for i in ids], dtype=np.uint8
                ),
                "syn_cdists": np.zeros(len(ids), dtype=np.float32),
                "syn_locs": np.asarray(syn_locs, dtype=np.float32),
                # a flat culture puts everything on one section per swc type
                "syn_secs": np.asarray(
                    [0 if int(s) == 1 else 1 for s in swc_types], dtype=np.int16
                ),
                "syn_layers": np.full(len(ids), LAYER_INDEX, dtype=np.uint8),
                "swc_types": np.asarray(swc_types, dtype=np.uint8),
            }
        write_cell_attributes(
            filepath,
            population,
            cells,
            namespace="Synapse Attributes",
            comm=MPI.COMM_WORLD,
        )
