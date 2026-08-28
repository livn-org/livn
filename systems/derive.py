from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Literal

import h5py
import numpy as np
from generate_2d import (
    CONNECTIVITY_CHUNK,
    GRAPH_FORMAT_VERSION,
    REGION_NAMESPACE,
    StableRandom,
    SynapseRecord,
    create_neural_h5,
)
from machinable import Interface
from machinable.config import to_dict
from machinable.utils import save_file
from mpi4py import MPI
from neuroh5.io import read_population_ranges, write_cell_attributes, write_graph
from pydantic import BaseModel, Field, model_validator


def kernel_weights(distances: np.ndarray, kernel: str, sigma: float) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-(distances**2) / (2.0 * sigma**2))
    return np.exp(-distances / sigma)


def destination_index(edges) -> np.ndarray:
    block_index = edges["Destination Block Index"][:].astype(np.int64)
    block_pointer = edges["Destination Block Pointer"][:].astype(np.int64)
    pointer = edges["Destination Pointer"][:].astype(np.int64)
    out = []
    for b, first in enumerate(block_index):
        lo, hi = block_pointer[b], block_pointer[b + 1]
        counts = np.diff(pointer[lo : hi + 1])
        out.append(np.repeat(first + np.arange(len(counts), dtype=np.int64), counts))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


class DeriveSystem(Interface):
    """Thin a superset down to a shippable system."""

    class Config(BaseModel):
        superset: str
        output_directory: str
        kernel: str = "exponential"
        sigma: float = Field(..., gt=0.0)
        mean_degree: dict[str, float] = Field(default_factory=dict)
        variant: Literal["guarded", "interior"] = "interior"
        floor: dict[str, int] = Field(
            default_factory=dict,
            description=(
                "Per projection, the fewest inputs a postsynaptic cell may end "
                "up with. `EXC->INH: 1` because a mean of 4 is a mean and a "
                "Renshaw cell contacted by zero motoneurons is not a member of "
                "the population that measurement describes."
            ),
        )
        degree_rule: Literal["fixed_probability", "fixed_degree"] = "fixed_probability"
        degree_reference: dict[str, dict] = Field(
            default_factory=dict,
            description=(
                "Per projection, the composition and presynaptic share the "
                "degree above was measured at. Declarative: `mean_degree` "
                "arrives already resolved, so neither this nor `degree_rule` "
                "changes what is built. They are here so a shipped graph "
                "states which rule chose its degrees instead of leaving that "
                "to be reconstructed from a table in another repository -- "
                "the two rules disagree about the E<->EI relationship, and a "
                "fitted result is not interpretable without knowing which one "
                "it assumed."
            ),
        )
        mea: dict | None = None
        random_seed: int = 123

        @model_validator(mode="after")
        def _validate(self):
            if self.kernel not in ("exponential", "gaussian"):
                raise ValueError(f"unknown kernel {self.kernel!r}")
            return self

    @property
    def cells_filepath(self) -> str:
        return os.path.join(self.config.output_directory, "cells.h5")

    @property
    def connections_filepath(self) -> str:
        return os.path.join(self.config.output_directory, "connections.h5")

    def _source(self) -> dict:
        with open(os.path.join(self.config.superset, "graph.json")) as f:
            graph = json.load(f)
        version = int(graph.get("version", 0))
        if version != GRAPH_FORMAT_VERSION:
            raise ValueError(
                f"{self.config.superset!r} is a v{version} graph; derive needs "
                f"v{GRAPH_FORMAT_VERSION}. Rebuild the superset"
            )
        connections = next(iter(graph["connections"].values()))["config"]
        return {
            "architecture": graph["architecture"]["config"],
            "synapses": connections["synapses"],
            "population_definitions": connections["population_definitions"],
        }

    def _cells(self, populations: list[str]) -> dict:
        """Coordinates and the recorded-region mask, per population."""
        path = os.path.join(self.config.superset, "cells.h5")
        out = {}
        with h5py.File(path, "r") as h:
            for pop in populations:
                base = f"Populations/{pop}"
                if base not in h:
                    out[pop] = {
                        "xy": np.zeros((0, 2), np.float32),
                        "keep": np.zeros(0, bool),
                    }
                    continue
                g = h[f"{base}/Generated Coordinates"]
                xy = np.column_stack(
                    [
                        g["X Coordinate/Attribute Value"][:],
                        g["Y Coordinate/Attribute Value"][:],
                    ]
                )
                z = g["Z Coordinate/Attribute Value"][:]
                region = f"{base}/{REGION_NAMESPACE}/interior/Attribute Value"

                interior = (
                    h[region][:].astype(bool)
                    if region in h
                    else np.ones(len(xy), dtype=bool)
                )
                keep = (
                    np.ones(len(xy), dtype=bool)
                    if self.config.variant == "guarded"
                    else interior
                )
                out[pop] = {"xy": xy, "z": z, "keep": keep}
        return out

    def _edges(self, pre: str, post: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(pre_index, post_index, distance)` per edge, population-local."""
        path = os.path.join(self.config.superset, "connections.h5")
        with h5py.File(path, "r") as h:
            group = f"Projections/{post}/{pre}"
            if group not in h:
                return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float64),)
            g = h[group]
            src = g["Edges/Source Index"][:].astype(np.int64)
            dst = destination_index(g["Edges"])
            return src, dst, g["Connections/distance"][:].astype(np.float64)

    def _target_mass(self, pre_xy: np.ndarray, post_xy: np.ndarray, same: bool):
        """Per-postsynaptic-cell kernel mass under the target kernel."""
        n_post = len(post_xy)
        mass = np.zeros(n_post, dtype=np.float64)
        peak = 0.0
        chunk = max(1, min(n_post, CONNECTIVITY_CHUNK))
        for lo in range(0, n_post, chunk):
            hi = min(lo + chunk, n_post)
            d = np.linalg.norm(
                pre_xy[:, None, :] - post_xy[None, lo:hi, :], axis=2
            ).astype(np.float32)
            w = kernel_weights(d, self.config.kernel, self.config.sigma)
            if same:
                rows = np.arange(lo, hi)
                inside = rows < w.shape[0]
                w[rows[inside], np.arange(hi - lo)[inside]] = 0.0
            mass[lo:hi] = w.sum(axis=0)
            peak = max(peak, float(w.max(initial=0.0)))
        return mass, peak

    def _keep_probability(self, source: dict, pre: str, post: str, cells: dict, degree):
        """`(p_target(edge), p_source(edge))` for the edges."""
        spec = source["synapses"][post][pre]
        k1, s1 = spec["kernel"]["kernel"], float(spec["kernel"]["sigma"])
        amp1 = float(spec["kernel"]["amplitude"])

        pre_keep, post_keep = cells[pre]["keep"], cells[post]["keep"]
        pre_xy = cells[pre]["xy"][pre_keep]
        post_xy = cells[post]["xy"][post_keep]

        src, dst, dist = self._edges(pre, post)

        pre_new = np.cumsum(pre_keep) - 1
        post_new = np.cumsum(post_keep) - 1
        alive = pre_keep[src] & post_keep[dst]
        src, dst, dist = pre_new[src[alive]], post_new[dst[alive]], dist[alive]

        mass, peak = self._target_mass(pre_xy, post_xy, same=(pre == post))
        w2 = kernel_weights(dist, self.config.kernel, self.config.sigma)

        if self.config.variant == "interior":
            with np.errstate(divide="ignore", invalid="ignore"):
                p2 = np.where(mass[dst] > 0, degree * w2 / mass[dst], 0.0)
        else:
            total = mass.sum()
            amp2 = (degree * len(post_xy) / total) if total > 0 else 0.0
            if amp2 * peak > 1.0:
                raise ValueError(
                    f"{pre}->{post} at degree {degree:g} needs a connection "
                    f"probability of {amp2 * peak:.3g}; the target kernel "
                    "cannot deliver it"
                )
            p2 = amp2 * w2

        p1 = amp1 * kernel_weights(dist, k1, s1)
        return src, dst, dist, p2, p1, len(post_xy)

    def __call__(self):
        os.makedirs(self.config.output_directory, exist_ok=True)
        for path in (self.cells_filepath, self.connections_filepath):
            if os.path.isfile(path):
                os.remove(path)

        source = self._source()
        definitions = source["population_definitions"]
        populations = list(definitions)
        cells = self._cells(populations)
        counts = {p: int(cells[p]["keep"].sum()) for p in populations}
        distributions = {p: {"2d": counts[p]} for p in populations}

        degrees = dict(self.config.mean_degree)
        flags: dict[str, dict[str, bool]] = {p: {} for p in populations}
        for post in populations:
            for pre in populations:
                if degrees.get(f"{pre}->{post}", 0.0) > 0.0:
                    flags[post][pre] = True

        for path in (self.cells_filepath, self.connections_filepath):
            create_neural_h5(path, distributions, flags, definitions)
        ranges = read_population_ranges(str(self.cells_filepath))[0]

        for pop in populations:
            keep = cells[pop]["keep"]
            if not keep.any():
                continue
            start, _ = ranges[pop]
            xy, z = cells[pop]["xy"][keep], cells[pop]["z"][keep]
            write_cell_attributes(
                self.cells_filepath,
                pop,
                {
                    int(start + i): {
                        "X Coordinate": np.asarray([xy[i, 0]], np.float32),
                        "Y Coordinate": np.asarray([xy[i, 1]], np.float32),
                        "Z Coordinate": np.asarray([z[i]], np.float32),
                        "U Coordinate": np.asarray([xy[i, 0]], np.float32),
                        "V Coordinate": np.asarray([xy[i, 1]], np.float32),
                        "L Coordinate": np.asarray([z[i]], np.float32),
                    }
                    for i in range(len(xy))
                },
                namespace="Generated Coordinates",
                comm=MPI.COMM_WORLD,
            )

        records = {
            p: {
                int(g): SynapseRecord()
                for g in range(ranges[p][0], ranges[p][0] + counts[p])
            }
            for p in populations
        }
        synapse_config: dict[str, dict] = {p: {} for p in populations}

        for post in populations:
            for pre in populations:
                degree = float(degrees.get(f"{pre}->{post}", 0.0))
                if degree <= 0.0 or not counts[pre] or not counts[post]:
                    continue
                spec = source["synapses"][post][pre]
                src, dst, dist, p2, p1, n_post = self._keep_probability(
                    source, pre, post, cells, degree
                )

                with np.errstate(divide="ignore", invalid="ignore"):
                    q = np.where(p1 > 0, p2 / p1, 0.0)
                if q.size and q.max() > 1.0 + 1e-9:
                    raise ValueError(
                        f"{pre}->{post} asks for a graph denser than the "
                        f"superset at some distance (keep probability "
                        f"{q.max():.3g} > 1). It is outside the envelope the "
                        f"superset was generated at; widen the superset"
                    )

                draws = StableRandom(self.config.random_seed, "derive", pre, post)
                take = draws.random(q.size).reshape(q.shape) < q

                floor = int(self.config.floor.get(f"{pre}->{post}", 0))
                if floor:
                    take = self._apply_floor(take, dst, p2, n_post, floor)

                self._write_projection(
                    pre, post, spec, src[take], dst[take], dist[take], ranges, records
                )
                synapse_config[post][pre] = self._spec_for(spec, degree, n_post)

        for pop in populations:
            self._write_synapse_attributes(pop, records[pop])
        self._write_graph_json(source, distributions, counts, synapse_config, cells)
        self._write_provenance(source)
        self._write_mea()
        return self

    @staticmethod
    def _apply_floor(take, dst, p2, n_post, floor):
        """Give a postsynaptic cell left with none its likeliest partners back."""
        have = np.bincount(dst[take], minlength=n_post)
        short = np.flatnonzero(have < floor)
        if not short.size:
            return take
        order = np.argsort(dst, kind="stable")
        bounds = np.searchsorted(dst[order], short)
        ends = np.searchsorted(dst[order], short, side="right")
        for j, lo, hi in zip(short, bounds, ends, strict=False):
            rows = order[lo:hi]
            if not rows.size:
                continue  # nothing to give back; the superset has no candidate
            best = rows[np.argsort(-p2[rows])][: floor - int(have[j])]
            take[best] = True
        return take

    def _write_projection(self, pre, post, spec, src, dst, dist, ranges, records):
        pre_start, post_start = ranges[pre][0], ranges[post][0]
        section = (spec.get("sections") or ["soma"])[0]
        swc = 1 if section == "soma" else 4
        syn_type = 0 if spec.get("type", "excitatory") == "excitatory" else 1

        edges: dict[int, tuple] = {}
        order = np.argsort(dst, kind="stable")
        src, dst, dist = src[order], dst[order], dist[order]
        for j in np.unique(dst):
            rows = np.flatnonzero(dst == j)
            gid = int(post_start + j)
            record = records[post][gid]
            syn_ids = [
                record.add(syn_type, float(dist[r]), 0 if swc == 1 else 1, swc)
                for r in rows
            ]
            edges[gid] = (
                (pre_start + src[rows]).astype(np.uint32),
                {
                    "Connections": {"distance": dist[rows].astype(np.float32)},
                    "Synapses": {"syn_id": np.asarray(syn_ids, np.uint32)},
                },
            )
        if edges:
            write_graph(
                self.connections_filepath,
                src_pop_name=pre,
                dst_pop_name=post,
                edges=edges,
                comm=MPI.COMM_WORLD,
            )

    def _spec_for(self, spec, degree, n_post):
        derived = {k: v for k, v in spec.items() if k != "kernel"}
        derived["kernel"] = {
            **spec["kernel"],
            "kernel": self.config.kernel,
            "sigma": float(self.config.sigma),
            "amplitude": None
            if self.config.variant == "interior"
            else spec["kernel"]["amplitude"],
            "mean_degree": float(degree),
            "normalisation": (
                "per_cell" if self.config.variant == "interior" else "global"
            ),
        }
        return derived

    def _write_synapse_attributes(self, pop, records):
        cell_dict = {}
        for gid, record in records.items():
            n = len(record.syn_ids)
            cell_dict[gid] = {
                "syn_ids": np.asarray(record.syn_ids, np.uint32),
                "syn_types": np.asarray(record.syn_types, np.uint8),
                "syn_cdists": np.asarray(record.syn_cdists, np.float32),
                "syn_locs": np.full(n, 0.5, np.float32),
                "syn_secs": np.asarray(record.syn_secs, np.int16),
                "syn_layers": np.zeros(n, np.uint8),
                "swc_types": np.asarray(record.swc_types, np.uint8),
            }
        write_cell_attributes(
            self.cells_filepath,
            pop,
            cell_dict,
            namespace="Synapse Attributes",
            comm=MPI.COMM_WORLD,
        )

    def _uuid(self, role: str, payload) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(role.encode())
        digest.update(self._digest(self.config.superset).encode())
        digest.update(
            json.dumps(
                {
                    "kernel": self.config.kernel,
                    "sigma": float(self.config.sigma),
                    "mean_degree": to_dict(self.config.mean_degree),
                    "variant": self.config.variant,
                    "floor": to_dict(self.config.floor),
                    "random_seed": int(self.config.random_seed),
                    "payload": payload,
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        return str(uuid.UUID(bytes=digest.digest()))

    def _write_graph_json(self, source, distributions, counts, synapse_config, cells):
        architecture = dict(source["architecture"])
        if self.config.variant == "interior":
            architecture["margin"] = 0.0
        kept = [
            (cells[p]["xy"][cells[p]["keep"]], cells[p]["z"][cells[p]["keep"]])
            for p in cells
            if cells[p]["keep"].any()
        ]
        if kept:
            xy = np.concatenate([k[0] for k in kept])
            z = np.concatenate([k[1] for k in kept])
            lo = [float(xy[:, 0].min()), float(xy[:, 1].min())]
            hi = [float(xy[:, 0].max()), float(xy[:, 1].max())]
            architecture["area"] = [lo, hi]
            architecture["area_kwargs"] = {
                "x_range": [lo[0], hi[0]],
                "y_range": [lo[1], hi[1]],
            }
            architecture["layer_extents"] = {
                "2D": [
                    [lo[0], lo[1], float(z.min())],
                    [hi[0], hi[1], float(z.max())],
                ]
            }
        architecture["cell_distributions"] = distributions
        architecture["cell_counts"] = {p: int(c) for p, c in counts.items()}
        architecture["cells_filepath"] = "./cells.h5"
        save_file(
            os.path.join(self.config.output_directory, "graph.json"),
            {
                "version": GRAPH_FORMAT_VERSION,
                "architecture": {
                    "uuid": self._uuid("architecture", architecture),
                    "config": architecture,
                },
                "distances": {
                    "culture2d": {
                        "uuid": self._uuid("distances", architecture),
                        "config": {
                            "coordinate_namespace": "Generated Coordinates",
                            "cell_distributions": distributions,
                            "layer_extents": architecture["layer_extents"],
                        },
                    }
                },
                "synapse_forest": {},
                "connections": {
                    "culture2d": {
                        "uuid": self._uuid("connections", synapse_config),
                        "config": {
                            "coordinates_namespace": "Generated Coordinates",
                            "connectivity_namespace": "Connections",
                            "distances_namespace": "Connections",
                            "population_definitions": source["population_definitions"],
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

    @staticmethod
    def _digest(path: str) -> str:
        """Content hash of a superset, so a rebuild of it can be checked."""
        h = hashlib.blake2b(digest_size=16)
        for name in ("cells.h5", "connections.h5"):
            h.update(name.encode())
            datasets: list[str] = []
            with h5py.File(os.path.join(path, name), "r") as f:

                def collect(key, obj, into=datasets):
                    if isinstance(obj, h5py.Dataset):
                        into.append(key)

                f.visititems(collect)
                for key in sorted(datasets):
                    value = np.ascontiguousarray(f[key][()])
                    h.update(key.encode())
                    h.update(f"{value.dtype}{value.shape}".encode())
                    h.update(
                        value.tobytes()
                        if value.dtype != object
                        else repr(value.tolist()).encode()
                    )
        return h.hexdigest()

    def _write_provenance(self, source):
        with open(os.path.join(self.config.superset, "provenance.json")) as f:
            upstream = json.load(f)
        upstream_connectivity = upstream.get("config", {}).get("connectivity", {})
        document = {
            "derived_from": {
                "degree_rule": self.config.degree_rule,
                "degree_reference": to_dict(self.config.degree_reference),
                "superset": os.path.basename(os.path.abspath(self.config.superset)),
                "superset_sha": self._digest(self.config.superset),
                "generator": upstream.get("generator"),
                "seed": upstream.get("config", {}).get("random_seed"),
                "sigma": upstream_connectivity.get("sigma"),
                "mean_degree": upstream_connectivity.get("mean_degree"),
                "margin": upstream.get("config", {}).get("margin"),
            },
            "config": {
                "kernel": self.config.kernel,
                "sigma": float(self.config.sigma),
                "mean_degree": to_dict(self.config.mean_degree),
                "variant": self.config.variant,
                "floor": to_dict(self.config.floor),
                "random_seed": int(self.config.random_seed),
                "mea_channels": (
                    len(self.config.mea["electrode_coordinates"])
                    if self.config.mea
                    else 0
                ),
            },
            "version": GRAPH_FORMAT_VERSION,
        }
        with open(
            os.path.join(self.config.output_directory, "provenance.json"), "w"
        ) as f:
            json.dump(document, f, indent=2, sort_keys=True, default=str)

    def _write_mea(self):
        if not self.config.mea:
            return

        with open(os.path.join(self.config.output_directory, "mea.json"), "w") as f:
            json.dump(to_dict(self.config.mea), f)
