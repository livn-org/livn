from __future__ import annotations

import json
import math
import os
import re
import secrets
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from livn.utils import P

if TYPE_CHECKING:
    from livn.run import Run
    from livn.types import Env

RSF = "rsf"
FORMAT = 1
PRODUCER = "livn"
COMPONENT = "rsf"
RECORD_KEY = f"{RSF}_provenance"
POINT_SOURCE = f"{RSF}.point_source/1"
OPERATOR_PATH = f"traces.zarr/{RSF}/observe.json"

DEFAULT_GAIN_UV = 0.195
PROBEINTERFACE_VERSION = "0.3.2"

MASK_FIELDS = ("electrode_positions", "source_positions")

_INT16 = np.iinfo(np.int16)


class RSFError(ValueError):
    """A run that cannot be saved as a session."""


def _zarr():
    hint = "livn.rsf needs zarr>=3; install it with `pip install 'zarr>=3'`"
    try:
        import zarr
    except ImportError as e:
        raise ImportError(hint) from e
    major = str(getattr(zarr, "__version__", "0")).split(".")[0]
    if not major.isdigit() or int(major) < 3:
        raise ImportError(f"{hint} (found zarr {zarr.__version__})")
    return zarr


@dataclass
class _Rows:
    keys: list[str]
    sources: list[str]
    sections: list[str] | None
    values: np.ndarray  # (n_frames, n_rows) float32


def store(
    run: Run | None,
    path: str | os.PathLike,
    env: Env,
    *,
    withhold: Sequence[Mapping] | None = None,
    voltage: bool = False,
    currents: bool = False,
    stimulus: Mapping | None = None,
    gain_uv: float = DEFAULT_GAIN_UV,
    label: str | None = None,
) -> str | None:
    """Write ``run`` as an RSF session directory at ``path``.

    The channel signal is projected from the run's membrane currents through the env's MEA, or
    taken from ``run.potential`` when the run holds one (a run read back by :func:`load`).

    Under MPI the caller gathers first; this writes on the root rank and is a no-op elsewhere, so
    ``store(run.gather(env.comm), path, env)`` is the call that works on every rank.

    Args:
        run: The run, or ``None`` (off the root rank after a gather)
        path: The session directory; must not exist, or be empty
        env: The env that produced the run: names the simulator and supplies the MEA, the
            recording coordinates, the simulated gids and the system
        withhold: Stage transforms applied in order to the channel signal before quantization,
            ``{"mask": {"fields": [...]}}`` or ``{"noise": {"distribution": "gaussian", "params":
            {"sigma_uv": s}, "seed": n}}``
        voltage: Also write ``ground_truth/voltage.zarr``
        currents: Also write ``ground_truth/currents.zarr``
        stimulus: ``{"cls", "kwargs"}`` of what drove the run; ``None`` for spontaneous activity
        gain_uv: uV per LSB of the stored traces; a sample outside int16 is an error
        label: Free text for people; defaults to the system's name

    Returns:
        The session directory on the root rank, ``None`` elsewhere
    """
    if run is None or not P.is_root(comm=getattr(env, "comm", None)):
        return None
    _zarr()

    plan = _prepare(
        run,
        path,
        env,
        withhold=withhold,
        voltage=voltage,
        currents=currents,
        stimulus=stimulus,
        gain_uv=gain_uv,
        label=label,
    )
    _write(plan)
    return plan["path"]


def _prepare(run, path, env, *, withhold, voltage, currents, stimulus, gain_uv, label):
    path = os.fspath(path)
    _check_location(path)
    _check_not_batched(run)

    simulator = _simulator_name(env)
    operator, stimulates = _point_source_operator(env)
    channel_ids = [_id_str(c) for c in np.asarray(operator.electrode_coordinates)[:, 0]]

    gain_uv = float(gain_uv)
    if not gain_uv > 0 or not math.isfinite(gain_uv):
        raise RSFError(
            f"gain_uv must be a positive number of µV per LSB, not {gain_uv}"
        )

    # the channel signal, (n_channels, T) in uV
    signal_series = run.channels.get("current")
    if signal_series is not None and signal_series.values is not None:
        uv, row_ids = _project(run, env, operator), list(channel_ids)
        dt, t0 = float(signal_series.dt), float(signal_series.t0)
    elif getattr(run, "potential", None) is not None:
        potential = run.potential
        row_ids = [_id_str(c) for c in np.asarray(potential.ids)]
        unknown = sorted(set(row_ids) - set(channel_ids))
        if unknown:
            raise RSFError(
                f"run.potential has rows {unknown[:5]} that are not electrodes of the env's MEA"
            )
        uv = np.asarray(potential.values, dtype=np.float64)
        dt, t0 = float(potential.dt), float(potential.t0)
    else:
        raise RSFError(
            "a session is a channel-space recording, and this run holds nothing to put in "
            f"traces.zarr: its channels are {sorted(run.channels) or 'none'}. Call "
            "env.record_membrane_current() before env.run(...) so the MEA signal can be "
            "projected from the membrane currents."
        )

    fs = _sampling_frequency(dt)
    n_frames = int(uv.shape[1])
    tick0 = round(t0 / dt)

    sources = [int(g) for g in np.asarray(env.simulated_gids(everywhere=True)).ravel()]
    known_sources = {str(g) for g in sources}

    stages, masked = _withhold_stages(withhold, row_ids)
    rng_rows = list(row_ids)
    for item in stages:
        transform = item["transform"]
        if "mask" in transform:
            dropped = {
                f.split(":", 1)[1]
                for f in transform["mask"]["fields"]
                if f.startswith("channel:")
            }
            keep = [i for i, key in enumerate(rng_rows) if key not in dropped]
            uv = uv[keep]
            rng_rows = [rng_rows[i] for i in keep]
        else:
            noise = transform["noise"]
            rng = np.random.default_rng(noise["seed"])
            uv = uv + rng.normal(0.0, noise["params"]["sigma_uv"], size=uv.shape)
    row_ids = rng_rows
    if not row_ids:
        raise RSFError(
            "withhold dropped every channel; a session needs at least one row"
        )
    traces = _quantize(uv, gain_uv)

    # ground truth
    spikes = None
    if run.spikes is not None:
        spikes = _spike_frames(
            run.spikes, t0=t0, dt=dt, n_frames=n_frames, sources=sources
        )

    truth: dict[str, _Rows] = {}
    if voltage:
        truth["voltage"] = _series_rows(
            run.channels.get("voltage"), "voltage", dt, t0, n_frames, known_sources
        )
    if currents:
        truth["currents"] = _series_rows(
            run.channels.get("current"), "current", dt, t0, n_frames, known_sources
        )

    # documents and records
    started_at = _now()
    recording_id = uuid.uuid4().hex
    group = {"id": uuid.uuid4().hex, "sequence": 0}
    spec = _spec(env)
    model = _model(env)
    seed = getattr(env, "seed", None)
    seed = None if seed is None else int(seed)

    observe_stage = {
        "role": "acquisition",
        "status": "known",
        "transform": {"observe": {"operator": OPERATOR_PATH}},
        "attribution": "livn MEA point-source model",
    }
    record = {
        "format": FORMAT,
        "recording_id": recording_id,
        "env": {
            "substrate": "simulated",
            "system": {"simulation": {"simulator": simulator, "spec": spec}},
            "model": model,
            "seed": seed,
        },
        "space": "channel",
        "rows": {key: [observe_stage, *stages] for key in row_ids},
        "started_at": started_at,
        "first_sample_tick": int(tick0),
        "group": group,
    }

    observe = _observe_document(operator, stimulates, masked)
    source_document: dict[str, Any] = {"ids": sources}
    if "source_positions" not in masked:
        source_document["coordinates_um"] = _source_coordinates(env, sources)
    simulation = {
        "simulator": {"name": simulator, "version": _version()},
        "spec": spec,
        "model": model,
        "seed": seed,
        "sources": source_document,
        "stimulus": None if stimulus is None else _plain(stimulus),
    }
    regenerate = _regenerate(env)
    if regenerate:
        simulation["regenerate"] = regenerate

    group_attrs = {
        "format": FORMAT,
        "recording_id": recording_id,
        "created_at": started_at,
        "label": _label(env) if label is None else str(label),
        "producer": _producer(),
        "backend_id": simulator,
        "group": group,
        "compression": {
            "codec": "none",
            "level": 0,
            "shuffle": "none",
            "lsb_truncate_bits": 0,
        },
        "compressed_bytes": 0,
    }

    probe = None
    if "electrode_positions" not in masked:
        probe = _probe(operator, row_ids)

    stores = []
    if spikes is not None:
        stores.append("spikes")
    stores.extend(truth)

    for name, document in (
        ("observe.json", observe),
        ("simulation.json", simulation),
        ("the record", record),
    ):
        _json(document, name)

    return {
        "path": path,
        "fs": fs,
        "n_frames": n_frames,
        "row_ids": row_ids,
        "traces": traces,
        "gain_uv": gain_uv,
        "tick0": int(tick0),
        "spikes": spikes,
        "sources": sources,
        "truth": truth,
        "record": record,
        "observe": observe,
        "simulation": simulation,
        "group_attrs": group_attrs,
        "probe": probe,
        "stores": [
            {
                "path": f"ground_truth/{name}.zarr",
                "space": "source",
                "written_by": PRODUCER,
            }
            for name in stores
        ],
    }


def _write(plan: dict) -> None:
    zarr = _zarr()

    path = plan["path"]
    os.makedirs(path, exist_ok=True)
    truth_record = {
        **plan["record"],
        "space": "source",
    }

    if plan["spikes"] is not None or plan["truth"]:
        os.makedirs(os.path.join(path, "ground_truth"), exist_ok=True)

    if plan["spikes"] is not None:
        sample_index, unit_index = plan["spikes"]
        store = os.path.join(path, "ground_truth", "spikes.zarr")
        root = zarr.open_group(
            store,
            mode="w",
            attributes={
                "sampling_frequency": plan["fs"],
                "num_segments": 1,
                "annotations": {
                    RECORD_KEY: {
                        **truth_record,
                        "rows": {str(g): [] for g in plan["sources"]},
                    }
                },
            },
        )
        _array(root, "unit_ids", np.asarray(plan["sources"], dtype=np.int64))
        _array(root, "spikes/sample_index", sample_index)
        _array(root, "spikes/unit_index", unit_index)
        _array(
            root,
            "spikes/segment_slices",
            np.asarray([[0, len(sample_index)]], dtype=np.int64),
        )
        _minimal_group(store, plan["record"]["recording_id"])

    for name, rows in plan["truth"].items():
        store = os.path.join(path, "ground_truth", f"{name}.zarr")
        root = zarr.open_group(
            store,
            mode="w",
            attributes={
                "sampling_frequency": plan["fs"],
                "num_segments": 1,
                "annotations": {
                    RECORD_KEY: {**truth_record, "rows": {key: [] for key in rows.keys}}
                },
            },
        )
        _strings(root, "channel_ids", rows.keys)
        _array(
            root, "traces_seg0", rows.values, chunks=_chunks(plan["fs"], rows.values)
        )
        if rows.sections is not None:
            _strings(root, "properties/source", rows.sources)
            _strings(root, "properties/section", rows.sections)
        _minimal_group(store, plan["record"]["recording_id"])

    # traces.zarr, and last its rsf/zarr.json
    store = os.path.join(path, "traces.zarr")
    attributes = {
        "sampling_frequency": plan["fs"],
        "num_segments": 1,
        "annotations": {RECORD_KEY: plan["record"]},
    }
    if plan["probe"] is not None:
        attributes["probe"] = plan["probe"]
    root = zarr.open_group(store, mode="w", attributes=attributes)
    n_rows = len(plan["row_ids"])
    _strings(root, "channel_ids", plan["row_ids"])
    _array(
        root, "traces_seg0", plan["traces"], chunks=_chunks(plan["fs"], plan["traces"])
    )
    _array(
        root,
        "properties/gain_to_uV",
        np.full(n_rows, plan["gain_uv"], dtype=np.float64),
    )
    _array(root, "properties/offset_to_uV", np.zeros(n_rows, dtype=np.float64))
    _strings(root, "properties/channel", plan["row_ids"])
    _strings(root, "properties/band", ["wide"] * n_rows)

    rsf = os.path.join(store, RSF)
    # each table its own hierarchy, so rsf/zarr.json is not created before it is complete
    anchors = zarr.open_group(os.path.join(rsf, "tick_anchors"), mode="w")
    _array(anchors, "frame", np.asarray([0], dtype=np.uint64))
    _array(anchors, "tick", np.asarray([plan["tick0"]], dtype=np.uint64))
    markers = zarr.open_group(os.path.join(rsf, "stim_markers"), mode="w")
    _array(markers, "tick", np.zeros(0, dtype=np.uint64))
    _strings(markers, "channel", [])
    dropped = zarr.open_group(os.path.join(rsf, "dropped_spans"), mode="w")
    _array(dropped, "frame_start", np.zeros(0, dtype=np.uint64))
    _array(dropped, "n_frames", np.zeros(0, dtype=np.uint32))
    _strings(dropped, "channel", [])

    _write_json(os.path.join(rsf, "observe.json"), plan["observe"])
    _write_json(os.path.join(rsf, "simulation.json"), plan["simulation"])
    _write_json(os.path.join(rsf, "stores.json"), plan["stores"])
    _group_node(rsf, {**plan["group_attrs"], "finished_at": _now()})


def _check_location(path: str) -> None:
    if os.path.lexists(path):
        if not os.path.isdir(path):
            raise RSFError(f"{path} exists and is not a directory")
        if os.listdir(path):
            raise RSFError(
                f"{path} is not empty; a session is written to a new location"
            )
        return
    parent = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(os.path.abspath(path))
    if os.path.isdir(parent):
        folded = name.casefold()
        twins = [n for n in os.listdir(parent) if n != name and n.casefold() == folded]
        if twins:
            raise RSFError(
                f"{name!r} differs only by case from {twins[0]!r} in {parent}; "
                "two names in one location must not be case twins"
            )


def _check_not_batched(run) -> None:
    for name, channel in run.channels.items():
        padded = getattr(channel, "padded", None)
        if padded is not None and np.ndim(padded.times) > 2:
            raise RSFError(f"{name!r} is batched; a session is one sample")
        values = getattr(channel, "values", None) if channel.kind == "series" else None
        if values is not None and np.ndim(values) != 2:
            raise RSFError(
                f"{name!r} has shape {tuple(np.shape(values))}; a session holds one sample of "
                "(rows, frames) series"
            )


def _simulator_name(env) -> str:
    match = re.match(r"^livn\.backend\.([a-z0-9_]+)", type(env).__module__)
    if match is None:
        raise RSFError(
            f"cannot name the simulator of a {type(env).__module__}.{type(env).__qualname__}; "
            "a session is saved with the env of a livn backend"
        )
    return f"livn_{match.group(1)}"


def _point_source_operator(env):
    from livn.io import MEA, ComposedIO, PointSourceModel

    io = getattr(env, "io", None)
    operator, stimulates = io, True
    if isinstance(io, ComposedIO):
        operator, stimulates = io.outputs, io.inputs is io.outputs
    if not isinstance(operator, MEA):
        name = "None" if operator is None else type(operator).__qualname__
        raise RSFError(
            f"the recording IO is {name}; only a point-source MEA has an observation operator "
            f"the format defines ({POINT_SOURCE})"
        )
    if not isinstance(operator.volume_conductor, PointSourceModel):
        raise RSFError(
            f"the MEA's volume conductor is {type(operator.volume_conductor).__qualname__}, "
            f"not livn's PointSourceModel, so it is not {POINT_SOURCE}"
        )
    return operator, stimulates


def _sampling_frequency(dt: float) -> float:
    fs = 1000.0 / float(dt)
    if abs(fs - round(fs)) < 1e-6:
        fs = float(round(fs))
    return fs


def _project(run, env, operator) -> np.ndarray:
    ids = np.asarray(run.current_ids).astype(np.int64).ravel()
    values = np.asarray(run.current)

    coordinates = np.asarray(env.recording_coordinates())
    coordinates_of: dict[int, list[int]] = {}
    for row, gid in enumerate(coordinates[:, 0].astype(np.int64).tolist()):
        coordinates_of.setdefault(gid, []).append(row)

    picked, taken = [], {}
    for gid in ids.tolist():
        k = taken.get(gid, 0)
        own = coordinates_of.get(gid, [])
        if k >= len(own):
            raise RSFError(
                f"gid {gid} has more membrane-current rows than the {len(own)} recording "
                "coordinate(s) the env places it at, so its signal cannot be projected"
            )
        picked.append(own[k])
        taken[gid] = k + 1

    distances = operator.distances(coordinates[np.asarray(picked, dtype=np.int64)])
    return np.asarray(operator.potential_recording(distances, values), dtype=np.float64)


def _withhold_stages(withhold, row_ids) -> tuple[list[dict], set[str]]:
    stages, masked = [], set()
    for item in withhold or ():
        if not isinstance(item, Mapping) or len(item) != 1:
            raise RSFError(
                f"a withhold item is one transform, {{'mask': ...}} or {{'noise': ...}}, "
                f"not {item!r}"
            )
        ((kind, params),) = item.items()
        if kind == "mask":
            fields = params.get("fields") if isinstance(params, Mapping) else None
            if not isinstance(fields, (list, tuple)) or not fields:
                raise RSFError(f"a mask names the fields it withholds, got {params!r}")
            for f in fields:
                if f in MASK_FIELDS:
                    continue
                if isinstance(f, str) and f.startswith("channel:"):
                    if f.split(":", 1)[1] not in row_ids:
                        raise RSFError(f"{f!r} names no row of the session")
                    continue
                raise RSFError(
                    f"livn does not know how to withhold {f!r}; it withholds "
                    f"{', '.join(MASK_FIELDS)} and 'channel:<id>'"
                )
            masked.update(fields)
            transform = {"mask": {"fields": [str(f) for f in fields]}}
        elif kind == "noise":
            if (
                not isinstance(params, Mapping)
                or params.get("distribution") != "gaussian"
            ):
                raise RSFError(f"livn adds only gaussian noise, got {params!r}")
            sigma = (params.get("params") or {}).get("sigma_uv")
            if (
                not isinstance(sigma, (int, float))
                or not sigma >= 0
                or not math.isfinite(sigma)
            ):
                raise RSFError(
                    f"gaussian noise needs params.sigma_uv >= 0, got {params!r}"
                )
            seed = params.get("seed")
            if seed is None:
                seed = secrets.randbits(63)
            transform = {
                "noise": {
                    "distribution": "gaussian",
                    "params": {"sigma_uv": float(sigma)},
                    "seed": int(seed),
                }
            }
        else:
            raise RSFError(f"livn does not know how to withhold by {kind!r}")
        stages.append(
            {
                "role": "analysis",
                "status": "known",
                "transform": transform,
                "attribution": "livn.rsf.store",
            }
        )
    return stages, masked


def _quantize(uv: np.ndarray, gain_uv: float) -> np.ndarray:
    scaled = np.round(uv / gain_uv)
    if scaled.size:
        peak = float(np.max(np.abs(uv)))
        if not np.all(np.isfinite(scaled)):
            raise RSFError("the channel signal holds a NaN or an infinity")
        if scaled.max() > _INT16.max or scaled.min() < _INT16.min:
            raise RSFError(
                f"the channel signal peaks at {peak:.6g} µV, beyond the int16 range at "
                f"{gain_uv} µV/LSB; pass gain_uv >= {peak / _INT16.max:.6g}"
            )
    return np.ascontiguousarray(scaled.astype(np.int16).T)


def _spike_frames(
    events, *, t0, dt, n_frames, sources
) -> tuple[np.ndarray, np.ndarray]:
    """``(sample_index, unit_index)``, ascending by frame and then by unit."""
    ids, times = events.ids, events.times
    if ids is None or times is None:
        ids, times = np.zeros(0, dtype=np.int64), np.zeros(0)
    ids = np.asarray(ids).astype(np.int64).ravel()
    times = np.asarray(times, dtype=np.float64).ravel()

    offset = round((float(events.t0) - t0) / dt)
    frames = np.floor(times / dt + 1e-9).astype(np.int64) + offset

    end = (n_frames - offset) * dt
    frames[(frames == n_frames) & (np.abs(times - end) < 1e-6)] = n_frames - 1
    outside = (frames < 0) | (frames >= n_frames)
    if outside.any():
        i = int(np.flatnonzero(outside)[0])
        raise RSFError(
            f"gid {ids[i]} spikes at {times[i] + events.t0} ms, outside the "
            f"{n_frames} frames of the session ({int(outside.sum())} spike(s) outside)"
        )

    unit_of = {g: i for i, g in enumerate(sources)}
    missing = sorted(set(ids.tolist()) - set(unit_of))
    if missing:
        raise RSFError(
            f"gids {missing[:5]} spiked but are not simulated gids of the env"
        )
    units = np.asarray([unit_of[g] for g in ids.tolist()], dtype=np.int64)

    order = np.lexsort((units, frames))
    return frames[order], units[order]


def _series_rows(series, name, dt, t0, n_frames, known_sources) -> _Rows:
    if series is None or series.values is None:
        raise RSFError(
            f"{name}=True but the run has no {name} channel; record it before env.run(...)"
        )
    if abs(float(series.dt) - dt) > 1e-9:
        raise RSFError(
            f"the {name} series is sampled at dt={series.dt} ms and the channel signal at "
            f"dt={dt} ms; ground truth shares the traces' frame axis (record both at one dt)"
        )
    if abs(float(series.t0) - t0) > 1e-9 or series.values.shape[1] != n_frames:
        raise RSFError(
            f"the {name} series covers {series.values.shape[1]} frames from t0={series.t0}, "
            f"the channel signal {n_frames} frames from t0={t0}"
        )

    sources = [str(int(g)) for g in np.asarray(series.ids).ravel()]
    sections = getattr(series, "sections", None)
    sections = (
        None if sections is None else [str(s) for s in np.asarray(sections).ravel()]
    )
    multi = len(set(sources)) < len(sources)
    if not multi:
        keys = sources
    elif sections is None:
        flag = "currents" if name == "current" else name
        raise RSFError(
            f"{flag}=True on a multi-compartment run whose {name} rows carry no section names "
            f"(this backend does not record them); save with {flag}=False"
        )
    else:
        keys = [f"{s}/{c}" for s, c in zip(sources, sections, strict=True)]
    if len(set(keys)) < len(keys):
        raise RSFError(f"the {name} series has duplicate rows")
    outside = sorted(set(sources) - known_sources)
    if outside:
        raise RSFError(
            f"the {name} series has rows of gids {outside[:5]} the env does not simulate"
        )

    return _Rows(
        keys=keys,
        sources=sources,
        sections=sections,
        values=np.ascontiguousarray(np.asarray(series.values, dtype=np.float32).T),
    )


def _observe_document(operator, stimulates: bool, masked: set[str]) -> dict:
    conductor = operator.volume_conductor
    document: dict[str, Any] = {"kind": POINT_SOURCE}
    coordinates = np.asarray(operator.electrode_coordinates, dtype=np.float64)
    if "electrode_positions" in masked:
        document["electrode_ids"] = [_id_value(c) for c in coordinates[:, 0]]
    else:
        document["electrode_coordinates_um"] = [
            [_id_value(row[0]), float(row[1]), float(row[2]), float(row[3])]
            for row in coordinates
        ]
    document["output_radius_um"] = float(operator.output_radius)
    document["recording_gain"] = float(conductor.recording_gain)
    document["min_distance_um"] = float(conductor.min_distance_um)
    document["detection_threshold"] = float(operator.detection_threshold)
    if stimulates:
        document["input_radius_um"] = float(operator.input_radius)
        document["stimulation_gain"] = float(conductor.stimulation_gain)
        document["electrode_radius_um"] = float(conductor.electrode_radius_um)
        document["culture_height_um"] = float(conductor.culture_height_um)
    return document


def _probe(operator, row_ids: list[str]) -> dict:
    coordinates = np.asarray(operator.electrode_coordinates, dtype=np.float64)
    position = {_id_str(row[0]): (float(row[1]), float(row[2])) for row in coordinates}
    radius = float(operator.volume_conductor.electrode_radius_um)
    n = len(row_ids)
    return {
        "specification": "probeinterface",
        "version": PROBEINTERFACE_VERSION,
        "probes": [
            {
                "ndim": 2,
                "si_units": "um",
                "annotations": {"model_name": "livn MEA", "manufacturer": ""},
                "contact_annotations": {},
                "contact_positions": [list(position[key]) for key in row_ids],
                "contact_plane_axes": [[[1.0, 0.0], [0.0, 1.0]]] * n,
                "contact_shapes": ["circle"] * n,
                "contact_shape_params": [{"radius": radius}] * n,
                "contact_ids": list(row_ids),
                "device_channel_indices": list(range(n)),
            }
        ],
    }


def _source_coordinates(env, sources: list[int]) -> list[list[float]]:
    soma = {
        int(row[0]): [float(row[1]), float(row[2]), float(row[3])]
        for row in np.asarray(env.active_neuron_coordinates())
    }
    recording: dict[int, list] = {}
    for row in np.asarray(env.recording_coordinates()):
        recording.setdefault(int(row[0]), []).append(
            [float(row[1]), float(row[2]), float(row[3])]
        )
    out = []
    for gid in sources:
        rows = recording.get(gid, [])
        if len(rows) == 1:
            out.append(rows[0])
        elif gid in soma:
            out.append(soma[gid])
        else:
            raise RSFError(f"gid {gid} has no coordinates in the env's system")
    return out


def _spec(env) -> str | None:
    from livn.system import identity

    try:
        spec = identity(env.system)
    except Exception:
        return None
    return None if spec in (None, "") else str(spec)


def _model(env) -> str | None:
    model = getattr(env, "model", None)
    if model is None:
        return None
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _regenerate(env) -> dict:
    from livn.types import _describe

    regenerate: dict[str, Any] = {}
    system = getattr(env, "system", None)
    try:
        described = _plain(_describe(system))
        json.dumps(described, allow_nan=False)
    except Exception:
        described = None
    if described is not None:
        regenerate["system_spec"] = described

    directory = getattr(system, "uri", None)
    if isinstance(directory, (str, os.PathLike)):
        provenance = os.path.join(os.fspath(directory), "provenance.json")
        if os.path.isfile(provenance):
            with open(provenance) as f:
                regenerate["system_provenance"] = json.load(f)

    stream = getattr(env, "noise_stream", None)
    if stream is not None:
        regenerate["noise_stream"] = int(stream)
    return regenerate


def _label(env) -> str:
    name = getattr(getattr(env, "system", None), "name", None)
    return name if isinstance(name, str) else ""


def _producer() -> dict:
    return {"name": PRODUCER, "version": _version(), "component": COMPONENT}


def _version() -> str:
    from livn import __version__

    return str(__version__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _id_str(value) -> str:
    if isinstance(value, str):
        return value
    f = float(value)
    if f.is_integer():
        return str(int(f))
    raise RSFError(f"id {value!r} is neither an integer nor a string")


def _id_value(value):
    s = _id_str(value)
    return int(s) if re.fullmatch(r"-?[0-9]+", s) else s


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _json(document, name: str) -> str:
    try:
        return json.dumps(_plain(document), allow_nan=False, indent=2)
    except ValueError as e:
        raise RSFError(
            f"{name} holds a NaN or an infinity, which no document may: {e}"
        ) from e
    except TypeError as e:
        raise RSFError(f"{name} holds a value that is not JSON: {e}") from e


def _write_json(path: str, document) -> None:
    text = _json(document, os.path.basename(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def _group_node(directory: str, attributes: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    node = {"zarr_format": 3, "node_type": "group", "attributes": _plain(attributes)}
    with open(os.path.join(directory, "zarr.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(node, allow_nan=False, indent=2) + "\n")


def _minimal_group(store: str, recording_id: str) -> None:
    now = _now()
    _group_node(
        os.path.join(store, RSF),
        {
            "format": FORMAT,
            "recording_id": recording_id,
            "created_at": now,
            "producer": _producer(),
            "finished_at": now,
        },
    )


def _chunks(fs: float, values: np.ndarray) -> tuple[int, int]:
    return (max(1, min(round(fs), values.shape[0])), max(1, values.shape[1]))


def _array(group, name: str, values: np.ndarray, chunks=None) -> None:
    values = np.asarray(values)
    if chunks is None:
        chunks = tuple(max(1, s) for s in values.shape)
    array = group.create_array(
        name,
        shape=values.shape,
        dtype=values.dtype,
        chunks=chunks,
        compressors=None,
        fill_value=0,
    )
    if values.size:
        array[...] = values


def _strings(group, name: str, values: Sequence[str]) -> None:
    array = group.create_array(
        name, shape=(len(values),), dtype=str, chunks=(max(1, len(values)),)
    )
    if len(values):
        array[:] = np.asarray([str(v) for v in values], dtype=object)


@dataclass(frozen=True)
class Session:
    run: Run
    io: Any
    record: dict
    simulation: dict | None


def load(path: str | os.PathLike) -> Session:
    zarr = _zarr()

    from livn.run import Run, Series

    path = os.fspath(path)
    traces_path = os.path.join(path, "traces.zarr")
    if not os.path.isfile(os.path.join(traces_path, "zarr.json")):
        raise RSFError(f"{path} is not a session: it has no traces.zarr/zarr.json")
    group_attrs = _read_node(os.path.join(traces_path, RSF))
    if not group_attrs.get("finished_at"):
        raise RSFError(
            f"{path} is not complete (traces.zarr has no finished_at); it may still be written"
        )

    root = zarr.open_group(traces_path, mode="r")
    attrs = _read_node(traces_path)
    fs = float(attrs["sampling_frequency"])
    dt = 1000.0 / fs
    record = (attrs.get("annotations") or {}).get(RECORD_KEY) or {}
    tick = record.get("first_sample_tick")
    t0 = float(tick) * dt if tick is not None else 0.0

    keys = [str(k) for k in _read(root, "channel_ids")]
    stored = np.asarray(_read(root, "traces_seg0"))
    gain = np.asarray(_read(root, "properties/gain_to_uV"), dtype=np.float64)
    offset = np.asarray(_read(root, "properties/offset_to_uV"), dtype=np.float64)
    values = (stored.astype(np.float64) * gain + offset).T.astype(np.float32)
    n_frames = stored.shape[0]
    duration = n_frames * dt

    potential = Series(ids=_ids_array(keys), values=values, dt=dt, t0=t0)
    run = Run(t0=t0, duration=duration, potential=potential)

    spikes_path = os.path.join(path, "ground_truth", "spikes.zarr")
    if _complete(spikes_path):
        sorting = zarr.open_group(spikes_path, mode="r")
        units = np.asarray(_read(sorting, "unit_ids"))
        sample_index = np.asarray(_read(sorting, "spikes/sample_index"), dtype=np.int64)
        unit_index = np.asarray(_read(sorting, "spikes/unit_index"), dtype=np.int64)
        order = np.argsort(sample_index, kind="stable")
        run = run.add_spikes(
            _ids_array(units[unit_index[order]]), sample_index[order] * dt
        )

    for name, channel in (("voltage", "voltage"), ("currents", "current")):
        store = os.path.join(path, "ground_truth", f"{name}.zarr")
        if not _complete(store):
            continue
        truth = zarr.open_group(store, mode="r")
        ids = [str(k) for k in _read(truth, "channel_ids")]
        sections = None
        if _has(truth, "properties/section"):
            sections = np.asarray([str(s) for s in _read(truth, "properties/section")])
        if _has(truth, "properties/source"):
            ids = [str(s) for s in _read(truth, "properties/source")]
        samples = np.asarray(_read(truth, "traces_seg0"), dtype=np.float32).T
        run = run.add(
            channel, _ids_array(ids), samples, dt=dt, kind="series", sections=sections
        )

    documents = os.path.join(traces_path, RSF)
    simulation = None
    if os.path.isfile(os.path.join(documents, "simulation.json")):
        with open(os.path.join(documents, "simulation.json"), encoding="utf-8") as f:
            simulation = json.load(f)

    return Session(
        run=run,
        io=_read_operator(os.path.join(documents, "observe.json")),
        record=record,
        simulation=simulation,
    )


def _read(group, name: str) -> np.ndarray:
    zarr = _zarr()

    node = group.get(name)
    if not isinstance(node, zarr.Array):
        raise RSFError(f"{group.store_path}/{name} is not an array")
    return np.asarray(node[...])


def _has(group, name: str) -> bool:
    return group.get(name) is not None


def _read_node(directory: str) -> dict:
    node_path = os.path.join(directory, "zarr.json")
    if not os.path.isfile(node_path):
        return {}
    with open(node_path, encoding="utf-8") as f:
        return json.load(f).get("attributes") or {}


def _complete(store: str) -> bool:
    return bool(_read_node(os.path.join(store, RSF)).get("finished_at"))


def _ids_array(ids) -> np.ndarray:
    """Integer ids when every id is an integer, else strings."""
    strings = [
        _id_str(i) if not isinstance(i, str) else i for i in np.asarray(ids).tolist()
    ]
    if all(re.fullmatch(r"-?[0-9]+", s) for s in strings):
        return np.asarray([int(s) for s in strings], dtype=np.int64)
    return np.asarray(strings)


def _read_operator(path: str):
    from livn.io import MEA

    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    if (
        document.get("kind") != POINT_SOURCE
        or "electrode_coordinates_um" not in document
    ):
        return None
    coordinates = document["electrode_coordinates_um"]
    if not all(isinstance(row[0], int) for row in coordinates):
        return None  # livn's MEA numbers its electrodes
    conductor = {
        "recording_gain": document["recording_gain"],
        "min_distance_um": document["min_distance_um"],
    }
    for field in ("stimulation_gain", "electrode_radius_um", "culture_height_um"):
        if field in document:
            conductor[field] = document[field]
    kwargs: dict[str, Any] = {
        "electrode_coordinates": np.asarray(coordinates, dtype=np.float64),
        "output_radius": document["output_radius_um"],
        "volume_conductor": conductor,
        "detection_threshold": document.get("detection_threshold", 0.0),
    }
    if "input_radius_um" in document:
        kwargs["input_radius"] = document["input_radius_um"]
    return MEA(**kwargs)
