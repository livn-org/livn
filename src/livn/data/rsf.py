from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from livn.data import Missing
from livn.utils import P

if TYPE_CHECKING:
    from livn.io import IO
    from livn.policy import Policy
    from livn.run import Events, Run, Series
    from livn.types import Env

RSF = "rsf"
FORMAT = 1
PRODUCER = "livn"
COMPONENT = "rsf"
RECORD_KEY = f"{RSF}_provenance"
POINT_SOURCE = f"{RSF}.point_source/1"
LIVN_IO = "livn.io/1"
OPERATOR_PATH = f"traces.zarr/{RSF}/observe.json"
INPUTS = "inputs"

DEFAULT_GAIN_UV = 0.195
PROBEINTERFACE_VERSION = "0.3.2"

MASK_FIELDS = ("electrode_positions", "source_positions")

_INT16 = np.iinfo(np.int16)


class RSFError(ValueError):
    """A run that cannot be saved as a session."""


def _zarr():
    hint = "livn.data.rsf needs zarr>=3; install it with `pip install 'zarr>=3'`"
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
    stimulus: Mapping | Policy | None = None,
    inputs: np.ndarray | None = None,
    inputs_dt: float | None = None,
    gain_uV: float = DEFAULT_GAIN_UV,
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
        withhold: Transforms that drop what a session would reveal, applied after the IO's
            stack and before quantization: ``{"mask": {"fields": [...]}}``. Noise is the
            IO's own: its ``Noise`` stages are applied here with their ``seed`` and
            recorded in the row stages, in stack order
        voltage: Also write ``ground_truth/voltage.zarr``
        currents: Also write ``ground_truth/currents.zarr``
        stimulus: ``{"cls", "kwargs"}`` of what drove the run, or the channel-space
            :class:`~livn.policy.Policy` itself (whose spec is recorded and whose rendered
            array is stored as the input series unless ``inputs`` is given); ``None`` for
            spontaneous activity
        inputs: The delivered per-channel input, ``(timestep, n_channels)`` over the stimulating
            IO's channels, starting at the run's ``t0``. It is stored at the traces' frame rate
            (zero-order hold; zero past its end) so :meth:`RsfRecording.inputs` can serve it
        inputs_dt: Sampling interval of ``inputs`` in ms; defaults to the policy's ``dt``
        gain_uV: uV per LSB of the stored traces; a sample outside int16 is an error
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
        inputs=inputs,
        inputs_dt=inputs_dt,
        gain_uV=gain_uV,
        label=label,
    )
    _write(plan)
    return plan["path"]


def _prepare(
    run,
    path,
    env,
    *,
    withhold,
    voltage,
    currents,
    stimulus,
    inputs,
    inputs_dt,
    gain_uV,
    label,
):
    path = os.fspath(path)
    _check_location(path)
    _check_not_batched(run)

    simulator = _simulator_name(env)
    io, operator, stimulates = _observation(env)
    channel_ids = [_id_str(c) for c in np.asarray(operator.electrode_coordinates)[:, 0]]

    gain_uV = float(gain_uV)
    if not gain_uV > 0 or not math.isfinite(gain_uV):
        raise RSFError(
            f"gain_uV must be a positive number of µV per LSB, not {gain_uV}"
        )

    # the channel signal, (n_channels, T) in uV
    signal_series = run.channels.get("current")
    projected = None
    through_stack = False
    if signal_series is not None and signal_series.values is not None:
        dt, t0 = float(signal_series.dt), float(signal_series.t0)
        (uv, projected), row_ids = _project(run, env, io, dt), list(channel_ids)
        through_stack = True
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

    stack_stages = _stack_stages(io) if through_stack else []
    stages, masked = _withhold_stages(withhold, row_ids)
    rng_rows = list(row_ids)
    for item in stages:
        dropped = {
            f.split(":", 1)[1]
            for f in item["transform"]["mask"]["fields"]
            if f.startswith("channel:")
        }
        keep = [i for i, key in enumerate(rng_rows) if key not in dropped]
        uv = uv[keep]
        rng_rows = [rng_rows[i] for i in keep]
    row_ids = rng_rows
    if not row_ids:
        raise RSFError(
            "withhold dropped every channel; a session needs at least one row"
        )
    traces = _quantize(uv, gain_uV)

    stimulus, delivered = _stimulus(stimulus, inputs, inputs_dt)
    if delivered is not None:
        delivered = _delivered(io, delivered, dt, n_frames)

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
        "rows": {key: [observe_stage, *stack_stages, *stages] for key in row_ids},
        "started_at": started_at,
        "first_sample_tick": int(tick0),
        "group": group,
    }

    observe = _observe_document(io, operator, stimulates, masked)
    source_document: dict[str, Any] = {"ids": sources}
    if "source_positions" not in masked:
        source_document["coordinates"] = _source_coordinates(env, sources)
    simulation = {
        "simulator": {"name": simulator, "version": _version()},
        "spec": spec,
        "model": model,
        "seed": seed,
        "sources": source_document,
        "stimulus": None if stimulus is None else _plain(stimulus),
    }
    if delivered is not None:
        simulation["inputs"] = {
            "channels": delivered["ids"],
            "units": delivered["units"],
            "source_dt": delivered["source_dt"],
        }
    if projected is not None:
        simulation["trace_sections"] = projected["contributing"]
        if projected["unplaced"]:
            simulation["unplaced_sections"] = projected["unplaced"]
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
        "gain_uV": gain_uV,
        "tick0": int(tick0),
        "spikes": spikes,
        "sources": sources,
        "truth": truth,
        "inputs": delivered,
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
        np.full(n_rows, plan["gain_uV"], dtype=np.float64),
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
    if plan["inputs"] is not None:
        delivered = zarr.open_group(
            os.path.join(rsf, INPUTS),
            mode="w",
            attributes={
                "units": plan["inputs"]["units"],
                "source_dt": plan["inputs"]["source_dt"],
            },
        )
        _strings(delivered, "channel_ids", plan["inputs"]["ids"])
        _array(
            delivered,
            "values",
            plan["inputs"]["values"],
            chunks=_chunks(plan["fs"], plan["inputs"]["values"]),
        )
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


def _observation(env):
    from livn.io import MEA, ComposedIO

    io = getattr(env, "io", None)
    device, stimulates = io, True
    if isinstance(io, ComposedIO):
        device, stimulates = io.outputs, io.inputs is io.outputs
    if not isinstance(device, MEA):
        name = "None" if device is None else type(device).__qualname__
        raise RSFError(
            f"the recording IO is {name}; a session is recorded through an MEA "
            f"(its medium and transfers are stored with it)"
        )
    device.medium.require("electrical")  # no electrical part, nothing to record
    return io, device, stimulates


def _transfers(io) -> tuple:
    return tuple(getattr(io, "transfers", ()))


def _stack_stages(io) -> list[dict]:
    """The row stages of the IO's transfers, in signal order."""
    stages = []
    for transfer in _transfers(io):
        document = transfer.serialize()
        if document["cls"] == "livn.io.Noise":
            transform = {
                "noise": {
                    "distribution": "gaussian",
                    "params": {"sigma_uV": float(document["sigma_uV"])},
                    "seed": int(document["seed"]),
                }
            }
        else:
            transform = {"transfer": document}
        stages.append(
            {
                "role": "acquisition",
                "status": "known",
                "transform": transform,
                "attribution": document["cls"],
            }
        )
    return stages


def _stimulus(stimulus, inputs, inputs_dt):
    """`(spec, (values, dt, units) | None)`: the recorded spec and the delivered series."""
    from livn.policy import Policy

    units = None
    if isinstance(stimulus, Policy):
        spec = {
            "cls": f"{type(stimulus).__module__}.{type(stimulus).__qualname__}",
            "kwargs": stimulus.serialize(),
        }
        units = stimulus.input_units
        if inputs is None:
            inputs = stimulus.render()
            inputs_dt = inputs_dt or getattr(stimulus, "dt", None)
        stimulus = spec
    if inputs is None:
        return stimulus, None
    if inputs_dt is None or not float(inputs_dt) > 0:
        raise RSFError(
            "inputs needs the sampling interval of its rows: pass inputs_dt in ms"
        )
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 2:
        raise RSFError(
            f"inputs is (timestep, n_channels), not shape {tuple(values.shape)}"
        )
    return stimulus, {"values": values, "dt": float(inputs_dt), "units": units}


def _delivered(io, given: dict, dt: float, n_frames: int) -> dict:
    """The input series on the traces' frame grid, with the channels it drives."""
    from livn.io import ComposedIO

    device = io.inputs if isinstance(io, ComposedIO) and io.inputs is not None else io
    ids = [_id_str(c) for c in np.asarray(device.channel_ids).ravel()]
    values, dt_in = given["values"], given["dt"]
    if values.shape[1] != len(ids):
        raise RSFError(
            f"inputs has {values.shape[1]} channels and the stimulating IO has {len(ids)}"
        )
    units = given["units"]
    if units is None:
        units = getattr(device, "input_units", None)

    # frame i is at i * dt after the run starts; the input holds its row until the next
    rows = np.floor(np.arange(n_frames) * dt / dt_in + 1e-9).astype(np.int64)
    inside = rows < len(values)
    series = np.zeros((n_frames, len(ids)), dtype=np.float32)
    series[inside] = values[rows[inside]]
    return {"ids": ids, "values": series, "source_dt": dt_in, "units": units}


def _sampling_frequency(dt: float) -> float:
    fs = 1000.0 / float(dt)
    if abs(fs - round(fs)) < 1e-6:
        fs = float(round(fs))
    return fs


def _project(run, env, io, dt=None) -> tuple[np.ndarray, dict[str, list[str]] | None]:
    operator = getattr(io, "outputs", io)
    dt = run.current_dt if dt is None else dt
    ids = np.asarray(run.current_ids).astype(np.int64).ravel()
    values = np.asarray(run.current)
    row_sections = run.current_sections
    row_sections = (
        None
        if row_sections is None
        else [str(s) for s in np.asarray(row_sections).ravel()]
    )

    coordinates = np.asarray(env.recording_coordinates())
    owners = coordinates[:, 0].astype(np.int64).tolist()
    names = getattr(env, "recording_sections", lambda: None)()
    by_name = row_sections is not None and names is not None
    if by_name:
        names = [str(n) for n in np.asarray(names).ravel()]
        at = {
            (gid, name): row
            for row, (gid, name) in enumerate(zip(owners, names, strict=True))
        }
    coordinates_of: dict[int, list[int]] = {}
    for row, gid in enumerate(owners):
        coordinates_of.setdefault(gid, []).append(row)

    picked, kept, unplaced, taken = [], [], [], {}
    for r, gid in enumerate(ids.tolist()):
        if by_name:
            row = at.get((gid, row_sections[r]))
            if row is None:
                unplaced.append(row_sections[r])
                continue
            picked.append(row)
            kept.append(r)
            continue
        k = taken.get(gid, 0)
        own = coordinates_of.get(gid, [])
        if k >= len(own):
            raise RSFError(
                f"gid {gid} has more membrane-current rows than the {len(own)} recording "
                "coordinate(s) the env places it at, so its signal cannot be projected"
            )
        picked.append(own[k])
        kept.append(r)
        taken[gid] = k + 1

    if not picked:
        raise RSFError("no membrane-current row has a recording coordinate")
    distances = operator.distances(coordinates[np.asarray(picked, dtype=np.int64)])

    uv = np.asarray(
        operator.potential_recording(distances, values[np.asarray(kept)], dt=dt),
        dtype=np.float64,
    )
    for transfer in _transfers(io):
        uv = (
            transfer.apply_seeded(uv)
            if hasattr(transfer, "apply_seeded")
            else np.asarray(transfer(uv, dt=dt), dtype=np.float64)
        )
    if not by_name:
        return uv, None
    return uv, {
        "contributing": list(dict.fromkeys(row_sections[r] for r in kept)),
        "unplaced": list(dict.fromkeys(unplaced)),
    }


def _withhold_stages(withhold, row_ids) -> tuple[list[dict], set[str]]:
    stages, masked = [], set()
    for item in withhold or ():
        if not isinstance(item, Mapping) or len(item) != 1:
            raise RSFError(
                f"a withhold item is one transform, {{'mask': ...}}, not {item!r}"
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
        else:
            raise RSFError(f"livn does not know how to withhold by {kind!r}")
        stages.append(
            {
                "role": "analysis",
                "status": "known",
                "transform": transform,
                "attribution": "livn.data.rsf.store",
            }
        )
    return stages, masked


def _quantize(uv: np.ndarray, gain_uV: float) -> np.ndarray:
    scaled = np.round(uv / gain_uV)
    if scaled.size:
        peak = float(np.max(np.abs(uv)))
        if not np.all(np.isfinite(scaled)):
            raise RSFError("the channel signal holds a NaN or an infinity")
        if scaled.max() > _INT16.max or scaled.min() < _INT16.min:
            raise RSFError(
                f"the channel signal peaks at {peak:.6g} µV, beyond the int16 range at "
                f"{gain_uV} µV/LSB; pass gain_uV >= {peak / _INT16.max:.6g}"
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


def _observe_document(io, operator, stimulates: bool, masked: set[str]) -> dict:
    from livn.io import Noise, PointSource

    medium = operator.medium
    electrical = operator.electrical
    plain_point_source = type(electrical) is PointSource and all(
        isinstance(t, Noise) for t in _transfers(io)
    )
    document: dict[str, Any] = {"kind": POINT_SOURCE if plain_point_source else LIVN_IO}
    coordinates = np.asarray(operator.electrode_coordinates, dtype=np.float64)
    if "electrode_positions" in masked:
        document["electrode_ids"] = [_id_value(c) for c in coordinates[:, 0]]
    else:
        document["electrode_coordinates"] = [
            [_id_value(row[0]), float(row[1]), float(row[2]), float(row[3])]
            for row in coordinates
        ]
    document["output_radius"] = float(operator.output_radius)
    document["detection_threshold"] = float(operator.detection_threshold)
    if plain_point_source:
        document["recording_gain"] = float(electrical.recording_gain)
        document["min_distance"] = float(electrical.min_distance)
    if stimulates:
        document["input_radius"] = float(operator.input_radius)
        document["electrode_radius"] = float(operator.electrode_radius)
        if plain_point_source:
            document["stimulation_gain"] = float(
                electrical.stimulation_gain(operator.electrode_radius)
            )
            document["culture_height"] = float(electrical.height)
    document["livn"] = {
        "medium": medium.serialize(),
        "transfers": [t.serialize() for t in _transfers(io)],
    }
    return document


def _probe(operator, row_ids: list[str]) -> dict:
    coordinates = np.asarray(operator.electrode_coordinates, dtype=np.float64)
    position = {_id_str(row[0]): (float(row[1]), float(row[2])) for row in coordinates}
    radius = float(operator.electrode_radius)
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


class RsfRecording:
    """A session opened lazily: the zarr arrays stay on disk and :meth:`read` touches only
    the chunks of the window it is asked for, so it is cheap to call from many threads
    (a read is a plain zarr slice). Pickling keeps the path and reopens on the other side.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._open()

    def _open(self) -> None:
        zarr = _zarr()

        path = self.path
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
        self.fs = float(attrs["sampling_frequency"])
        self._dt = 1000.0 / self.fs
        self.record = (attrs.get("annotations") or {}).get(RECORD_KEY) or {}
        tick = self.record.get("first_sample_tick")
        self._t0 = float(tick) * self._dt if tick is not None else 0.0

        self._keys = [str(k) for k in _read(root, "channel_ids")]
        self.channel_ids = _ids_array(self._keys)
        self._column = {key: i for i, key in enumerate(self._keys)}
        self._traces = _array_of(root, "traces_seg0")  # stays on disk
        self._gain = np.asarray(_read(root, "properties/gain_to_uV"), dtype=np.float64)
        self._offset = np.asarray(
            _read(root, "properties/offset_to_uV"), dtype=np.float64
        )

        documents = os.path.join(traces_path, RSF)
        self.simulation = None
        if os.path.isfile(os.path.join(documents, "simulation.json")):
            with open(
                os.path.join(documents, "simulation.json"), encoding="utf-8"
            ) as f:
                self.simulation = json.load(f)
        self.io = _read_operator(os.path.join(documents, "observe.json"))
        self._inputs_path = os.path.join(documents, INPUTS)
        self.bad_channels = np.zeros(0, dtype=np.int64)
        self.metadata = {
            "format": RSF,
            "path": path,
            "record": self.record,
            "simulation": self.simulation,
            "label": group_attrs.get("label"),
            "gain_uV": float(self._gain[0]) if len(self._gain) else None,
        }

    def __getstate__(self) -> dict:
        return {"path": self.path}

    def __setstate__(self, state: dict) -> None:
        self.path = state["path"]
        self._open()

    @property
    def segments(self) -> int:
        return 1

    def num_samples(self, segment: int = 0) -> int:
        self._segment(segment)
        return int(self._traces.shape[0])

    def read(self, segment, start, length, channels=None) -> np.ndarray:
        """Samples ``[start, start + length)`` of the given channel ids (all by default), as
        float32 ``(length, channels)`` in uV."""
        self._segment(segment)
        start, length = _window(start, length, int(self._traces.shape[0]))
        raw = np.asarray(self._traces[start : start + length, :])
        gain, offset = self._gain, self._offset
        if channels is not None:
            columns = self._columns(channels)
            raw, gain, offset = raw[:, columns], gain[columns], offset[columns]
        return (raw.astype(np.float64) * gain + offset).astype(np.float32)

    def spikes(self, segment: int = 0) -> Events:
        self._segment(segment)
        from livn.run import Events

        ids, samples = self._sorting()
        return Events(
            ids=ids,
            times=samples * self._dt,
            t0=self._t0,
            duration=self.num_samples() * self._dt,
        )

    def inputs(self, segment, start, length) -> np.ndarray:
        """The delivered input on the frame grid, float32 ``(length, channels)``."""
        self._segment(segment)
        if not os.path.isdir(self._inputs_path):
            raise Missing(
                f"{self.path} stores no input series: it was saved without `inputs` "
                "(or a policy as `stimulus`)"
            )
        values = _array_of(_zarr().open_group(self._inputs_path, mode="r"), "values")
        start, length = _window(start, length, int(values.shape[0]))
        return np.asarray(values[start : start + length, :], dtype=np.float32)

    def ground_truth(self, name: str, segment: int = 0):
        """``"spikes"`` (Events), ``"voltage"`` or ``"currents"`` (Series), whole segment."""
        self._segment(segment)
        if name == "spikes":
            return self.spikes()
        if name in ("current", "currents"):
            name = "currents"
        elif name != "voltage":
            raise Missing(f"an rsf session holds no ground truth {name!r}")
        series = self._truth_series(name, 0, self.num_samples())
        if series is None:
            raise Missing(f"{self.path} stores no ground_truth/{name}.zarr")
        return series

    def to_run(self, segment=0, start=0, length=None) -> Run:
        from livn.run import Run, Series

        self._segment(segment)
        n = self.num_samples()
        start, length = _window(start, n - start if length is None else length, n)
        dt, t0 = self._dt, self._t0 + start * self._dt

        potential = Series(
            ids=self.channel_ids,
            values=self.read(0, start, length).T,
            dt=dt,
            t0=t0,
        )
        run = Run(t0=t0, duration=length * dt, potential=potential)

        if _complete(os.path.join(self.path, "ground_truth", "spikes.zarr")):
            ids, samples = self._sorting()
            keep = (samples >= start) & (samples < start + length)
            run = run.add_spikes(ids[keep], (samples[keep] - start) * dt)

        for name, channel in (("voltage", "voltage"), ("currents", "current")):
            series = self._truth_series(name, start, length)
            if series is not None:
                run = run.add(
                    channel,
                    series.ids,
                    series.values,
                    dt=dt,
                    kind="series",
                    sections=series.sections,
                )
        return run

    def _segment(self, segment: int) -> None:
        if segment != 0:
            raise IndexError(f"an rsf session has one segment, not segment {segment}")

    def _columns(self, channels) -> np.ndarray:
        try:
            return np.asarray(
                [self._column[_id_str(c)] for c in np.asarray(channels).tolist()]
            )
        except KeyError as e:
            raise KeyError(f"{e.args[0]!r} is not a channel of {self.path}") from None

    def _sorting(self) -> tuple[np.ndarray, np.ndarray]:
        """``(unit ids, frames)`` of the ground-truth spikes, ordered by frame."""
        store = os.path.join(self.path, "ground_truth", "spikes.zarr")
        if not _complete(store):
            raise Missing(f"{self.path} stores no ground_truth/spikes.zarr")
        sorting = _zarr().open_group(store, mode="r")
        units = np.asarray(_read(sorting, "unit_ids"))
        sample_index = np.asarray(_read(sorting, "spikes/sample_index"), dtype=np.int64)
        unit_index = np.asarray(_read(sorting, "spikes/unit_index"), dtype=np.int64)
        order = np.argsort(sample_index, kind="stable")
        return _ids_array(units[unit_index[order]]), sample_index[order]

    def _truth_series(self, name: str, start: int, length: int) -> Series | None:
        from livn.run import Series

        store = os.path.join(self.path, "ground_truth", f"{name}.zarr")
        if not _complete(store):
            return None
        truth = _zarr().open_group(store, mode="r")
        ids = [str(k) for k in _read(truth, "channel_ids")]
        sections = None
        if _has(truth, "properties/section"):
            sections = np.asarray([str(s) for s in _read(truth, "properties/section")])
        if _has(truth, "properties/source"):
            ids = [str(s) for s in _read(truth, "properties/source")]
        array = _array_of(truth, "traces_seg0")
        samples = np.asarray(array[start : start + length, :], dtype=np.float32).T
        return Series(
            ids=_ids_array(ids),
            values=samples,
            dt=self._dt,
            t0=self._t0 + start * self._dt,
            sections=sections,
        )


def _window(start, length, n: int) -> tuple[int, int]:
    start, length = int(start), int(length)
    if start < 0 or length < 0 or start + length > n:
        raise IndexError(f"the window [{start}, {start + length}) is outside 0..{n}")
    return start, length


@dataclass(frozen=True)
class Session:
    """A session opened by :func:`load`. Nothing is read until it is asked for: ``run``
    materialises the whole session as a :class:`~livn.run.Run`; ``recording`` is the lazy
    view to read windows from."""

    recording: RsfRecording

    @cached_property
    def run(self) -> Run:
        return self.recording.to_run()

    @property
    def io(self) -> IO | None:
        return self.recording.io

    @property
    def record(self) -> dict:
        return self.recording.record

    @property
    def simulation(self) -> dict | None:
        return self.recording.simulation


def load(path: str | os.PathLike) -> Session:
    return Session(RsfRecording(path))


def _read(group, name: str) -> np.ndarray:
    zarr = _zarr()

    node = group.get(name)
    if not isinstance(node, zarr.Array):
        raise RSFError(f"{group.store_path}/{name} is not an array")
    return np.asarray(node[...])


def _array_of(group, name: str):
    """The zarr array itself, not its contents."""
    zarr = _zarr()

    node = group.get(name)
    if not isinstance(node, zarr.Array):
        raise RSFError(f"{group.store_path}/{name} is not an array")
    return node


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
    from livn.io import MEA, ComposedIO, Medium, PointSource

    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    if (
        document.get("kind") not in (POINT_SOURCE, LIVN_IO)
        or "electrode_coordinates" not in document
    ):
        return None
    coordinates = document["electrode_coordinates"]
    if not all(isinstance(row[0], int) for row in coordinates):
        return None  # livn's MEA numbers its electrodes
    livn = document.get("livn")
    transfers = []
    if livn is not None:
        medium = Medium(**{k: v for k, v in livn["medium"].items() if k != "cls"})
        transfers = livn["transfers"]
    else:
        radius = document.get("electrode_radius", 15.0)
        height = document.get("culture_height", 50.0)
        sigma = 1e6 / (
            4.0 * math.pi * document["recording_gain"] * document["min_distance"]
        )
        rho = None
        if "stimulation_gain" in document:
            h_eff = (radius**2 + height**2) ** 0.5
            rho = document["stimulation_gain"] * 4.0 * math.pi * h_eff / 1000.0
        medium = Medium(
            electrical=PointSource(
                sigma_s_per_m=sigma,
                height=height,
                min_distance=document["min_distance"],
                stimulation_resistivity_ohm_m=rho,
            )
        )
    kwargs: dict[str, Any] = {
        "electrode_coordinates": np.asarray(coordinates, dtype=np.float64),
        "output_radius": document["output_radius"],
        "medium": medium,
        "detection_threshold": document.get("detection_threshold", 0.0),
    }
    if "input_radius" in document:
        kwargs["input_radius"] = document["input_radius"]
    if "electrode_radius" in document:
        kwargs["electrode_radius"] = document["electrode_radius"]
    mea = MEA(**kwargs)
    if not transfers:
        return mea
    return ComposedIO(None, [mea, *transfers], medium=medium)
