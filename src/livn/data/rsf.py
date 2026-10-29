from __future__ import annotations

import itertools
import json
import math
import os
import re
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

DEFAULT_GAIN_UV = 0.195
PROBEINTERFACE_VERSION = "0.3.2"

MASK_FIELDS = ("electrode_positions", "source_positions")

TRACE_DTYPES = ("int16", "int32")
NOTE_KINDS = ("label", "description", "note")
CALIBRATION = "stimulus_calibration"

CONDITION = "condition"
CHANNEL = "channel"
CHANNELS_DOCUMENT = "channels.json"
# the kinds of channel whose bits can mark a stimulus, the digital input first
MARKING_KINDS = ("digital_in", "stim_sideband")
CONDITIONS = "conditions"
CONDITION_COLUMNS = ("frame", "quantity", "item", "value", "units", "basis")


PROBE_FRAME = "probe"
RECORD_MEMBERS = frozenset(
    (
        "started_at",
        "kind",
        "channels",
        "marker_channel",
        "marker_edge",
        "timing",
        "placement_lag_ms",
        "origin",
        "index",
    )
)

CALIBRATION_FIT_TOLERANCE = 5.0

CONDITION_BASIS_RANK = {"measured": 2, "setpoint": 1}
WAVELENGTH_BASES = ("measured", "stated", "device_default")

TIMING_RANK = {
    "marker": 4,
    "from_marker": 3,
    "device_clock": 2,
    "schedule": 1,
    "simulated": 1,
}
STIM_COLUMNS = ("frame", "channel", "value", "program", "event", "timing")
PLANNED_COLUMNS = ("t_ms", "channel", "value", "event", "load")

PROGRAM_KEYS = (
    "inputs",
    "inputs_dt",
    "policy",
    "start",
    "start_frame",
    "stop",
    "device",
    "channels",
    "origin",
    "members",
    "started_at",
)
RECORD_MEMBERS = (
    "started_at",
    "kind",
    "channels",
    "marker_channel",
    "marker_edge",
    "timing",
    "placement_lag_ms",
    "origin",
    "planned",
    "index",
)
CHANNEL_SOURCES = {"light": "livn PatternedLight", "electrode": "livn MEA"}

EPOCH = "epoch"
EPOCH_KINDS = ("protocol", "block", "baseline", "spontaneous", "spot_mapping")
EPOCH_KEYS = (
    "key",
    "item",
    "frame",
    "end_frame",
    "start",
    "end",
    "kind",
    "text",
    "parent",
    "programs",
    "origin",
    "author",
)
NOTE_KEYS = ("text", "frame", "author", "at", "item")
PREP_KEYS = ("id", "label", "kind", "species", "age", "expression", "models")

CONNECTIONS = "connections"
STIMULUS_COUPLING = "stimulus_coupling"
CONNECTION_WEIGHT_UNITS = "1"
CONNECTION_WEIGHT_REFERENCE = (
    "the synapse's unitary conductance, which the weight scales"
)
OPSIN_CONDUCTANCES = {
    "RhO3c": ("g0_pS", "pS", 1.0),
    "ChR2H134R": ("gmax", "S/cm2", 4e-4),
}

PRODUCER_NAME = re.compile(r"[a-z0-9_-]+")
DOCUMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json")
HEX_ITEM = re.compile(r"[0-9a-f]{32}")


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
    dtype: str = "int16",
    label: str | None = None,
    author: str | None = None,
    recording=None,
    conditions=None,
    programs: Sequence[Mapping] | None = None,
    prep: Mapping | None = None,
    description: str | None = None,
    notes: Sequence[Mapping] | None = None,
    epochs: Sequence[Mapping] | None = None,
    populations: bool = False,
    expression: Mapping | bool | None = None,
    connections: Mapping | bool = False,
    stimulus_coupling: Mapping | bool = False,
    ext: Mapping[str, Mapping[str, Any]] | None = None,
    zone: str | bool | None = None,
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
            array is the delivered input unless ``inputs`` is given); ``None`` for
            spontaneous activity
        inputs: The delivered per-channel input, ``(timestep, n_channels)`` over the stimulating
            IO's channels in its input units, starting at the run's ``t0``. It is held on the
            traces' frame grid (zero-order hold; zero past its end).
        inputs_dt: Sampling interval of ``inputs`` in ms; defaults to the policy's ``dt``
        gain_uV: uV per LSB of the stored traces; a sample outside ``dtype`` is an error
            (livn never clips, so every row's ``clipped_samples`` is 0)
        dtype: ``"int16"`` or ``"int32"``, the integer type of ``traces_seg0``
        label: Free text for people; defaults to the system's name.
        author: Who made the session. ``None`` writes no ``author``.
        conditions: Further :class:`livn.conditions.Conditions` of the projection
        programs: The stimulus as several programs, one stimulation record each, in place
            of ``stimulus`` and ``inputs``. Each is a mapping of

            - ``inputs`` ``(timestep, n)`` in the device's input units, over all of the
              device's channels or over ``channels``, and ``inputs_dt`` in ms; or
              ``policy``, a channel-space :class:`~livn.policy.Policy` whose rendering and
              ``dt`` they default to;
            - ``start`` (ms) or ``start_frame``, where it begins in the session (0);
            - ``stop`` (ms after its start): a program stopped early closes with its plan's
              last event at that frame, and its later edges are not delivered;
            - ``device``, its device's index among the stimulating devices (0);
            - ``channels``, the device channel ids it drives (by default those it sets
              non-zero): the record's channels, and no others;
            - ``origin``, what generated it (``{"generator", "software", "params",
              "seed"}``), by default the policy's;
            - ``members``, ``{"<producer>": {...}}``, other producers' members of the record;
            - ``started_at``, the record's timestamp, by default the session's start plus
              the program's start.

            Each record states its plan (``planned``), its channels by site ``name`` and
            ``source``, and its delivered rows; a channel is driven by one program at a
            time
        prep: The simulated preparation, ``{"id", "label", "kind", "species", "age",
            "expression", "models"}``, every member optional: ``id`` is minted when not
            given, ``models`` is the physical preparation it stands for. ``age`` is at
            frame 0; an ``expression`` entry may carry an ``age`` of its own
        description: The session's description, its first ``description`` version
        notes: ``[{"text", "frame", "author", "at", "item"}]``, each a ``note`` version
        epochs: The session's named spans, ``[{"key" or "item", "frame" or "start" (ms),
            "end_frame" or "end" (ms), "kind", "text", "parent", "programs", "origin",
            "author"}]``. ``key`` names an epoch within this call and is minted into an
            ``item``; a given ``item`` (32 hex digits) is kept, so one run's epoch can be
            written into several sessions. ``parent`` is a key or an item, ``programs``
            indices into ``programs`` (``None`` when not said, ``[]`` for none)
        populations: Write each source's population into ``simulation.json``
        expression: Each source's opsin expression in ``simulation.json``: ``True`` takes
            it from the model's opsin configuration (its conductance, ``0`` in a population
            without the opsin), or ``{construct: {"units", "values", "reference"}}`` with
            ``values`` parallel to the sources or keyed by gid (``0`` for one not named)
        connections: Write ``ground_truth/connections.zarr``: ``True`` from
            ``env.connections()`` (each weight the one in force, its strength applied, as a
            multiple of the synapse's unitary conductance; the physical delay), or
            ``{"pre", "post", "weight", "units", "reference", "sign", "delay", "receptor",
            "inputs"}`` as columns, ``reference`` saying what a weight in units ``"1"`` is of
        stimulus_coupling: Write ``ground_truth/stimulus_coupling.zarr``, what each record
            channel at full scale puts on each source: ``True`` from the stimulating
            devices, or ``{"gain", "units", "channel_ids", "source_ids"}``. Not written
            under a mask on ``source_positions``
        ext: ``{"<producer>": {"<name>.json": document}}``, other producers' documents,
            written to ``rsf/ext/<producer>/``
        zone: The IANA zone the session was made in, or ``True`` for the host's when it
            can be told; ``None`` writes none

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
        dtype=dtype,
        label=label,
        author=author,
        recording=recording,
        conditions=conditions,
        programs=programs,
        prep=prep,
        description=description,
        notes=notes,
        epochs=epochs,
        populations=populations,
        expression=expression,
        connections=connections,
        stimulus_coupling=stimulus_coupling,
        ext=ext,
        zone=zone,
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
    dtype,
    label,
    author=None,
    recording=None,
    conditions=None,
    programs=None,
    prep=None,
    description=None,
    notes=None,
    epochs=None,
    populations=False,
    expression=None,
    connections=False,
    stimulus_coupling=False,
    ext=None,
    zone=None,
):
    from livn.io.recording import as_composition

    path = os.fspath(path)
    _check_location(path)
    _check_not_batched(run)
    _check_author(author)

    simulator = _simulator_name(env)
    io, operator, stimulates = _observation(env)
    recording = io.recording if recording is None else as_composition(recording)
    channel_ids = [_id_str(c) for c in np.asarray(operator.electrode_coordinates)[:, 0]]

    gain_uV = float(gain_uV)
    if not gain_uV > 0 or not math.isfinite(gain_uV):
        raise RSFError(
            f"gain_uV must be a positive number of µV per LSB, not {gain_uV}"
        )

    zone = _zone(zone)
    prep = None if prep is None else _simulated_prep(prep)
    extra_documents = _ext_documents(ext)
    if description is not None and not isinstance(description, str):
        raise RSFError(f"description is text, not {description!r}")

    signal_series = run.channels.get("current")
    planned_programs = None
    if programs is not None:
        if stimulus is not None or inputs is not None:
            raise RSFError(
                "programs= states the whole stimulus; pass it instead of stimulus= and "
                "inputs=, with each program's policy or inputs"
            )
        dt_signal, n_signal = _signal_grid(run, signal_series)
        planned_programs, series = _plan_programs(io, programs, dt_signal, n_signal)
        delivered = {"values": series, "dt": dt_signal, "units": None}
    else:
        stimulus, delivered = _stimulus(stimulus, inputs, inputs_dt)

    # the channel signal, (n_channels, T) in uV
    projected = None
    through_stack = False
    if signal_series is not None and signal_series.values is not None:
        dt, t0 = float(signal_series.dt), float(signal_series.t0)
        (uv, projected), row_ids = (
            _project(
                run,
                env,
                io,
                dt,
                recording=recording,
                delivered=delivered,
                conditions=conditions,
            ),
            list(channel_ids),
        )
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

    stack_stages = _stack_stages(recording) if through_stack else []
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
    traces = _quantize(uv, gain_uV, dtype)

    if delivered is not None and planned_programs is None:
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
    system: dict[str, Any] = {"simulator": simulator, "spec": spec}
    if prep is not None:
        system["prep"] = prep
    record = {
        "format": FORMAT,
        "recording_id": recording_id,
        "env": {
            "substrate": "simulated",
            "system": {"simulation": system},
            "model": model,
            "seed": seed,
        },
        "space": "channel",
        "rows": {key: [observe_stage, *stack_stages, *stages] for key in row_ids},
        "started_at": started_at,
        "first_sample_tick": int(tick0),
        "group": group,
    }

    observe = _observe_document(io, operator, stimulates, masked, recording)
    source_document: dict[str, Any] = {"ids": sources}
    if "source_positions" not in masked:
        source_document["coordinates_um"] = _source_coordinates(env, sources)
    if populations:
        source_document["populations"] = _source_populations(env, sources)
    if expression is not None and expression is not False:
        source_document["expression"] = _source_expression(env, sources, expression)
    simulation = {
        "simulator": {"name": simulator, "version": _version()},
        "spec": spec,
        "model": model,
        "seed": seed,
        "sources": source_document,
        "stimulus": None if stimulus is None else _plain(stimulus),
    }
    if prep is not None:
        simulation["prep"] = prep

    channel_devices: dict[str, tuple] = {}
    planned_tables: list[dict | None] = []
    record_of: dict[int, int] = {}
    if planned_programs is None:
        stimulation, stim_events = _programs(
            io, delivered, stimulus, masked, started_at, channel_devices
        )
    else:
        stimulation, stim_events, planned_tables, record_of = _program_records(
            planned_programs, masked, started_at, dt, channel_devices
        )
    cited = [i for i, record in enumerate(stimulation) if "kind" in record]
    if cited:
        cite = {"record": cited[0]} if len(cited) == 1 else {"records": cited}
        simulation["stimulus"] = {**(simulation["stimulus"] or {}), **cite}

    run_extras: dict[str, Any] = {}
    if projected is not None:
        run_extras["trace_sections"] = projected["contributing"]
        if projected["unplaced"]:
            run_extras["unplaced_sections"] = projected["unplaced"]
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
    if author is not None:
        group_attrs["author"] = author
    if zone is not None:
        group_attrs["zone"] = zone

    probe = None
    if "electrode_positions" not in masked:
        probe = _probe(operator, row_ids)

    conditions_table = _simulated_conditions(env, simulator)

    tables: dict[str, dict] = {}
    if connections is not None and connections is not False:
        tables[CONNECTIONS] = _connection_table(env, sources, connections)
    if stimulus_coupling is not None and stimulus_coupling is not False:
        if "source_positions" in masked:
            warnings.warn(
                "stimulus_coupling.zarr is not written: it would reveal the source "
                "positions the mask withholds",
                stacklevel=3,
            )
        else:
            tables[STIMULUS_COUPLING] = _coupling_table(
                env, sources, stimulation, channel_devices, stimulus_coupling
            )

    stores = []
    if spikes is not None:
        stores.append("spikes")
    stores.extend(truth)

    ext = {"simulation.json": run_extras}
    if observe["kind"] == POINT_SOURCE:
        ext["observe.json"] = observe.pop("livn")

    instant = _instant(started_at)
    note_versions: list[tuple[str, dict]] = []
    if description is not None:
        note_versions.append(
            ("description", _note_version("session", description, instant, zone))
        )
    note_versions += [
        ("note", version) for version in _notes(notes, n_frames, instant, zone)
    ]
    note_versions += [
        (EPOCH, version)
        for version in _epoch_versions(
            epochs, n_frames, dt, record_of, stimulation, instant, zone
        )
    ]

    for name, document in (
        ("observe.json", observe),
        ("simulation.json", simulation),
        ("the record", record),
        *(("a stimulation record", r) for r in stimulation),
        *ext.items(),
        *((f"a {kind} version", v) for kind, v in note_versions),
    ):
        _json(document, name)

    return {
        "path": path,
        "fs": fs,
        "n_frames": n_frames,
        "row_ids": row_ids,
        "traces": traces,
        "gain_uV": gain_uV,
        "label_version": None
        if label is None
        else _note_version("session", str(label), instant, zone),
        "notes": note_versions,
        "ext": ext,
        "producer_ext": extra_documents,
        "tick0": int(tick0),
        "spikes": spikes,
        "sources": sources,
        "truth": truth,
        "tables": tables,
        "stimulation": stimulation,
        "planned": planned_tables,
        "stim_events": stim_events,
        "record": record,
        "observe": observe,
        "simulation": simulation,
        "group_attrs": group_attrs,
        "probe": probe,
        "conditions": conditions_table,
        "stores": [
            {
                "path": f"ground_truth/{name}.zarr",
                "space": "source",
                "written_by": PRODUCER,
            }
            for name in stores
        ]
        + [
            {"path": f"ground_truth/{name}.zarr", "space": None, "written_by": PRODUCER}
            for name in tables
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

    if plan["spikes"] is not None or plan["truth"] or plan["tables"]:
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

    for name, table in plan["tables"].items():
        store = os.path.join(path, "ground_truth", f"{name}.zarr")
        _write_table_store(zarr, store, name, table)
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
    _array(root, "properties/clipped_samples", np.zeros(n_rows, dtype=np.uint64))

    rsf = os.path.join(store, RSF)
    # each table its own hierarchy, so rsf/zarr.json is not created before it is complete
    anchors = zarr.open_group(os.path.join(rsf, "tick_anchors"), mode="w")
    _array(anchors, "frame", np.asarray([0], dtype=np.uint64))
    _array(anchors, "tick", np.asarray([plan["tick0"]], dtype=np.uint64))
    markers = zarr.open_group(os.path.join(rsf, "stim_markers"), mode="w")
    _array(markers, "tick", np.zeros(0, dtype=np.uint64))
    _strings(markers, "channel", [])
    events = zarr.open_group(os.path.join(rsf, "stim_events"), mode="w")
    for name in STIM_COLUMNS:
        column = plan["stim_events"][name]
        if name in ("channel", "timing"):
            _strings(events, name, column)
        else:
            _array(events, name, column)
    if plan["stimulation"]:
        _group_node(os.path.join(rsf, "stimulation"), {})
        for index, attribute in enumerate(plan["stimulation"]):
            node = os.path.join(rsf, "stimulation", f"{index:04d}")
            _group_node(node, {"stimulation": attribute})
            planned = plan["planned"][index] if plan["planned"] else None
            if planned is not None:
                table = zarr.open_group(os.path.join(node, "planned"), mode="w")
                _array(table, "t_ms", np.asarray(planned["t_ms"], dtype=np.float64))
                _strings(table, "channel", planned["channel"])
                _array(table, "value", np.asarray(planned["value"], dtype=np.float32))
                _array(table, "event", np.asarray(planned["event"], dtype=np.uint32))
                _array(table, "load", np.asarray(planned["load"], dtype=np.uint32))
    dropped = zarr.open_group(os.path.join(rsf, "dropped_spans"), mode="w")
    _array(dropped, "frame_start", np.zeros(0, dtype=np.uint64))
    _array(dropped, "n_frames", np.zeros(0, dtype=np.uint32))
    _strings(dropped, "channel", [])

    gaps = zarr.open_group(os.path.join(rsf, "gaps"), mode="w")
    _array(gaps, "frame", np.zeros(0, dtype=np.uint64))
    _array(gaps, "n_frames", np.zeros(0, dtype=np.uint64))

    if plan["conditions"] is not None:
        rows = plan["conditions"]["rows"]
        table = zarr.open_group(
            os.path.join(rsf, CONDITIONS),
            mode="w",
            attributes={"items": plan["conditions"]["items"]},
        )
        _array(table, "frame", np.asarray([r[0] for r in rows], dtype=np.uint64))
        _strings(table, "quantity", [r[1] for r in rows])
        _strings(table, "item", [r[2] for r in rows])
        _array(table, "value", np.asarray([r[3] for r in rows], dtype=np.float64))
        _strings(table, "units", [r[4] for r in rows])
        _strings(table, "basis", [r[5] for r in rows])

    _write_json(os.path.join(rsf, "observe.json"), plan["observe"])
    _write_json(os.path.join(rsf, "simulation.json"), plan["simulation"])
    _write_json(os.path.join(rsf, "stores.json"), plan["stores"])
    os.makedirs(os.path.join(rsf, "ext", PRODUCER), exist_ok=True)
    for name, document in plan["ext"].items():
        _write_json(os.path.join(rsf, "ext", PRODUCER, name), document)
    for producer, documents in plan["producer_ext"].items():
        os.makedirs(os.path.join(rsf, "ext", producer), exist_ok=True)
        for name, document in documents.items():
            _write_json(os.path.join(rsf, "ext", producer, name), document)
    if plan["label_version"] is not None:
        _add_note(rsf, "label", plan["label_version"])
    for kind, version in plan["notes"]:
        _add_note(rsf, kind, version)
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


def _check_author(author) -> None:
    if author is None:
        return
    if not isinstance(author, str) or not author.strip():
        raise RSFError(
            f"author is who made the session, as they want to be named, not {author!r}; "
            "pass None to leave it unknown"
        )
    if "@" in author:
        raise RSFError(f"author {author!r} looks like an email address")


def _simulator_name(env) -> str:
    match = re.match(r"^livn\.backend\.([a-z0-9_]+)", type(env).__module__)
    if match is None:
        raise RSFError(
            f"cannot name the simulator of a {type(env).__module__}.{type(env).__qualname__}; "
            "a session is saved with the env of a livn backend"
        )
    return f"livn_{match.group(1)}"


def _simulated_conditions(env, simulator: str) -> dict | None:
    celsius = getattr(env, "_celsius", None)
    if not isinstance(celsius, int | float) or isinstance(celsius, bool):
        return None
    if not math.isfinite(float(celsius)):
        return None
    return {
        "rows": [(0, "temperature", "network", float(celsius), "C", "setpoint")],
        "items": {"network": {"source": simulator}},
    }


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


def _transfers(recording) -> tuple:
    from livn.io.recording import Chain

    return tuple(recording.transfers) if isinstance(recording, Chain) else ()


def _stack_stages(recording) -> list[dict]:
    stages = []
    for transfer in _transfers(recording):
        document = transfer.serialize()
        if document["cls"] == "livn.io.Noise":
            transform = {
                "noise": {
                    "distribution": "gaussian",
                    "params": {"sigma_uv": float(document["sigma_uV"])},
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


def _livn_scale(kind: str, full_scale: float) -> float:
    """livn's quantity for an RSF stimulus value of 1.0 on a channel of ``kind``."""
    return -float(full_scale) if kind == "electrode" else float(full_scale)


def _stimulating_devices(io) -> list:
    from livn.io import ComposedIO, InputDevices

    device = io.inputs if isinstance(io, ComposedIO) else io
    return list(device.devices) if isinstance(device, InputDevices) else [device]


def _no_events() -> dict:
    return {
        "frame": np.zeros(0, dtype=np.uint64),
        "channel": [],
        "value": np.zeros(0, dtype=np.float32),
        "program": np.zeros(0, dtype=np.uint32),
        "event": np.zeros(0, dtype=np.uint32),
        "timing": [],
    }


def _programs(
    io, delivered, spec, masked: set[str], started_at: str, channel_devices=None
):
    """``(records, stim_events)``: one stimulation record per stimulating device, in device
    order (a ``ComposedIO``'s ``InputDevices`` are several), and the rows of every program.

    A device RSF has no channel object for (a ``LightArray``'s fibres command source power,
    not a fraction of an irradiance at the sample) gets a record without the neutral members.
    """
    from livn.io import MEA, InputDevices, PatternedLight

    if delivered is None:
        return [], _no_events()
    devices = _stimulating_devices(io)
    values = delivered["values"]
    parts = InputDevices._of(devices).split(values) if len(devices) > 1 else [values]
    records, tables = [], []
    for program, (device, part) in enumerate(zip(devices, parts, strict=True)):
        livn = {
            "channels": [_id_value(c) for c in np.asarray(device.channel_ids).ravel()],
            "units": delivered["units"]
            if len(devices) == 1
            else getattr(device, "input_units", None),
            "source_dt": delivered["source_dt"],
            "stimulus": None if spec is None else _plain(spec),
        }
        record: dict[str, Any] = {"started_at": started_at}
        if isinstance(device, MEA):
            kind = "electrode"
        elif isinstance(device, PatternedLight):
            kind = "light"
        else:
            records.append({**record, PRODUCER: livn})
            continue
        prefix = kind if len(devices) == 1 else f"{kind}/{program}"
        channels, fractions = _channels(device, kind, prefix, part, masked)
        if channel_devices is not None:
            for column, channel in enumerate(channels):
                channel_devices[channel["id"]] = (device, column, kind)
        records.append(
            {
                **record,
                "kind": kind,
                "channels": channels,
                "marker_channel": None,
                "marker_edge": None,
                "timing": "simulated",
                "placement_lag_ms": 0,
                PRODUCER: livn,
            }
        )
        tables.append(_events(program, [c["id"] for c in channels], fractions))
    if not tables:
        return records, _no_events()
    return records, {
        name: (
            [v for t in tables for v in t[name]]
            if name in ("channel", "timing")
            else np.concatenate([t[name] for t in tables])
        )
        for name in STIM_COLUMNS
    }


def _channels(device, kind: str, prefix: str, values: np.ndarray, masked: set[str]):
    """The channel objects of one device, and its commands as fractions of full scale."""
    unplaced = "electrode_positions" in masked
    channels = []
    if kind == "light":
        scale = float(device.irradiance)
        if not scale > 0 or not math.isfinite(scale):
            raise RSFError(
                f"the PatternedLight's irradiance is {scale}; a session states it as the "
                "channels' full scale, which is a positive number of mW/mm2"
            )
        spots = np.asarray(device.spots, dtype=np.float64)
        labels = device.labels or [None] * len(spots)
        for row, label in zip(spots, labels, strict=True):
            channels.append(
                {
                    "id": f"{prefix}/{_id_str(row[0])}",
                    "shape": {"kind": "unplaced"}
                    if unplaced
                    else {
                        "kind": "disc",
                        "x": float(row[1]),
                        "y": float(row[2]),
                        "radius": float(row[3]),
                    },
                    "full_scale": {"units": "mW/mm2", "value": scale},
                    "bipolar": False,
                    "wavelength_nm": float(device.wavelength),
                    "label": label,
                }
            )
        low, knob = 0.0, "the PatternedLight's irradiance"
    else:
        scale = float(device.max_current_uA)
        for row in np.asarray(device.electrode_coordinates, dtype=np.float64):
            contact = _id_value(row[0])
            channels.append(
                {
                    "id": f"{prefix}/{_id_str(row[0])}",
                    "shape": {"kind": "unplaced"}
                    if unplaced
                    else {
                        "kind": "contact",
                        "contact_id": contact,
                        "x": float(row[1]),
                        "y": float(row[2]),
                    },
                    "full_scale": {"units": "uA", "value": scale},
                    "bipolar": True,
                    "wavelength_nm": None,
                    "label": None,
                }
            )
        low, knob = -1.0, "MEA(max_current_uA=...)"

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RSFError("the delivered input holds a NaN or an infinity")
    # a PatternedLight's command is already a fraction of its irradiance, the full scale
    fractions = values if kind == "light" else values / _livn_scale(kind, scale)
    if fractions.size and (fractions.max() > 1 + 1e-6 or fractions.min() < low - 1e-6):
        peak = float(np.abs(values).max())
        raise RSFError(
            f"the delivered {kind} input reaches {peak:.6g} {device.input_units}, beyond "
            f"its full scale {scale:g} ({'mW/mm2' if kind == 'light' else 'uA'}); a "
            f"session states it as a fraction of that, so raise {knob}"
        )
    return channels, np.clip(fractions, low, 1.0).astype(np.float32)


def _events(program: int, ids: list[str], fractions: np.ndarray) -> dict:
    n_frames, n = fractions.shape
    previous = np.zeros_like(fractions)
    previous[1:] = fractions[:-1]
    frames, columns = np.nonzero(
        fractions != previous
    )  # ordered by frame, then channel
    values = fractions[frames, columns]
    if n_frames and fractions[-1].any():
        frames = np.concatenate([frames, np.full(n, n_frames)])
        columns = np.concatenate([columns, np.arange(n)])
        values = np.concatenate([values, np.zeros(n, dtype=np.float32)])
    edges = np.unique(frames)
    return {
        "frame": frames.astype(np.uint64),
        "channel": [ids[c] for c in columns.tolist()],
        "value": values.astype(np.float32),
        "program": np.full(len(frames), program, dtype=np.uint32),
        "event": np.searchsorted(edges, frames).astype(np.uint32),
        "timing": ["simulated"] * len(frames),
    }


def _signal_grid(run, signal_series) -> tuple[float, int]:
    """The frame interval and count of the channel signal, before it is projected."""
    if signal_series is not None and signal_series.values is not None:
        return float(signal_series.dt), int(np.shape(signal_series.values)[-1])
    potential = getattr(run, "potential", None)
    if potential is not None:
        return float(potential.dt), int(np.shape(potential.values)[1])
    raise RSFError(
        "a session is a channel-space recording, and this run holds nothing to put in "
        f"traces.zarr: its channels are {sorted(run.channels) or 'none'}. Call "
        "env.record_membrane_current() before env.run(...) so the MEA signal can be "
        "projected from the membrane currents."
    )


def _nearest(x) -> np.ndarray:
    """The nearest integer, halves up, as the format rounds a time onto frames."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5 + 1e-9).astype(np.int64)


def _site_names(device, kind: str) -> list[str]:
    """Each channel's site name: a spot's label (else its id), an electrode's contact."""
    ids = [_id_str(c) for c in np.asarray(device.channel_ids).ravel()]
    labels = getattr(device, "labels", None) if kind == "light" else None
    names = list(labels) if labels else ids
    if len(set(names)) != len(names):
        raise RSFError(
            f"the {kind} channels' names {names} repeat; a session names each stimulus "
            "site once (give the PatternedLight distinct labels)"
        )
    return names


def _program_fractions(device, kind: str, values: np.ndarray, what: str) -> np.ndarray:
    """A program's commands as fractions of each channel's full scale."""
    if kind == "light":
        scale = float(device.irradiance)
        if not scale > 0 or not math.isfinite(scale):
            raise RSFError(
                f"the PatternedLight's irradiance is {scale}; a session states it as the "
                "channels' full scale, which is a positive number of mW/mm2"
            )
        fractions, low, knob = values, 0.0, "the PatternedLight's irradiance"
        units = "mW/mm2"
    else:
        scale = float(device.max_current_uA)
        fractions = values / _livn_scale(kind, scale)
        low, knob, units = -1.0, "MEA(max_current_uA=...)", "uA"
    if fractions.size and (fractions.max() > 1 + 1e-6 or fractions.min() < low - 1e-6):
        peak = float(np.abs(values).max())
        raise RSFError(
            f"{what} reaches {peak:.6g} {device.input_units}, beyond its full scale "
            f"{scale:g} ({units}); a session states it as a fraction of that, so raise "
            f"{knob}"
        )
    return np.clip(fractions, low, 1.0).astype(np.float32)


def _planned_edges(fractions: np.ndarray, dt_in: float, ids: list[str]) -> dict:
    """What a program was to deliver, ``planned``'s five columns: an event per row of the
    rendering where a channel changes, and last an event that sets every channel to 0."""
    n_rows, n = fractions.shape
    previous = np.zeros_like(fractions)
    previous[1:] = fractions[:-1]
    rows, columns = np.nonzero(fractions != previous)  # by row, then channel
    values = fractions[rows, columns]
    if len(rows):
        last = rows[-1]
        closing = (rows == last).sum() == n and not fractions[last].any()
    else:
        closing = False
    if not closing:
        rows = np.concatenate([rows, np.full(n, n_rows)])
        columns = np.concatenate([columns, np.arange(n)])
        values = np.concatenate([values, np.zeros(n, dtype=np.float32)])
    event = np.searchsorted(np.unique(rows), rows)
    return {
        "t_ms": rows.astype(np.float64) * float(dt_in),
        "channel": [ids[c] for c in columns.tolist()],
        "value": values.astype(np.float32),
        "event": event.astype(np.uint32),
        "load": np.zeros(len(rows), dtype=np.uint32),
    }


def _delivered_rows(
    planned: dict, start_frame: int, dt: float, n_frames: int, stop
) -> dict:
    """A plan placed on the session's frames from ``start_frame``: every row at the
    aligned first event's frame plus its offset, rounded; a program the session ends or
    ``stop`` cuts closes with its plan's last event there and delivers nothing after."""
    t = np.asarray(planned["t_ms"], dtype=np.float64)
    event = np.asarray(planned["event"], dtype=np.int64)
    first = int(np.argmin(event))
    frame0 = start_frame + int(_nearest(t[first] / dt))
    frames = frame0 + _nearest((t - t[first]) / dt)
    last = int(event.max())
    end = n_frames
    if stop is not None:
        stop = float(stop)
        if not stop >= 0 or not math.isfinite(stop):
            raise RSFError(f"a program's stop is ms after its start, not {stop}")
        end = min(end, start_frame + int(_nearest(stop / dt)))
    closing = event == last
    truncated = bool(np.any(~closing & (frames >= end)) or frames[closing].max() > end)
    keep = np.ones(len(t), dtype=bool)
    if truncated:
        keep = closing | (frames < end)
        frames = np.where(closing, end, frames)
    return {
        "frame": frames[keep].astype(np.uint64),
        "channel": [c for c, k in zip(planned["channel"], keep, strict=True) if k],
        "value": np.asarray(planned["value"], dtype=np.float32)[keep],
        "event": event[keep].astype(np.uint32),
        "stopped": truncated,
    }


def _policy_origin(policy) -> dict:
    seed = getattr(policy, "seed", None)
    return {
        "generator": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "software": {"name": PRODUCER, "version": _version()},
        "params": _plain(policy.serialize()),
        "seed": int(seed) if isinstance(seed, int | np.integer) else None,
    }


def _check_origin(origin, what: str) -> dict:
    if not isinstance(origin, Mapping):
        raise RSFError(f"{what}'s origin is an object, not {origin!r}")
    origin = _plain(origin)
    generator = origin.get("generator")
    software = origin.get("software")
    if not isinstance(generator, str) or not generator:
        raise RSFError(f"{what}'s origin names its generator, a non-empty string")
    if not isinstance(software, Mapping) or not isinstance(software.get("name"), str):
        raise RSFError(
            f"{what}'s origin says which software generated it: "
            '{"software": {"name": ..., "version": ...}}'
        )
    version = software.get("version")
    if version is not None and not isinstance(version, str):
        raise RSFError(f"{what}'s origin software version is a string or None")
    seed = origin.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise RSFError(f"{what}'s origin seed is an integer or None, not {seed!r}")
    out = {
        "generator": generator,
        "software": {"name": software["name"], "version": version},
        "params": origin.get("params"),
        "seed": seed,
    }
    _json(out, f"{what}'s origin")
    return out


def _check_producer(name, what: str) -> str:
    if not isinstance(name, str) or not PRODUCER_NAME.fullmatch(name) or name == RSF:
        raise RSFError(
            f"{what} {name!r} is not a producer name ([a-z0-9_-]+, never 'rsf')"
        )
    if name == PRODUCER:
        raise RSFError(f"{what} {name!r} is livn's own; livn writes its own members")
    return name


def _check_members(members, what: str) -> dict:
    if members is None:
        return {}
    if not isinstance(members, Mapping):
        raise RSFError(f"{what}'s members are {{producer: {{...}}}}, not {members!r}")
    out = {}
    for name, member in members.items():
        if name in RECORD_MEMBERS:
            raise RSFError(
                f"{what}'s member {name!r} is the format's, not a producer's"
            )
        _check_producer(name, f"{what}'s member")
        if not isinstance(member, Mapping):
            raise RSFError(f"{what}'s member {name!r} is an object")
        out[name] = _plain(member)
    return out


def _plan_programs(io, programs, dt: float, n_frames: int):
    """Each program's plan and delivered rows, and the delivered input on the frame grid
    over every stimulating channel, in the devices' input units (for the projection)."""
    from livn.io import MEA, PatternedLight
    from livn.policy import Policy

    if isinstance(programs, Mapping) or not isinstance(programs, Sequence):
        raise RSFError("programs is a list of programs, each a mapping")
    if not programs:
        raise RSFError("programs is empty; pass None for a session without stimulus")
    devices = _stimulating_devices(io)
    offsets = np.cumsum([0, *(int(d.num_channels) for d in devices)])
    series = np.zeros((n_frames, int(offsets[-1])), dtype=np.float32)
    holding: dict[int, list] = {}
    plans = []
    for k, entry in enumerate(programs):
        what = f"program {k}"
        if not isinstance(entry, Mapping):
            raise RSFError(f"{what} is a mapping, not {entry!r}")
        unknown = sorted(set(entry) - set(PROGRAM_KEYS))
        if unknown:
            raise RSFError(
                f"{what} has {unknown}; a program takes {', '.join(PROGRAM_KEYS)}"
            )
        index = entry.get("device", 0)
        if not isinstance(index, int | np.integer) or not 0 <= index < len(devices):
            raise RSFError(
                f"{what}'s device is an index into the {len(devices)} stimulating "
                f"device(s), not {index!r}"
            )
        index = int(index)
        device = devices[index]
        kind = (
            "electrode"
            if isinstance(device, MEA)
            else "light"
            if isinstance(device, PatternedLight)
            else None
        )

        policy = entry.get("policy")
        values, dt_in = entry.get("inputs"), entry.get("inputs_dt")
        units, spec = getattr(device, "input_units", None), None
        if policy is not None:
            if not isinstance(policy, Policy):
                raise RSFError(f"{what}'s policy is a livn Policy, not {policy!r}")
            try:
                policy.as_input_units(units)
            except ValueError as e:
                raise RSFError(f"{what}: {e}") from None
            spec = {
                "cls": f"{type(policy).__module__}.{type(policy).__qualname__}",
                "kwargs": policy.serialize(),
            }
            if values is None:
                values = policy.render()
                dt_in = dt_in or getattr(policy, "dt", None)
        if values is None:
            raise RSFError(f"{what} has neither inputs nor a policy")
        if dt_in is None or not float(dt_in) > 0:
            raise RSFError(f"{what}'s inputs need inputs_dt, their interval in ms")
        dt_in = float(dt_in)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise RSFError(
                f"{what}'s inputs are (timestep, n_channels), not {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise RSFError(f"{what}'s inputs hold a NaN or an infinity")

        ids = [_id_str(c) for c in np.asarray(device.channel_ids).ravel()]
        chosen = entry.get("channels")
        if chosen is not None:
            chosen = [_id_str(c) for c in chosen]
            missing = [c for c in chosen if c not in ids]
            if missing or len(set(chosen)) != len(chosen) or not chosen:
                raise RSFError(
                    f"{what}'s channels {chosen} are not distinct channels of its "
                    f"device ({ids})"
                )
        if values.shape[1] == len(ids):
            full = values
        elif chosen is not None and values.shape[1] == len(chosen):
            full = np.zeros((len(values), len(ids)))
            full[:, [ids.index(c) for c in chosen]] = values
        else:
            raise RSFError(
                f"{what}'s inputs have {values.shape[1]} channels and its device "
                f"{len(ids)}"
            )
        if chosen is None:
            columns = [c for c in range(len(ids)) if full[:, c].any()]
            if not columns:
                raise RSFError(
                    f"{what} sets no channel; name the channels it drives in channels="
                )
        else:
            columns = sorted(ids.index(c) for c in chosen)
            others = [c for c in range(len(ids)) if c not in columns]
            if full[:, others].any():
                raise RSFError(f"{what} sets channels that are not its channels=")

        if "start" in entry and "start_frame" in entry:
            raise RSFError(f"{what} starts at a time (start) or a frame (start_frame)")
        if "start_frame" in entry:
            start_frame = entry["start_frame"]
            if not isinstance(start_frame, int | np.integer):
                raise RSFError(f"{what}'s start_frame is an integer frame")
            start_frame = int(start_frame)
        else:
            start = float(entry.get("start", 0.0))
            if not math.isfinite(start):
                raise RSFError(f"{what}'s start is ms, not {start}")
            start_frame = int(_nearest(start / dt))
        if not 0 <= start_frame < n_frames:
            raise RSFError(
                f"{what} starts at frame {start_frame}, outside the session's "
                f"{n_frames} frames"
            )

        origin = entry.get("origin")
        if origin is None and policy is not None:
            origin = _policy_origin(policy)
        origin = None if origin is None else _check_origin(origin, what)
        started_at = entry.get("started_at")
        if started_at is not None:
            try:
                _instant(started_at)
            except (TypeError, ValueError):
                raise RSFError(
                    f"{what}'s started_at is an RFC 3339 timestamp, not {started_at!r}"
                ) from None

        plan: dict[str, Any] = {
            "given": k,
            "device_index": index,
            "device": device,
            "kind": kind,
            "start_frame": start_frame,
            "origin": origin,
            "members": _check_members(entry.get("members"), what),
            "started_at": started_at,
            "livn": {
                "channels": [_id_value(ids[c]) for c in columns],
                "units": units,
                "source_dt": dt_in,
                "stimulus": None if spec is None else _plain(spec),
                "device": index,
            },
        }
        if kind is None:
            # a device RSF has no channel object for: a record without rows
            frames = np.arange(n_frames - start_frame)
            rows = np.floor(frames * dt / dt_in + 1e-9).astype(np.int64)
            inside = rows < len(full)
            block = series[start_frame:, offsets[index] : offsets[index + 1]]
            block[inside] = full[rows[inside]]
            plans.append(plan)
            continue

        names = _site_names(device, kind)
        prefix = kind if len(devices) == 1 else f"{kind}/{index}"
        record_ids = [f"{prefix}/{names[c]}" for c in columns]
        fractions = _program_fractions(device, kind, full[:, columns], what)
        planned = _planned_edges(fractions, dt_in, record_ids)
        delivered = _delivered_rows(
            planned, start_frame, dt, n_frames, entry.get("stop")
        )
        factor = 1.0 if kind == "light" else _livn_scale(kind, device.max_current_uA)
        for frame, channel, value, event in zip(
            delivered["frame"].tolist(),
            delivered["channel"],
            delivered["value"].tolist(),
            delivered["event"].tolist(),
            strict=True,
        ):
            column = int(offsets[index]) + columns[record_ids.index(channel)]
            holding.setdefault(column, []).append(
                (frame, start_frame, k, event, value * factor, channel)
            )
        plan.update(
            columns=columns,
            names=[names[c] for c in columns],
            ids=record_ids,
            planned=planned,
            delivered=delivered,
        )
        plans.append(plan)

    # a channel is driven by one program at a time
    for column, rows in holding.items():
        spans: dict[int, tuple[int, int]] = {}
        for frame, _, k, *_ in rows:
            lo, hi = spans.get(k, (frame, frame))
            spans[k] = (min(lo, frame), max(hi, frame))
        ordered = sorted(spans.items(), key=lambda item: item[1])
        for (a, (a_lo, a_hi)), (b, (b_lo, b_hi)) in itertools.combinations(ordered, 2):
            if a_lo < b_hi and b_lo < a_hi:
                raise RSFError(
                    f"programs {a} and {b} both drive {rows[0][5]!r} between frames "
                    f"{max(a_lo, b_lo)} and {min(a_hi, b_hi)}; a channel is driven by "
                    "one program at a time"
                )
        # the held value, by frame, then the program's start and its events
        rows.sort(key=lambda r: r[:4])
        for i, (frame, *_, value, _) in enumerate(rows):
            stop = rows[i + 1][0] if i + 1 < len(rows) else n_frames
            series[min(frame, n_frames) : min(stop, n_frames), column] = value
    return plans, series


def _site_channel(device, kind: str, column: int, id_: str, name: str, unplaced):
    """One channel object of a program's record, with its site's ``name`` and ``source``."""
    if kind == "light":
        row = np.asarray(device.spots, dtype=np.float64)[column]
        labels = device.labels or [None] * device.num_channels
        return {
            "id": id_,
            "shape": {"kind": "unplaced"}
            if unplaced
            else {
                "kind": "disc",
                "x": float(row[1]),
                "y": float(row[2]),
                "radius": float(row[3]),
            },
            "full_scale": {"units": "mW/mm2", "value": float(device.irradiance)},
            "bipolar": False,
            "wavelength_nm": float(device.wavelength),
            "label": labels[column],
            "name": name,
            "source": CHANNEL_SOURCES[kind],
        }
    row = np.asarray(device.electrode_coordinates, dtype=np.float64)[column]
    return {
        "id": id_,
        "shape": {"kind": "unplaced"}
        if unplaced
        else {
            "kind": "contact",
            "contact_id": _id_value(row[0]),
            "x": float(row[1]),
            "y": float(row[2]),
        },
        "full_scale": {"units": "uA", "value": float(device.max_current_uA)},
        "bipolar": True,
        "wavelength_nm": None,
        "label": None,
        "name": name,
        "source": CHANNEL_SOURCES[kind],
    }


def _program_records(plans, masked, started_at: str, dt: float, channel_devices):
    """``(records, stim_events, planned, record_of)``: one record per program in start
    order, the rows of all, each record's plan, and each given program's record index."""
    order = sorted(range(len(plans)), key=lambda k: (plans[k]["start_frame"], k))
    record_of = {plans[k]["given"]: i for i, k in enumerate(order)}
    session_start = _instant(started_at)
    unplaced = "electrode_positions" in masked
    records, planned_tables, tables = [], [], []
    for index, k in enumerate(order):
        plan = plans[k]
        stamp = plan["started_at"] or (
            session_start + timedelta(milliseconds=plan["start_frame"] * dt)
        ).astimezone(UTC).isoformat(timespec="microseconds")
        record: dict[str, Any] = {"started_at": stamp}
        if plan["kind"] is None:
            records.append({**record, PRODUCER: plan["livn"], **plan["members"]})
            planned_tables.append(None)
            continue
        channels = [
            _site_channel(plan["device"], plan["kind"], column, id_, name, unplaced)
            for column, id_, name in zip(
                plan["columns"], plan["ids"], plan["names"], strict=True
            )
        ]
        for column, channel in zip(plan["columns"], channels, strict=True):
            channel_devices[channel["id"]] = (plan["device"], column, plan["kind"])
        record.update(
            kind=plan["kind"],
            channels=channels,
            marker_channel=None,
            marker_edge=None,
            timing="simulated",
            placement_lag_ms=0,
        )
        if plan["origin"] is not None:
            record["origin"] = plan["origin"]
        record[PRODUCER] = plan["livn"]
        record.update(plan["members"])
        records.append(record)
        planned_tables.append(plan["planned"])
        rows = plan["delivered"]
        n = len(rows["frame"])
        tables.append(
            {
                "frame": rows["frame"],
                "channel": rows["channel"],
                "value": rows["value"],
                "program": np.full(n, index, dtype=np.uint32),
                "event": rows["event"],
                "timing": ["simulated"] * n,
            }
        )
    if not tables:
        return records, _no_events(), planned_tables, record_of
    events = {
        name: (
            [v for t in tables for v in t[name]]
            if name in ("channel", "timing")
            else np.concatenate([t[name] for t in tables])
        )
        for name in STIM_COLUMNS
    }
    return records, events, planned_tables, record_of


# the session's preparation, notes, epochs and documents


def _zone(zone) -> str | None:
    """The IANA zone to write: the one given, the host's for ``True`` when it can be
    told, or none."""
    import zoneinfo

    if zone is None or zone is False:
        return None
    if zone is True:
        zone = _host_zone()
        if zone is None:
            return None
    if not isinstance(zone, str):
        raise RSFError(f"zone is an IANA time zone name, not {zone!r}")
    try:
        zoneinfo.ZoneInfo(zone)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise RSFError(f"zone {zone!r} is not an IANA time zone") from None
    return zone


def _host_zone() -> str | None:
    """The host's IANA zone as its configuration names it, or ``None``; never a guess
    from an offset."""
    import zoneinfo

    try:
        import tzlocal  # ty: ignore[unresolved-import]

        name = tzlocal.get_localzone_name()
    except Exception:
        name = os.environ.get("TZ", "").lstrip(":") or None
        if name is None:
            target = os.path.realpath("/etc/localtime")
            _, found, tail = target.partition("zoneinfo/")
            name = tail if found else None
    if not name:
        return None
    try:
        zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        return None
    return name


def _age(age, what: str) -> dict | None:
    if age is None:
        return None
    if (
        not isinstance(age, Mapping)
        or not _number(age.get("value"))
        or not age["value"] >= 0
        or not isinstance(age.get("units"), str)
        or not age["units"]
    ):
        raise RSFError(f"{what} is {{'value': >= 0, 'units': str}}, not {age!r}")
    return {"value": age["value"], "units": age["units"]}


def _simulated_prep(prep) -> dict:
    """``env.system.simulation.prep``, with an ``id`` minted when none is given."""
    if not isinstance(prep, Mapping):
        raise RSFError(f"prep is a mapping of {', '.join(PREP_KEYS)}, not {prep!r}")
    unknown = sorted(set(prep) - set(PREP_KEYS))
    if unknown:
        raise RSFError(f"prep has {unknown}; a prep has {', '.join(PREP_KEYS)}")
    out: dict[str, Any] = {}
    id_ = prep.get("id")
    out["id"] = uuid.uuid4().hex if id_ is None else id_
    for name in ("id", "label", "kind", "species", "models"):
        value = out["id"] if name == "id" else prep.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise RSFError(f"prep {name} is a non-empty string or None, not {value!r}")
        out[name] = value
    if out["models"] is not None and out["models"] == out["id"]:
        raise RSFError(
            "prep models names the physical preparation the simulation stands for, "
            f"never its own id {out['id']!r}"
        )
    out["age"] = _age(prep.get("age"), "prep age")
    expression = prep.get("expression")
    if expression is not None:
        if not isinstance(expression, Sequence) or isinstance(expression, str):
            raise RSFError("prep expression is a list of constructs")
        entries = []
        for entry in expression:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("construct"), str
            ):
                raise RSFError(f"an expression entry names its construct: {entry!r}")
            method = entry.get("method")
            if not isinstance(method, str) or not method:
                raise RSFError(f"expression {entry['construct']!r} states its method")
            since = entry.get("since")
            if since is not None and (
                not isinstance(since, str)
                or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", since)
            ):
                raise RSFError(f"expression since is a YYYY-MM-DD date, not {since!r}")
            item: dict[str, Any] = {
                "construct": entry["construct"],
                "method": method,
                "since": since,
            }
            age = _age(entry.get("age"), "an expression age")
            if age is not None:
                if out["age"] is not None and age["units"] != out["age"]["units"]:
                    warnings.warn(
                        f"expression {entry['construct']!r} states its age in "
                        f"{age['units']!r} and the prep in {out['age']['units']!r}",
                        stacklevel=4,
                    )
                item["age"] = age
            entries.append(item)
        out["expression"] = entries
    _json(out, "prep")
    return {k: v for k, v in out.items() if v is not None or k in ("label", "models")}


def _ext_documents(ext) -> dict[str, dict[str, Any]]:
    if ext is None:
        return {}
    if not isinstance(ext, Mapping):
        raise RSFError("ext is {producer: {name.json: document}}")
    out = {}
    for producer, documents in ext.items():
        _check_producer(producer, "ext producer")
        if not isinstance(documents, Mapping):
            raise RSFError(f"ext {producer!r} is {{name.json: document}}")
        for name, document in documents.items():
            if not isinstance(name, str) or not DOCUMENT_NAME.fullmatch(name):
                raise RSFError(
                    f"ext {producer!r} names its documents plainly, 'name.json', not "
                    f"{name!r}"
                )
            _json(document, f"rsf/ext/{producer}/{name}")
        out[producer] = {name: _plain(d) for name, d in documents.items()}
    return out


def _note_author(author, what: str):
    if author is None:
        return None
    try:
        _check_author(author)
    except RSFError as e:
        raise RSFError(f"{what}: {e}") from None
    return author


def _note_frame(frame, n_frames: int, what: str):
    if frame is None:
        return None
    if not isinstance(frame, int | np.integer) or not 0 <= frame <= n_frames:
        raise RSFError(f"{what}'s frame is a frame of 0..{n_frames}, not {frame!r}")
    return int(frame)


def _notes(notes, n_frames: int, instant: datetime, zone) -> list[dict]:
    out = []
    for k, note in enumerate(notes or ()):
        what = f"note {k}"
        if not isinstance(note, Mapping) or not isinstance(note.get("text"), str):
            raise RSFError(f"{what} is {{'text', 'frame', 'author', 'at'}}")
        unknown = sorted(set(note) - set(NOTE_KEYS))
        if unknown:
            raise RSFError(f"{what} has {unknown}; a note takes {', '.join(NOTE_KEYS)}")
        item = note.get("item") or uuid.uuid4().hex
        if not isinstance(item, str) or not HEX_ITEM.fullmatch(item):
            raise RSFError(f"{what}'s item is 32 lowercase hex digits, not {item!r}")
        at = instant
        if note.get("at") is not None:
            try:
                at = _instant(note["at"])
            except (TypeError, ValueError):
                raise RSFError(f"{what}'s at is an RFC 3339 timestamp") from None
        version = _note_version(item, note["text"], at, zone)
        version["author"] = _note_author(note.get("author"), what)
        version["frame"] = _note_frame(note.get("frame"), n_frames, what)
        out.append(version)
    return out


def _epoch_versions(
    epochs, n_frames: int, dt: float, record_of, stimulation, instant, zone
) -> list[dict]:
    """One ``epoch`` version per epoch, keys minted into items and parents resolved."""
    if not epochs:
        return []
    items: dict[str, str] = {}
    given = []
    for k, epoch in enumerate(epochs):
        what = f"epoch {k}"
        if not isinstance(epoch, Mapping):
            raise RSFError(f"{what} is a mapping of {', '.join(EPOCH_KEYS)}")
        unknown = sorted(set(epoch) - set(EPOCH_KEYS))
        if unknown:
            raise RSFError(
                f"{what} has {unknown}; an epoch takes {', '.join(EPOCH_KEYS)}"
            )
        item, key = epoch.get("item"), epoch.get("key")
        if item is not None and key is not None:
            raise RSFError(f"{what} is named by a key or an item, not both")
        if item is not None:
            if not isinstance(item, str) or not HEX_ITEM.fullmatch(item):
                raise RSFError(
                    f"{what}'s item is 32 lowercase hex digits, not {item!r}"
                )
        else:
            item = uuid.uuid4().hex
        name = key if key is not None else item
        if name in items:
            raise RSFError(f"{what} repeats the epoch {name!r}")
        items[name] = item
        if key is not None:
            items.setdefault(item, item)
        given.append((what, item, epoch))

    def frame_of(epoch, frame_key, time_key, what):
        if frame_key in epoch and time_key in epoch:
            raise RSFError(f"{what} gives {frame_key} or {time_key}, not both")
        if time_key in epoch:
            value = epoch[time_key]
            if value is None:
                return None
            return int(_nearest(float(value) / dt))
        return epoch.get(frame_key)

    out = []
    spans = {}
    for what, item, epoch in given:
        frame = frame_of(epoch, "frame", "start", what)
        end = frame_of(epoch, "end_frame", "end", what)
        if frame is None:
            raise RSFError(f"{what} starts at a frame (frame=) or a time (start=)")
        frame = _note_frame(frame, n_frames, what)
        end = _note_frame(end, n_frames, what)
        if end is not None and end < frame:
            raise RSFError(f"{what} ends at frame {end}, before it starts at {frame}")
        kind = epoch.get("kind")
        if not isinstance(kind, str) or not kind:
            raise RSFError(f"{what} has a kind: one of {', '.join(EPOCH_KINDS)}")
        if kind not in EPOCH_KINDS:
            warnings.warn(
                f"{what}'s kind {kind!r} is not one of {', '.join(EPOCH_KINDS)}; a "
                "reader shows it as text",
                stacklevel=4,
            )
        text = epoch.get("text", "")
        if not isinstance(text, str):
            raise RSFError(f"{what}'s text is a string")
        parent = epoch.get("parent")
        if parent is not None:
            if parent in items:
                parent = items[parent]
            elif not isinstance(parent, str) or not HEX_ITEM.fullmatch(parent):
                raise RSFError(f"{what}'s parent {parent!r} names no epoch")
            if parent == item:
                raise RSFError(f"{what} is its own parent")
        programs = epoch.get("programs")
        if programs is not None:
            if isinstance(programs, str) or not isinstance(programs, Sequence):
                raise RSFError(f"{what}'s programs are indices into programs=")
            indices = []
            for p in programs:
                if not isinstance(p, int | np.integer) or int(p) not in record_of:
                    raise RSFError(
                        f"{what}'s programs name {p!r}, which is not an index into "
                        "programs="
                    )
                index = record_of[int(p)]
                if "kind" not in stimulation[index]:
                    raise RSFError(
                        f"{what} names program {p}, whose record has no neutral members"
                    )
                indices.append(index)
            programs = sorted(indices)
        origin = epoch.get("origin")
        origin = None if origin is None else _check_origin(origin, what)
        version = _note_version(item, text, instant, zone)
        version.update(
            author=_note_author(epoch.get("author"), what),
            frame=frame,
            end_frame=end,
            kind=kind,
            parent=parent,
            programs=programs,
            origin=origin,
        )
        spans[item] = (what, frame, end)
        out.append(version)
    for version in out:
        parent = version["parent"]
        if parent not in spans:
            continue
        what, frame, end = spans[version["item"]]
        _, p_frame, p_end = spans[parent]
        if frame < p_frame or (end is not None and p_end is not None and end > p_end):
            raise RSFError(
                f"{what} [{frame}, {end}) lies outside its parent [{p_frame}, {p_end})"
            )
    return out


def _population_of(env, sources: list[int]) -> list[str | None]:
    ranges = getattr(getattr(env, "system", None), "population_ranges", None)
    if not isinstance(ranges, Mapping):
        raise RSFError("the env's system states no population_ranges")
    out = []
    for gid in sources:
        found = None
        for name, (start, count) in ranges.items():
            if int(start) <= gid < int(start) + int(count):
                found = str(name)
                break
        out.append(found)
    return out


def _source_populations(env, sources: list[int]) -> list[str | None]:
    return _population_of(env, sources)


def _opsin_conductance(env) -> tuple[str, str, float, list] | None:
    """``(construct, units, value, populations)`` of the model's opsin, or ``None``."""
    model = getattr(env, "model", None)
    config = None
    for name in ("opsin_config", "native_opsin_config", "neuron_opsin_config"):
        hook = getattr(model, name, None)
        if callable(hook):
            config = hook() or None
            break
    if not config:
        return None
    construct = str(config.get("mechanism", "RhO3c"))
    params = dict(config.get("params") or {})
    known = OPSIN_CONDUCTANCES.get(construct)
    if known is None or not _number(params.get(known[0], known[2])):
        raise RSFError(
            f"the model's opsin {construct!r} states no conductance livn can read; pass "
            "expression={construct: {'units', 'values'}}"
        )
    name, units, default = known
    value = params.get(name, default)
    populations = config.get("populations")
    return (
        construct,
        units,
        float(value),
        None if populations is None else list(populations),
    )


def _source_expression(env, sources: list[int], expression) -> dict:
    if expression is True:
        found = _opsin_conductance(env)
        if found is None:
            raise RSFError(
                "expression=True, but the env's model attaches no opsin; pass "
                "expression={construct: {'units', 'values'}}"
            )
        construct, units, value, covered = found
        owners = _population_of(env, sources)
        values = [
            value if covered is None or owner in covered else 0.0 for owner in owners
        ]
        return {construct: {"units": units, "values": values}}
    if not isinstance(expression, Mapping):
        raise RSFError("expression is True or {construct: {'units', 'values'}}")
    out = {}
    for construct, entry in expression.items():
        what = f"expression {construct!r}"
        if not isinstance(construct, str) or not isinstance(entry, Mapping):
            raise RSFError(f"{what} is {{'units', 'values', 'reference'}}")
        units = entry.get("units")
        if not isinstance(units, str) or not units:
            raise RSFError(f"{what} states its units")
        values = entry.get("values")
        if isinstance(values, Mapping):
            by_gid = {int(g): v for g, v in values.items()}
            unknown = sorted(set(by_gid) - set(sources))
            if unknown:
                raise RSFError(f"{what} names gids {unknown[:5]} that are not sources")
            values = [by_gid.get(g, 0.0) for g in sources]
        values = np.asarray(values, dtype=np.float64).ravel()
        if len(values) != len(sources):
            raise RSFError(
                f"{what} has {len(values)} values for {len(sources)} sources"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise RSFError(f"{what}'s values are finite and >= 0")
        item: dict[str, Any] = {"units": units, "values": values.tolist()}
        reference = entry.get("reference")
        if reference is not None:
            if not isinstance(reference, str) or not reference:
                raise RSFError(f"{what}'s reference is text")
            item["reference"] = reference
        elif units == "1":
            raise RSFError(
                f"{what} is a fraction (units '1') and says of what in reference"
            )
        out[construct] = item
    return out


def _connection_table(env, sources: list[int], given) -> dict:
    """``connections.zarr``'s columns and its ``inputs``."""
    source_ids = [str(g) for g in sources]
    if given is True:
        comm = getattr(env, "comm", None)
        size = comm.Get_size() if hasattr(comm, "Get_size") else 1
        if size > 1:
            raise RSFError(
                "connections=True reads this rank's connections only; under MPI gather "
                "the table and pass it as connections={...}"
            )
        table = env.connections()
        if table.get("weight") is None:
            raise RSFError(
                f"{type(env).__name__}.connections() states no weight; pass "
                "connections={...}"
            )
        pre_gid = np.asarray(table["pre_gid"], dtype=np.int64)
        known = set(sources)
        pre_population = [str(p) for p in np.asarray(table["pre_population"]).tolist()]
        post_population = [
            str(p) for p in np.asarray(table["post_population"]).tolist()
        ]
        pre = [
            str(g) if g in known else f"{population or 'input'}/{g}"
            for g, population in zip(pre_gid.tolist(), pre_population, strict=True)
        ]
        synapses = (getattr(env.system, "connections_config", None) or {}).get(
            "synapses", {}
        )
        signs = []
        for post_name, pre_name in zip(post_population, pre_population, strict=True):
            spec = (synapses.get(post_name) or {}).get(pre_name)
            if not isinstance(spec, Mapping) or "type" not in spec:
                raise RSFError(
                    f"the system states no synapse type for {pre_name} -> {post_name}, "
                    "so the sign of its connections is unknown"
                )
            signs.append(-1 if spec["type"] == "inhibitory" else 1)
        given = {
            "pre": pre,
            "post": [str(g) for g in np.asarray(table["post_gid"]).tolist()],
            "weight": np.asarray(table["weight"], dtype=np.float64),
            "units": CONNECTION_WEIGHT_UNITS,
            "reference": CONNECTION_WEIGHT_REFERENCE,
            "sign": signs,
            "inputs": sorted({p for p in pre if p not in source_ids}),
        }
        if table.get("delay") is not None:
            given["delay"] = np.asarray(table["delay"], dtype=np.float32)
        receptor = [str(r) for r in np.asarray(table.get("receptor", [])).tolist()]
        if receptor and all(receptor) and len(receptor) == len(pre):
            given["receptor"] = receptor
    if not isinstance(given, Mapping):
        raise RSFError("connections is True or {'pre', 'post', 'weight', ...}")
    missing = [k for k in ("pre", "post", "weight", "units", "sign") if k not in given]
    if missing:
        raise RSFError(f"connections has no {missing}")
    pre = [_id_str(p) for p in given["pre"]]
    post = [_id_str(p) for p in given["post"]]
    weight = np.asarray(given["weight"], dtype=np.float64).ravel()
    sign = np.asarray(given["sign"]).ravel()
    n = len(pre)
    if not (len(post) == len(weight) == len(sign) == n):
        raise RSFError("connections' columns are of one length")
    if not np.all(np.isin(sign, (1, -1))):
        raise RSFError("connections' sign is 1 (excitatory) or -1 (inhibitory)")
    if not isinstance(given["units"], str) or not given["units"]:
        raise RSFError("connections' weight states its units")
    inputs = [str(i) for i in given.get("inputs") or ()]
    if set(inputs) & set(source_ids):
        raise RSFError("connections' inputs are inputs without a source, not sources")
    known = set(source_ids)
    outside = sorted({p for p in post if p not in known})
    if outside:
        raise RSFError(f"connections' post {outside[:5]} are not sources")
    unnamed = sorted({p for p in pre if p not in known and p not in inputs})
    if unnamed:
        raise RSFError(
            f"connections' pre {unnamed[:5]} are neither sources nor named in inputs"
        )
    if not np.all(np.isfinite(weight)):
        raise RSFError("connections' weight holds a NaN or an infinity")
    out = {
        "arrays": {
            "pre": pre,
            "post": post,
            "weight": weight,
            "sign": sign.astype(np.int8),
        },
        "array_attributes": {"weight": {"units": given["units"]}},
        "attributes": {"inputs": inputs},
    }
    reference = given.get("reference")
    if reference is not None:
        if not isinstance(reference, str) or not reference:
            raise RSFError("connections' reference says what a weight of 1 is, as text")
        out["array_attributes"]["weight"]["reference"] = reference
    elif given["units"] == "1":
        raise RSFError(
            "connections' weight is a fraction (units '1') and says of what in reference"
        )
    if given.get("delay") is not None:
        delay = np.asarray(given["delay"], dtype=np.float32).ravel()
        if len(delay) != n:
            raise RSFError("connections' delay is one per connection")
        out["arrays"]["delay"] = delay
    if given.get("receptor") is not None:
        receptor = [str(r) for r in given["receptor"]]
        if len(receptor) != n:
            raise RSFError("connections' receptor is one per connection")
        out["arrays"]["receptor"] = receptor
    return out


def _coupling_table(env, sources, stimulation, channel_devices, given) -> dict:
    """``stimulus_coupling.zarr``: what each record channel at full scale puts on each
    source, linearly."""
    record_ids = [
        c["id"] for r in stimulation if "kind" in r for c in r.get("channels") or ()
    ]
    record_ids = list(dict.fromkeys(record_ids))
    if not record_ids:
        raise RSFError(
            "stimulus_coupling needs stimulus channels, and the session has none"
        )
    source_ids = [str(g) for g in sources]
    if given is True:
        kinds = {channel_devices[c][2] for c in record_ids}
        if len(kinds) > 1:
            raise RSFError(
                "stimulus_coupling states one unit, and the session has light and "
                "electrode channels; pass stimulus_coupling={...} for one kind"
            )
        (kind,) = kinds
        xyz = np.asarray(_source_coordinates(env, sources), dtype=np.float64)
        rows = np.column_stack([np.asarray(sources, dtype=np.float64), xyz])
        reached: dict[int, np.ndarray] = {}
        gain = np.zeros((len(record_ids), len(sources)), dtype=np.float32)
        for i, channel in enumerate(record_ids):
            device, column, kind = channel_devices[channel]
            if id(device) not in reached:
                reached[id(device)] = np.asarray(device.reach(rows), dtype=np.float64)
            factor = (
                1.0 if kind == "light" else _livn_scale(kind, device.max_current_uA)
            )
            gain[i] = reached[id(device)][column] * factor
        given = {
            "gain": gain,
            "units": "mW/mm2" if kind == "light" else "mV",
            "channel_ids": record_ids,
            "source_ids": source_ids,
        }
    if not isinstance(given, Mapping):
        raise RSFError(
            "stimulus_coupling is True or {'gain', 'units', 'channel_ids', 'source_ids'}"
        )
    channel_ids = [str(c) for c in given.get("channel_ids") or ()]
    sids = [_id_str(s) for s in given.get("source_ids") or ()]
    gain = np.asarray(given.get("gain"), dtype=np.float32)
    if gain.shape != (len(channel_ids), len(sids)):
        raise RSFError(
            f"stimulus_coupling's gain is [channels, sources], "
            f"({len(channel_ids)}, {len(sids)}), not {gain.shape}"
        )
    unknown = sorted(set(channel_ids) - set(record_ids))
    if unknown:
        raise RSFError(f"stimulus_coupling names channels {unknown[:5]} of no record")
    outside = sorted(set(sids) - set(source_ids))
    if outside:
        raise RSFError(f"stimulus_coupling names {outside[:5]}, which are not sources")
    units = given.get("units")
    if not isinstance(units, str) or not units:
        raise RSFError("stimulus_coupling's gain states its units")
    if not np.all(np.isfinite(gain)):
        raise RSFError("stimulus_coupling's gain holds a NaN or an infinity")
    return {
        "arrays": {"gain": gain, "channel_ids": channel_ids, "source_ids": sids},
        "array_attributes": {"gain": {"units": units}},
        "attributes": {},
    }


def _write_table_store(zarr, store: str, name: str, table: dict) -> None:
    root = zarr.open_group(store, mode="w", attributes=_plain(table["attributes"]))
    for column, values in table["arrays"].items():
        if isinstance(values, list):
            _strings(root, column, values)
        else:
            _array(root, column, np.asarray(values))
        attributes = table["array_attributes"].get(column)
        if attributes:
            root[column].attrs.update(attributes)


def _sampling_frequency(dt: float) -> float:
    fs = 1000.0 / float(dt)
    if abs(fs - round(fs)) < 1e-6:
        fs = float(round(fs))
    return fs


def _project(
    run, env, io, dt=None, *, recording=None, delivered=None, conditions=None
) -> tuple[np.ndarray, dict[str, list[str]] | None]:
    from livn.conditions import Conditions
    from livn.io.recording import Chain, apply, as_composition, fill
    from livn.types import place_rows

    recording = io.recording if recording is None else as_composition(recording)
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
    if names is not None:
        names = [str(n) for n in np.asarray(names).ravel()]
    by_name = row_sections is not None and names is not None
    try:
        picked, kept, unplaced = place_rows(owners, names, ids, row_sections)
    except ValueError as error:
        raise RSFError(str(error)) from None

    if not picked:
        raise RSFError("no membrane-current row has a recording coordinate")
    rows = coordinates[np.asarray(picked, dtype=np.int64)]
    facts = {
        "membrane_currents_uA": values[np.asarray(kept)],
        "neuron_coordinates": rows,
        "gids": ids[np.asarray(kept)],
        "dt": dt,
    }
    if delivered is not None:
        facts["inputs"] = _delivered(io, delivered, dt, values.shape[-1])["values"]
    given = fill(Conditions() if conditions is None else conditions, "run", **facts)

    if isinstance(recording, Chain):
        uv = np.asarray(recording.head_only()(given, io=io), dtype=np.float64)
        for name, transfer in recording.items()[1:]:
            uv = (
                transfer.apply_seeded(uv)
                if hasattr(transfer, "apply_seeded")
                else np.asarray(
                    apply(transfer, given, signal=uv, io=io, name=name),
                    dtype=np.float64,
                )
            )
    else:
        uv = np.asarray(recording(given, io=io), dtype=np.float64)
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


def _quantize(uv: np.ndarray, gain_uV: float, dtype: str = "int16") -> np.ndarray:
    if dtype not in TRACE_DTYPES:
        raise RSFError(f"traces are int16 or int32, not {dtype!r}")
    limits = np.iinfo(dtype)
    scaled = np.round(uv / gain_uV)
    if scaled.size:
        peak = float(np.max(np.abs(uv)))
        if not np.all(np.isfinite(scaled)):
            raise RSFError("the channel signal holds a NaN or an infinity")
        if scaled.max() > limits.max or scaled.min() < limits.min:
            hint = "" if dtype == "int32" else ", or dtype='int32'"
            raise RSFError(
                f"the channel signal peaks at {peak:.6g} µV, beyond the {dtype} range at "
                f"{gain_uV} µV/LSB; pass gain_uV >= {peak / limits.max:.6g}{hint}"
            )
    return np.ascontiguousarray(scaled.astype(dtype).T)


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


def _observe_document(
    io, operator, stimulates: bool, masked: set[str], recording=None
) -> dict:
    from livn.io import Noise, PointSource
    from livn.io.recording import is_stack

    recording = io.recording if recording is None else recording
    medium = operator.medium
    electrical = operator.electrical
    stack = is_stack(recording)
    plain_point_source = (
        type(electrical) is PointSource
        and stack
        and all(isinstance(t, Noise) for t in _transfers(recording))
    )
    document: dict[str, Any] = {"kind": POINT_SOURCE if plain_point_source else LIVN_IO}
    coordinates = np.asarray(operator.electrode_coordinates, dtype=np.float64)
    if "electrode_positions" in masked:
        document["electrode_ids"] = [_id_value(c) for c in coordinates[:, 0]]
    else:
        document["electrode_coordinates_um"] = [
            [_id_value(row[0]), float(row[1]), float(row[2]), float(row[3])]
            for row in coordinates
        ]
    document["output_radius_um"] = float(operator.output_radius)
    document["detection_threshold"] = float(operator.detection_threshold)
    if plain_point_source:
        document["recording_gain"] = float(electrical.recording_gain)
        document["min_distance_um"] = float(electrical.min_distance)
    if stimulates:
        document["input_radius_um"] = float(operator.input_radius)
        document["electrode_radius_um"] = float(operator.electrode_radius)
        if plain_point_source:
            document["stimulation_gain"] = float(
                electrical.stimulation_gain(operator.electrode_radius)
            )
            document["culture_height_um"] = float(electrical.height)
    document["livn"] = {
        "medium": medium.serialize(),
        "transfers": [t.serialize() for t in _transfers(recording)],
    }
    if not stack:
        document["livn"]["recording"] = recording.serialize()
        inputs = io.serialize().get("inputs") if hasattr(io, "inputs") else None
        if inputs is not None:
            document["livn"]["inputs"] = inputs
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


def _note_version(
    item: str, text: str, at: datetime | None = None, zone: str | None = None
) -> dict:
    """A version under ``rsf/notes/``."""
    at = datetime.now(UTC) if at is None else at
    return {
        "item": item,
        "at": _stamp(at),
        "zone": zone,
        "author": None,
        "removed": False,
        "text": text,
        "frame": None,
    }


def _stamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _add_note(rsf_dir: str, kind: str, version: dict) -> str:
    """Write ``version`` as a new ``<stamp>-<uuid>.json`` of ``kind``."""
    folder = os.path.join(rsf_dir, "notes", kind)
    os.makedirs(folder, exist_ok=True)
    stamp = _instant(version["at"]).strftime("%Y%m%dT%H%M%S.%fZ")
    path = os.path.join(folder, f"{stamp}-{uuid.uuid4().hex}.json")
    text = _json(version, f"a {kind} version")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return path


def _instant(stamp: str) -> datetime:
    """An RFC 3339 timestamp as an aware datetime; digits past the microsecond are dropped."""
    match = re.fullmatch(r"(.*T\d\d:\d\d:\d\d)(\.\d+)?(Z|z|[+-]\d\d:\d\d)", stamp)
    if match is None:
        raise ValueError(f"{stamp!r} is not an RFC 3339 timestamp")
    head, fraction, zone = match.groups()
    fraction = (fraction or "")[:7]
    zone = "+00:00" if zone in ("Z", "z") else zone
    value = datetime.fromisoformat(head + fraction + zone)
    if value.tzinfo is None:
        raise ValueError(f"{stamp!r} has no offset")
    return value


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
class StimMarkers:
    tick: np.ndarray
    frame: np.ndarray
    channel: np.ndarray

    def __len__(self) -> int:
        return len(self.tick)


@dataclass(frozen=True)
class StimEvents:
    frame: np.ndarray
    channel: np.ndarray
    value: np.ndarray
    program: np.ndarray
    event: np.ndarray
    timing: np.ndarray

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class Planned:
    time: np.ndarray
    channel: np.ndarray
    value: np.ndarray
    event: np.ndarray
    load: np.ndarray

    def __len__(self) -> int:
        return len(self.time)


@dataclass(frozen=True)
class Condition:
    quantity: str
    item: str
    value: float
    units: str | None
    basis: str
    origin: str
    frames: np.ndarray
    values: np.ndarray
    source: str | None = None
    method: str | None = None
    text: str | None = None
    at: str | None = None
    author: str | None = None


class RsfRecording:
    """A session opened lazily: the zarr arrays stay on disk and :meth:`read` touches only
    the chunks of the window it is asked for, so it is cheap to call from many threads
    (a read is a plain zarr slice). Pickling keeps the path and reopens on the other side.

    ``growing=True`` opens a store that is still being written (no ``finished_at``): its
    tables are read to their shortest column, and :attr:`stimulus_final_to` says how far its
    delivered stimulus is final.
    """

    def __init__(self, path: str | os.PathLike, *, growing: bool = False):
        self.path = os.fspath(path)
        self.growing = bool(growing)
        self._open()

    def _open(self) -> None:
        zarr = _zarr()

        path = self.path
        traces_path = os.path.join(path, "traces.zarr")
        if not os.path.isfile(os.path.join(traces_path, "zarr.json")):
            raise RSFError(f"{path} is not a session: it has no traces.zarr/zarr.json")
        documents = os.path.join(traces_path, RSF)
        group_attrs = _read_node(documents)
        if not group_attrs:
            raise RSFError(
                f"{path} has no traces.zarr/rsf/ group: it is not an RSF session (a "
                "SpikeInterface recording, or a store written before RSF 1)"
            )
        if group_attrs.get("format") != FORMAT:
            raise RSFError(
                f"{path} is RSF format {group_attrs.get('format')!r}; livn reads format {FORMAT}"
            )
        self.finished = bool(group_attrs.get("finished_at"))
        if not self.finished and not self.growing:
            raise RSFError(
                f"{path} is not complete (traces.zarr has no finished_at); it may still be "
                "written: open it with growing=True to follow it"
            )

        root = zarr.open_group(traces_path, mode="r")
        attrs = _read_node(traces_path)
        self.fs = float(attrs["sampling_frequency"])
        self._dt = 1000.0 / self.fs
        self.record = (attrs.get("annotations") or {}).get(RECORD_KEY) or {}
        if self.record.get("format", FORMAT) != FORMAT:
            raise RSFError(
                f"{path} has a provenance record of format {self.record.get('format')!r}; "
                f"livn reads format {FORMAT}"
            )
        if self.record and self.record.get("recording_id") != group_attrs.get(
            "recording_id"
        ):
            raise RSFError(
                f"{path}: the record of traces.zarr names recording "
                f"{self.record.get('recording_id')!r} and its rsf/ group "
                f"{group_attrs.get('recording_id')!r}; such a store does not conform"
            )
        self.producer = dict(group_attrs.get("producer") or {})
        self.tick_rate = float(group_attrs.get("hardware_tick_rate_hz") or self.fs)
        self._r = self.tick_rate / self.fs
        tick = self.record.get("first_sample_tick")
        self.first_sample_tick = None if tick is None else int(tick)
        self._t0 = 0.0 if tick is None else int(tick) / self.tick_rate * 1000.0

        self._keys = [str(k) for k in _read(root, "channel_ids")]
        self.channel_ids = _ids_array(self._keys)
        self._column = {key: i for i, key in enumerate(self._keys)}
        self._traces = _array_of(root, "traces_seg0")  # stays on disk
        if str(self._traces.dtype) not in TRACE_DTYPES:
            raise RSFError(
                f"{path}: traces_seg0 is {self._traces.dtype}; RSF traces are int16 or int32"
            )
        self._gain = np.asarray(_read(root, "properties/gain_to_uV"), dtype=np.float64)
        self._offset = np.asarray(
            _read(root, "properties/offset_to_uV"), dtype=np.float64
        )
        self._root = root

        anchors = _table(root, "tick_anchors", ("frame", "tick"))
        if anchors is None:
            raise RSFError(
                f"{path} has no rsf/tick_anchors table: it is not an RSF store"
            )
        self._anchor_frames = anchors["frame"].astype(np.int64)
        self._anchor_ticks = anchors["tick"].astype(np.uint64)
        if np.any(np.diff(self._anchor_frames) <= 0):
            raise RSFError(f"{path}: rsf/tick_anchors is not sorted by frame")
        # each anchor's tick as an exact offset from the first, so 64-bit ticks keep their digits
        self._anchor_offsets = np.asarray(
            [int(t) - int(self._anchor_ticks[0]) for t in self._anchor_ticks],
            dtype=np.float64,
        )
        gaps = _table(root, "gaps", ("frame", "n_frames"))
        self.gaps = (
            None
            if gaps is None
            else np.stack(
                [gaps["frame"].astype(np.int64), gaps["n_frames"].astype(np.int64)],
                axis=1,
            ).reshape(-1, 2)
        )
        if self.gaps is not None and np.any(np.diff(self.gaps[:, 0]) <= 0):
            raise RSFError(
                f"{path}: rsf/gaps is not sorted by frame with one gap per frame, so the "
                "frames' times cannot be read from it"
            )
        dropped = _table(root, "dropped_spans", ("frame_start", "n_frames", "channel"))
        if dropped is None:
            raise RSFError(
                f"{path} has no rsf/dropped_spans table: it is not an RSF store"
            )
        self._dropped = dropped

        self.simulation = None
        if os.path.isfile(os.path.join(documents, "simulation.json")):
            with open(
                os.path.join(documents, "simulation.json"), encoding="utf-8"
            ) as f:
                self.simulation = json.load(f)
        self._documents = documents
        self._held = None
        max_current_uA = self._electrode_range()
        self.io = _read_operator(
            os.path.join(documents, "observe.json"),
            os.path.join(documents, "ext", PRODUCER, "observe.json"),
            max_current_uA=max_current_uA,
        )
        self.electrode_map = dict(group_attrs.get("electrode_map") or {})
        self._contacts = _probe_contacts(
            attrs.get("probe") or _json_file(os.path.join(documents, "probe.json")),
            self._keys,
            self.row_property("channel"),
            set(self.channels()),
        )
        self.electrode_rows = {c["id"]: c["row"] for c in self._contacts}
        if self.io is None:
            self.io = _probe_io(self._contacts, max_current_uA)
        self.bad_channels = np.zeros(0, dtype=np.int64)
        self.created_at = group_attrs.get("created_at")
        self.zone = _string(group_attrs.get("zone"))
        self.author = _string(group_attrs.get("author"))
        self.derived_from = _derived_from(group_attrs.get("derived_from"))
        self.substrate = _string((self.record.get("env") or {}).get("substrate"))
        self.prep = _prep((self.record.get("env") or {}).get("system"))
        notes = _read_notes(os.path.join(documents, "notes"))
        self.label = _current_label(notes, group_attrs.get("label", ""))
        self.description = _current_text(notes.get("description", {}).get("session"))
        self.notes = sorted(
            (
                {
                    "item": item,
                    **{k: v[k] for k in ("text", "frame", "author", "zone", "at")},
                    "carried_from": _string(v.get("carried_from")),
                }
                for item, v in (
                    (item, _current(versions))
                    for item, versions in notes.get("note", {}).items()
                )
                if v is not None and not v["removed"]
            ),
            key=lambda note: _instant(note["at"]),
        )
        self.metadata = {
            "format": RSF,
            "path": path,
            "record": self.record,
            "simulation": self.simulation,
            "label": self.label,
            "description": self.description,
            "producer": self.producer,
            "gain_uV": float(self._gain[0]) if len(self._gain) else None,
            "started_at": self.started_at,
            "zone": self.zone,
            "author": self.author,
            "derived_from": self.derived_from,
            "prep": self.prep,
        }

    @property
    def started_at(self) -> str | None:
        derived = (self.derived_from or {}).get("started_at")
        if derived is not None:
            return derived
        return _string(self.record.get("started_at")) or _string(self.created_at)

    def __getstate__(self) -> dict:
        return {"path": self.path, "growing": self.growing}

    def __setstate__(self, state: dict) -> None:
        self.path = state["path"]
        self.growing = state.get("growing", False)
        self._open()

    @property
    def segments(self) -> int:
        return 1

    def num_samples(self, segment: int = 0) -> int:
        self._segment(segment)
        return int(self._traces.shape[0])

    def read(self, segment, start, length, channels=None) -> np.ndarray:
        """Samples ``[start, start + length)`` of the given channel ids (all by default), as
        float32 ``(length, channels)`` in uV, each row through its own gain and offset."""
        self._segment(segment)
        start, length = _window(start, length, int(self._traces.shape[0]))
        raw = np.asarray(self._traces[start : start + length, :])
        gain, offset = self._gain, self._offset
        if channels is not None:
            columns = self._columns(channels)
            raw, gain, offset = raw[:, columns], gain[columns], offset[columns]
        return (raw.astype(np.float64) * gain + offset).astype(np.float32)

    def row_property(self, name: str) -> np.ndarray | None:
        """The per-row array ``properties/<name>`` (``"channel"``, ``"band"``,
        ``"impedance"``, ...), or ``None`` when the store has none."""
        if not _has(self._root, f"properties/{name}"):
            return None
        values = _read(self._root, f"properties/{name}")
        if values.dtype == object:
            values = np.asarray([str(v) for v in values])
        return values

    @property
    def clipped_samples(self) -> np.ndarray | None:
        """Per row, the samples clipped to the integer range when quantizing; ``None`` when
        the producer did not count them."""
        values = self.row_property("clipped_samples")
        return None if values is None else values.astype(np.uint64)

    # the device clock and the frames that hold signal

    def tick_at(self, frames) -> np.ndarray:
        """The device tick of each frame, float64 (fractional when a frame spans a non-integer
        number of ticks); NaN before the first tick anchor or without a clock. Exact up to
        float64's 2^53: the ticks of a long recording keep their digits only as differences,
        which :meth:`elapsed` uses."""
        frames = np.asarray(frames, dtype=np.float64)
        k = self._anchor_of_frames(frames)
        out = np.full(frames.shape, np.nan)
        ok = k >= 0
        base = float(self._anchor_ticks[0]) if len(self._anchor_ticks) else 0.0
        out[ok] = (
            base
            + self._anchor_offsets[k[ok]]
            + (frames[ok] - self._anchor_frames[k[ok]]) * self._r
        )
        return out

    def frame_at(self, ticks) -> np.ndarray:
        """The frame each device tick falls on, float64: ``anchor.frame + (tick -
        anchor.tick) / r`` for the last anchor at or before the tick. NaN for a tick before
        the first anchor, one in a gap (past the frames its anchor governs) and one past the
        last frame."""
        ticks = np.asarray(ticks).astype(np.uint64)
        out = np.full(ticks.shape, np.nan)
        if not len(self._anchor_ticks) or not ticks.size:
            return out
        if np.any(np.diff(self._anchor_ticks.astype(np.float64)) < 0):
            raise RSFError(
                f"{self.path}: the tick anchors go backwards, so a tick names no frame"
            )
        k = np.searchsorted(self._anchor_ticks, ticks, side="right") - 1
        ok = k >= 0
        delta = np.zeros(ticks.shape, dtype=np.float64)
        delta[ok] = (ticks[ok] - self._anchor_ticks[k[ok]]).astype(np.float64)
        frame = np.where(
            ok, self._anchor_frames[np.maximum(k, 0)] + delta / self._r, np.nan
        )
        ends = np.append(self._anchor_frames[1:], self.num_samples())
        inside = ok & (frame < ends[np.maximum(k, 0)])
        out[inside] = frame[inside]
        return out

    def elapsed(self, frames) -> np.ndarray:
        """Seconds from frame 0 to each frame on the source's clock: through the tick anchors
        when the store has a clock, else frame counts plus the frames ``rsf/gaps`` says were
        lost before each frame (``n_frames`` 0, unknown, counts as none)."""
        frames = np.asarray(frames, dtype=np.float64)
        if len(self._anchor_frames):
            first = float(self._anchor_frames[0])
            k = self._anchor_of_frames(frames)
            ticks = np.where(
                k >= 0,
                self._anchor_offsets[np.maximum(k, 0)]
                + (frames - self._anchor_frames[np.maximum(k, 0)]) * self._r,
                (frames - first) * self._r,
            )
            return first / self.fs + ticks / self.tick_rate
        lost = np.zeros(frames.shape)
        if self.gaps is not None and len(self.gaps):
            cumulative = np.cumsum(self.gaps[:, 1])
            k = np.searchsorted(self.gaps[:, 0], frames, side="right") - 1
            lost = np.where(k >= 0, cumulative[np.maximum(k, 0)], 0.0)
        return (frames + lost) / self.fs

    def frame_of_elapsed(self, seconds) -> np.ndarray:
        """The inverse of :meth:`elapsed`: the fractional frame at each time after frame 0,
        NaN for a time in a gap or outside the stored frames."""
        seconds = np.asarray(seconds, dtype=np.float64)
        n = self.num_samples()
        if len(self._anchor_frames):
            first = float(self._anchor_frames[0])
            offsets = (seconds - first / self.fs) * self.tick_rate
            k = np.searchsorted(self._anchor_offsets, offsets, side="right") - 1
            frame = np.where(
                k >= 0,
                self._anchor_frames[np.maximum(k, 0)]
                + (offsets - self._anchor_offsets[np.maximum(k, 0)]) / self._r,
                seconds * self.fs,
            )
            ends = np.append(self._anchor_frames[1:], n)
            stale = (k >= 0) & (frame >= ends[np.maximum(k, 0)])
        else:
            frame = seconds * self.fs
            stale = np.zeros(seconds.shape, dtype=bool)
            if self.gaps is not None and len(self.gaps):
                starts = self.gaps[:, 0] + np.concatenate(
                    [[0], np.cumsum(self.gaps[:-1, 1])]
                )
                lost = np.cumsum(self.gaps[:, 1])
                at = seconds * self.fs
                k = np.searchsorted(starts, at, side="right") - 1
                ok = k >= 0
                frame = np.where(ok, at - lost[np.maximum(k, 0)], at)
                stale = ok & (
                    at < starts[np.maximum(k, 0)] + self.gaps[np.maximum(k, 0), 1]
                )
        return np.where(stale | (frame < 0) | (frame >= n), np.nan, frame)

    def valid_spans(self, segment: int = 0, channels=None) -> np.ndarray:
        """``(k, 2)`` int64 ``[start, stop)`` frame ranges in which consecutive frames are
        adjacent in time (no gap, no jump of the device clock) and the given channels (every
        row by default) hold delivered samples (no dropped span). A window inside one span
        is contiguous, real signal."""
        self._segment(segment)
        n = self.num_samples()
        cuts = {0, n}
        if self.gaps is not None:
            cuts.update(int(f) for f in self.gaps[:, 0])
        cuts.update(int(f) for f in self._anchor_frames[1:])
        cuts = sorted(c for c in cuts if 0 <= c <= n)
        spans = [(a, b) for a, b in itertools.pairwise(cuts) if b > a]

        rows = (
            set(self._keys)
            if channels is None
            else {self._keys[c] for c in self._columns(channels)}
        )
        holes = [
            (int(s), int(s) + int(m))
            for s, m, key in zip(
                self._dropped["frame_start"],
                self._dropped["n_frames"],
                self._dropped["channel"],
                strict=True,
            )
            if str(key) in rows and int(m) > 0
        ]
        for lo, hi in sorted(holes):
            kept = []
            for a, b in spans:
                if hi <= a or lo >= b:
                    kept.append((a, b))
                    continue
                if lo > a:
                    kept.append((a, lo))
                if hi < b:
                    kept.append((hi, b))
            spans = kept
        return np.asarray(spans, dtype=np.int64).reshape(-1, 2)

    def stim_markers(self, segment: int = 0) -> StimMarkers:
        """The stimulation edges the acquisition detected, sorted by tick and placed on frames
        (:class:`StimMarkers`)."""
        self._segment(segment)
        table = _table(self._root, "stim_markers", ("tick", "channel"))
        if table is None:
            raise RSFError(
                f"{self.path} has no rsf/stim_markers table: it is not an RSF store"
            )
        ticks = table["tick"].astype(np.uint64)
        if len(ticks) and not len(self._anchor_ticks):
            raise RSFError(
                f"{self.path} has stim markers and no tick anchor, so they name no frame"
            )
        order = np.argsort(ticks, kind="stable")
        ticks = ticks[order]
        channels = np.asarray([str(c) for c in table["channel"][order]])
        return StimMarkers(tick=ticks, frame=self.frame_at(ticks), channel=channels)

    def ext(self, producer: str, name: str):
        """The JSON document ``rsf/ext/<producer>/<name>``, or ``None``: what a producer keeps
        that the format does not define (opaque to everyone else)."""
        path = os.path.join(self._documents, "ext", producer, name)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def sortings(self) -> list[str]:
        """The session's sorting stores besides its ground truth, as paths relative to it."""
        found = []
        for entry in sorted(os.listdir(self.path)):
            here = os.path.join(self.path, entry)
            if entry.endswith(".zarr"):
                candidates = [entry] if entry != "traces.zarr" else []
            elif os.path.isdir(here) and entry not in ("ground_truth", "uncertainty"):
                candidates = [
                    f"{entry}/{name}"
                    for name in sorted(os.listdir(here))
                    if name.endswith(".zarr")
                ]
            else:
                candidates = []
            found += [
                c
                for c in candidates
                if os.path.isfile(os.path.join(self.path, c, "unit_ids", "zarr.json"))
            ]
        return found

    def spikes(self, segment: int = 0, sorting: str | None = None) -> Events:
        """The ground-truth spikes, or with ``sorting`` those of one of :meth:`sortings`, named
        by its path or, when that is unambiguous, by its name."""
        self._segment(segment)
        from livn.run import Events

        ids, samples = self._sorting(sorting)
        return Events(
            ids=ids,
            times=samples * self._dt,
            t0=self._t0,
            duration=self.num_samples() * self._dt,
        )

    def stimulation_records(self) -> list[dict]:
        """The ``stimulation`` attribute of every ``rsf/stimulation/<NNNN>/``, in start order,
        each as written plus its ``index`` (the ``program`` of its ``stim_events`` rows). A
        record with ``kind`` has the neutral members; the producer's own member (``livn``,
        ...) is opaque to every other reader."""
        root = os.path.join(self._documents, "stimulation")
        if not os.path.isdir(root):
            return []
        records = []
        for name in sorted(os.listdir(root)):
            if not re.fullmatch(r"[0-9]{4}", name):
                continue
            record = _read_node(os.path.join(root, name)).get("stimulation")
            if isinstance(record, Mapping):
                records.append({**record, "index": int(name)})
        return records

    def planned(self, index: int) -> Planned | None:
        """The plan of the stimulation record ``index``, ``rsf/stimulation/NNNN/planned``
        (:class:`Planned`: ``time``, ms from the program's start, ``channel``, ``value``,
        ``event`` and ``load``), or ``None`` when the record states none."""
        table = _table(
            self._root, f"stimulation/{int(index):04d}/planned", PLANNED_COLUMNS
        )
        if table is None:
            return None
        return Planned(
            time=np.asarray(table["t_ms"], dtype=np.float64),
            channel=np.asarray([str(c) for c in table["channel"]], dtype=str),
            value=np.asarray(table["value"]).astype(np.float32),
            event=np.asarray(table["event"]).astype(np.int64),
            load=np.asarray(table["load"]).astype(np.int64),
        )

    def check_delivered(self, index: int, tolerance: int = 1) -> dict:
        """What the record ``index`` delivered against its plan."""
        plan = self.planned(index)
        report: dict[str, Any] = {
            "index": int(index),
            "planned": plan is not None,
            "aligned": [],
            "missing": [],
            "extra": [],
            "values": [],
            "mistimed": [],
            "stop": None,
            "ok": plan is not None,
        }
        if plan is None:
            return report
        rows = self.stim_events()
        mine = rows.program == int(index)
        delivered = {
            (int(e), str(c)): (int(f), float(v), str(t))
            for e, c, f, v, t in zip(
                rows.event[mine],
                rows.channel[mine],
                rows.frame[mine],
                rows.value[mine],
                rows.timing[mine],
                strict=True,
            )
        }
        planned = {
            (int(e), str(c)): (float(t), float(v), int(load))
            for e, c, t, v, load in zip(
                plan.event, plan.channel, plan.time, plan.value, plan.load, strict=True
            )
        }
        report["extra"] = sorted(set(delivered) - set(planned))
        for key in sorted(set(planned) & set(delivered)):
            if abs(planned[key][1] - delivered[key][1]) > 1e-6:
                report["values"].append((*key, planned[key][1], delivered[key][1]))

        def expected(key):
            frame_a, t_a = anchors[planned[key][2]]
            return frame_a + int(_nearest((planned[key][0] - t_a) * self.fs / 1000.0))

        anchors: dict[int, tuple[int, float]] = {}
        shared = [key for key in delivered if key in planned]
        for load in sorted({p[2] for p in planned.values()}):
            keys = [key for key in shared if planned[key][2] == load]
            if not keys:
                continue
            best = max(TIMING_RANK.get(delivered[key][2], 0) for key in keys)
            anchor = min(
                key for key in keys if TIMING_RANK.get(delivered[key][2], 0) == best
            )
            anchors[load] = (delivered[anchor][0], planned[anchor][0])
            report["aligned"].append(
                {
                    "load": load,
                    "event": anchor[0],
                    "frame": delivered[anchor][0],
                    "timing": delivered[anchor][2],
                }
            )

        # a stop: the plan's last event delivered, early or with keys before it missing
        last = int(plan.event.max()) if len(plan) else None
        absent = sorted(set(planned) - set(delivered))
        closing = [key for key in shared if key[0] == last]
        cut = [key for key in absent if last is not None and key[0] < last]
        early = any(delivered[key][0] < expected(key) - tolerance for key in closing)
        stopped = bool(closing) and bool(cut or early)
        if stopped:
            report["stop"] = {
                "frame": min(delivered[key][0] for key in closing),
                "event": last,
                "undelivered": cut,
            }
            absent = [key for key in absent if key not in cut]
        report["missing"] = absent

        for key in sorted(shared):
            if stopped and key[0] == last:
                continue  # the closing event of a stopped program is at the stop
            if abs(delivered[key][0] - expected(key)) > tolerance:
                report["mistimed"].append((*key, delivered[key][0], expected(key)))
        report["ok"] = not (
            report["extra"]
            or report["missing"]
            or report["mistimed"]
            or report["values"]
        )
        return report

    def stimulus_channels(self) -> list[dict]:
        """The channels of every record with the neutral members, each the channel object
        plus its record's ``kind`` and ``index``, once per ``id`` (one id is one shape and one
        unit in every record that uses it), in order of first appearance: the columns of
        :meth:`inputs`. For a session livn wrote, the stimulating IO's channels in order."""
        seen: dict[str, dict] = {}
        for record in self.stimulation_records():
            if "kind" not in record:
                continue
            for channel in record.get("channels") or ():
                if isinstance(channel, Mapping) and isinstance(channel.get("id"), str):
                    seen.setdefault(
                        channel["id"],
                        {**channel, "kind": record["kind"], "index": record["index"]},
                    )
        return list(seen.values())

    def stim_events(self, resolve: bool = True) -> StimEvents:
        """The rows of ``rsf/stim_events``, read to the shortest column.

        Resolved by default: of the rows of one key ``(program, event, channel)`` the one whose
        timing source ranks highest wins (``marker`` > ``from_marker`` > ``device_clock`` >
        ``schedule``; ``simulated`` stands alone; an unknown source lowest), ties to the later
        row, and the winners are ordered by frame, then program, event and channel.
        ``resolve=False`` gives every row in table order. A store from before the table has
        none."""
        table = _table(self._root, "stim_events", STIM_COLUMNS)
        if table is None:
            table = _no_events()
        rows = StimEvents(
            frame=np.asarray(table["frame"]).astype(np.int64),
            channel=np.asarray([str(c) for c in table["channel"]], dtype=str),
            value=np.asarray(table["value"]).astype(np.float32),
            program=np.asarray(table["program"]).astype(np.int64),
            event=np.asarray(table["event"]).astype(np.int64),
            timing=np.asarray([str(t) for t in table["timing"]], dtype=str),
        )
        if not resolve or not len(rows):
            return rows
        _, channel = np.unique(rows.channel, return_inverse=True)
        sources, timing = np.unique(rows.timing, return_inverse=True)
        rank = np.asarray([TIMING_RANK.get(str(t), 0) for t in sources])[timing]
        n = len(rows)
        # within a key, the best source last and of equal sources the later row last
        order = np.lexsort((np.arange(n), rank, channel, rows.event, rows.program))
        key = np.stack([rows.program[order], rows.event[order], channel[order]])
        last = np.ones(n, dtype=bool)
        last[:-1] = np.any(key[:, 1:] != key[:, :-1], axis=0)
        kept = order[last]
        kept = kept[
            np.lexsort(
                (channel[kept], rows.event[kept], rows.program[kept], rows.frame[kept])
            )
        ]
        return StimEvents(**{name: getattr(rows, name)[kept] for name in STIM_COLUMNS})

    def stimulus_full_scales(self) -> dict[str, dict]:
        """What a value of 1.0 delivers, per channel id of :meth:`stimulus_channels`:
        ``{"units", "value", "source", "measured_at", "method", "text"}``. The channel's current
        ``stimulus_calibration`` version (``source`` ``"calibration"``) unless there is none or
        it is removed; else the record's ``full_scale`` (``source`` ``"record"``), whose value
        is ``None`` for light nobody measured: such a channel is known only relatively."""
        out: dict[str, dict] = {}
        for channel in self.stimulus_channels():
            scale = channel.get("full_scale") or {}
            out[channel["id"]] = {
                "units": scale.get("units"),
                "value": scale.get("value"),
                "source": "record",
                "measured_at": None,
                "method": None,
                "text": None,
            }
        versions = _read_notes(
            os.path.join(self._documents, "notes"), (CALIBRATION,)
        ).get(CALIBRATION, {})
        for item, history in versions.items():
            current = _current(history)
            if item not in out or current is None or current["removed"]:
                continue
            if current["full_scale"].get("units") != out[item]["units"]:
                continue  # a calibration states the channel's own units
            out[item] = {
                "units": current["full_scale"]["units"],
                "value": current["full_scale"]["value"],
                "source": "calibration",
                "measured_at": current["measured_at"],
                "method": current["method"],
                "text": current["text"],
            }
        return out

    def stimulus_wavelengths(self) -> dict[str, dict]:
        """The wavelength of each channel of :meth:`stimulus_channels`, in nm:
        ``{"value", "basis", "source"}``. The channel's current ``stimulus_calibration``
        version's ``wavelength_nm`` when that version has one (``source`` ``"calibration"``),
        else the record's ``wavelength_nm`` (``source`` ``"record"``), else ``None`` with
        ``source`` ``None``. ``basis`` comes with the value from the same place: its
        ``wavelength_basis``, one of ``measured`` (a spectrometer), ``stated`` (a person) and
        ``device_default`` (the instrument's documented source), or ``None`` when that place
        does not say. Channels that are not light have no wavelength and are ``None`` too."""
        out: dict[str, dict] = {}
        for channel in self.stimulus_channels():
            nm = _wavelength(channel.get("wavelength_nm"))
            out[channel["id"]] = {
                "value": nm,
                "basis": None
                if nm is None
                else _string(channel.get("wavelength_basis")),
                "source": None if nm is None else "record",
            }
        versions = _read_notes(
            os.path.join(self._documents, "notes"), (CALIBRATION,)
        ).get(CALIBRATION, {})
        for item, history in versions.items():
            current = _current(history)
            if item not in out or current is None or current["removed"]:
                continue
            nm = _wavelength(current.get("wavelength_nm"))
            if nm is None:
                continue
            out[item] = {
                "value": nm,
                "basis": _string(current.get("wavelength_basis")),
                "source": "calibration",
            }
        return out

    def stimulus_calibrations(self) -> dict[str, dict]:
        """The current ``stimulus_calibration`` version of each channel that has one (not
        removed), by channel id: ``{"full_scale", "measured_at", "method", "text",
        "wavelength_nm", "wavelength_basis", "latency_ms", "position", "at", "author"}``,
        each ``None`` where the version does not say. ``latency_ms`` is how long after the
        delivered edge the stimulus reaches the sample, stated by the version's measurement.
        ``position`` is where the version places the channel in the probe frame, ``{"x",
        "y", "radius", "uncertainty", "basis"}``."""
        versions = _read_notes(
            os.path.join(self._documents, "notes"), (CALIBRATION,)
        ).get(CALIBRATION, {})
        out = {}
        for item, history in sorted(versions.items()):
            current = _current(history)
            if current is None or current["removed"]:
                continue
            out[item] = {
                "full_scale": dict(current["full_scale"]),
                "measured_at": _string(current.get("measured_at")),
                "method": _string(current.get("method")),
                "text": current["text"],
                "wavelength_nm": _wavelength(current.get("wavelength_nm")),
                "wavelength_basis": _string(current.get("wavelength_basis")),
                "latency_ms": _latency(current.get("latency_ms")),
                "position": _noted_position(current, self.electrode_map.get("wiring")),
                "at": current["at"],
                "author": current["author"],
            }
        return out

    def epochs(self) -> list[dict]:
        """The session's named spans, the current version of each ``rsf/notes/epoch/``
        item that is not removed, ordered by ``frame`` and then ``end_frame``: ``{"item",
        "frame", "end_frame", "kind", "text", "parent", "programs", "origin", "open",
        "parent_missing", "author", "at", "carried_from"}``.

        An epoch still open in a complete store ends at the store's length, with ``open``
        ``True``; in a growing one ``end_frame`` is ``None``. A ``parent`` that names no
        current epoch is read as ``None``, with ``parent_missing`` ``True``."""
        versions = _read_notes(os.path.join(self._documents, "notes"), (EPOCH,)).get(
            EPOCH, {}
        )
        current = {
            item: version
            for item, version in (
                (item, _current(history)) for item, history in versions.items()
            )
            if version is not None and not version["removed"]
        }
        n = self.num_samples()
        out = []
        for item, version in current.items():
            parent = version["parent"]
            missing = parent is not None and parent not in current
            end = version["end_frame"]
            out.append(
                {
                    "item": item,
                    "frame": int(version["frame"]),
                    "end_frame": n if end is None and self.finished else end,
                    "kind": version["kind"],
                    "text": version["text"],
                    "parent": None if missing else parent,
                    "programs": None
                    if version["programs"] is None
                    else [int(p) for p in version["programs"]],
                    "origin": version["origin"],
                    "open": end is None,
                    "parent_missing": missing,
                    "author": version["author"],
                    "at": version["at"],
                    "carried_from": _string(version.get("carried_from")),
                }
            )
        return sorted(
            out,
            key=lambda e: (
                e["frame"],
                math.inf if e["end_frame"] is None else e["end_frame"],
                e["item"],
            ),
        )

    def epoch_of(self, index: int) -> list[dict]:
        """The epochs the stimulation record ``index`` belongs to: those that name it in
        ``programs``, and of those whose ``programs`` is ``None`` the ones whose span holds
        the record's first delivered frame."""
        rows = self.stim_events()
        frames = rows.frame[rows.program == int(index)]
        first = int(frames.min()) if len(frames) else None
        out = []
        for epoch in self.epochs():
            if epoch["programs"] is not None:
                if int(index) in epoch["programs"]:
                    out.append(epoch)
                continue
            end = epoch["end_frame"]
            if (
                first is not None
                and epoch["frame"] <= first
                and (end is None or first < end)
            ):
                out.append(epoch)
        return out

    def sources(self) -> dict | None:
        """The simulated sources of ``simulation.json``: ``{"ids", "coordinates_um",
        "populations", "expression"}``, each ``None`` where the run record does not say
        (``coordinates_um`` under a mask on ``source_positions``); ``None`` for a session
        that is not a simulation. ``expression`` is ``{construct: {"units", "values",
        "reference"}}``."""
        if not isinstance(self.simulation, Mapping):
            return None
        sources = self.simulation.get("sources")
        if not isinstance(sources, Mapping):
            return None
        ids = sources.get("ids")
        expression = sources.get("expression")
        if isinstance(expression, Mapping):
            expression = {
                str(construct): {
                    "units": _string(entry.get("units")),
                    "values": np.asarray(entry.get("values"), dtype=np.float64),
                    "reference": _string(entry.get("reference")),
                }
                for construct, entry in expression.items()
                if isinstance(entry, Mapping)
            }
        else:
            expression = None
        coordinates = sources.get("coordinates_um")
        populations = sources.get("populations")
        return {
            "ids": None if ids is None else _ids_array(ids),
            "coordinates_um": None
            if coordinates is None
            else np.asarray(coordinates, dtype=np.float64).reshape(-1, 3),
            "populations": None
            if not isinstance(populations, list)
            else [_string(p) for p in populations],
            "expression": expression,
        }

    def connections(self) -> dict | None:
        """``ground_truth/connections.zarr``, the simulated network at frame 0: ``{"pre",
        "post", "weight", "units", "reference", "sign", "delay", "receptor", "inputs"}``
        (``reference``, ``delay`` and ``receptor`` ``None`` when the store has none; ``pre``
        a source id or one of ``inputs``, the inputs without a source), or ``None`` when the
        session has none."""
        root = self._table_store(CONNECTIONS)
        if root is None:
            return None
        weight = _array_of(root, "weight")
        return {
            "pre": np.asarray([str(p) for p in _read(root, "pre")], dtype=str),
            "post": np.asarray([str(p) for p in _read(root, "post")], dtype=str),
            "weight": np.asarray(weight[...], dtype=np.float64),
            "units": _string(weight.attrs.get("units")),
            "reference": _string(weight.attrs.get("reference")),
            "sign": np.asarray(_read(root, "sign")).astype(np.int8),
            "delay": np.asarray(_read(root, "delay"), dtype=np.float32)
            if _has(root, "delay")
            else None,
            "receptor": np.asarray([str(r) for r in _read(root, "receptor")], dtype=str)
            if _has(root, "receptor")
            else None,
            "inputs": [str(i) for i in root.attrs.get("inputs") or ()],
        }

    def stimulus_coupling(self) -> dict | None:
        """``ground_truth/stimulus_coupling.zarr``: what each stimulus channel at full scale
        puts on each source, ``{"gain" [channels, sources], "units", "channel_ids",
        "source_ids"}``; ``None`` when the session has none (as under a mask on
        ``source_positions``)."""
        root = self._table_store(STIMULUS_COUPLING)
        if root is None:
            return None
        gain = _array_of(root, "gain")
        return {
            "gain": np.asarray(gain[...], dtype=np.float32),
            "units": _string(gain.attrs.get("units")),
            "channel_ids": [str(c) for c in _read(root, "channel_ids")],
            "source_ids": [str(c) for c in _read(root, "source_ids")],
        }

    def _table_store(self, name: str):
        store = os.path.join(self.path, "ground_truth", f"{name}.zarr")
        if not _complete(store):
            return None
        return _zarr().open_group(store, mode="r")

    def channels(self) -> dict[str, dict]:
        """The session's described channels that are not electrodes, by channel name, from
        ``rsf/channels.json``: ``{"kind", "contact_id", "label", "units", "scale", "bits",
        "part", "carried_from"}``, each ``None`` where the document does not say, with ``bits``
        ``{index: {"weight", "label", "source", "marks", "carried_from"}}`` keyed by the bit's
        integer index (``{}`` for a channel without bits).

        The current ``rsf/notes/channel/`` version of the item ``<channel>`` restates the
        channel's ``units``, ``scale`` and ``label``, and that of ``<channel>/<bit>`` (the bit a
        decimal index after the last ``/``) a bit's ``source``, ``label`` and ``marks``; a
        version never changes a ``kind`` or a ``weight``, and a version that names a channel
        or a bit the document does not describe is ignored. ``carried_from`` is the source
        session's ``recording_id`` of a version a derived session carried. The electrodes are
        not here: what a row of the wiring is stays the wiring's. No document: ``{}``, the store
        says nothing about its other channels."""
        path = os.path.join(self._documents, CHANNELS_DOCUMENT)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                document = json.load(f)
        except (OSError, ValueError):
            return {}
        if not isinstance(document, Mapping) or document.get("kind") != "channels":
            return {}
        described = document.get("channels")
        if not isinstance(described, Mapping):
            return {}
        out = {
            str(name): _channel_description(description)
            for name, description in described.items()
            if isinstance(description, Mapping)
        }

        versions = _read_notes(os.path.join(self._documents, "notes"), (CHANNEL,)).get(
            CHANNEL, {}
        )
        for item, history in versions.items():
            current = _current(history)
            if current is None or current["removed"]:
                continue
            carried = _string(current.get("carried_from"))
            if item in out:
                out[item].update(
                    units=_string(current.get("units")),
                    scale=float(current["scale"])
                    if _number(current.get("scale"))
                    else None,
                    label=_string(current.get("label")),
                    carried_from=carried,
                )
                continue
            name, _, index = item.rpartition("/")
            if not re.fullmatch(r"[0-9]+", index) or name not in out:
                continue
            bit = out[name]["bits"].get(int(index))
            if bit is None:
                continue  # a version gives no bit its weight
            bit.update(
                source=_string(current.get("source")),
                label=_string(current.get("label")),
                marks=_string(current.get("marks")),
                carried_from=carried,
            )
        return out

    def marking_bit(self, record: Mapping | int) -> dict | None:
        """The bit whose edges mark the stimulation record ``record`` (a record of
        :meth:`stimulation_records`, or its index), or ``None`` when no described channel
        says so: a ``digital_in`` or ``stim_sideband`` channel of :meth:`channels` with a bit
        whose ``marks`` is a channel ``id`` of the record or the record's ``kind`` (``light``,
        ``electrode``). Found by ``marks``, never by ``source``. Of several, the record's
        ``marker_channel`` first, then a bit that names one of its channels over one that names
        its kind, then ``digital_in`` before ``stim_sideband``, then by name and index.

        Returns ``{"channel", "bit", "weight", "label", "source", "marks", "gain_to_uV"}``. A
        caller finds the edges by thresholding the stored row (``traces_seg0`` as written) at
        half the ``weight``; one that reads scaled traces (:meth:`read`) divides by the row's
        ``gain_to_uV`` first, ``None`` when the channel is not a row of the store."""
        if not isinstance(record, Mapping):
            index = int(record)
            record = next(
                (r for r in self.stimulation_records() if r["index"] == index), None
            )
            if record is None:
                raise IndexError(f"{self.path} has no stimulation record {index}")
        ids = {
            c["id"]
            for c in record.get("channels") or ()
            if isinstance(c, Mapping) and isinstance(c.get("id"), str)
        }
        kind = record.get("kind")
        candidates = []
        for name, channel in self.channels().items():
            if channel["kind"] not in MARKING_KINDS:
                continue
            for index, bit in channel["bits"].items():
                marks = bit["marks"]
                if marks is None or not (marks in ids or (kind and marks == kind)):
                    continue
                rank = (
                    name != record.get("marker_channel"),
                    marks not in ids,
                    MARKING_KINDS.index(channel["kind"]),
                    name,
                    index,
                )
                candidates.append((rank, name, index, bit))
        if not candidates:
            return None
        _, name, index, bit = min(candidates, key=lambda c: c[0])
        column = self._column.get(name)
        return {
            "channel": name,
            "bit": index,
            "weight": bit["weight"],
            "label": bit["label"],
            "source": bit["source"],
            "marks": bit["marks"],
            "gain_to_uV": None if column is None else float(self._gain[column]),
        }

    def environment(
        self,
        frame: int | None = None,
        *,
        start: int | None = None,
        length: int | None = None,
        basis: str | None = None,
        segment: int = 0,
    ) -> dict[tuple[str, str], Condition]:
        """The session's conditions (e.g. temperature, ...), one :class:`Condition` per
        ``(quantity, item)``, at ``frame``, over the window ``[start, start + length)``, or by
        default over the whole session."""
        self._segment(segment)
        n = self.num_samples()
        if frame is not None:
            if start is not None or length is not None:
                raise ValueError("environment() takes a frame or a window, not both")
            a = int(frame)
            if a < 0 or a > n:
                raise IndexError(f"frame {a} is outside 0..{n}")
            b = a + 1
        else:
            a = 0 if start is None else int(start)
            a, length = _window(a, n - a if length is None else length, n)
            b = a + max(length, 1)

        out: dict[tuple[str, str], Condition] = {}
        table = _table(self._root, CONDITIONS, CONDITION_COLUMNS)
        if table is not None and len(table["frame"]):
            items = _read_node(os.path.join(self._documents, CONDITIONS)).get("items")
            items = items if isinstance(items, Mapping) else {}
            out.update(_conditions_from_table(table, items, a, b, basis))

        versions = _read_notes(
            os.path.join(self._documents, "notes"), (CONDITION,)
        ).get(CONDITION, {})
        for key, history in versions.items():
            current = _current(history)
            if current is None or current["removed"]:
                continue
            quantity, item = key.split("/", 1)
            if (quantity, item) in out:
                continue
            if basis is not None and current["basis"] != basis:
                continue
            value = float(current["value"])
            out[(quantity, item)] = Condition(
                quantity=quantity,
                item=item,
                value=value,
                units=current["units"],
                basis=current["basis"],
                origin="note",
                frames=np.asarray([a], dtype=np.int64),
                values=np.asarray([value], dtype=np.float64),
                method=_string(current.get("method")),
                text=current["text"],
                at=current["at"],
                author=current["author"],
            )
        return dict(sorted(out.items()))

    @property
    def stimulus_final_to(self) -> int:
        """The frame before which the delivered stimulus is final: the store's length once it
        is complete; while it grows, the length less the longest ``placement_lag_ms`` of its
        records, before which no more ``schedule`` row is appended ("Stimulation records").
        A record whose ``timing`` is ``marker`` may still have its rows re-placed by
        better-placed ones when the store completes."""
        n = self.num_samples()
        if self.finished:
            return n
        lags = [
            float(record.get("placement_lag_ms") or 0.0)
            for record in self.stimulation_records()
            if "kind" in record
        ]
        lag = math.ceil(max(lags, default=0.0) * self.fs / 1000.0 - 1e-9)
        return max(0, n - lag)

    def inputs(
        self, segment, start, length, *, physical: bool = False, latency: bool = False
    ) -> np.ndarray:
        """The delivered stimulus on the frame grid, float32 ``(length, channels)``, one column
        per :meth:`stimulus_channels` channel: each channel delivers 0 before its first event
        and holds an event's value until its next (the resolved :meth:`stim_events`).

        Values are as RSF stores them, fractions of each channel's full scale: light in
        ``[0, 1]``, electrodes in ``[-1, 1]`` with cathodic current positive. ``physical=True``
        multiplies by :meth:`stimulus_full_scales` into livn's quantities: mW/mm2 for light,
        uA for electrodes with cathodic current negative as livn's MEA reads it (the sign
        flips here and nowhere else); it raises when a channel's full scale is unknown.

        ``latency=True`` gives the stimulus as it reaches the sample: each channel's edges
        later by the ``latency_ms`` of its current calibration, when one states it
        (:meth:`stimulus_calibrations`), rounded to the nearest frame.

        A growing store's rows are final up to :attr:`stimulus_final_to`.
        """
        self._segment(segment)
        start, length = _window(start, length, self.num_samples())
        channels, held = self._held_stimulus()
        shifts = np.zeros(len(channels), dtype=np.int64)
        if latency:
            calibrations = self.stimulus_calibrations()
            for column, channel in enumerate(channels):
                ms = (calibrations.get(channel["id"]) or {}).get("latency_ms")
                if ms:
                    shifts[column] = int(_nearest(ms * self.fs / 1000.0))
        out = np.zeros((length, len(channels)), dtype=np.float32)
        at = np.arange(start, start + length)
        for column, (frames, values) in enumerate(held):
            if not len(frames):
                continue
            frames = frames + shifts[column]
            k = np.searchsorted(frames, at, side="right") - 1
            out[:, column] = np.where(k >= 0, values[np.maximum(k, 0)], 0.0)
        if not physical:
            return out
        scales = self.stimulus_full_scales()
        unknown = [c["id"] for c in channels if scales[c["id"]]["value"] is None]
        if unknown:
            raise Missing(
                f"{self.path}: the full scale of {unknown[:5]} is unknown (no calibration), "
                "so their stimulus is known only relatively; read physical=False"
            )
        factor = np.asarray(
            [_livn_scale(c["kind"], scales[c["id"]]["value"]) for c in channels],
            dtype=np.float64,
        )
        return (out * factor).astype(np.float32)

    def stimulus_frame(self, record: Mapping | int) -> dict:
        """Which frame a stimulation record's light shapes are in, and how they reach the
        probe's: ``{"frame", "placement", "by", "why"}``.

        RSF places a channel's ``shape`` in the probe's coordinates, and a record that says
        nothing else is read so (``frame`` ``"probe"``, ``placement`` ``None``). A producer
        may state otherwise in its own member's ``target``: a ``frame`` and a camera
        ``calibration`` taken on some electrode layout. When that layout is the session's
        electrode map the spots are in the probe frame already. When it is another layout and
        the calibration's points are labelled with electrodes the probe has, the affine map
        from those points to their contacts is ``placement`` (a 3x3 matrix, ``by``
        ``"calibration"``). Otherwise the spots are in a frame of their own, which ``frame``
        names (the stated frame and the calibration layout's hash), and ``placement`` is
        ``None``: only a placement given to :meth:`stimulus_io` puts them over the
        electrodes. ``why`` says which of these it was."""
        if not isinstance(record, Mapping):
            found = [r for r in self.stimulation_records() if r["index"] == int(record)]
            if not found:
                raise KeyError(f"{self.path} has no stimulation record {record}")
            record = found[0]
        discs = [
            c["id"]
            for c in record.get("channels") or ()
            if isinstance(c, Mapping) and (c.get("shape") or {}).get("kind") == "disc"
        ]
        noted = self._noted_positions()
        if discs and all(i in noted for i in discs):
            return {
                "frame": PROBE_FRAME,
                "placement": None,
                "by": "note",
                "why": (
                    f"the newest stimulus_calibration version of each of its {len(discs)} "
                    "discs states its position in the probe frame"
                ),
            }
        return _spot_frame(record, self.electrode_map.get("wiring"), self._contacts)

    def _noted_positions(self) -> dict[str, dict]:
        """The positions the channels' current calibration versions state, by channel id."""
        return {
            item: calibration["position"]
            for item, calibration in self.stimulus_calibrations().items()
            if calibration["position"] is not None
        }

    def stimulus_io(self, medium=None, *, placement=None):
        """livn's stimulating devices for :meth:`stimulus_channels`, in their order, so the
        IO's channels are the columns of :meth:`inputs`.

        Each :class:`~livn.io.PatternedLight` carries ``frame``, the frame its spots are in
        (``"probe"`` when they are over the electrodes of :attr:`io`), and ``placement``,
        ``{"matrix", "by"}`` or ``None``, the transform its recorded discs went through
        (:meth:`stimulus_frame`). Spots left in a frame of their own raise a warning.

        A channel whose newest ``stimulus_calibration`` version states its position in the
        probe frame is put there instead of where its record's ``shape`` and frame say,
        with the shape's radius unless the version states one (``position`` in
        :meth:`stimulus_calibrations`). A device whose spots all have one is ``frame``
        ``"probe"`` with ``placement`` ``{"matrix": None, "by": "note"}``; when only some
        do, ``placement["noted"]`` lists them.

        ``placement`` overrides all of that: a 2D transform from the recorded disc
        coordinates into the probe frame, applied to every light channel: a similarity or
        affine as a 2x3 ``[[a, b, tx], [c, d, ty]]`` or a 3x3 matrix, or a 3x3 projective
        one (a camera map, say). A centre maps through it and a radius scales by the square
        root of its local area scale. Electrode contacts are in the map's frame and stay put."""
        from livn.io import MEA, InputDevices, PatternedLight

        given = None if placement is None else _placement_matrix(placement)
        channels = self.stimulus_channels()
        if not channels:
            raise Missing(f"{self.path} has no delivered stimulus")
        records = {r["index"]: r for r in self.stimulation_records()}
        scales = self.stimulus_full_scales()
        wavelengths = self.stimulus_wavelengths()
        noted = self._noted_positions()
        runs: list[list[tuple]] = []
        for channel in channels:
            group = (
                channel["index"],
                channel["kind"],
                wavelengths[channel["id"]]["value"],
                scales[channel["id"]]["value"],
            )
            if runs and runs[-1][0][0] == group:
                runs[-1].append((group, channel))
            else:
                runs.append([(group, channel)])
        devices = []
        for run in runs:
            (_, kind, wavelength, scale), _ = run[0]
            members = [channel for _, channel in run]
            extra = {} if medium is None else {"medium": medium}
            if kind == "light":
                if wavelength is not None:
                    extra["wavelength"] = float(wavelength)
                # what the record states itself; a calibration version's position wins
                stated = _spot_frame(
                    records[run[0][0][0]],
                    self.electrode_map.get("wiring"),
                    self._contacts,
                )
                if given is not None:
                    matrix, by, frame = given, "caller", PROBE_FRAME
                    ids = []
                else:
                    matrix, by, frame = (
                        stated["placement"],
                        stated["by"],
                        stated["frame"],
                    )
                    ids = [c["id"] for c in members if c["id"] in noted]
                    if len(ids) == len(members):
                        matrix, by, frame = None, "note", PROBE_FRAME
                if frame != PROBE_FRAME:
                    warnings.warn(
                        f"{self.path}: the spots of stimulation record {run[0][0][0]} "
                        f"{'without a stated position ' if ids else ''}are in their own "
                        f"frame {frame!r}, not the probe's ({stated['why']}); their "
                        "PatternedLight is not placed over the electrodes. Pass "
                        "stimulus_io(placement=...) to place them",
                        stacklevel=2,
                    )
                members = [
                    _noted(c, noted[c["id"]])
                    if c["id"] in ids
                    else c
                    if matrix is None
                    else _placed(c, matrix)
                    for c in members
                ]
                device = PatternedLight.from_channels(
                    members,
                    irradiance=1.0 if scale is None else float(scale),
                    **extra,
                )
                device.frame = frame
                device.placement = (
                    None
                    if matrix is None and not ids
                    else {
                        "matrix": None if matrix is None else matrix.copy(),
                        "by": by,
                        **({"noted": ids} if ids and by != "note" else {}),
                    }
                )
                devices.append(device)
                continue
            rows = []
            for channel in members:
                shape = channel.get("shape") or {}
                if shape.get("kind") != "contact":
                    raise ValueError(
                        f"channel {channel['id']!r} is {shape.get('kind')!r}, not a contact: "
                        "its session withholds where it acts (a mask on electrode_positions), "
                        "so there is no electrode to model"
                    )
                contact = str(shape["contact_id"])
                if not re.fullmatch(r"-?[0-9]+", contact):
                    raise ValueError(
                        f"channel {channel['id']!r} is contact {contact!r}; livn's MEA "
                        "numbers its electrodes"
                    )
                rows.append([int(contact), float(shape["x"]), float(shape["y"]), 0.0])
            devices.append(MEA(rows, max_current_uA=float(scale), **extra))
        return devices[0] if len(devices) == 1 else InputDevices(devices)

    def _held_stimulus(self) -> tuple[list[dict], list[tuple[np.ndarray, np.ndarray]]]:
        """The channels and, per channel, its resolved events' ``(frames, values)`` in the
        order they hold: by frame, then program (start order) and event."""
        if self._held is not None:
            return self._held
        channels = self.stimulus_channels()
        if not channels:
            raise Missing(f"{self.path} has no delivered stimulus")
        rows = self.stim_events()
        # a row names a channel of the record its program is
        own = np.zeros(len(rows), dtype=bool)
        for record in self.stimulation_records():
            if "kind" in record:
                ids = [c.get("id") for c in record.get("channels") or ()]
                own |= (rows.program == record["index"]) & np.isin(rows.channel, ids)
        order = np.lexsort((rows.event, rows.program, rows.frame))
        order = order[own[order]]
        held = []
        for channel in channels:
            mine = order[rows.channel[order] == channel["id"]]
            held.append((rows.frame[mine], rows.value[mine]))
        if self.finished:
            self._held = (channels, held)
        return channels, held

    def _electrode_range(self) -> float | None:
        """The one full scale of the session's electrode channels, if they share one."""
        scales = self.stimulus_full_scales()
        values = {
            scales[c["id"]]["value"]
            for c in self.stimulus_channels()
            if c["kind"] == "electrode"
        }
        return values.pop() if len(values) == 1 and None not in values else None

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

    def conditions(self, segment=0, start=0, length=None):
        """The conditions of the stored run over a window, to project it again."""
        from livn.conditions import Conditions

        self._segment(segment)
        n = self.num_samples()
        start, length = _window(start, n - start if length is None else length, n)
        currents = self._truth_series("currents", start, length)
        inputs = None
        if self.stimulus_channels():
            inputs = self._device_inputs(segment, start, length)
        return Conditions(
            membrane_currents_uA=None if currents is None else currents.values,
            gids=None if currents is None else np.asarray(currents.ids),
            dt=self._dt,
            inputs=inputs,
            t0=self._t0 + start * self._dt,
            segment=segment,
            start=start,
        )

    def _device_inputs(self, segment, start, length) -> np.ndarray:
        fractions = self.inputs(segment, start, length)
        scales = self.stimulus_full_scales()
        factor = np.asarray(
            [
                _livn_scale(c["kind"], scales[c["id"]]["value"])
                if c["kind"] == "electrode"
                else 1.0
                for c in self.stimulus_channels()
            ],
            dtype=np.float64,
        )
        return (fractions * factor).astype(np.float32)

    def _segment(self, segment: int) -> None:
        if segment != 0:
            raise IndexError(f"an rsf session has one segment, not segment {segment}")

    def _anchor_of_frames(self, frames: np.ndarray) -> np.ndarray:
        """Index of the last tick anchor at or before each frame, -1 before the first."""
        return np.searchsorted(self._anchor_frames, frames, side="right") - 1

    def _columns(self, channels) -> np.ndarray:
        try:
            return np.asarray(
                [self._column[_id_str(c)] for c in np.asarray(channels).tolist()]
            )
        except KeyError as e:
            raise KeyError(f"{e.args[0]!r} is not a channel of {self.path}") from None

    def _sorting(self, sorting: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``(unit ids, frames)`` of the ground-truth spikes or of a sorting, ordered by frame."""
        if sorting is None:
            relative = "ground_truth/spikes.zarr"
        else:
            stores = self.sortings()
            named = [
                s
                for s in stores
                if sorting in (s, s.removesuffix(".zarr"))
                or os.path.basename(s) in (sorting, f"{sorting}.zarr")
            ]
            if len(named) != 1:
                which = "matches several" if named else "is none"
                raise Missing(
                    f"{sorting!r} {which} of the sortings of {self.path}: {stores}"
                )
            relative = named[0]
        store = os.path.join(self.path, relative)
        if not _complete(store):
            raise Missing(f"{self.path} stores no finished {relative}")
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

    def conditions(self, segment=0, start=0, length=None):
        """See :meth:`RsfRecording.conditions`."""
        return self.recording.conditions(segment, start, length)

    def environment(self, frame=None, **kwargs) -> dict[tuple[str, str], Condition]:
        return self.recording.environment(frame, **kwargs)

    def channels(self) -> dict[str, dict]:
        return self.recording.channels()

    def marking_bit(self, record) -> dict | None:
        return self.recording.marking_bit(record)


def load(path: str | os.PathLike) -> Session:
    return Session(RsfRecording(path))


def same_preparation(a: RsfRecording, b: RsfRecording) -> bool:
    """Whether two sessions are of one preparation: equal ``substrate`` and ``prep.id``. A
    physical preparation and a simulation of it are two (the simulation ``models`` it)."""
    if a.prep is None or b.prep is None or a.prep["id"] is None:
        return False
    return a.substrate == b.substrate and a.prep["id"] == b.prep["id"]


def preparations(recordings) -> dict[tuple[str | None, str], list[RsfRecording]]:
    """The sessions of each preparation, keyed by ``(substrate, prep id)``, each ordered by
    the preparation's age and then by ``created_at``. A session without a prep id is left
    out."""
    out: dict[tuple[str | None, str], list[RsfRecording]] = {}
    for recording in recordings:
        prep = recording.prep
        if prep is None or prep["id"] is None:
            continue
        out.setdefault((recording.substrate, prep["id"]), []).append(recording)

    def order(recording):
        age = recording.prep["age"]
        created = recording.created_at
        return (
            age is None,
            0.0 if age is None else age["value"],
            _instant(created) if isinstance(created, str) else datetime.max,
        )

    return {key: sorted(group, key=order) for key, group in out.items()}


def join_epochs(recordings) -> list[dict]:
    """The epochs of several sessions joined by ``item``: one run that spans sessions
    (the days of a preparation, the members of a group) is one entry, ``{"item", "kind",
    "text", "parent", "origin", "sessions": [{"recording_id", "frame", "end_frame",
    "programs", "open"}]}``, sessions in the order given. ``kind``, ``text``, ``parent``
    and ``origin`` are those of the first session that has the epoch."""
    joined: dict[str, dict] = {}
    for recording in recordings:
        recording_id = recording.record.get("recording_id")
        for epoch in recording.epochs():
            entry = joined.setdefault(
                epoch["item"],
                {
                    "item": epoch["item"],
                    "kind": epoch["kind"],
                    "text": epoch["text"],
                    "parent": epoch["parent"],
                    "origin": epoch["origin"],
                    "sessions": [],
                },
            )
            entry["sessions"].append(
                {
                    "recording_id": recording_id,
                    "frame": epoch["frame"],
                    "end_frame": epoch["end_frame"],
                    "programs": epoch["programs"],
                    "open": epoch["open"],
                }
            )
    return list(joined.values())


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


def _read_operator(
    path: str, ext_path: str | None = None, max_current_uA: float | None = None
):
    from livn.io import MEA, ComposedIO, Medium, PointSource

    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    if (
        document.get("kind") not in (POINT_SOURCE, LIVN_IO)
        or "electrode_coordinates_um" not in document
    ):
        return None
    coordinates = document["electrode_coordinates_um"]
    if not all(isinstance(row[0], int) for row in coordinates):
        return None  # livn's MEA numbers its electrodes
    livn = document.get("livn")
    if livn is None and ext_path is not None and os.path.isfile(ext_path):
        with open(ext_path, encoding="utf-8") as f:
            livn = json.load(f)
    transfers = []
    recording = stimulating = None
    if livn is not None:
        medium = Medium(**{k: v for k, v in livn["medium"].items() if k != "cls"})
        transfers = livn["transfers"]
        recording = livn.get("recording")
        stimulating = livn.get("inputs")
    else:
        radius = document.get("electrode_radius_um", 15.0)
        height = document.get("culture_height_um", 50.0)
        sigma = 1e6 / (
            4.0 * math.pi * document["recording_gain"] * document["min_distance_um"]
        )
        rho = None
        if "stimulation_gain" in document:
            h_eff = (radius**2 + height**2) ** 0.5
            rho = document["stimulation_gain"] * 4.0 * math.pi * h_eff / 1000.0
        medium = Medium(
            electrical=PointSource(
                sigma_s_per_m=sigma,
                height=height,
                min_distance=document["min_distance_um"],
                stimulation_resistivity_ohm_m=rho,
            )
        )
    kwargs: dict[str, Any] = {
        "electrode_coordinates": np.asarray(coordinates, dtype=np.float64),
        "output_radius": document["output_radius_um"],
        "medium": medium,
        "detection_threshold": document.get("detection_threshold", 0.0),
    }
    if "input_radius_um" in document:
        kwargs["input_radius"] = document["input_radius_um"]
    if "electrode_radius_um" in document:
        kwargs["electrode_radius"] = document["electrode_radius_um"]
    if max_current_uA is not None:
        kwargs["max_current_uA"] = max_current_uA
    mea = MEA(**kwargs)
    if recording is not None:
        return ComposedIO(stimulating, mea, medium=medium, recording=recording)
    if not transfers:
        return mea
    return ComposedIO(None, [mea, *transfers], medium=medium)


def _json_file(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _probe_contacts(probe, keys: list[str], names, described: set[str]) -> list[dict]:
    """The probe's connected contacts that are electrodes, in row order: ``{"id", "row",
    "x", "y", "radius"}``, ``row`` the row's key and ``radius`` ``None`` for a shape that is
    not a circle. A row ``rsf/channels.json`` describes (a reference, an auxiliary input) is
    not an electrode, whatever the probe says."""
    if not isinstance(probe, Mapping):
        return []
    contacts: list[tuple[int, dict]] = []
    for one in probe.get("probes") or ():
        if not isinstance(one, Mapping):
            continue
        ids = one.get("contact_ids") or ()
        positions = one.get("contact_positions") or ()
        rows = one.get("device_channel_indices") or ()
        shapes = one.get("contact_shapes") or ()
        params = one.get("contact_shape_params") or ()
        for k, (id_, position, row) in enumerate(
            zip(ids, positions, rows, strict=False)
        ):
            if not isinstance(row, int) or not 0 <= row < len(keys):
                continue  # unconnected
            name = keys[row] if names is None else str(names[row])
            if name in described or keys[row] in described:
                continue
            radius = None
            if k < len(shapes) and shapes[k] == "circle" and k < len(params):
                value = (params[k] or {}).get("radius")
                if isinstance(value, int | float) and not isinstance(value, bool):
                    radius = float(value)
            contacts.append(
                (
                    row,
                    {
                        "id": str(id_),
                        "row": keys[row],
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "radius": radius,
                    },
                )
            )
    contacts.sort(key=lambda c: c[0])
    return [contact for _, contact in contacts]


def _probe_io(contacts: list[dict], max_current_uA: float | None):
    """An MEA of the probe's electrodes at their positions, in row order; ``None`` when there
    are none, or when an id is not a number (livn's MEA numbers its electrodes)."""
    from livn.io import MEA

    if not contacts or not all(re.fullmatch(r"-?[0-9]+", c["id"]) for c in contacts):
        return None
    if len({int(c["id"]) for c in contacts}) != len(contacts):
        return None
    kwargs: dict[str, Any] = {}
    radii = {c["radius"] for c in contacts}
    if len(radii) == 1 and None not in radii:
        kwargs["electrode_radius"] = radii.pop()
    if max_current_uA is not None:
        kwargs["max_current_uA"] = max_current_uA
    return MEA(
        [[int(c["id"]), c["x"], c["y"], 0.0] for c in contacts],
        **kwargs,
    )


def _stated_target(record: Mapping) -> Mapping | None:
    """The ``target`` a producer's own member of ``record`` states a frame in, if any."""
    for key, member in record.items():
        if key in RECORD_MEMBERS or not isinstance(member, Mapping):
            continue
        target = member.get("target")
        if isinstance(target, Mapping) and isinstance(target.get("frame"), str):
            return target
    return None


def _spot_frame(record: Mapping, wiring, contacts: list[dict]) -> dict:
    """See :meth:`RsfRecording.stimulus_frame`."""

    def own(frame, why):
        return {"frame": frame, "placement": None, "by": None, "why": why}

    def probe(why, placement=None, by="record"):
        return {"frame": PROBE_FRAME, "placement": placement, "by": by, "why": why}

    target = _stated_target(record)
    if target is None:
        return probe("the record states no other frame, and RSF's is the probe's")
    stated = target["frame"]
    key = re.sub(r"[^a-z]", "", stated.lower())
    if key in ("probe", "probeinterfaceum"):
        return probe(f"the record states the frame {stated!r}")
    if key != "sampleum":
        return own(stated, f"the record states the device frame {stated!r}")
    calibration = target.get("calibration")
    if not isinstance(calibration, Mapping):
        return own(
            stated,
            f"the record states the frame {stated!r} and no calibration that relates it "
            "to the probe",
        )
    layout = _string(calibration.get("wiring"))
    name = stated if layout is None else f"{stated}:{layout[:12]}"
    if layout is not None and layout == wiring:
        return probe(
            f"the record's frame {stated!r} was calibrated on the session's own electrode map"
        )
    fit = _labelled_fit(calibration.get("sample_points"), contacts)
    if isinstance(fit, str):
        return own(
            name,
            f"the record's frame {stated!r} is calibrated on another electrode layout "
            f"({layout or 'not named'}), and {fit}",
        )
    matrix, n, rms = fit
    return probe(
        f"the record's frame {stated!r} is calibrated on another electrode layout "
        f"({layout or 'not named'}); its {n} labelled calibration points fit their probe "
        f"contacts to {rms:.2g} um RMS",
        placement=matrix,
        by="calibration",
    )


def _labelled_fit(points, contacts: list[dict]):
    """The affine map from a calibration's labelled points to the probe contacts their labels
    name, as ``(matrix, n, rms)``, or why there is none. A label names a contact by its id,
    with or without a leading letter prefix (``"e21"`` for contact ``"21"``)."""
    position = {c["id"]: (c["x"], c["y"]) for c in contacts}
    source, destination = [], []
    for point in points if isinstance(points, Sequence) else ():
        if not isinstance(point, Mapping):
            continue
        label, to = _string(point.get("label")), point.get("to")
        if label is None or not isinstance(to, Sequence) or len(to) != 2:
            continue
        contact = label if label in position else re.sub(r"^[A-Za-z]+", "", label)
        if contact in position:
            source.append([float(to[0]), float(to[1]), 1.0])
            destination.append(position[contact])
    if len(source) < 3:
        return (
            f"only {len(source)} of its calibration points name an electrode of the "
            "probe, too few to fit"
        )
    a = np.asarray(source)
    b = np.asarray(destination, dtype=np.float64)
    if np.linalg.matrix_rank(a, tol=1e-6 * max(1.0, np.abs(a).max())) < 3:
        return "its labelled calibration points are collinear"
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((a @ solution - b) ** 2, axis=1))))
    if rms > CALIBRATION_FIT_TOLERANCE:
        return (
            f"its labelled calibration points fit their probe contacts only to {rms:.3g} "
            "um RMS"
        )
    matrix = np.eye(3)
    matrix[:2, :] = solution.T
    return matrix, len(source), rms


def _placement_matrix(placement) -> np.ndarray:
    """A 2x3 affine or a 3x3 (affine or projective) transform as a 3x3 matrix, checked."""
    matrix = np.asarray(placement, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(
            f"a placement is a 2D transform, 2x3 or 3x3, not shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("a placement is finite and not singular")
    return matrix / matrix[2, 2] if matrix[2, 2] != 0 else matrix


def _placed(channel: Mapping, matrix: np.ndarray) -> dict:
    """A light channel with its disc's centre moved by ``matrix``, its radius scaled by the
    square root of the transform's local area scale there."""
    shape = channel.get("shape") or {}
    if shape.get("kind") != "disc" or any(shape.get(k) is None for k in "xy"):
        return dict(channel)  # what is not a disc, from_channels refuses as it is
    x, y = float(shape["x"]), float(shape["y"])
    u, v, w = matrix @ np.asarray([x, y, 1.0])
    if w == 0:
        raise ValueError(
            f"the placement sends channel {channel.get('id')!r} to infinity"
        )
    out = {**shape, "x": float(u / w), "y": float(v / w)}
    if shape.get("radius") is not None:
        # the Jacobian of (u/w, v/w) at the centre; for an affine, the linear part
        jacobian = (matrix[:2, :2] - np.outer([u / w, v / w], matrix[2, :2])) / w
        scale = abs(float(np.linalg.det(jacobian))) ** 0.5
        out["radius"] = float(shape["radius"]) * scale
    return {**channel, "shape": out}


def _noted_position(version: Mapping, wiring) -> dict | None:
    """The position a ``stimulus_calibration`` version states for its channel, or ``None``.

    The format has no member for it, so it is livn's own, under ``ext.livn``:
    ``position_um`` ``{"x", "y"}`` in the probe frame (``frame`` ``"probe"``), and
    optionally ``radius_um``, ``position_uncertainty_um``, ``position_basis`` and the
    ``electrode_map`` ``{"design", "wiring"}`` it was placed on. A position placed on
    another electrode map than the session's is not in this probe's frame and is ignored.
    """
    ext = version.get("ext")
    stated = ext.get(PRODUCER) if isinstance(ext, Mapping) else None
    if not isinstance(stated, Mapping) or stated.get("frame") != PROBE_FRAME:
        return None
    position = stated.get("position_um")
    if not isinstance(position, Mapping):
        return None
    x, y = position.get("x"), position.get("y")
    if not (_number(x) and _number(y) and math.isfinite(x) and math.isfinite(y)):
        return None
    on = stated.get("electrode_map")
    placed_on = _string(on.get("wiring")) if isinstance(on, Mapping) else None
    if placed_on is not None and wiring is not None and placed_on != wiring:
        return None

    def positive(value):
        ok = _number(value) and math.isfinite(value) and value > 0
        return float(value) if ok else None

    return {
        "x": float(x),
        "y": float(y),
        "radius": positive(stated.get("radius_um")),
        "uncertainty": positive(stated.get("position_uncertainty_um")),
        "basis": _string(stated.get("position_basis")),
    }


def _noted(channel: Mapping, position: Mapping) -> dict:
    """A light channel with its disc at a ``position`` a calibration version states; the
    radius stays the shape's unless the version states one."""
    shape = channel.get("shape") or {}
    if shape.get("kind") != "disc":
        return dict(channel)
    out = {**shape, "x": position["x"], "y": position["y"]}
    if position["radius"] is not None:
        out["radius"] = position["radius"]
    return {**channel, "shape": out}


def _table(group, name: str, columns: Sequence[str]) -> dict[str, np.ndarray] | None:
    zarr = _zarr()

    node = group.get(f"{RSF}/{name}")
    if not isinstance(node, zarr.Group):
        return None
    values = {column: _read(node, column) for column in columns}
    n = min((len(v) for v in values.values()), default=0)
    return {column: v[:n] for column, v in values.items()}


_NOTE_KEY = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{32}\.json")
_NOTE_MEMBERS = {
    "item": str,
    "at": str,
    "removed": bool,
    "text": str,
}


def _read_notes(
    directory: str, kinds: Sequence[str] = NOTE_KINDS
) -> dict[str, dict[str, list[tuple]]]:
    out: dict[str, dict[str, list[tuple]]] = {}
    for kind in kinds:
        folder = os.path.join(directory, kind)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not _NOTE_KEY.fullmatch(name):
                continue
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    version = json.load(f)
                if not isinstance(version, dict) or any(
                    not isinstance(version.get(m), t) for m, t in _NOTE_MEMBERS.items()
                ):
                    continue
                if not all(m in version for m in ("zone", "author", "frame")):
                    continue
                at = _instant(version["at"])
            except (OSError, ValueError):
                continue
            item = version["item"]
            if kind == "note":
                if not re.fullmatch(r"[0-9a-f]{32}", item):
                    continue
            elif kind == CALIBRATION:
                if not _calibration(version):
                    continue
            elif kind == CONDITION:
                if not _condition(version):
                    continue
            elif kind == CHANNEL:
                if not item or version["frame"] is not None:
                    continue
            elif kind == EPOCH:
                if not _epoch(version):
                    continue
            elif item != "session":
                continue
            out.setdefault(kind, {}).setdefault(item, []).append((at, name, version))
    return out


def _epoch(version: dict) -> bool:
    """An ``epoch`` version: a hex ``item``, an integer ``frame`` and ``end_frame`` at or
    after it (or ``null``), a ``kind``, and ``parent``, ``programs`` and ``origin``."""
    if not HEX_ITEM.fullmatch(version["item"]):
        return False
    frame, end = version["frame"], version.get("end_frame", False)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        return False
    if end is not None and (
        not isinstance(end, int) or isinstance(end, bool) or end < frame
    ):
        return False
    if not isinstance(version.get("kind"), str):
        return False
    if not all(m in version for m in ("parent", "programs", "origin")):
        return False
    parent = version["parent"]
    if parent is not None and not isinstance(parent, str):
        return False
    programs = version["programs"]
    if programs is not None and (
        not isinstance(programs, list)
        or not all(isinstance(p, int) and not isinstance(p, bool) for p in programs)
    ):
        return False
    return version["origin"] is None or isinstance(version["origin"], dict)


def _latency(value) -> float | None:
    """A stimulus edge's latency in ms, or ``None`` when none is stated."""
    if not _number(value) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _calibration(version: dict) -> bool:
    scale = version.get("full_scale")
    if (
        not version["item"]
        or version["frame"] is not None
        or not isinstance(scale, dict)
    ):
        return False
    if "measured_at" not in version or "method" not in version:
        return False
    if version["removed"]:
        return not version["text"]
    value = scale.get("value")
    if value is None:
        return (
            _wavelength(version.get("wavelength_nm")) is not None
            or _latency(version.get("latency_ms")) is not None
        )
    return _number(value) and value > 0


def _channel_description(description: Mapping) -> dict:
    """A channel description of ``rsf/channels.json`` with every member, ``None`` where
    absent; bits keyed by integer index, those without a positive integer ``weight`` left
    out (a reader cannot find their edges)."""
    bits: dict[int, dict] = {}
    described = description.get("bits")
    if isinstance(described, Mapping):
        for index, bit in described.items():
            if not re.fullmatch(r"[0-9]+", str(index)) or not isinstance(bit, Mapping):
                continue
            weight = bit.get("weight")
            if not _number(weight) or weight <= 0 or weight != int(weight):
                continue
            bits[int(index)] = {
                "weight": int(weight),
                "label": _string(bit.get("label")),
                "source": _string(bit.get("source")),
                "marks": _string(bit.get("marks")),
                "carried_from": None,
            }
    scale = description.get("scale")
    return {
        "kind": _string(description.get("kind")),
        "contact_id": _string(description.get("contact_id")),
        "label": _string(description.get("label")),
        "units": _string(description.get("units")),
        "scale": float(scale) if _number(scale) else None,
        "bits": dict(sorted(bits.items())),
        "part": _string(description.get("part")),
        "carried_from": None,
    }


def _condition(version: dict) -> bool:
    """A ``condition`` version: ``item`` ``<quantity>/<item>``, ``frame`` null, ``units`` and
    ``basis`` strings, ``value`` a number (``null`` only when ``removed``) and ``method``."""
    quantity, _, item = version["item"].partition("/")
    if not quantity or not item or version["frame"] is not None:
        return False
    if not isinstance(version.get("units"), str) or not isinstance(
        version.get("basis"), str
    ):
        return False
    if "method" not in version:
        return False
    if version["removed"]:
        return True
    value = version.get("value")
    return _number(value) and math.isfinite(value)


def _conditions_from_table(
    table: dict[str, np.ndarray],
    items: Mapping,
    a: int,
    b: int,
    basis: str | None,
) -> dict[tuple[str, str], Condition]:
    """The table's conditions over the frames ``[a, b)``: per ``(quantity, item)`` the steps
    of the best basis in effect there (``measured``, then ``setpoint``, then any other)."""
    frames = np.asarray(table["frame"]).astype(np.int64)
    values = np.asarray(table["value"]).astype(np.float64)
    columns = {
        name: [str(v) for v in table[name]]
        for name in ("quantity", "item", "units", "basis")
    }
    rows: dict[tuple[str, str, str], list[int]] = {}
    for k in range(len(frames)):
        key = (columns["quantity"][k], columns["item"][k], columns["basis"][k])
        if basis is not None and key[2] != basis:
            continue
        rows.setdefault(key, []).append(k)

    best: dict[tuple[str, str], tuple] = {}
    for (quantity, item, b_name), ks in rows.items():
        ks = np.asarray(ks)
        # by frame; of rows at one frame the later in the table holds
        ks = ks[np.lexsort((ks, frames[ks]))]
        f, v = frames[ks], values[ks]
        last = np.ones(len(f), dtype=bool)
        last[:-1] = f[1:] != f[:-1]
        ks, f, v = ks[last], f[last], v[last]
        first = np.searchsorted(f, a, side="right") - 1
        stop = np.searchsorted(f, b, side="left")
        lo = max(first, 0)
        steps_f, steps_v, steps_k = f[lo:stop], v[lo:stop], ks[lo:stop]
        if not len(steps_f):
            continue  # no row of this basis in effect over the frames asked for
        rank = (CONDITION_BASIS_RANK.get(b_name, 0), b_name)
        if (quantity, item) in best and best[(quantity, item)][0] >= rank:
            continue
        best[(quantity, item)] = (rank, b_name, steps_f, steps_v, steps_k)

    out: dict[tuple[str, str], Condition] = {}
    for (quantity, item), (_, b_name, steps_f, steps_v, steps_k) in best.items():
        begins = np.maximum(steps_f, a)
        ends = np.append(begins[1:], b)
        weights = (ends - begins).astype(np.float64)
        value = (
            float(np.average(steps_v, weights=weights))
            if weights.sum() > 0
            else float(steps_v[-1])
        )
        entry = items.get(item)
        source = entry.get("source") if isinstance(entry, Mapping) else None
        out[(quantity, item)] = Condition(
            quantity=quantity,
            item=item,
            value=value,
            units=columns["units"][int(steps_k[0])],
            basis=b_name,
            origin="table",
            frames=steps_f,
            values=steps_v,
            source=_string(source),
        )
    return out


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _string(value) -> str | None:
    return value if isinstance(value, str) else None


def _wavelength(value) -> float | None:
    """A wavelength in nm, or ``None`` when ``value`` is none."""
    if not _number(value) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _derived_from(value) -> dict | None:
    """The ``derived_from`` attribute with every member RSF defines, ``None`` where absent:
    ``recording_id``, ``session``, the source's ``started_at``, ``created_at`` and ``author``
    and the derivation's ``changes`` (``[{"what", "how", "params"}]``) when it says them."""
    if not isinstance(value, Mapping):
        return None
    changes = value.get("changes")
    if isinstance(changes, list):
        changes = [
            {
                "what": _string(change.get("what")),
                "how": _string(change.get("how")),
                "params": dict(change["params"])
                if isinstance(change.get("params"), Mapping)
                else {},
            }
            for change in changes
            if isinstance(change, Mapping)
        ]
    else:
        changes = None
    return {
        "recording_id": _string(value.get("recording_id")),
        "session": _string(value.get("session")),
        "started_at": _string(value.get("started_at")),
        "created_at": _string(value.get("created_at")),
        "author": _string(value.get("author")),
        "changes": changes,
    }


def _prep(system) -> dict | None:
    """The preparation of ``env.system``, with every member RSF defines, ``None`` where
    absent: ``id``, ``label``, ``kind``, ``species``, ``age`` (``{"value", "units"}``) and
    ``expression`` (``[{"construct", "method", "since"}]``, with an ``age`` where an entry
    states one). A physical ``prep``, or a ``simulation``'s ``prep``, which also has
    ``models``: the physical preparation the simulation stands for. ``None`` for a session
    whose system names no preparation."""
    if not isinstance(system, Mapping):
        return None
    simulated = False
    prep = system.get("prep")
    if not isinstance(prep, Mapping):
        simulation = system.get("simulation")
        prep = simulation.get("prep") if isinstance(simulation, Mapping) else None
        simulated = True
    if not isinstance(prep, Mapping):
        return None
    expression = prep.get("expression")
    if isinstance(expression, list):
        entries = []
        for e in expression:
            if not isinstance(e, Mapping):
                continue
            entry = {
                "construct": _string(e.get("construct")),
                "method": _string(e.get("method")),
                "since": _string(e.get("since")),
            }
            age = _read_age(e.get("age"))
            if age is not None:
                entry["age"] = age
            entries.append(entry)
        expression = entries
    else:
        expression = None
    out = {
        "id": _string(prep.get("id")),
        "label": _string(prep.get("label")),
        "kind": _string(prep.get("kind")),
        "species": _string(prep.get("species")),
        "age": _read_age(prep.get("age")),
        "expression": expression,
    }
    if simulated:
        models = _string(prep.get("models"))
        out["models"] = None if models == out["id"] else models
    return out


def _read_age(age) -> dict | None:
    if (
        isinstance(age, Mapping)
        and _number(age.get("value"))
        and isinstance(age.get("units"), str)
    ):
        return {"value": float(age["value"]), "units": age["units"]}
    return None


def _current(versions: list[tuple] | None) -> dict | None:
    if not versions:
        return None
    return max(versions, key=lambda v: (v[0], v[1]))[2]


def _current_text(versions: list[tuple] | None) -> str:
    version = _current(versions)
    return "" if version is None or version["removed"] else str(version["text"])


def _current_label(notes: dict, attribute: str) -> str:
    version = _current(notes.get("label", {}).get("session"))
    if version is None or version["removed"]:
        return str(attribute or "")
    return str(version["text"])
