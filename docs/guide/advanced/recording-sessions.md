# Recording sessions

`livn.data.rsf` saves a run as a session in the recording session format (RSF) and reads sessions back, simulated or recorded, through the same calls.

It needs `zarr` 3 or later, which livn does not install for you (`pip install 'zarr>=3'`).

## Writing a session

```python
from livn.data.rsf import load, store

env = env.record_spikes()
env = env.record_membrane_current()
env = env.record_voltage()

run = env.run(1000)
store(run, "sessions/spontaneous-1", env, voltage=True, currents=True)
```

Under MPI, make sure to gather first `store(run.gather(env.comm), path, env)` so it writes on the root rank only.

Optional keywords:

- `label=`, `author=` and `description=`: the session's name, who made it and a description.
- `notes=[{"text", "frame", "author"}]`: notes about the session or about a frame of it.
- `withhold=[{"mask": {"fields": [...]}}]`: what the session must not reveal (`electrode_positions`, `source_positions`, `channel:<id>`).
- `gain_uV=` and `dtype=`: how the traces are quantised.
- `zone=`: an IANA time zone, or `True` for the host's.
- `ext={"<producer>": {"<name>.json": document}}`: documents of another program, read back with `recording.ext(producer, name)`.

The traces are projected through the env's [recording composition](/guide/concepts/io#recording-compositions) (`env.io.recording`, or `store(..., recording=Chain(...))`). To project a stored run again, use:

```python
session = load("sessions/spontaneous-1")
env.potential_recording(session.io.recording, session.conditions())
```

## Reading a session

```python
session = load("sessions/spontaneous-1")
session.run.potential      # the channel signal in uV, one row per electrode
session.run.spike_times    # on the frame grid of the signal
session.io                 # the MEA it was recorded through
session.simulation["seed"] # metadata
session.record             # the provenance record
```

To read windows without loading the whole session, open it with `livn.data.open`. The same calls work for an RSF session and for an Arrow dataset, local or on the Hugging Face hub:

```python
from livn.data import open as open_recording

recording = open_recording("sessions/spontaneous-1")
recording = open_recording("./systems/datasets/spe1/c16")
recording = open_recording("hf://frthjf/spe1-zhao2026-benchmark/c16", split="test")

window = recording.read(0, start=30_000, length=15_000)     # float32 [frames, channels], uV
units = recording.spikes(0)                                  # Events, ids = unit ids
```

`spikes`, `inputs`, `ground_truth` and `to_run(segment, start, length)` read windows the same way. A recording can be read from several threads and pickled.

`livn.data.arrow.store(run, path, env=env, ground_truth=("spikes", "voltage"))` writes a run as an Arrow dataset instead, one row per call.

## Stimulated sessions

To keep what drove the run, pass either `stimulus=policy` or `inputs=array, inputs_dt=ms` in the stimulating IO's input units. Reading the delivered stimulus works the same on a recorded session:

```python
recording = load("sessions/stimulated-1").recording
recording.stimulus_channels()      # the channels, the columns of inputs()
recording.inputs(0, 0, 1000)       # fractions of full scale, [frames, channels]
recording.inputs(0, 0, 1000, physical=True)  # mW/mm2 and uA (cathodic negative)
recording.stimulus_full_scales()   # {id: {"units", "value", ...}}
recording.stim_events()            # the delivered edges
recording.stimulus_io()            # PatternedLight / MEA / InputDevices of the channels
```

When a channel's calibration (`recording.stimulus_calibrations()`) states a `latency_ms` from the delivered edge to the sample, `inputs(..., latency=True)` applies it, and so do [windows](/optimization/fit#windows-of-a-recording) read with `inputs=True`.

### Several programs

A stimulus made of several programs, such as the blocks of a protocol or light and electrodes driven together, is stored with `programs=` in place of `stimulus=` and `inputs=`. Each program gives a `policy`, or `inputs` with `inputs_dt`, and optionally:

- `start` (ms) or `start_frame`;
- `stop`: ms after its start, for a program stopped early;
- `device`: an index among the stimulating devices;
- `channels`: the channels it drives (by default those it sets non-zero);
- `origin`: where the program came from (by default the policy's);
- `members`: other programs' fields for its record.

A channel is driven by one program at a time.

```python
store(run, "sessions/day-14", env, programs=[
    {"policy": mapping_policy},
    {"inputs": block, "inputs_dt": 0.1, "start": 500.0, "stop": 120.0},
])

recording = load("sessions/day-14").recording
recording.stimulation_records()[1]["channels"][0]["name"]  # a channel's site name, e.g. a spot label
recording.planned(1)            # what program 1 was to deliver
recording.check_delivered(1)    # missing, extra and mistimed edges; an early stop is reported as a stop
```

Match a program's channels by their `name`, the stimulus site, which stays the same across sessions.

## Preparations and epochs

`prep={"id", "label", "kind", "species", "age", "expression", "models"}` states the simulated preparation: `age` is its age at the start of the session, `models` the physical preparation it stands for, and `id` is generated when not given. Give the same `id` to every session of one preparation, for example its days in vitro:

```python
from livn.data import rsf

rsf.same_preparation(a, b)        # are two sessions of one preparation?
rsf.preparations(recordings)      # sessions grouped by preparation, ordered by age
recording.prep                    # the preparation of one session
```

`epochs=[...]` names spans of the session. Each epoch is `{"key" or "item", "frame" or "start", "end_frame" or "end", "kind", "text", "parent", "programs", "origin"}`, with `kind` one of `protocol`, `block`, `baseline`, `spontaneous` or `spot_mapping`, `parent` the key or item of the epoch it lies within, and `programs` indices into `programs=`. To write one run into several sessions, pass the same `item` in each; `rsf.join_epochs(recordings)` reads it back as one. `recording.epochs()` and `recording.epoch_of(index)` read a single session.

## Ground truth

- Spikes, voltages and currents per cell `voltage=True` and `currents=True`
- Populations and opsin expression per cell via `populations=True` and `expression=True` (or `expression={construct: {"units", "values", "reference"}}`)
- Connections at the start of the run via `connections=True`. Weights are multiples of the synapse's unitary conductance and presynaptic inputs that are not cells are listed separately
- Stimulus coupling via `stimulus_coupling=True` (not written when `source_positions` are withheld)

Read them back with `recording.ground_truth(name)`, `recording.sources()`, `recording.connections()` and `recording.stimulus_coupling()`.

## SpikeInterface

A session's traces open in SpikeInterface with `spikeinterface.read_zarr("sessions/spontaneous-1/traces.zarr")`, and its ground-truth spikes are a SpikeInterface sorting.
