# Snapshots

A snapshot saves a running environment's full simulation state, so that a run can be forked at a decision point and continued under different stimuli, or under different noise, as often as needed. Every branch starts from exactly the same network state; nothing has to be re-simulated from `t = 0`.

```python
from livn import make

env = make("EI")
env.record_spikes()

env.run(500)                    # bring the network to the decision point
snapshot = env.save_state()     # t = 500 ms

responses = {}
for name, stimulus in candidate_stimuli.items():
    env.restore_state(snapshot)
    responses[name] = env.run(200, stimulus)   # continues from t = 500 ms
```

`run()` after a restore continues from `snapshot.t`, with spike times relative to it as in any continued run.

## What is restored

The snapshot covers everything the next time step depends on:

- membrane potentials, channel gates, ion concentrations and opsin states
- synaptic states and, under plasticity, the learned weights
- spikes still in flight along a delayed connection, which arrive after the restore as they would have
- the position of every noise stream
- stimulus already scheduled past the snapshot, and the field or light the last step left in place

## Noise replicates

`restore_state(snapshot)` continues the snapshot's noise exactly. Passing `noise_stream` moves every noise source onto another stream from that point on, so several independent realisations can follow one snapshot. The numbering mirrors `run_many()` where the snapshot's own stream is the uninterrupted run, and each other stream is a reproducible alternative.

```python
replicates = []
for m in range(8):
    env.restore_state(snapshot, noise_stream=snapshot.noise_stream + m)
    replicates.append(env.run(200, stimulus))
# replicates[0] is the uninterrupted continuation; the rest differ only in their noise
```

Combining the two gives K actions times M noise replicates from one decision point:

```python
for action in actions:
    for m in range(M):
        env.restore_state(snapshot, noise_stream=snapshot.noise_stream + m)
        outcome = env.run(200, action)
```

## Rules

- **Save after the first `run()`.** Before it there is nothing to save; `clear()` is the snapshot at `t = 0`.
- **A snapshot restores only into the environment that saved it,** any number of times. It lives in memory and holds that environment's resources until it is garbage collected, `snapshot.release()` is called, or the environment is closed.
- **Set up recording before saving.** The recorders at restore must be the ones the snapshot was taken with; the recording buffers start empty after a restore, so no earlier samples are repeated.
- **Change parameters after restoring, not before.** A restore puts the synapses (weights, delays and their parameters) and the noise sources back as they were at the save, but leaves membrane parameters and holding currents as they are now. To give a branch other parameters, set them once the state is back.
- `clear()`, other runs, other stimuli and reseeding between save and restore are all fine.

## Backends

| backend | snapshots |
|---|---|
| `neuron` | yes, also across MPI ranks: each rank saves and restores its own part of the network. Not under CoreNEURON. |
| `native` | yes (CPU build) |
| `diffrax`, `brian2` | not yet: `save_state()` raises `NotImplementedError` |

Ask with `livn.types.supports(env, "snapshot")` rather than by backend name.

With a [Distributed Environment](/guide/advanced/distributed), `save_state()` has every worker save its own env and returns a handle; the states stay on the workers. `restore_state(handle, noise_stream=...)` puts every worker back, and `release_state(handle)` frees them. Each worker restores the state *it* had, so this forks a set of workers that were at the same point; wait for outstanding calls before saving or restoring.

::: tip Custom mechanisms
Under NEURON, a mechanism that carries a value from one time step to the next in an `ASSIGNED` variable must declare it `RANGE` for a snapshot to see it, and a noise mechanism must expose its random stream's position through `noiseSequence()` and `setNoiseSequence()`, as the bundled `Gfluct3` does.
:::
