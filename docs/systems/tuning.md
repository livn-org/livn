# Tuning Systems

After [generating](/systems/generate) a system, the synaptic weights and noise parameters need to be tuned so that the network produces biologically realistic dynamics. livn provides a surrogate-assisted optimization pipeline that automatically searches for parameters that match target neural activity metrics.

::: tip Prerequisites
This section requires the `systems/` subpackage and its dependencies (`uv sync --package systems`). Familiarity with [Models](/guide/concepts/model) (especially synaptic weights and noise parameters) is assumed.
:::

## Why tune?

A freshly generated system has bare connectivity - the synaptic weights and background noise levels are not yet calibrated. Without tuning, the network may be:

- **Quiescent**: Too little excitation, no spontaneous activity
- **Runaway**: Too much excitation, pathological hypersynchrony
- **Unrealistic**: Wrong firing rate balance, absent oscillations, or non-critical dynamics

Tuning finds parameters that produce target dynamics, i.e. the values a recording of the real preparation reports, such as:

- a mean firing rate and how irregular the spiking is
- how much of the population fires at all, and how correlated it is
- burst rate, synchrony and population timescale within measured bands
- near-critical dynamics with a branching ratio around 1, power-law avalanches

## How it works

livn uses **surrogate-assisted multi-objective optimization** via the [dmosopt](https://github.com/dmosopt/dmosopt) library:

1. **Initial sampling**: Random parameter configurations are simulated
2. **Surrogate model**: A Transformer-based neural network learns to predict activity metrics from parameters
3. **Evolutionary optimization**: An evolutionary algorithm proposes new parameter configurations guided by the surrogate
4. **Simulation evaluation**: Promising candidates are simulated to validate predictions
5. **Iteration**: Steps 2-4 repeat for multiple epochs

The `target` config option specifies a **tuning target** that defines the parameter search space, the optimization objectives, and the constraints. livn ships with `systems.targets.EI.Culture` as the default target for the cultures. It takes the values it fits as arguments, so tuning against your own recordings does not require writing one from scratch.

## Tuning Targets

::: tip
Fitting a culture to your own measurements usually needs no code at all since the built-in [`Culture`](#systems-targets-ei-culture) target takes the measured values as arguments; see [tuning against your own
measurements](#tuning-against-your-own-measurements). Read on when you need a protocol it does not cover.
:::

A tuning target is a class that subclasses `TuningTargets` and defines three things:

1. **Search space** - which parameters to optimize and their bounds
2. **Objectives** - what metrics to minimize (returned as `(objective_value, feature_value)` tuples)
3. **Constraints** - hard constraints that valid solutions must satisfy

### Minimal example

```python
from systems.targets.protocol import TuningTargets
from livn.decoding import MeanFiringRate, Slice

class MyTarget(TuningTargets):
    def __init__(self):
        super().__init__()
        self.target_mfr = 3.0
        self.duration = 10000.0
        self.warmup = 2000.0

    # --- Search space ---

    def _weight_space(self, model):
        return {
            "EXC_EXC-dend-AMPA-weight": [0.05, 10.0, self.transform_log10],
            "INH_EXC-soma-AMPA-weight": [0.05, 10.0, self.transform_log10],
            "EXC_INH-soma-GABA_A-weight": [0.05, 10.0, self.transform_log10],
        }

    def _noise_space(self, model):
        return {
            "noise-g_e0": [0.0002, 0.02, self.transform_log10],
            "noise-std_e": [0.0001, 0.05, self.transform_log10],
        }

    def _protocol_space(self, model):
        return {}  # additional protocol-specific parameters

    # --- Evaluation ---

    def objective_names(self):
        return ["mfr"]

    def constraint_names(self):
        return ["not_quiescent"]

    def __call__(self, env):
        """Run a simulation and return (objectives, constraints)."""
        total = int(self.warmup + self.duration)
        env.record_spikes()
        env.record_voltage()
        data = env.run(total)

        # Compute objectives: dict of name -> [(objective_value, feature_value)]
        recording = Slice(start=self.warmup, stop=total)(data)
        mfr = MeanFiringRate(duration=self.duration)(recording, env)
        rate = mfr["rate_hz"] if mfr else 0.0
        objectives = {"mfr": [((rate - self.target_mfr) ** 2, rate)]}

        # Compute constraints: dict of name -> [(constraint_value, feature_value)]
        # Positive = satisfied, negative = violated
        constraints = {
            "not_quiescent": [(1.0 if rate > 0.1 else -1.0, rate)],
        }

        return objectives, constraints
```

### Search space definition

Override `_weight_space()`, `_noise_space()`, and `_protocol_space()` to define the parameters the optimizer will search over. Each receives the `model` the run was configured with — useful for deriving keys from what the model actually builds, and ignorable otherwise. Each entry maps a parameter name to its bounds:

```python
def _weight_space(self, model):
    return {
        "param_name": [min, max],               # identity transform
        "param_name": [min, max, transform_fn],  # with transform
    }
```

The optional third element is a transform function applied to the bounds before optimization. This is useful for parameters that span multiple orders of magnitude. Built-in transforms:

| Transform | Forward | Inverse | Use case |
|-----------|---------|---------|----------|
| `transform_identity` | x | x | Default, linear parameters |
| `transform_log10` | log10(x) | 10^x | Parameters spanning orders of magnitude |
| `transform_log1p` | log10(1+x) | 10^x - 1 | Like log10 but handles zero |

The parameter names must match the names expected by [`env.set_params()`](/guide/concepts/env#parameters), which routes each key by its prefix:

| prefix | goes to | example |
|---|---|---|
| `weight-`, or no prefix | `env.set_weights()` | `EXC_INH-soma-GABA_A-weight` |
| `noise-` | `env.set_noise()` | `noise-g_e0` |
| `cells-` | `env.cells.set_params()` | `cells-soma.g_pas`, `cells-EXC:soma.g_pas` |

Synaptic weight parameters follow the convention `{post}_{pre}-{section}-{mechanism}-weight` and are the search space's default, so an unprefixed name is read as a weight. The **postsynaptic** population comes first, because that is the population the synapse belongs to: on the shipped cultures the three keys are

| Key | Reads as |
|-----|----------|
| `EXC_EXC-dend-AMPA-weight` | EXC→EXC, onto the dendrite |
| `INH_EXC-soma-AMPA-weight` | EXC→INH, onto the soma |
| `EXC_INH-soma-GABA_A-weight` | INH→EXC, onto the soma |

A key naming a section or mechanism the network does not have selects nothing and is applied silently, so `system.weight_names` is worth checking against the graph you are tuning. Parameters of the point process itself rather than of a connection (`tau_rec`, `U`, `tau_decay`) drop the source: `EXC-dend-AMPA-tau_rec`.

`cells-` reaches the physical parameters of the cells themselves rather than the synapses between them, and applies one value to every cell of every population unless the name says otherwise. Its names are the ones the cells expose (`env.cells[gid].get_params()`), which on the NEURON backend are `"<section type>.<name>"` under the same section types weight keys select on. Searching over them tunes the cell model alongside the network:

```python
def _weight_space(self, model):
    return {
        "EXC_EXC-dend-AMPA-weight": [0.05, 10.0, self.transform_log10],
        "cells-soma.g_pas": [1e-5, 1e-3, self.transform_log10],
    }
```

To scope by population, use:

```python
def _protocol_space(self, model):
    return {
        "cells-EXC:dend.gmax_KCa": [1e-5, 1e-3, self.transform_log10],
    }
```

::: tip
Cell parameters are per cell, so `set_params()` can only broadcast one value across all of them. To give each cell its own value, address them directly with `env.cells.set_params()` or `env.cells[gid].set_params()`, see [cell parameters](/guide/concepts/env#cell-parameters).
:::

### Objectives and constraints

When the optimizer evaluates a candidate parameter set, it:

1. Calls `env.set_params(target.transform_params(x))` to apply the parameters (decoding them from optimization space via inverse transforms)
2. Calls `target(env)` which must return `(objectives, constraints)`

**Objectives** are values to minimize. Each entry is a list of `(objective_value, feature_value)` tuples (one per trial). The optimizer minimizes the mean `objective_value` across trials and logs the mean `feature_value` for analysis.

**Constraints** determine feasibility. Each entry is a list of `(constraint_value, feature_value)` tuples where positive `constraint_value` means the constraint is satisfied and negative means it is violated. Infeasible solutions are discarded.

### Consuming protocol-specific parameters

If your target introduces parameters that should not be passed to `env.set_params()` (e.g., stimulus amplitude), override the target's own `set_params()`.

```python
def _protocol_space(self, model):
    return {"stim_amplitude": [0.1, 5.0]}

def set_params(self, params):
    remaining = params.copy()
    self.amplitude = remaining.pop("stim_amplitude")
    return remaining  # only env-level params remain
```

::: tip
This is unaffected by the prefixes above as whatever the target does not consume is handed to `env.set_params()`, which then routes it to weights, noise or cells as usual.
:::

## Built-in targets

### `systems.targets.EI.Culture`

The default target for cultures measures a free-running network and scores it against a handful of values you can set. Give it a `stimulus` and it also delivers a pulse train after the measured window and fits the network's recruitment curve, read with `livn.decoding.RecruitmentCurve`.

**Objectives** — what the optimizer minimizes. Each is the squared distance between the measured value and its target (`mfr` in log space):

| Name | Default | Description |
|------|---------|-------------|
| `mfr` | 1.0 Hz | Mean firing rate |
| `isi_cv` | 1.2 | Coefficient of variation of the inter-spike intervals — how irregular the spiking is |
| `active_fraction` | 1.0 | Fraction of units that fire at all |
| `mean_channel_correlation` | *unset* | Mean pairwise correlation. Only becomes an objective when you give it a value |

To fit fewer, pass `skip_objectives`.

**Constraints** — hard gates a solution has to satisfy. Each is a class constant you can override:

| Constraint | Constants |
|------------|-----------|
| not runaway / not quiescent / is stable | `MAX_POP_RATE_PER_UNIT_HZ`, `MIN_POP_RATE_PER_UNIT_HZ`, `STABILITY_MARGIN` |
| firing rates in band | `MAX_NEURON_RATE_HZ`, `MIN_MEAN_RATE_HZ`, `MAX_MEAN_RATE_HZ` |
| synchrony | `SYNCHRONY_BAND`, `MIN_SYNC_PEAK`, `MAX_SYNC_PEAK` |
| bursting | `MIN_BURST_RATE_HZ`, `MAX_BURST_RATE_HZ` |
| liveness | `MIN_ACTIVE_FRACTION`, `MIN_POPULATION_ACTIVE` |
| timescale and criticality | `POP_TAU_BAND_MS`, `BRANCHING_RATIO_BAND`, `MIN_AVALANCHE_R2` |

**Search space** — derived from the graph rather than declared, so it follows whatever projections the system actually has (a model built with `short_term_depression=True` adds that mechanism's `tau_rec` and `U` to it). The recurrent excitatory weight is searched on its own scale and every other weight as a ratio to it (`...-weight_ratio`), which keeps the E/I balance separable from the overall drive; the OU background (`noise-g_e0`, `noise-g_i0`, `noise-std_e`, `noise-std_i`, `noise-tau_e`, `noise-tau_i`) is searched alongside it. Two optional coordinates: `adaptation=True` frees the cell's calcium-dependent potassium current and calcium removal rate, and `ignition=True` searches the recurrent weight along the measured `weight × g_e0` ignition boundary instead of on its own axis.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `targets` | see above | The values being fitted |
| `overrides` | `{}` | Constraint constants, by name; `{"targets": {...}}` also works. An unknown name raises |
| `feature_bands` | `{}` | `{feature: (lo, hi)}` used to *rank* the front, not to gate it. Matched against the recorded feature columns, which are the objective names |
| `duration` / `warmup` | 30 000 / 1 000 ms | Measured window, and the settling time before it |
| `readout` | `"neurons"` | `"channels"` measures through an array instead of per neuron |
| `mea` | `None` | Array geometry (`electrode_coordinates`, `input_radius`, `output_radius`), required for `readout="channels"` |
| `skip_objectives` / `skip_constraints` | `()` | Names to leave out |
| `adaptation` / `ignition` | `False` | The optional search coordinates above |
| `stimulus` | `None` | A `Protocol` (a `livn.policy.PulseSweepPolicy` plus the baseline, the recovery time and the driving electrode) to deliver after the measured window. Requires `readout='channels'`. Adds the `stimulus_threshold` objective and the `io-volume_conductor-stimulation_gain` coordinate. The sweep is `len(amplitudes) * repeats * trial_ms` of extra simulation, and `trial_ms` has to leave `recovery_ms` of quiet between one response and the next pulse's baseline &mdash; spend the budget on spacing before repeats |
| `stimulus_threshold` | `{}` | The bracket the culture's own recruitment crossed in, as `livn.decoding.recruitment_threshold` reports it |
| `gate_stimulus` | `True` | Deliver the sweep only to candidates the measured window leaves feasible. |

## Tuning against your own measurements

```python
from machinable import get

tuner = get("tune", {
    "system": "./systems/graphs/EI",
    "target": ["systems.targets.EI.Culture", {
        # what the recording says the network does
        "targets": {
            "mfr": 0.51,                        # Hz
            "isi_cv": 0.78,
            "active_fraction": 0.48,
            "mean_channel_correlation": 0.29,   # only if you measured one
        },
        # the bands a solution has to stay inside
        "overrides": {
            "MIN_MEAN_RATE_HZ": 0.45,
            "MAX_MEAN_RATE_HZ": 0.89,
            "MAX_NEURON_RATE_HZ": 7.4,
            "MAX_SYNC_PEAK": 0.05,
            "MAX_BURST_RATE_HZ": 0.47,
            "BRANCHING_RATIO_BAND": [1.25, 1.37],
        },
        # ranking rather than gating: a solution inside every band comes out
        # above one that merely scores well. Names are the objective features
        "feature_bands": {
            "mfr": [0.45, 0.89],
            "isi_cv": [0.70, 0.95],
        },
        "duration": 30000.0,
    }],
})
tuner.launch()
```

The same on the command line, where the target is a `[path, options]` pair:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    target='["systems.targets.EI.Culture", {"targets": {"mfr": 0.51, "isi_cv": 0.78}}]' \
    **resources='{"-n": 2}' \
    --launch
```

Which numbers you need depends on what you can measure reliably. Only `targets` is really required since every constraint has a default.

::: warning
Measure the simulation the way you measured the culture. With `readout="neurons"` the metrics are computed per cell; with `readout="channels"` they are computed on spikes pooled per electrode, which is what an MEA recording gives you. The two do not produce the same `mfr` for the same network, so a target measured on channels has to be fitted on channels:

```python
"target": ["systems.targets.EI.Culture", {
    "readout": "channels",
    "mea": {
        "electrode_coordinates": [[0, 200.0, 200.0, 5.0], [1, 400.0, 200.0, 5.0]],
        "input_radius": 50.0,
        "output_radius": 50.0,
    },
    "targets": {"mfr": 0.51},
}],
```
:::


## Writing custom tuning targets

You can write your own `TuningTargets` subclass to tune a system against your own experimental data or a different activity regime. Place your target module anywhere importable (e.g., inside `systems/targets/` for project-level targets, or any Python package on your path).

### Tuning against experimental recordings

A common use case is matching simulation output to experimental MEA recordings. For example, suppose you have recorded spontaneous activity from a cortical organoid and want to tune a simulated system to reproduce its firing statistics:

```python
# systems/targets/my_organoid.py
import numpy as np
from systems.targets.protocol import TuningTargets
from livn.decoding import MeanFiringRate, Slice, LFP

class OrganoidMatch(TuningTargets):
    """Tune to match experimental organoid recordings."""

    def __init__(
        self,
        recording_mfr: float = 2.3,       # measured mean firing rate (Hz)
        recording_burst_rate: float = 0.05, # measured burst rate (Hz)
        duration: float = 20000.0,
        warmup: float = 2000.0,
    ):
        self.recording_mfr = recording_mfr
        self.recording_burst_rate = recording_burst_rate
        self.duration = duration
        self.warmup = warmup
        super().__init__()

    def _weight_space(self, model):
        return {
            "EXC_EXC-dend-AMPA-weight": [0.05, 10.0, self.transform_log10],
            "INH_EXC-soma-AMPA-weight": [0.05, 10.0, self.transform_log10],
            "EXC_INH-soma-GABA_A-weight": [0.05, 10.0, self.transform_log10],
        }

    def _noise_space(self, model):
        return {
            "noise-g_e0": [0.0002, 0.02, self.transform_log10],
            "noise-std_e": [0.0001, 0.05, self.transform_log10],
        }

    def objective_names(self):
        return ["mfr", "burst_rate"]

    def constraint_names(self):
        return ["not_quiescent", "not_runaway"]

    def __call__(self, env):
        total = int(self.warmup + self.duration)
        env.record_spikes()
        env.record_voltage()
        env.record_membrane_current()
        data = env.run(total)

        recording = Slice(start=self.warmup, stop=total)(data)
        # Mean firing rate objective
        mfr_result = MeanFiringRate(duration=self.duration)(recording, env)
        rate = mfr_result["rate_hz"] if mfr_result else 0.0
        mfr_obj = (rate - self.recording_mfr) ** 2

        objectives = {
            "mfr": [(mfr_obj, rate)],
            "burst_rate": [(0.0, 0.0)],  # placeholder; implement burst detection
        }

        constraints = {
            "not_quiescent": [(1.0 if rate > 0.1 else -1.0, rate)],
            "not_runaway": [(1.0 if rate < 50.0 else -1.0, rate)],
        }

        return objectives, constraints
```

Then run:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    target=systems.targets.my_organoid.OrganoidMatch \
    **resources='{"-n": 2}' \
    --launch
```

### Extending the built-in Culture target

```python
# systems/targets/my_culture.py
from systems.targets.EI import Culture

class MyCulture(Culture):
    """The values measured on our own preparation."""

    # constraint constants are ordinary class attributes
    MIN_MEAN_RATE_HZ = 2.0
    MAX_MEAN_RATE_HZ = 8.0

    def __init__(self, **kwargs):
        kwargs.setdefault("targets", {"mfr": 5.0, "isi_cv": 1.6})
        super().__init__(**kwargs)
```

## Running the tuner

### Via the CLI

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    target=systems.targets.EI.Culture \
    **resources='{"-n": 2}' \
    --launch
```

The `mpi` execution module handles `mpirun` automatically. `-n` specifies the total number of MPI ranks; at least 2 are required (one controller + one or more workers). Each worker uses `nprocs_per_worker` ranks, so the total should be `1 + num_workers * nprocs_per_worker`.

To fit a [rung](/guide/concepts/system#subselections) rather than the whole culture, name it as selection:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    selection=e1 \
    **resources='{"-n": 2}' \
    --launch
```

The result is written to `params/e1.json`.

To use a custom target, specify its dotted import path:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    target=systems.targets.my_organoid.OrganoidMatch \
    **resources='{"-n": 2}' \
    --launch
```

For larger runs with multiple workers:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    nprocs_per_worker=4 \
    **resources='{"-n": 65}' \
    --launch
```

On Slurm clusters, use the `slurm` execution module instead:

```sh
livn systems slurm tune \
    system=./systems/graphs/EI \
    nprocs_per_worker=4 \
    **resources='{"--nodes": 2, "--ntasks-per-node": 56, "-p": "normal", "-t": "4:00:00"}' \
    --launch
```

The execution module handles MPI launch commands, job submission, and resource allocation automatically. See the [machinable execution docs](https://machinable.org/guide/execution) for details.

### Sizing the run automatically

Picking `nprocs_per_worker`, `--nodes` and `--ntasks-per-node` by hand means knowing how much memory a worker needs, which depends on how many synapses the selection wires. `autosize` works it out for `~ca1`, `~EI` and `~E_only` automatically and you can set `autosize=True` to use it elsewhere.

To preview it before committing to a job:

```sh
livn systems tune '~ca1(selection="e3")' --sizing
```

```
  system        ./systems/graphs/CA1
  selection     e3
  node          128.0 GiB x 56 cores, planned to 90%
  worker        26.1 GiB over 13 rank(s)
  layout        22 node(s) x 55 rank(s) = 1210 ranks (110.6 GiB used per node)
  workers       93 for 100 samples per epoch  -- epochs will queue
  evaluations   284 in the first epoch (142 dims x n_initial=2), then 100 per epoch x 4 = 684 in all
```

To override the defaults, use:

```sh
LIVN_WORKER_MEMORY_MAX=128 LIVN_CORES_PER_NODE=56 livn systems slurm tune ~ca1 --launch
```

or `worker_memory_max` and `cores_per_node`, and `max_nodes` in code, as well as:

```python
class MyCulture(Culture):
    MIN_RANKS_PER_WORKER = 2  # sizing may add ranks, not go below this
```

- Ranks per worker — the fewest that bring a rank's share of the network under the node's per-core memory.
- Nodes — enough that the worker count reaches the samples an epoch draws. Workers past that would idle.
- Ranks per node — as dense as the node's memory and cores allow, subject to `(total ranks − 1)` dividing by ranks per worker, which is what the controller-plus-workers layout requires.

::: warning
`ranks` means total ranks to the `mpi` module (`-n`) and ranks per node to `slurm` (`--ntasks-per-node`). The two agree on a single node, so for a local `mpi` run pass `max_nodes=1` and the printed `-n` is correct:

```sh
livn systems tune '~ca1(selection="e1", max_nodes=1)' --sizing
```
:::

### Via Python

```python
from machinable import get

tuner = get("tune", {
    "system": "./systems/graphs/EI",
    "target": "systems.targets.EI.Culture",
    "trials": 1,
    "nprocs_per_worker": 1,
})
tuner.launch()
```

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `system` | `./systems/graphs/EI` | Path to the generated system, or an `int` for that many unconnected cells |
| `selection` | `None` | Stored subselection to build instead of the whole system |
| `model` | `None` | Model class (None = system default) |
| `target` | `systems.targets.EI.Culture` | Dotted path to a `TuningTargets` subclass, or `[path, options]` |
| `trials` | `1` | Simulation trials per evaluation |
| `nprocs_per_worker` | `1` | MPI ranks per simulation worker (ignored when `autosize` is on) |
| `autosize` | `False` | Size ranks and nodes from the selection; see [above](#sizing-the-run-automatically) |
| `worker_memory_max` | `None` | GiB per node, else `LIVN_WORKER_MEMORY_MAX`, else this machine |
| `cores_per_node` | `None` | Ranks a node can run, else `LIVN_CORES_PER_NODE` / `SLURM_CPUS_ON_NODE` / this machine |
| `max_nodes` | `None` | Cap on the node count; `1` for a local `mpi` run |
| `n_initial` | `100` | Initial samples **per search dimension** |
| `population_size` | `100` | Evolutionary population |
| `num_generations` | `10` | Generations per epoch |
| `n_epochs` | `10` | Optimizer epochs (epoch 0 is the initial sampling) |
| `surrogate` | `{}` | Extra surrogate settings, passed through as `surrogate_*` |

::: warning
`n_initial` is a multiplier, not a count: dmosopt draws `n_initial × (number of search dimensions)` initial samples. On an 11-parameter space the default is ~1100 simulations before the surrogate gets a turn, so lower it for a short run.
:::

### Inspecting results

After optimization, inspect and extract the best parameters:

```sh
livn systems tune system=./systems/graphs/EI --inspect
```

Or in Python:

```python
tuner.inspect()
```

This ranks all evaluated solutions and reports the front. A run produces a front, not an answer, so selecting one solution requires promotion:

```sh
livn systems tune system=./systems/graphs/EI "--promote('default', loc=0)"
```

which writes `params/default.json` (or `params/<selection>.json` when the run used one):

```json
{
    "ReducedCalciumSomaDendrite": {
        "default": {
            "params": {
                "EXC_EXC-dend-AMPA-weight": 0.31,
                "INH_EXC-soma-AMPA-weight": 2.909,
                "EXC_INH-soma-GABA_A-weight": 9.407,
                "noise-g_e0": 1.0,
                "noise-std_e": 0.329
            },
            "meta": {"loc": 0, "source": "...", "space": ["..."]}
        }
    }
}
```

`meta` records where the solution came from, including its position in the ranked front.

These parameters are then applied by `livn.make()` or `env.apply_default_params()`.

`--export` writes the whole front to a `front.json` next to the run, which `--promote(front=...)` can bank from later without the run being at hand. Before promoting it is often worth looking at the dynamics since a solution can satisfy every scalar target and still be degenerate:

```sh
livn systems tune system=./systems/graphs/EI --export

livn systems mpi **resources='{"-n": 8}' run \
    system=./systems/graphs/EI \
    "~front('systems/storage/.../front.json', 0)" \
    decoding='["livn.decoding.GatherAndMerge", {"duration": 65000, "voltages": false, "membrane_currents": false}]' \
    figure='["plots.Raster", {"warmup": 5000}]' \
    --launch
```

## Tips

- **Start small**: Tune a small rung first (`selection=e1`), then move to the full culture. A rung is its own network, so its result is a starting point for the next one, not a set to carry over
- **Use multiple trials**: Set `trials > 1` to reduce variance in the evaluation metrics
- **Check for stability**: After tuning, run extended simulations (>10s) to verify the parameters produce stable dynamics
- **Iterate**: The first round of tuning may not find optimal parameters; re-run with narrowed search bounds around promising regions
- **Match your data**: When tuning against experimental recordings, start with the metrics you can measure most reliably (e.g., firing rate) before adding more complex objectives (e.g., LFP spectra, avalanche statistics)
- **Log-transform weight parameters**: Synaptic weights typically span orders of magnitude; use `transform_log10` (or `transform_log1p`, where a bound sits at zero) to help the optimizer explore the space efficiently
