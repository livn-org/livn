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

The `target` config option specifies a **tuning target**, i.e. the search problem. The target owns the parameter search space, the objectives, the constraints, the network the fit runs on and what one evaluation costs the scheduler; `tune` only states how to search it. livn ships with `targets.EI` (`Culture`) as the default target for the cultures, configured by the measurements it fits, so tuning against your own recordings does not require writing one from scratch.

## Tuning Targets

A tuning target is a [machinable interface](https://machinable.org/guide/interface) that subclasses `Target` and defines four things:

1. **Configuration** - a typed `Config` holding everything the problem states, with defaults
2. **Search space** - which parameters to optimize and their bounds
3. **Objectives** - what metrics to minimize (returned as `(objective_value, feature_value)` tuples)
4. **Constraints** - hard constraints that valid solutions must satisfy

Because the configuration is a `Config`, it is validated, composable through `~versions` and override dicts, and recorded with the run.

### Minimal example

```python
from pydantic import BaseModel
from systems.targets.protocol import Target
from livn.decoding import MeanFiringRate, Slice

class MyTarget(Target):
    class Config(Target.Config):
        target_mfr: float = 3.0
        duration: float = 10000.0
        warmup: float = 2000.0

    def _configure(self):
        self.target_mfr = self.config.target_mfr
        self.duration = self.config.duration
        self.warmup = self.config.warmup

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

### `targets.EI` (`Culture`)

The default target for cultures measures a free-running network and scores it against a handful of values you can set. Give it a `stimulus` and it also delivers a pulse train after the measured window and fits the network's recruitment curve, read with `livn.decoding.RecruitmentCurve`.

**Objectives** — squared distance between the measured value and its target. `mfr`, `isi_cv` and `active_fraction` are always scored; the burst family is scored wherever the culture bursts, and where it does not, `fano_factor`, `mean_channel_correlation` and `max_synchronous_peak` are scored in its place.

**Constraints**

| Constraint | Constants |
|------------|-----------|
| not runaway / not quiescent / is stable | `MAX_POP_RATE_PER_UNIT_HZ`, `MIN_POP_RATE_PER_UNIT_HZ`, `STABILITY_MARGIN` |
| firing rates in band | `MAX_NEURON_RATE_HZ`, `MIN_MEAN_RATE_HZ`, `MAX_MEAN_RATE_HZ` |
| synchrony | `SYNCHRONY_BAND`, `MIN_SYNC_PEAK`, `MAX_SYNC_PEAK` |
| bursting | `MIN_BURST_RATE_HZ`, `MAX_BURST_RATE_HZ` |
| liveness | `MIN_ACTIVE_FRACTION`, `MIN_POPULATION_ACTIVE` |
| timescale and criticality | `POP_TAU_BAND_MS`, `BRANCHING_RATIO_BAND`, `MIN_AVALANCHE_R2` |

**Search space** - the recurrent excitatory weight is searched on its own scale and every other weight as a ratio to it (`...-weight_ratio`), which keeps the E/I balance separable from the overall drive; the OU background is searched as a total conductance and an I:E ratio (`noise-g_total`, `noise-g_ratio`) with the two correlation times, the vesicle pool as `U` and `tau_rec`, and the cell's calcium-dependent adaptation half a decade each side of the culture-like cell. With an evoked block it also searches the stimulation gain.

## Tuning against your own measurements

Extract your recording as a target document (see `systems/targets/schema.py`)

```python
from machinable import get

tuner = get("tune", ["~fit(observation='…/my-target.json')"])
tuner.launch()
```

## Writing custom tuning targets

You can write your own `Target` subclass to tune a system against your own experimental data or a different activity regime. Place your target module anywhere importable (e.g., inside `systems/targets/` for project-level targets).

### Tuning against experimental recordings

A common use case is matching simulation output to experimental MEA recordings. For example, suppose you have recorded spontaneous activity from a cortical organoid and want to tune a simulated system to reproduce its firing statistics:

```python
# systems/targets/my_organoid.py
import numpy as np
from systems.targets.protocol import Target
from livn.decoding import MeanFiringRate, Slice, LFP

class OrganoidMatch(Target):
    """Tune to match experimental organoid recordings."""

    class Config(Target.Config):
        recording_mfr: float = 2.3          # measured mean firing rate (Hz)
        recording_burst_rate: float = 0.05  # measured burst rate (Hz)
        duration: float = 20000.0
        warmup: float = 2000.0

    def _configure(self):
        self.recording_mfr = self.config.recording_mfr
        self.recording_burst_rate = self.config.recording_burst_rate
        self.duration = self.config.duration
        self.warmup = self.config.warmup

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
    target=targets.my_organoid \
    **resources='{"-n": 2}' \
    --launch
```
## Running the tuner

### Via the CLI

```sh
livn systems mpi tune '~fit(observation="…/E_E-sample2_15.json")' \
    **resources='{"-n": 3}' \
    --launch
```

The `mpi` execution module handles `mpirun` automatically. `-n` specifies the total number of MPI ranks; at least 2 are required (one controller + one or more workers). Each worker uses the target's `sizing.nprocs_per_worker` ranks unless `autosize` works out a layout, so the total is `1 + num_workers * nprocs_per_worker` — 3 for one worker of the two ranks `targets.EI` asks for.

To fit a [rung](/guide/concepts/system#subselections) rather than the whole culture, name it as selection:

```sh
livn systems mpi tune '~fit(observation="…")' \
    selection=e1 \
    **resources='{"-n": 3}' \
    --launch
```

To use a custom target, name its module:

```sh
livn systems mpi tune \
    system=./systems/graphs/EI \
    target=targets.my_organoid \
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
livn systems slurm \
    **resources='{"--nodes": 2, "--ntasks-per-node": 56, "-p": "normal", "-t": "4:00:00"}' \
    tune \
    system=./systems/graphs/EI \
    nprocs_per_worker=4 \
    --launch
```

The execution module handles MPI launch commands, job submission, and resource allocation automatically. See the [machinable execution docs](https://machinable.org/guide/execution) for details.

### Sizing the run automatically

Picking `nprocs_per_worker`, `--nodes` and `--ntasks-per-node` by hand means knowing how much memory a worker needs, which depends on how many synapses the selection wires. `autosize` is on by default and works it out from the target's own memory model; a target that cannot price its own network falls back to `nprocs_per_worker`.

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
LIVN_MIN_RANKS_PER_WORKER=8 livn systems slurm tune '~fit(observation="…")' --launch
```

```python
from systems.targets.protocol import Sizing

class MyCulture(Culture):
    class Config(Culture.Config):
        # sizing may add ranks, not go below this
        sizing: Sizing = Sizing(min_ranks_per_worker=2, n_initial=25, n_epochs=10)
```

::: warning
`ranks` means total ranks to the `mpi` module (`-n`) and ranks per node to `slurm` (`--ntasks-per-node`). The two agree on a single node, so for a local `mpi` run pass `max_nodes=1` and the printed `-n` is correct:

```sh
livn systems tune '~ca1(selection="e1")' max_nodes=1 --sizing
```
:::

### Via Python

```python
from machinable import get

tuner = get("tune", {
    "target": ["targets.EI", {"observation": "…/E_E-sample2_15.json"}],
    "trials": 1,
})
tuner.launch()
```

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `target` | `targets.EI` | The problem: a module, then `~versions` and override dicts |
| `system` | `None` | Override the network the target states |
| `selection` | `None` | Stored subselection to build instead of the whole system |
| `model` | `None` | Override the model the target states |
| `trials` | `1` | Simulation trials per evaluation |
| `nprocs_per_worker` | `None` | MPI ranks per worker when `autosize` is off; `None` takes the target's |
| `autosize` | `True` | Size ranks and nodes from the target's memory model; see [above](#sizing-the-run-automatically) |
| `worker_memory_max` | `None` | GiB per node, else `LIVN_WORKER_MEMORY_MAX`, else this machine |
| `cores_per_node` | `None` | Ranks a node can run, else `LIVN_CORES_PER_NODE` / `SLURM_CPUS_ON_NODE` / this machine |
| `max_nodes` | `None` | Cap on the node count; `1` for a local `mpi` run |
| `n_initial` | `None` | Initial samples **per search dimension**; `None` takes the target's |
| `population_size` | `100` | Evolutionary population |
| `num_generations` | `10` | Generations per epoch |
| `n_epochs` | `None` | Optimizer epochs (epoch 0 is the initial sampling); `None` takes the target's |
| `optimizer` | `nsga2` | Which dmosopt optimizer runs the search |
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

which writes `env.json` beside the graph (or `env-<selection>-<group>.json` when the run named either):

```json
{
    "system": {"cls": "livn.system.NeuroH5System", "kwargs": {"uri": "."}},
    "model": {"cls": "livn.models.rcsd.ReducedCalciumSomaDendrite", "kwargs": {}},
    "io": null,
    "selection": null,
    "params": {
        "EXC_EXC-dend-AMPA-weight": 0.31,
        "INH_EXC-soma-AMPA-weight": 2.909,
        "EXC_INH-soma-GABA_A-weight": 9.407,
        "noise-g_e0": 1.0,
        "noise-std_e": 0.329
    },
    "meta": {"loc": 0, "source": "...", "space": ["..."]}
}
```

`meta` records where the solution came from, including its position in the ranked front.

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
