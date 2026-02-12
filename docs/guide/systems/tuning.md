# Tuning Systems

After [generating](/guide/systems/generate) a system, the synaptic weights and noise parameters need to be tuned so that the network produces biologically realistic dynamics. livn provides a surrogate-assisted optimization pipeline that automatically searches for parameters that match target neural activity metrics.

::: tip Prerequisites
This section requires the `systems/` subpackage and its dependencies (`uv sync --package systems`). Familiarity with [Models](/guide/concepts/model) (especially synaptic weights and noise parameters) is assumed.
:::

## Why tune?

A freshly generated system has bare connectivity - the synaptic weights and background noise levels are not yet calibrated. Without tuning, the network may be:

- **Quiescent**: Too little excitation, no spontaneous activity
- **Runaway**: Too much excitation, pathological hypersynchrony
- **Unrealistic**: Wrong firing rate balance, absent oscillations, or non-critical dynamics

Tuning finds parameters that produce **target dynamics**, such as:

- Mean firing rate ~1 Hz (spontaneous)
- Branching ratio ~1.0 (near-critical dynamics)
- Biologically plausible burst patterns and LFP spectra
- Power-law distributed neuronal avalanches

## How it works

livn uses **surrogate-assisted multi-objective optimization** via the [dmosopt](https://github.com/dmosopt/dmosopt) library:

1. **Initial sampling**: Random parameter configurations are simulated
2. **Surrogate model**: A Transformer-based neural network learns to predict activity metrics from parameters
3. **Evolutionary optimization**: An evolutionary algorithm proposes new parameter configurations guided by the surrogate
4. **Simulation evaluation**: Promising candidates are simulated to validate predictions
5. **Iteration**: Steps 2-4 repeat for multiple epochs

The `data` config option specifies a **tuning protocol** that defines the parameter search space, the optimization objectives, and the constraints. livn ships with `systems.data.EI.Spontaneous` as the default protocol for EXC-INH systems, but you can implement your own.

## The `TuneProtocol`

A tuning protocol is a class that subclasses `TuneProtocol` and defines three things:

1. **Search space** - which parameters to optimize and their bounds
2. **Objectives** - what metrics to minimize (returned as `(objective_value, feature_value)` tuples)
3. **Constraints** - hard constraints that valid solutions must satisfy

### Minimal example

```python
from systems.data.protocol import TuneProtocol
from livn.decoding import MeanFiringRate, Slice

class MyProtocol(TuneProtocol):
    def __init__(self):
        super().__init__()
        self.target_mfr = 3.0
        self.duration = 10000.0
        self.warmup = 2000.0

    # --- Search space ---

    def _weight_space(self):
        return {
            "EXC_EXC-hillock-AMPA-weight": [0.001, 20.0, self.transform_log1p],
            "EXC_INH-hillock-AMPA-weight": [0.001, 20.0, self.transform_log1p],
            "INH_EXC-soma-GABA_A-weight": [0.001, 12.0, self.transform_log1p],
            "INH_INH-soma-GABA_A-weight": [0.001, 12.0, self.transform_log1p],
        }

    def _noise_space(self):
        return {
            "noise-g_e0": [1.0, 5.0],
            "noise-std_e": [0.005, 0.5],
        }

    def _protocol_space(self):
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

        # Compute objectives: dict of name -> (objective_value, feature_value)
        recording = Slice(start=self.warmup, stop=total)(env, *data)
        mfr = MeanFiringRate(duration=self.duration)(env, *recording)
        rate = mfr["rate_hz"] if mfr else 0.0
        objectives = {"mfr": ((rate - self.target_mfr) ** 2, rate)}

        # Compute constraints: dict of name -> (constraint_value, feature_value)
        # Positive = satisfied, negative = violated
        constraints = {
            "not_quiescent": (1.0 if rate > 0.1 else -1.0, rate),
        }

        return objectives, constraints
```

### Search space definition

Override `_weight_space()`, `_noise_space()`, and `_protocol_space()` to define the parameters the optimizer will search over. Each entry maps a parameter name to its bounds:

```python
def _weight_space(self):
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

The parameter names must match the names expected by `env.set_params()`. Synaptic weight parameters follow the convention `{pre}_{post}-{section}-{mechanism}-weight` (e.g., `EXC_INH-hillock-AMPA-weight`). Noise parameters are prefixed with `noise-` (e.g., `noise-g_e0`).

### Objectives and constraints

When the optimizer evaluates a candidate parameter set, it:

1. Calls `env.set_params(protocol.transform_params(x))` to apply the parameters (decoding them from optimization space via inverse transforms)
2. Calls `protocol(env)` which must return `(objectives, constraints)`

**Objectives** are values to minimize. Each entry is a `(objective_value, feature_value)` tuple where the optimizer minimizes `objective_value` and logs `feature_value` for analysis.

**Constraints** determine feasibility. Each entry is a `(constraint_value, feature_value)` tuple where positive `constraint_value` means the constraint is satisfied and negative means it is violated. Infeasible solutions are discarded.

### Consuming protocol-specific parameters

If your protocol introduces parameters that should not be passed to `env.set_params()` (e.g., stimulus amplitude), override `set_params()`:

```python
def _protocol_space(self):
    return {"stim_amplitude": [0.1, 5.0]}

def set_params(self, params):
    remaining = params.copy()
    self.amplitude = remaining.pop("stim_amplitude")
    return remaining  # only weight/noise params remain
```

## Built-in protocols

### `systems.data.EI.Spontaneous`

The default protocol for EXC-INH systems optimizing for spontaneous (unstimulated) activity.

**Objectives:**

| Metric | Default target | Description |
|--------|---------------|-------------|
| `mfr` | 1.0 Hz | Mean firing rate |
| `branching_ratio` | 1.0 | Near-critical dynamics |
| `burst_rate` | 0.1 Hz | Network burst frequency |
| `burst_participation` | 0.4 | Fraction of neurons in bursts |
| `avalanche_power_law` | 0.6 (R^2) | Power-law fit quality |
| `delta_theta_ratio` | 1.6 | LFP delta/theta ratio |
| `spectral_slope` | -1.5 | 1/f spectral slope |
| LFP band powers | various | Delta, theta, alpha, beta, gamma |

**Constraints:** not runaway, not quiescent, stable activity, bounded firing rates, moderate synchrony.

Targets can be customized without subclassing by passing overrides to the constructor, since the protocol is instantiated internally via `data=systems.data.EI.Spontaneous`. To customize targets, subclass and override `DEFAULT_TARGETS`:

```python
from systems.data.EI import Spontaneous

class MySpontaneous(Spontaneous):
    DEFAULT_TARGETS = {
        **Spontaneous.DEFAULT_TARGETS,
        "mfr": 5.0,               # higher firing rate
        "branching_ratio": 0.95,   # slightly sub-critical
    }
```

## Running the tuner

### Via the CLI

```sh
livn systems tune \
    system=./systems/graphs/EI2 \
    data=systems.data.EI.Spontaneous \
    --launch
```

To use a custom protocol, specify its dotted import path:

```sh
livn systems tune \
    system=./systems/graphs/EI2 \
    data=my_module.MyProtocol \
    --launch
```

For large systems, run with MPI by prepending the `interface.remotes.mpi` execution module:

```sh
livn systems interface.remotes.mpi tune \
    system=./systems/graphs/EI2 \
    nprocs_per_worker=4 \
    **resources='{"--n": 64}' \
    --launch
```

On Slurm clusters, use the `slurm` (or `tacc` for TACC systems) execution module instead:

```sh
livn systems slurm tune \
    system=./systems/graphs/EI2 \
    nprocs_per_worker=4 \
    **resources='{"--nodes": 2, "--ntasks-per-node": 56, "-p": "normal", "-t": "4:00:00"}' \
    --launch
```

The execution module handles MPI launch commands, job submission, and resource allocation automatically. See the [machinable execution docs](https://machinable.org/guide/execution) for details.

### Via Python

```python
from machinable import get

tuner = get("tune", {
    "system": "./systems/graphs/EI2",
    "data": "systems.data.EI.Spontaneous",
    "trials": 1,
    "nprocs_per_worker": 1,
})
tuner.launch()
```

### Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `system` | `./systems/graphs/EI2` | Path to the generated system |
| `model` | `None` | Model class (None = system default) |
| `data` | `systems.data.EI.Spontaneous` | Dotted path to a `TuneProtocol` subclass |
| `trials` | `1` | Simulation trials per evaluation |
| `nprocs_per_worker` | `1` | MPI ranks per simulation worker |

The optimization runs 25 epochs with 100 initial random samples, a population size of 100, and 10 evolutionary generations per epoch.

### Inspecting results

After optimization, inspect and extract the best parameters:

```sh
livn systems tune system=./systems/graphs/EI2 --inspect
```

Or in Python:

```python
tuner.inspect()
```

This ranks all evaluated solutions by a composite score and saves the best configuration as `params.json` in the system directory:

```json
{
    "weight-EXC_EXC-hillock-AMPA-weight": 0.001,
    "weight-EXC_INH-hillock-AMPA-weight": 2.909,
    "weight-INH_EXC-soma-GABA_A-weight": 9.407,
    "noise-g_e0": 1.0,
    "noise-std_e": 0.329,
    ...
}
```

These parameters are then automatically loaded by `livn.make()` or `system.default_params()`.

## Tips

- **Start small**: Tune on EI1 or EI2 first, then transfer insights to larger systems
- **Use multiple trials**: Set `trials > 1` to reduce variance in the evaluation metrics
- **Check for stability**: After tuning, run extended simulations (>10s) to verify the parameters produce stable dynamics
- **Iterate**: The first round of tuning may not find optimal parameters; re-run with narrowed search bounds around promising regions
