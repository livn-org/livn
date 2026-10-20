# Generalized LIF (GLIF)

```python
from livn.models.glif import GLIF
from livn.system import predefined

GLIF(level=5, mechanism="hard")     # full Allen GLIF5
GLIF(level=1, mechanism="hard")     # leaky integrate-and-fire
GLIF(level=5, mechanism="escape")   # stochastic spike mechanism

GLIF.leaky_integrate_and_fire()     # the textbook LIF, parameters and all
```

`GLIF` is livn's point neuron. It is not one model among several but a space with two axes with a level (1–5, which parameters are in play) and a mechanism (`hard` or `escape`, how a spike is decided).

For fitting, a level is exactly a mask over the trainable parameter set:

```python
GLIF(level=3).trainable_params()
# ('tau_m', 'E_L', 'g_L', 'V_threshold_base', 'theta_decay_rate', 'asc_amp_1', ...)
```

## Units

GLIF is a delibarate exception to the [unit convention](/guide/concepts/units) so parameters keep their Allen Institute names without suffix. Everything is in **mV / pA / ms**: `tau_m` and `t_ref` in ms, `E_L` and the threshold in mV, `g_L` in nS (so `I/g_L` is a voltage), decay rates in 1/ms. A stimulus reaching `env.run` on the diffrax backend is scaled by `current_scale` (nA by default) before it is added.

## Mechanisms

Both mechanisms watch the same quantity, `V − Θ`, and differ only in what they do with it.

| | `hard` | `escape` |
|---|---|---|
| a spike is | the root of `V − Θ` | the root of `S = ∫λ dt`, with `λ = exp((V − Θ)/σ)/τ_s` |
| deterministic | yes | no: `S` restarts from `log(U) − α` after every spike |
| backends | diffrax, brian2 | diffrax |
| extra parameters | — | `sigma`, `tau_s`, `alpha` |

```python
GLIF(mechanism="escape", params={"sigma": 1.0})
```

## Networks

```python
env = Env(predefined("EI")), model=GLIF(level=1)).init()
env.module.network        # (cells, cells), w[pre, post]; None when unconnected
```

Weights must be set before `init()` on the diffrax backend since the connectivity is baked into the module:

```python
env = Env(predefined("S1")), model=GLIF()).set_weights({"EXC_EXC": 400.0}).init()
```

A network is one event-driven solve that integrates between events and restarts at every spike and every end of a refractory period, anywhere in the network:

```python
from livn.models.eventloop import SolverConfig

GLIF(config=SolverConfig(
    vectorized_events=True,   # the default: one event condition for the whole network
    dt_solver=0.1,            # Euler step between events, in ms
    max_segment_span=1.0,     # cut a solve 1 ms after its last restart
    checkpoints=1 << 30,      # reverse mode: store every segment and step
))
```

- `vectorized_events` (the default) hands the solver the largest of the cells' conditions instead of one condition per cell, so a step does not grow with the network, and the earliest crossing in a step is the one located. Root finding on the maximum relies on a monotone interpolant: use it with `Euler`. `vectorized_events=False` gives the solver one condition per cell, which other solvers need; then two cells crossing within one step fire in the order of their index, and the one located second starts the next solve above threshold and loses its spike.
- Recording nothing but spikes skips the sampled trace and keeps only the state at each segment's end; the spikes are the same.
- Batched with `vmap`, every window waits for the longest solve of an iteration, so a quiet stretch in one window holds up all of them; short segments bound that. `max_segment_span=math.inf` restarts only at events, which suits a single window.
- Reverse mode recomputes each loop about twice unless `checkpoints` is raised; the state of a segment is small and its steps are bounded, so storing all of them costs little memory.
- Events are located on the step's interpolant, so a coarse `dt_solver` moves spike times by its integration error, not by the step.

## Noise

`sigma_v` is a membrane diffusion amplitude, available to either mechanism. The Brownian path is a structural part of the solve, so it has to be asked for:

```python
GLIF(diffusion=True, params={"sigma_v": 1.0})
env.set_noise({"sigma_v": 1.0})   # equivalently, per run
```

## Batching

`num_samples` runs independent realisations of the same stimulus and every returned array gains a leading sample axis:

```python
run = env.run(100.0, stimulus, dt=0.1, num_samples=8)

run.voltage                  # (8, cells, T)
run.spikes.padded.times      # (8, cells, k)
run.spikes.raster(0.1)       # (8, cells, T)
```

Spikes come back as a rectangle — one row per cell, `inf`-padded — because that is the shape the solver can allocate before it knows how many events there will be. A batched run has no ragged form at all, since the number of spikes differs per sample, so `run.spike_times` raises there and points you at `.padded` or `.raster()`.

## Recordable states

Beyond spikes and voltage, GLIF exposes the rest of its state to [`env.record()`](/guide/concepts/env#recording). Each lands on the run's own `dt` grid as a `Series` channel:

| signal | shape | what it is |
|---|---|---|
| `threshold` | `[n_cells, T]` | Θ(t) = Θ_inf + θ_s(t) + θ_v(t), the value the voltage is compared against |
| `theta_s` | `[n_cells, T]` | the spike-driven component θ_s, zero at levels 1 and 3 |
| `theta_v` | `[n_cells, T]` | the voltage-coupled component θ_v, level 5 only |
| `AScurrents` | `[n_cells, T, 2]` | the two after-spike currents, zero at levels 1 and 2 |

```python
env = env.record_voltage()
env = env.record("threshold")

run = env.run(100)
run["threshold"].values          # [n_cells, T]
```

```python
env = env.record_voltage(dt=dt)
env = env.record("threshold")
env = env.record_spikes()

run = env.run(duration, stimulus, dt=dt)
run.spikes.padded.times     # exact event times, differentiable
```

## Inserting mechanisms

GLIF can be extended using `GlifMechanism`, an `equinox.Module` whose fields are its parameters, returned in the  `Model`'s `diffrax_mechanisms(env)`:

```python
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from livn.env import Env
from livn.models.glif import GLIF, GlifMechanism, cell_populations
from livn.stimulus import Stimulus


class LightCurrent(GlifMechanism):
    # its state columns, after the cell's own, and the stimulus modes it reads besides
    # the current
    states: tuple[str, ...] = eqx.field(static=True, default=("s",), kw_only=True)
    inputs: tuple[str, ...] = eqx.field(static=True, default=("irradiance",), kw_only=True)
    g: jnp.ndarray               # pA at full activation, per cell
    tau: float = 5.0             # parameters

    def drift(self, t, v, states, inputs):
        s = states[:, 0]
        ds = (inputs["irradiance"] * (1.0 - s) - s) / self.tau
        return self.g * s, ds[:, None]    # current into V in pA, d states / dt


class OptoGLIF(GLIF):
    def diffrax_mechanisms(self, env):
        populations = cell_populations(env)     # one name per cell, in module order
        return (LightCurrent(g=jnp.asarray(np.where(populations == "EXC", 50.0, 0.0))),)


env = Env(system, model=OptoGLIF(level=5)).init()
run = env.run(100.0, Stimulus.from_current(current) + Stimulus.from_irradiance(light))
```

| member | | |
|---|---|---|
| `states` | static field (`eqx.field(static=True, kw_only=True)`) | names of its state columns; also what [`env.record()`](/guide/concepts/env#recording) takes once the model lists them in `recordable_states()` |
| `inputs` | static field | modes it reads: `irradiance`, `conductance`, `current_density`, or `current` (the injected current in pA) |
| `initial(n_cells)` | optional | `[cells, len(states)]` at t0; zeros by default |
| `drift(t, v, states, inputs)` | required | `(current [cells] in pA, d states / dt [cells, len(states)])` between events; `inputs` is `{mode: [cells]}` at `t` |
| `on_spike(fired, states)` | optional | the states after the cells in `fired [cells]` spiked; only fired rows are kept |
| `presynaptic(states)` | optional | `[cells]` scaling each cell's outgoing synaptic jumps (from the states before `on_spike`), or `None` |

The cells are from `env.simulated_gids(everywhere=True)` with `cell_populations(env)` naming each one's population.

What a spike does to a mechanism:

- The voltage reset does not touch its states, so an opsin carries through a spike.
- `on_spike` acts at the spike time. The cell's own threshold and after-spike jumps land at the end of the refractory cut instead, which is Allen's ordering.
- Across the refractory cut the states keep integrating with V held at its reset value, as the threshold and after-spike currents keep decaying there. The current a mechanism injects is dropped while V is clamped, like every other current.

A stimulus mode is accepted only when a mechanism reads it so a GLIF Env without one refuses an `irradiance` stimulus. With no mechanisms the model is exactly the GLIF above. Mechanisms run on the diffrax backend only.

### Opsins

To use `livn.models.opsins.chr2` (ChR2[H134R], Williams et al. 2013) in GLIF, activate the `livn.models.opsins.ChR2H134R` mechanism:

```python
from livn.models.glif import GLIF, cell_populations
from livn.models.opsins.glif import ChR2H134R


class OptoGLIF(GLIF):
    def diffrax_mechanisms(self, env):
        exc = cell_populations(env) == "EXC"
        return (ChR2H134R(g_nS=jnp.asarray(np.where(exc, 40.0, 0.0)), celsius=34.0),)
```

