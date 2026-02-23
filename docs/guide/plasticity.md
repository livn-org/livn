# Synaptic Plasticity

In biological neural networks, the strength of a connection (synapse) between two neurons is not fixed but changes over time based on the activity of the neurons it connects. This property is called synaptic plasticity and is widely believed to be the cellular basis of learning and memory.

When a synapse gets stronger, it is called long-term potentiation (LTP). When it gets weaker, it is called long-term depression (LTD). Whether a synapse is potentiated or depressed depends on the relative timing and magnitude of activity in the pre- and postsynaptic neurons, a principle summarized as "neurons that fire together, wire together" (Hebbian learning).

livn implements a voltage-based form of spike-timing dependent plasticity (STDP), where the postsynaptic membrane voltage determines the direction and magnitude of weight changes. The learning rules are adapted from the Sigma3Exp2Syn mechanisms in [neuronpp](https://github.com/ziemowit-s/neuronpp).

## Excitatory vs inhibitory

Plasticity works differently for excitatory and inhibitory synapses:

**Excitatory synapses** (AMPA, NMDA) follow the classical Hebbian rule where synapses strengthen when the postsynaptic neuron is strongly depolarized (active) and weaken during moderate depolarization. This allows the network to reinforce connections that successfully drive postsynaptic firing.

**Inhibitory synapses** (GABA_A) use an inverted rule were synapses strengthen when the postsynaptic neuron is hyperpolarized (quiet) and weaken when the cell is active. The idea is that inhibitory connections that target already quiet neurons become stronger, helping maintain the excitation/inhibition (E/I) balance of the network. The inhibitory learning signal also decays four times faster than excitatory by default, making it more responsive to rapid changes in activity.

Both rules are enabled together and use per-connection weights so each synapse evolves independently based on its own spike timing.

| Label   | Mechanism             | Plastic? | Rule        |
|---------|-----------------------|----------|-------------|
| AMPA    | `StdpLinExp2Syn`      | Yes      | Excitatory  |
| NMDA    | `StdpLinExp2SynNMDA`  | Yes      | Excitatory  |
| GABA_A  | `StdpLinExp2SynInh`   | Yes      | Inhibitory  |
| GABA_B  | `LinExp2Syn`          | No       | —           |

## Homeostasis (synaptic scaling)

Hebbian learning has a fundamental instability problem: a synapse that drives a neuron to fire gets strengthened, which makes the neuron fire even more, which strengthens the synapse further, and so on. Left unchecked, this positive feedback loop leads to runaway excitation or complete silencing.

Biological neurons solve this through homeostatic synaptic scaling. Turrigiano+2008 proposed a model that rescales all incoming synaptic weights multiplicatively so the total stays roughly constant. This preserves the relative differences between synapses (what STDP learned) while keeping overall activity in a healthy range.

livn implements this via `normalize_weights()` as follows:

```
w_i <- w_i x (target / SUM_j w_j)
```

where `w_i` is each incoming weight and `SUM_j w_j` is their current sum.

Key properties:
- Preserves ratios such that relative weight differences learned by STDP are maintained
- Respects bounds such that individual weights stay within [w_min, w_max] via iterative clamping
- Each neuron is normalized independently

A typical training loop applies scaling periodically:

```python
from livn import make

env = make("EI1")

env.record_spikes()
env.record_voltage()
env.record_weights(dt=1.0)  # track weight evolution

env.enable_plasticity()

for epoch in range(10):
    env.run(100)             # Hebbian learning
    env.normalize_weights()  # homeostatic rescaling

weights = env.get_weights()
for (gid, syn_id, mech), w in list(weights.items())[:5]:
    print(f"  gid={gid} syn={syn_id} ({mech}): w={w:.4f}")
```

For a complete training example with plotting, see the [STDP Training example](../examples/stdp-training.md).

## Tuning tips

- For stronger plasticity, increase `A_ltp` / `A_ltd`
- For more selective LTP (excitatory), raise `theta_ltp` so only strong depolarizations trigger potentiation
- For more selective LTP (inhibitory), lower `theta_ltp` so only deeper hyperpolarization triggers potentiation
- For wider dynamic range, increase `w_max` and/or decrease `w_min`
- For slower adaptation, reduce `A_ltp` / `A_ltd` and increase `learning_tau`
- For homeostatic training, call `normalize_weights()` every 50–200 ms of simulation to prevent weight divergence
- To freeze after learning, use `disable_plasticity()` once the network has settled, then run evaluation with fixed weights

Learn more about [Tuning System](../systems/tuning.md).

## Parameters

The following parameters control the learning rule. They can be set via `enable_plasticity(config={...})` or by modifying the model's `neuron_plasticity_defaults()`.

### Excitatory defaults 

(from `ExcSigma3Exp2Syn`)

| Parameter          | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `A_ltp`            | 1.0     | LTP amplitude scaling                    |
| `A_ltd`            | 1.0     | LTD amplitude scaling                    |
| `theta_ltp`        | −45 mV  | Voltage threshold for LTP                |
| `theta_ltd`        | −60 mV  | Voltage threshold for LTD                |
| `ltp_sigmoid_half` | −40 mV  | Sigmoid half-activation for LTP          |
| `ltd_sigmoid_half` | −55 mV  | Sigmoid half-activation for LTD          |
| `learning_slope`   | 1.3     | Sigmoid slope                            |
| `learning_tau`     | 20      | Learning signal time scale               |
| `w_max`            | 5.0     | Maximum weight multiplier                |
| `w_min`            | 0.0001  | Minimum weight multiplier                |
| `w_init`           | 1.0     | Initial weight (set in MOD PARAMETER)    |

### Inhibitory defaults 

(from `InhSigma3Exp2Syn`)

| Parameter          | Default | Description                                        |
|--------------------|---------|----------------------------------------------------|
| `A_ltp`            | 1.0     | LTP amplitude scaling                              |
| `A_ltd`            | 1.0     | LTD amplitude scaling                              |
| `theta_ltp`        | −77 mV  | Voltage threshold for LTP (hyperpolarized)         |
| `theta_ltd`        | −70 mV  | Voltage threshold for LTD (hyperpolarized)         |
| `ltp_sigmoid_half` | −80 mV  | Sigmoid half-activation for LTP                    |
| `ltd_sigmoid_half` | −73 mV  | Sigmoid half-activation for LTD                    |
| `learning_slope`   | 1.2     | Sigmoid slope                                      |
| `learning_tau`     | 20      | Learning signal time scale                         |
| `w_max`            | 5.0     | Maximum weight multiplier                          |
| `w_min`            | 0.0001  | Minimum weight multiplier                          |
| `w_init`           | 1.0     | Initial weight (set in MOD PARAMETER)              |

Note: the inhibitory learning signal decays 4x faster than excitatory

---


## `enable_plasticity()`

```python
# using model defaults (both excitatory and inhibitory)
env.enable_plasticity()

# override parameters for all synapses
env.enable_plasticity({
    "A_ltp": 0.005,
    "A_ltd": 0.002,
    "w_max": 3.0,
})

# different parameters for each population
env.enable_plasticity({
    "EXC": {
        "A_ltp": 0.005,
        "A_ltd": 0.002,
        "theta_ltp": -40.0,
        "theta_ltd": -55.0,
    },
    "INH": {
        "A_ltp": 0.01,
        "theta_ltp": -77.0,
        "theta_ltd": -70.0,
    },
})
```

Population names (e.g. `"EXC"`, `"INH"`) are mapped to mechanism types via `model.neuron_plasticity_mechanism_groups()`

## `disable_plasticity()`

```python
env.enable_plasticity()
env.run(200)                 # weights evolve
env.disable_plasticity()
env.run(200)                 # weights frozen
```

## `normalize_weights()`

```python
# normalize to default target (mean w = 1)
env.normalize_weights()

# or set a custom target sum per neuron
env.normalize_weights(target=3.0)
```

## `env.get_weights()`

```python
weights = env.get_weights()
# {(0, 0, 'StdpLinExp2Syn'): 1.1, (0, 1, 'StdpLinExp2SynNMDA'): 0.987, ...}
```

## `record_weights()`

```python
env.record_weights(dt=1.0)
env.run(500)

for key, vec in env.w_recs.items():
    trace = np.array(vec.as_numpy())
    print(f"{key}: min={trace.min():.3f} max={trace.max():.3f}")
```

## Implementation details

- Weight changes are applied only when a presynaptic spike arrives, using the accumulated learning integral since the last spike.
- The voltage-based learning signal (`learning_w`) and its integral (`learn_int`) are shared across all connections to a given point process. Per-connection differentiation arises from the timing of when each connection samples this integral

### How the learning rule works

Each synapse monitors the postsynaptic voltage and computes two signals via sigmoidal activation functions:

- LTD signal is activated when voltage crosses `theta_ltd` (upward for excitatory, downward for inhibitory)
- LTP signal is activated when voltage crosses `theta_ltp` (same direction convention)

These are combined into a shared learning signal `learning_w` that decays exponentially. A running integral of `learning_w` (called `learn_int`) accumulates as a STATE variable.

Weight updates happen only at spike arrival. When a presynaptic spike reaches a connection:

1. `Delta = learn_int - last_int` (learning since this connection's last spike)
2. `w_plastic = w_plastic + Delta * w_plastic`
3. Clamp to `[w_min, w_max]`

Because different presynaptic neurons fire at different times, each connection samples a different portion of the learning integral, providing natural per-connection differentiation.

### Per-connection weight architecture

In NEURON, multiple presynaptic connections can share a single point process. Each incoming NetCon stores its own `w_plastic` (index 2) and `last_int` (index 3) in its weight vector, enabling per-connection Hebbian learning.

### NetCon weight vector layout

Each STDP-capable NetCon has 4 weight vector elements:

| Index | Name         | Description                                       |
|-------|--------------|---------------------------------------------------|
| 0     | `weight`     | Static weight from connectivity (set by system)   |
| 1     | `g_unit`     | Unitary conductance (set by system)               |
| 2     | `w_plastic`  | Per-connection plastic weight (STDP-modulated)    |
| 3     | `last_int`   | Per-connection snapshot of `learn_int` at last spike |
