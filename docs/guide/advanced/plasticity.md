# Synaptic Plasticity

In biological neural networks, the strength of a connection (synapse) between two neurons is not fixed but changes over time based on the activity of the neurons it connects. This property is called synaptic plasticity and is widely believed to be the cellular basis of learning and memory.

When a synapse gets stronger, it is called long-term potentiation (LTP). When it gets weaker, it is called long-term depression (LTD). Whether a synapse is potentiated or depressed depends on the relative timing and magnitude of activity in the pre- and postsynaptic neurons, a principle summarized as "neurons that fire together, wire together" (Hebbian learning).

livn implements a voltage-based form of spike-timing dependent plasticity (STDP), where the postsynaptic membrane voltage determines the direction and magnitude of weight changes. For details on the learning rules, parameters, and implementation, see the [Plasticity reference](/models/plasticity/stdp.md).

## Basic usage

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

For a complete training example with plotting, see the [STDP Training example](../../examples/stdp-training.md).

## API

### `enable_plasticity()`

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

### `disable_plasticity()`

```python
env.enable_plasticity()
env.run(200)                 # weights evolve
env.disable_plasticity()
env.run(200)                 # weights frozen
```

### `normalize_weights()`

```python
# normalize to default target (mean w = 1)
env.normalize_weights()

# or set a custom target sum per neuron
env.normalize_weights(target=3.0)
```

### `get_weights()`

```python
weights = env.get_weights()
# {(0, 0, 'StdpLinExp2Syn'): 1.1, (0, 1, 'StdpLinExp2SynNMDA'): 0.987, ...}
```

### `record_weights()`

```python
env.record_weights(dt=1.0)
env.run(500)

for key, vec in env.w_recs.items():
    trace = np.array(vec.as_numpy())
    print(f"{key}: min={trace.min():.3f} max={trace.max():.3f}")
```

## Tuning tips

- For stronger plasticity, increase `A_ltp` / `A_ltd`
- For more selective LTP (excitatory), raise `theta_ltp` so only strong depolarizations trigger potentiation
- For more selective LTP (inhibitory), lower `theta_ltp` so only deeper hyperpolarization triggers potentiation
- For wider dynamic range, increase `w_max` and/or decrease `w_min`
- For slower adaptation, reduce `A_ltp` / `A_ltd` and increase `learning_tau`
- For homeostatic training, call `normalize_weights()` every 50–200 ms of simulation to prevent weight divergence
- To freeze after learning, use `disable_plasticity()` once the network has settled, then run evaluation with fixed weights

Learn more about [Tuning Systems](../../systems/tuning.md).
