# Reduced Calcium Soma-Dendrite (RCSD)

The default model for the brian2, NEURON, and Diffrax backends. RCSD implements two-compartment neuron models with calcium dynamics, providing biophysically detailed excitatory and inhibitory cell types.

```python
from livn.models.rcsd import ReducedCalciumSomaDendrite

model = ReducedCalciumSomaDendrite()
```

## Cell types

### Excitatory: Booth-Rinzel-Kiehn motoneuron

A two-compartment motoneuron model with the following ion channels:

- **Na⁺**: Fast sodium (soma only)
- **K⁺**: Delayed rectifier (soma)
- **Ca²⁺**: L-type (dendrite) and N-type (both compartments)
- **KCa**: Calcium-dependent potassium (both compartments)

Calcium dynamics include influx via Ca²⁺ channels and extrusion via first-order kinetics, driving the KCa current. Soma and dendrite are coupled via gap-junction conductance.

```python
params = model.params("BoothRinzelKiehn-MN")
```

### Inhibitory: Pinsky-Rinzel interneuron

A fast-spiking interneuron with Ca²⁺ dynamics, KCa current, and simplified K⁺/Na⁺ gating compared to the excitatory cell.

```python
params = model.params("PinskyRinzel-PVBC")
```

## Dual-compartment stimulation

Because each neuron has a soma and dendrite compartment, `stimulus_coordinates` returns interleaved coordinates, doubling the number of stimulation targets:

```python
coords = model.stimulus_coordinates(system.neuron_coordinates)
# Shape: [2 * n_neurons, 4] - soma0, dend0, soma1, dend1, ...
```

The dendritic compartment is offset from the soma by `dx = 0.9 × L`, where L = 120 µm for motoneurons and L = 37.6 µm for interneurons.

## Synaptic dynamics

RCSD defines four synapse types with distinct kinetics:

| Synapse | Type | Rise time | Decay time | Voltage-dependent |
|---------|------|-----------|------------|-------------------|
| AMPA | Excitatory | ~0.1 ms | ~2 ms | No |
| NMDA | Excitatory | ~5 ms | ~30 ms | Yes (Mg²⁺ block) |
| GABA_A | Inhibitory | ~0.5 ms | ~5 ms | No |
| GABA_B | Inhibitory | ~50 ms | ~200 ms | No |

Synaptic weights are specified per projection (pre-population → post-population), target section, and synapse type:

```python
weights = {
    "EXC_EXC-hillock-AMPA-weight": 0.001,
    "EXC_INH-hillock-AMPA-weight": 2.9,
    "INH_EXC-soma-GABA_A-weight": 9.4,
}
env.set_weights(weights)
```

### Synaptic plasticity (STDP)

RCSD supports spike-timing-dependent plasticity via specialized synapse mechanisms:

- **Excitatory**: `StdpLinExp2Syn`, `StdpLinExp2SynNMDA`
- **Inhibitory**: `StdpLinExp2SynInh`

See the [Plasticity](/models/plasticity/stdp) reference and the [Plasticity guide](/guide/advanced/plasticity) for usage details.

## Background noise

RCSD uses an Ornstein-Uhlenbeck process (Gfluct3) to model fluctuating synaptic conductances. The noise is spatially split: the soma receives inhibitory noise only, while the dendrite receives excitatory noise only.

```python
noise_params = {
    "g_e0": 1.0,       # mean excitatory conductance
    "g_i0": 1.2,       # mean inhibitory conductance
    "std_e": 0.33,      # excitatory conductance std
    "std_i": 0.36,      # inhibitory conductance std
    "tau_e": 33.0,      # excitatory time constant (ms)
    "tau_i": 28.5,      # inhibitory time constant (ms)
}
env.set_noise(noise_params)
```

## Diffrax backend

When used with the Diffrax backend, RCSD provides a `MotoneuronCulture` Equinox module for differentiable simulation. It supports both current and conductance input modes and returns `(time, soma_voltage, dend_voltage, soma_current, dend_current, final_state)`.

```python
model = ReducedCalciumSomaDendrite()
module = model.diffrax_module()
```
