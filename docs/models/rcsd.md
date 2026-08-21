# Reduced Calcium Soma-Dendrite (RCSD)

The default model for the brian2, NEURON, and Diffrax backends. RCSD pairs a two-compartment motoneuron with a spinal inhibitory interneuron, both carrying calcium dynamics and a calcium-dependent potassium current.

```python
from livn.models.rcsd import ReducedCalciumSomaDendrite

model = ReducedCalciumSomaDendrite()
```

| Option | Default | Meaning |
|--------|---------|---------|
| `input_mode` | `None` | Override how the cell interprets a stimulus (`current_density`, `conductance`, `current`, `irradiance`). Only needed on the JAX backend, which bakes the choice into the compute graph |
| `refractory_period` | `2.0` | Spike-detector dead time in ms |
| `renshaw_phenotype` | `"invitro"` | Which tuned Renshaw parameter set the `INH` population uses (`"invitro"` or `"perry"`) |
| `short_term_depression` | `False` | Wire AMPA through `DepLinExp2Syn` (Tsodyks-Markram depression, per presynaptic stream) instead of `StdpLinExp2Syn` |

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

### Inhibitory: V1 Renshaw cell

`INH` is a spinal V1 Renshaw cell, the recurrent-inhibition subtype motoneurons wire onto preferentially ([Hoang et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6590086/)).

The cell is single-compartment and uses the same Booth-Rinzel-Kiehn channel formalism as the motoneuron:

- **Na⁺**: fast sodium (`Nas`)
- **K⁺**: delayed rectifier (`Kdr`) plus an A-type current (`Ka_v1in`), which the motoneuron does not have
- **Ca²⁺**: N-type (`CaN`) with first-order calcium accumulation
- **KCa**: calcium-dependent potassium, the afterhyperpolarization that paces repetitive firing
- plus `pas`, a `constant` resting-current pin

There are two parameter sets, selected by `renshaw_phenotype`:

| | `invitro` (default) | `perry` |
|---|---|---|
| Parameter set | `V1In-Renshaw-InVitro` | `V1In-Renshaw-Perry` |
| Fitted against | ES-cell-derived Renshaw cells in vitro (Hoang et al. 2018) | neonatal mouse Renshaw cells (Perry et al. 2015) |
| Input resistance target | ~420 MΩ | ~233 MΩ |
| `V_rest` | −60.0 mV | −50.5 mV |
| `V_threshold` | −50.0 mV | −30.8 mV |
| Soma diameter | 19.4 µm | 16.4 µm |

```python
model = ReducedCalciumSomaDendrite(renshaw_phenotype="perry")
params = model.params("V1In-Renshaw-Perry")
```

### Recurrent inhibition

The two cell types form a spinal recurrent-inhibition loop where motoneurons excite Renshaw cells cholinergically, Renshaw cells inhibit motoneurons glycinergically, on the motoneuron soma.

### Legacy: Pinsky-Rinzel interneuron

`params("PinskyRinzel-PVBC")` is still defined and used by the brian2 backend. The diffrax backend simulates motoneurons alone (see below).

## Compartments and stimulation

`stimulus_coordinates` returns two coordinates per neuron, interleaved, for every population:

```python
coords = model.stimulus_coordinates(system.neuron_coordinates)
# Shape: [2 * n_neurons, 4] - soma0, dend0, soma1, dend1, ...
```

The second coordinate sits `dx = 0.9 × L` from the soma along x, with L = 120 µm, the motoneuron's total length. `recording_coordinates` returns the same layout.

Renshaw cells have no second compartment, so the NEURON backend drops any stimulus column naming a section the cell does not have, and the membrane-current row for that slot stays zero.

## Synaptic dynamics

RCSD declares four synapse types and the NEURON mechanism each maps to:

| Synapse | Type | Mechanism | Voltage-dependent |
|---------|------|-----------|-------------------|
| AMPA | Excitatory | `StdpLinExp2Syn`, or `DepLinExp2Syn` with `short_term_depression=True` | No |
| NMDA | Excitatory | `StdpLinExp2SynNMDA` | Yes (Mg²⁺ block) |
| GABA_A | Inhibitory | `StdpLinExp2SynInh` | No |
| GABA_B | Inhibitory | `LinExp2Syn` | No |

```python
weights = {
    "EXC_EXC-hillock-AMPA-weight": 0.001,   # MN -> MN, on the motoneuron dendrite
    "INH_EXC-soma-AMPA-weight": 2.9,        # MN -> Renshaw, cholinergic
    "EXC_INH-soma-GABA_A-weight": 9.4,      # Renshaw -> MN, glycinergic, on the soma
}
env.set_weights(weights)
```

`hillock` is the motoneuron's dendritic compartment. A synapse placed off-soma on a Renshaw cell resolves to its soma, since that is the only section it has. `env.weight_names` lists the keys a given network accepts.

### Synaptic plasticity (STDP)

RCSD supports spike-timing-dependent plasticity via specialized synapse mechanisms:

- **Excitatory**: `StdpLinExp2Syn`, `StdpLinExp2SynNMDA`
- **Inhibitory**: `StdpLinExp2SynInh`

See the [Plasticity](/models/plasticity/stdp) reference and the [Plasticity guide](/guide/advanced/plasticity) for usage details.

## Opsin configuration

RCSD includes built-in opsin (channelrhodopsin) support for [optical stimulation](/guide/advanced/optical-stimulation). The `opsin_config()` method controls which opsin mechanism is inserted and where:

```python
model = ReducedCalciumSomaDendrite()
model.opsin_config()
# {'mechanism': 'RhO3c', 'sections': ['soma'], 'wavelength_nm': 473.0}
```

Override in a subclass to customize or disable:

```python
class NoOpsin(ReducedCalciumSomaDendrite):
    def opsin_config(self):
        return None  # disable opsins
```


## Background noise

RCSD uses an Ornstein-Uhlenbeck process (Gfluct3) to model fluctuating synaptic conductances. On the two-compartment motoneuron the noise is spatially split: the soma receives inhibitory noise only, the dendrite excitatory noise only.

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

The Renshaw cell is exempt from the split: its soma is its only site, so it carries both the excitatory and the inhibitory component. Applying the somatic half alone would pin it near `E_i` (−75 mV) and it would never fire.

## Diffrax backend

When used with the Diffrax backend, RCSD provides a `MotoneuronCulture` Equinox module for differentiable simulation. It supports both current and conductance input modes and returns `(time, soma_voltage, dend_voltage, soma_current, dend_current, final_state)`. Only the motoneuron is implemented there; a network simulated through Diffrax is excitatory-only.

```python
model = ReducedCalciumSomaDendrite()
module = model.diffrax_module(env, key)
```
