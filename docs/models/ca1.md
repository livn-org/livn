# CA1 Hippocampal Pyramidal

A morphologically detailed model of hippocampal CA1 pyramidal neurons for the NEURON backend.

```python
from livn.models.ca1 import CA1

model = CA1()
```

## Cell types

### Pyramidal (PYR)

Multi-compartment pyramidal neurons with a 31-section morphology:

- **Soma** (root)
- **Axon initial segment** + axon
- **Hillock** (backwards projection)
- **Apical dendrites** (radiatum/lacunosum-moleculare layers)
- **Oblique dendrites** (oriens compartments)

### External inputs

Other populations (EC, CA2, CA3) are modeled as external inputs via `VecStim` point processes rather than biophysical cell models.

## Synaptic dynamics

CA1 uses:

- **LinExp2Syn**: Standard double-exponential synapse
- **LinExp2SynNMDA**: NMDA-specific synapse with voltage-dependent Mg²⁺ block

## Stimulation

The morphologically detailed structure enables section-specific stimulation targeting via Y-offsets from the soma, allowing precise spatial targeting along the dendritic tree.
