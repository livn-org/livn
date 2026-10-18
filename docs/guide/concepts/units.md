# Units

livn uses one rule for units everywhere in its API.

> A variable name without a unit is in its default unit. A name whose value is in any other unit carries that unit as a suffix, and the suffix is mandatory.

So `dt`, `duration` and `window` are in ms, `rate` is in Hz and `radius` is in µm. A quantity in anything else is explicit using a suffix: `amplitude_uA`, `sigma_uV`, `rate_per_ms`.

## Default units

| quantity | default |
|---|---|
| time, durations, `dt`, windows, bins, latencies | ms |
| membrane and extracellular potential | mV |
| length, diameter, distance, position | µm (area µm²) |
| current: point process, injected, whole-cell | nA |
| current density | mA/cm² |
| conductance: point process, synaptic weight, whole-cell | µS |
| conductance density | S/cm² |
| specific capacitance | µF/cm² |
| axial resistivity | Ω·cm |
| input or seal resistance | MΩ |
| concentration | mM |
| temperature | °C |
| frequency: firing, burst and sampling rates, filter corners | Hz |
| conduction and propagation velocity | µm/ms (numerically equal to mm/s and m/s) |
| irradiance | mW/mm² |
| wavelength | nm |
| angle | degrees |

## Suffixes

A quantity that is not in its default unit gets a suffix in SI case: `_uA`, `_uV`, `_pA`, `_nS`, `_pF`, `_pS`, `_mS_per_cm2`, `_ohm_m`, `_S_per_m`, `_per_ms`. A suffix is used only for a non-default unit. A default unit never gets one, so `window_ms`, `radius_um` and `rate_hz` are not valid names.

Examples: `threshold_uA`, `sigma_uV`, `gc_mS_per_cm2`, `bias_pA`, `g0_pS`, `stimulation_resistivity_ohm_m`, `max_rate_per_ms`.

## Seconds

Seconds are not used anywhere in the API; times are in ms. The only exception is a wall-clock interval, which carries `_s` (`log_interval_wall_s`).

## Exceptions

- GLIF. The parameters (`g_L`, `asc_amp_*`, synaptic weights, input current) keep the Allen Institute names and units so conductance in nS, current in pA, time in ms, potential in mV. They carry no suffix. See [GLIF](/models/glif).
- MEA. Recorded membrane currents are in µA and electrode potentials in µV, and carry the suffixes `_uA` and `_uV` (`membrane_currents_uA`, `sigma_uV`, `amplitude_uA`).
- External formats. The Allen `neuron_config` stays in SI (`dt=5e-5`, `asc_tau_array`, `C`, `R_input`); `to_neuron_config` and `from_neuron_config` convert at the boundary. Backend-native names (NEURON, brian2, ...) are not renamed. The brian2 `CellHandle` converts SI to the default units at its boundary.
- `_s` as soma or slow (`theta_s`, `I_memb_s`) is not a seconds suffix.
