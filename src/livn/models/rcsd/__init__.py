from __future__ import annotations

import math
import os

from livn.types import Model
from livn import types
from livn.backend import backend

_USES_JAX = False

if "ax" in backend():
    import jax.numpy as np

    _USES_JAX = True
else:
    import numpy as np


class ReducedCalciumSomaDendrite(Model):
    def __init__(self, input_mode: str | None = None):
        # Optional override for the underlying neuron's stimulus
        # interpretation; this is only needed for the JAX
        # backend that build the compute graph at
        # compile time
        if input_mode is not None and input_mode not in {
            "current_density",
            "conductance",
            "current",
            "irradiance",
        }:
            raise ValueError(
                f"Unknown input_mode {input_mode!r}; expected one of "
                f"'current_density', 'conductance', 'current', 'irradiance'."
            )
        self.input_mode = input_mode

    def prepare_stimulus(self, stimulus):
        modes = {
            "extracellular",
            "current",
            "current_density",
            "conductance",
            "irradiance",
        }
        if stimulus.input_mode not in modes:
            raise ValueError(
                f"ReducedCalciumSomaDendrite does not support input_mode "
                f"'{stimulus.input_mode}'. Supported: {modes}"
            )
        return stimulus

    def opsin_config(self):
        return {
            "mechanism": "RhO3c",
            "sections": ["soma"],
            "wavelength_nm": 473.0,
        }

    def neuron_opsin_config(self):
        return self.opsin_config()

    def diffrax_opsin_config(self):
        return self.opsin_config()

    def stimulus_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> types.Float[types.Array, "n_stim_coords ixyz=4"]:
        """
        Transform neuron coordinates for two-compartment model stimulation

            gid, x, y, z -> gid, x + pp * L, y, z

        Returns:
            [2*n_neurons, 4] with interleaved soma/dendrite coordinates
            soma0, dend0, soma1, dend1, ...
        """
        L = 120.0  # 37.6
        dx = 0.9 * L

        n_neurons = neuron_coordinates.shape[0]

        dend_coords = neuron_coordinates.copy()
        if _USES_JAX:
            dend_coords = dend_coords.at[:, 1].add(dx)
        else:
            dend_coords[:, 1] += dx

        # interleave soma0, dend0, soma1, dend1, ...
        stacked = np.stack([neuron_coordinates, dend_coords], axis=1)  # [n, 2, 4]
        interleaved_coords = stacked.reshape(2 * n_neurons, 4)

        return interleaved_coords

    def recording_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
        population: str | None = None,
    ) -> types.Float[types.Array, "n_stim_coords ixyz=4"]:
        return self.stimulus_coordinates(neuron_coordinates, population=population)

    # neuron

    def params(self, name: str):
        base = {
            "BoothRinzelKiehn-MN": {
                "Ltotal": 120.0,
                "dend_alpha_Caconc": 1,
                "dend_f_Caconc": 0.004,
                "dend_kCa_Caconc": 8,
                "e_pas": -62,
                "global_cm": 2.0,
                "global_diam": 5.0,
                "pp": 0.1,
                "soma_alpha_Caconc": 1,
                "soma_f_Caconc": 0.004,
                "soma_kCa_Caconc": 8,
                "cm_ratio": 1.1303897,
                "dend_g_pas": 6.165833e-05,
                "dend_gmax_CaL": 1.1316314e-05,
                "dend_gmax_CaN": 1e-05,
                "dend_gmax_KCa": 0.0019142649,
                "gc": 1.108122,
                "soma_g_pas": 1e-05,
                "soma_gmax_CaN": 0.0032349424,
                "soma_gmax_K": 0.10458818,
                "soma_gmax_KCa": 0.005655824,
                "soma_gmax_Na": 0.11399703,
                "V_rest": -57.4,
                "V_threshold": -37.0,
            },
            "PinskyRinzel-PVBC": {
                "Ltotal": 37.62028884887695,
                "cm_ratio": 3.903846025466919,
                "dend_beta_Caconc": 0.03191220387816429,
                "dend_d_Caconc": 17.446317672729492,
                "dend_g_pas": 0.0004252658982295543,
                "dend_gmax_Ca": 0.8048859238624573,
                "dend_gmax_KCa": 1.0,
                "gc": 23.3135986328125,
                "pp": 0.10000000149011612,
                "soma_g_pas": 0.0016516740433871746,
                "soma_gmax_K": 0.0010000000474974513,
                "soma_gmax_Na": 0.898166298866272,
                "e_pas": -62,
                "global_cm": 3.0,
                "global_diam": 10.0,
                "ic_constant": 0.013448839558146165,
                "V_rest": -60.0,
                "V_threshold": -37.0,
            },
        }[name]
        if self.input_mode is not None:
            base = {**base, "input_mode": self.input_mode}
        return base

    def neuron_template_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "templates")

    def neuron_mechanisms_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "mechanisms")

    def neuron_celltypes(self, celltypes):
        if "EXC" in celltypes:
            celltypes["EXC"]["template class"] = (
                "livn.models.rcsd.neuron.templates.BRK.BRK"
            )
            celltypes["EXC"]["template"] = "@" + celltypes["EXC"]["template class"]
            celltypes["EXC"]["mechanism"] = {
                "BoothRinzelKiehn": self.params("BoothRinzelKiehn-MN")
            }

        if "INH" in celltypes:
            celltypes["INH"]["template class"] = (
                "livn.models.rcsd.neuron.templates.PRN.PRN"
            )
            celltypes["INH"]["template"] = "@" + celltypes["INH"]["template class"]
            celltypes["INH"]["mechanism"] = {
                "PinskyRinzel": self.params("PinskyRinzel-PVBC")
            }

    def neuron_synapse_mechanisms(self):
        return {
            "AMPA": "StdpLinExp2Syn",
            "NMDA": "StdpLinExp2SynNMDA",
            "GABA_A": "StdpLinExp2SynInh",
            "GABA_B": "LinExp2Syn",
        }

    def neuron_synapse_rules(self):
        return {
            "Exp2Syn": {
                "mech_file": "exp2syn.mod",
                "mech_params": ["tau1", "tau2", "e"],
                "netcon_params": {"weight": 0},
                "netcon_state": {},
            },
            "LinExp2Syn": {
                "mech_file": "lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "LinExp2SynNMDA": {
                "mech_file": "lin_exp2synNMDA.mod",
                "mech_params": [
                    "tau_rise",
                    "tau_decay",
                    "e",
                    "mg",
                    "Kd",
                    "gamma",
                    "vshift",
                ],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {},
            },
            "StdpLinExp2Syn": {
                "mech_file": "stdp_lin_exp2syn.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
            "StdpLinExp2SynNMDA": {
                "mech_file": "stdp_lin_exp2synNMDA.mod",
                "mech_params": [
                    "tau_rise",
                    "tau_decay",
                    "e",
                    "mg",
                    "Kd",
                    "gamma",
                    "vshift",
                ],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
            "StdpLinExp2SynInh": {
                "mech_file": "stdp_lin_exp2syn_inh.mod",
                "mech_params": ["tau_rise", "tau_decay", "e"],
                "netcon_params": {
                    "weight": 0,
                    "g_unit": 1,
                    "w_plastic": 2,
                    "last_int": 3,
                },
                "netcon_state": {},
            },
        }

    def neuron_plasticity_defaults(self):
        """Default plasticity parameters by population

        Returns a nested dict ``{population_name: {param: value}}`` that will be applied
        to matching point processes when ``enable_plasticity()`` is called.

        Populations are mapped to mechanism types via ``neuron_plasticity_mechanism_groups()``

        Default values are taken from the Sigma3Exp2Syn mechanisms in neuronpp
        (https://github.com/ziemowit-s/neuronpp).
        """
        return {
            "EXC": {
                "A_ltp": 1.0,
                "A_ltd": 1.0,
                "theta_ltp": -45.0,
                "theta_ltd": -60.0,
                "ltp_sigmoid_half": -40.0,
                "ltd_sigmoid_half": -55.0,
                "learning_slope": 1.3,
                "learning_tau": 20.0,
                "w_max": 5.0,
                "w_min": 0.0001,
            },
            "INH": {
                "A_ltp": 1.0,
                "A_ltd": 1.0,
                "theta_ltp": -77.0,
                "theta_ltd": -70.0,
                "ltp_sigmoid_half": -80.0,
                "ltd_sigmoid_half": -73.0,
                "learning_slope": 1.2,
                "learning_tau": 20.0,
                "w_max": 5.0,
                "w_min": 0.0001,
            },
        }

    def neuron_plasticity_mechanism_groups(self):
        """Maps population/group names to sets of STDP mechanism class names.

        Used by ``enable_plasticity()`` to decide which parameter config
        to apply to each point process based on its mechanism type.
        """
        return {
            "EXC": {"StdpLinExp2Syn", "StdpLinExp2SynNMDA"},
            "INH": {"StdpLinExp2SynInh"},
        }

    def neuron_noise_mechanism(self, section):
        from neuron import h

        return h.Gfluct3(section), None

    def neuron_noise_configure(
        self,
        population,
        mechanism,
        state,
        std_e=0.0030,
        std_i=0.0066,
        g_e0=0.0121,
        g_i0=0.0573,
        tau_e=2.728,
        tau_i=10.49,
        E_e=0,
        E_i=-75,
    ):
        import math

        sec_name = mechanism.get_segment().sec.name()
        is_soma = "soma" in sec_name

        mechanism.tau_e = tau_e
        mechanism.tau_i = tau_i
        mechanism.E_e = E_e
        mechanism.E_i = E_i

        if is_soma:
            # inhibition only
            mechanism.std_e = 0
            mechanism.g_e0 = 0
            mechanism.std_i = std_i
            mechanism.g_i0 = g_i0
        else:
            # excitation only
            mechanism.std_e = std_e
            mechanism.g_e0 = g_e0
            mechanism.std_i = 0
            mechanism.g_i0 = 0

        mechanism.on = 1 if (mechanism.std_e > 0 or mechanism.std_i > 0) else 0

        # recompute INITIAL variables manually to ensure changes propagate mid-simulation
        h_val = mechanism.h
        if mechanism.tau_e > 0:
            mechanism.D_e = 2 * mechanism.std_e**2 / mechanism.tau_e
            mechanism.exp_e = math.exp(-h_val / mechanism.tau_e)
            mechanism.amp_e = mechanism.std_e * math.sqrt(
                max(0.0, 1.0 - math.exp(-2 * h_val / mechanism.tau_e))
            )
        else:
            mechanism.D_e = 0.0
            mechanism.exp_e = 0.0
            mechanism.amp_e = 0.0

        if mechanism.tau_i > 0:
            mechanism.D_i = 2 * mechanism.std_i**2 / mechanism.tau_i
            mechanism.exp_i = math.exp(-h_val / mechanism.tau_i)
            mechanism.amp_i = mechanism.std_i * math.sqrt(
                max(0.0, 1.0 - math.exp(-2 * h_val / mechanism.tau_i))
            )
        else:
            mechanism.D_i = 0.0
            mechanism.exp_i = 0.0
            mechanism.amp_i = 0.0

    def neuron_default_noise(self, system: str):
        return {
            "EI1": {
                "g_e0": 1.0,
                "g_i0": 1.2172681093215942,
                "std_e": 0.3290764391422272,
                "std_i": 0.35633188486099243,
                "tau_e": 33.00786209106445,
                "tau_i": 28.50772476196289,
            },
            "EI2": {
                "g_e0": 1.4662606716156006,
                "g_i0": 0.9061993360519409,
                "std_e": 0.47152602672576904,
                "std_i": 0.1969195306301117,
                "tau_e": 17.493135452270508,
                "tau_i": 7.105101585388184,
            },
            "EI3": {},
            "EI4": {},
        }[system]

    def neuron_default_weights(self, system: str):
        return {
            "EI1": {
                "EXC_EXC-hillock-AMPA-weight": 0.0010000000254350994,
                "EXC_EXC-hillock-NMDA-weight": 0.37764625228307414,
                "EXC_INH-hillock-AMPA-weight": 2.9091933347646908,
                "EXC_INH-hillock-NMDA-weight": 0.0010000000254350994,
                "INH_EXC-soma-GABA_A-weight": 9.406616405134113,
                "INH_INH-soma-GABA_A-weight": 8.710510071227473,
            },
            "EI2": {
                "EXC_EXC-hillock-AMPA-weight": 0.0010000000254350994,
                "EXC_EXC-hillock-NMDA-weight": 0.00398131980116756,
                "EXC_INH-hillock-AMPA-weight": 12.114758587424397,
                "EXC_INH-hillock-NMDA-weight": 0.31300935167465127,
                "INH_EXC-soma-GABA_A-weight": 0.5000229360632054,
                "INH_INH-soma-GABA_A-weight": 5.83084802642212,
            },
            "EI3": {},
            "EI4": {},
        }[system]

    # diffrax

    def diffrax_module(self, env, key):
        from livn.models.rcsd.diffrax.culture import MotoneuronCulture

        return MotoneuronCulture(
            num_neurons=len(env.system.gids),
            params=self.params("BoothRinzelKiehn-MN"),
            key=key,
        )

    # brian2

    def _brk_equations(self, offset, params):
        p = params

        # Geometry: coupling conductance
        # In NEURON: Ra is set so that the coupling between soma and dend gives gc
        # gc is in mS/cm2, we convert to brian2 units in the equations
        # gc_coupling = gc / (p * (1-p)) between compartments
        pp = p["pp"]
        Ltotal = p["Ltotal"]
        gc = p["gc"]
        cm = p["global_cm"]
        cm_ratio = p["cm_ratio"]
        diam = p["global_diam"]
        e_pas = p["e_pas"]

        # Compartment areas (um^2) and lengths
        L_soma = pp * Ltotal  # um
        # L_dend = (1 - pp) * Ltotal  # um
        area_soma = math.pi * diam * L_soma  # um^2
        # area_dend = math.pi * diam * L_dend  # um^2

        # Axial resistance: same formula as in BRK.py biophys()
        # Ra is set so coupling effective conductance = gc (mS/cm2)
        # The coupling current: I_coupling = gc/(p*(1-p)) * (V_other - V_self) in mS/cm2 * mV = uA/cm2
        # Convert to nA: I_nA = I_mA_per_cm2 * area_cm2 * 1000
        # But in brian2 we work with mV directly in the ODE
        # dV/dt = ... / (cm * area) where cm is uF/cm2
        # The voltage equation: C * dV/dt = -I_ionic + I_coupling + I_stim (all in mA/cm2 or equivalent)
        # Since brian2 works with units, we express everything in mV and ms

        # KTF factor for GHK (CaN uses 36/293.15, CaL uses 25/293.15)
        celsius = 6.3
        fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
        fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
        cao = 2.0  # mM external calcium

        # gc_eff: coupling in S/cm2 = gc * 1e-3 (since gc in mS/cm2)
        # I_coup_soma = gc_eff / (pp * (1-pp)) * (Vd - Vs)  [A/cm2]
        # In the ODE: dVs/dt = 1/C * (...) where C in uF/cm2
        # 1/C * I [mA/cm2] gives mV/ms since: mA/cm2 / (uF/cm2) = mA/uF = 1000 V/s = mV/ms (just right!)
        # gc contribution: gc in mS/cm2, so gc*(Vd-Vs)/(pp*(1-pp)) in mS/cm2 * mV = uA/cm2 = 0.001 mA/cm2
        gc_factor = gc / (pp * (1 - pp)) * 0.001  # convert to mA/cm2 per mV -> S/cm2

        ic_constant = p.get("ic_constant", 0.0)
        ic_constant_d = p.get("ic_constant_d", 0.0)

        # Unit conversion: synaptic conductance in uS, voltage in mV
        # g_syn * (V - e) gives nA;  convert to mA/cm2: * 1e-6 / area_cm2
        # Sign: NEURON i = g*(v-e) is outward current; we want -i for dV/dt
        syn_factor = 1e-6 / (area_soma * 1e-8)  # nA -> mA/cm2

        return f"""
        # Soma voltage
        dVs/dt = (1000.0/{cm * cm_ratio}) * (-I_Na - I_K - I_KCa_s - I_CaN_s - I_leak_s - I_coup_s + I_noise_s + I_stim_s + I_ext + I_syn + {ic_constant}) / ms : 1
        # Dendrite voltage
        dVd/dt = (1000.0/{cm}) * (-I_KCa_d - I_CaN_d - I_CaL_d - I_leak_d - I_coup_d + I_stim_d + {ic_constant_d}) / ms : 1

        # --- Soma currents (mA/cm2) ---
        I_Na = {p["soma_gmax_Na"]} * m_inf_s**3 * h_s * (Vs - E_Na) : 1
        I_K = {p["soma_gmax_K"]} * n_s**4 * (Vs - E_K) : 1
        I_KCa_s = {p["soma_gmax_KCa"]} * (Ca_s / (Ca_s + 0.0005)) * (Vs - E_K) : 1
        I_CaN_s = {p["soma_gmax_CaN"]} * mnS**2 * hnS * ghk_s : 1
        I_leak_s = {p["soma_g_pas"]} * (Vs - ({e_pas})) : 1

        # --- Dendrite currents (mA/cm2) ---
        I_KCa_d = {p["dend_gmax_KCa"]} * (Ca_d / (Ca_d + 0.0005)) * (Vd - E_K) : 1
        I_CaN_d = {p["dend_gmax_CaN"]} * mnD**2 * hnD * ghk_d_N : 1
        I_CaL_d = {p["dend_gmax_CaL"]} * ml_d * ghk_d_L : 1
        I_leak_d = {p["dend_g_pas"]} * (Vd - ({e_pas})) : 1

        # --- Coupling (mA/cm2) ---
        I_coup_s = {gc_factor} * (Vs - Vd) : 1
        I_coup_d = {gc_factor} * (Vd - Vs) : 1

        # --- GHK driving force for calcium ---
        ghk_s = -({fN}) * (1.0 - (Ca_s / {cao}) * exp(Vs / {fN})) * efun_s : 1
        efun_s = int(abs(Vs / {fN}) < 1e-4) * (1.0 - Vs / {fN} / 2.0) + int(abs(Vs / {fN}) >= 1e-4) * ((Vs / {fN}) / (exp(Vs / {fN}) - 1.0 + 1e-20)) : 1

        ghk_d_N = -({fN}) * (1.0 - (Ca_d / {cao}) * exp(Vd / {fN})) * efun_d_N : 1
        efun_d_N = int(abs(Vd / {fN}) < 1e-4) * (1.0 - Vd / {fN} / 2.0) + int(abs(Vd / {fN}) >= 1e-4) * ((Vd / {fN}) / (exp(Vd / {fN}) - 1.0 + 1e-20)) : 1

        ghk_d_L = -({fL}) * (1.0 - (Ca_d / {cao}) * exp(Vd / {fL})) * efun_d_L : 1
        efun_d_L = int(abs(Vd / {fL}) < 1e-4) * (1.0 - Vd / {fL} / 2.0) + int(abs(Vd / {fL}) >= 1e-4) * ((Vd / {fL}) / (exp(Vd / {fL}) - 1.0 + 1e-20)) : 1

        # --- Gating variables ---
        m_inf_s = 1.0 / (1.0 + exp(-(Vs + 35.0) / 7.8)) : 1
        dh_s/dt = (1.0 / (1.0 + exp((Vs + 55.0) / 7.0)) - h_s) / (30.0 / (exp((Vs + 50.0) / 15.0) + exp(-(Vs + 50.0) / 16.0))) / ms : 1
        dn_s/dt = (1.0 / (1.0 + exp(-(Vs + 28.0) / 15.0)) - n_s) / (7.0 / (exp((Vs + 40.0) / 40.0) + exp(-(Vs + 40.0) / 50.0))) / ms : 1

        # CaN gating (soma)
        dmnS/dt = (1.0 / (1.0 + exp((Vs + 30.0) / (-5.0))) - mnS) / (4.0 * ms) : 1
        dhnS/dt = (1.0 / (1.0 + exp((Vs + 45.0) / 5.0)) - hnS) / (40.0 * ms) : 1

        # CaN gating (dendrite)
        dmnD/dt = (1.0 / (1.0 + exp((Vd + 30.0) / (-5.0))) - mnD) / (4.0 * ms) : 1
        dhnD/dt = (1.0 / (1.0 + exp((Vd + 45.0) / 5.0)) - hnD) / (40.0 * ms) : 1

        # CaL gating (dendrite)
        dml_d/dt = (1.0 / (1.0 + exp((Vd + 40.0) / (-7.0))) - ml_d) / (60.0 * ms) : 1

        # --- Calcium dynamics ---
        dCa_s/dt = {p["soma_f_Caconc"]} * (clip(-{p["soma_alpha_Caconc"]} * (I_CaN_s - I_CaN_s_rest), 0, inf) - {p["soma_kCa_Caconc"]} * (Ca_s - 1e-5)) / ms : 1
        dCa_d/dt = {p["dend_f_Caconc"]} * (clip(-{p["dend_alpha_Caconc"]} * ((I_CaN_d + I_CaL_d) - I_Ca_d_rest), 0, inf) - {p["dend_kCa_Caconc"]} * (Ca_d - 1e-5)) / ms : 1

        I_CaN_s_rest : 1
        I_Ca_d_rest : 1

        # --- Reversal potentials ---
        E_Na : 1
        E_K : 1

        # --- Extracellular stimulus as current density (mA/cm2) ---
        # V_ext enters through passive conductance: I = g_pas * V_ext
        # (linearized approximation of NEURON extracellular mechanism)
        V_ext = stim(t, i + {offset}) / mV : 1
        I_stim_s = {p["soma_g_pas"]} * V_ext : 1
        I_stim_d = {p["dend_g_pas"]} * V_ext : 1

        # --- Synaptic conductances (dual-exponential, in uS) ---
        dA_ampa/dt = -A_ampa / (tau_rise_ampa * ms) : 1
        dB_ampa/dt = -B_ampa / (tau_decay_ampa * ms) : 1
        dA_nmda/dt = -A_nmda / (tau_rise_nmda * ms) : 1
        dB_nmda/dt = -B_nmda / (tau_decay_nmda * ms) : 1
        dA_gaba_a/dt = -A_gaba_a / (tau_rise_gaba_a * ms) : 1
        dB_gaba_a/dt = -B_gaba_a / (tau_decay_gaba_a * ms) : 1
        dA_gaba_b/dt = -A_gaba_b / (tau_rise_gaba_b * ms) : 1
        dB_gaba_b/dt = -B_gaba_b / (tau_decay_gaba_b * ms) : 1

        g_ampa = B_ampa - A_ampa : 1
        g_nmda = B_nmda - A_nmda : 1
        g_gaba_a = B_gaba_a - A_gaba_a : 1
        g_gaba_b = B_gaba_b - A_gaba_b : 1

        # NMDA Mg2+ block (Jahr & Stevens)
        mgblock = 1.0 / (1.0 + exp(nmda_gamma * -(Vs + nmda_vshift)) * (nmda_mg / nmda_Kd)) : 1
        nmda_mg : 1
        nmda_Kd : 1
        nmda_gamma : 1
        nmda_vshift : 1

        # Synaptic time constants (set per population during init)
        tau_rise_ampa : 1
        tau_decay_ampa : 1
        tau_rise_nmda : 1
        tau_decay_nmda : 1
        tau_rise_gaba_a : 1
        tau_decay_gaba_a : 1
        tau_rise_gaba_b : 1
        tau_decay_gaba_b : 1

        # Synaptic reversal potentials
        e_ampa : 1
        e_nmda : 1
        e_gaba_a : 1
        e_gaba_b : 1

        # Synaptic current (mA/cm2, sign: negative = inward = depolarizing for exc)
        I_syn_ampa = g_ampa * (Vs - e_ampa) * {syn_factor} : 1
        I_syn_nmda = g_nmda * mgblock * (Vs - e_nmda) * {syn_factor} : 1
        I_syn_gaba_a = g_gaba_a * (Vs - e_gaba_a) * {syn_factor} : 1
        I_syn_gaba_b = g_gaba_b * (Vs - e_gaba_b) * {syn_factor} : 1
        I_syn = -(I_syn_ampa + I_syn_nmda + I_syn_gaba_a + I_syn_gaba_b) : 1

        # --- STDP learning signal (excitatory rule, shared across connections) ---
        exc_ltd = int(Vs > theta_ltd_exc) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltd_exc) * log(slope_exc), -500, 500)))) : 1
        exc_ltp = int(Vs > theta_ltp_exc) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltp_exc) * log(slope_exc), -500, 500)))) : 1
        sig_sat_exc = 2.0 / (1.0 + exp(clip(-(-A_ltd_exc * exc_ltd + A_ltp_exc * 2.0 * exc_ltp) / (learning_tau_exc + 1e-20) * log(slope_exc), -500, 500))) - 1.0 : 1
        dlearning_w_exc/dt = -learning_w_exc / (4.0 * ms) + plasticity_on * sig_sat_exc / (125.0 * ms) : 1
        dlearn_int_exc/dt = learning_w_exc / ms : 1

        # --- STDP learning signal (inhibitory rule, shared across connections) ---
        inh_ltd = int(Vs < theta_ltd_inh) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltd_inh) * log(slope_inh), -500, 500)))) : 1
        inh_ltp = int(Vs < theta_ltp_inh) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltp_inh) * log(slope_inh), -500, 500)))) : 1
        sig_sat_inh = 2.0 / (1.0 + exp(clip(-(-A_ltd_inh * inh_ltd + A_ltp_inh * 2.0 * inh_ltp) / (learning_tau_inh + 1e-20) * log(slope_inh), -500, 500))) - 1.0 : 1
        dlearning_w_inh/dt = -learning_w_inh / ms + plasticity_on * sig_sat_inh / (125.0 * ms) : 1
        dlearn_int_inh/dt = learning_w_inh / ms : 1

        # STDP parameters (set by enable_plasticity)
        plasticity_on : 1
        theta_ltp_exc : 1
        theta_ltd_exc : 1
        half_ltp_exc : 1
        half_ltd_exc : 1
        slope_exc : 1
        A_ltp_exc : 1
        A_ltd_exc : 1
        learning_tau_exc : 1
        theta_ltp_inh : 1
        theta_ltd_inh : 1
        half_ltp_inh : 1
        half_ltd_inh : 1
        slope_inh : 1
        A_ltp_inh : 1
        A_ltd_inh : 1
        learning_tau_inh : 1

        # --- Noise as current density (mA/cm2) ---
        I_noise_s = (g_noise_e * (Vs - 0) + g_noise_i * (Vs - (-75))) * (-1e-6 / {area_soma * 1e-8}) : 1

        # --- Noise conductances (updated externally via run_regularly) ---
        g_noise_e : 1
        g_noise_i : 1
        g_e0 : 1
        g_i0 : 1
        tau_e : 1
        tau_i : 1
        amp_e : 1
        amp_i : 1

        # --- External current for optogenetic injection ---
        I : amp
        I_ext = I/amp * {1000.0 / (area_soma * 1e-8)} : 1
        noise_amplitude : 1

        # --- Total membrane current per compartment (mA/cm2, positive outward) ---
        I_memb_s = I_Na + I_K + I_KCa_s + I_CaN_s + I_leak_s + I_coup_s : 1
        I_memb_d = I_KCa_d + I_CaN_d + I_CaL_d + I_leak_d + I_coup_d : 1

        # v alias for voltage monitoring (Vs is dimensionless in mV, convert to volt)
        v = Vs * mV : volt
        """

    def _prn_equations(self, offset, params):
        p = params
        pp = p["pp"]
        Ltotal = p["Ltotal"]
        gc = p["gc"]
        cm = p["global_cm"]
        cm_ratio = p["cm_ratio"]
        diam = p["global_diam"]
        e_pas = p["e_pas"]

        L_soma = pp * Ltotal
        # L_dend = (1 - pp) * Ltotal
        area_soma = math.pi * diam * L_soma
        # area_dend = math.pi * diam * L_dend

        celsius = 6.3
        fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
        cao = 2.0

        gc_factor = gc / (pp * (1 - pp)) * 0.001  # S/cm2

        # Temperature correction for Q10=3
        Q10 = 3.0
        tcorr = Q10 ** ((celsius - 36.0) / 10.0)

        ic_constant = p.get("ic_constant", 0.0)
        ic_constant_d = p.get("ic_constant_d", 0.0)

        # Unit conversion: synaptic conductance in uS, voltage in mV
        syn_factor = 1e-6 / (area_soma * 1e-8)

        return f"""
        # Soma voltage
        dVs/dt = (1000.0/{cm * cm_ratio}) * (-I_Na - I_K - I_leak_s - I_coup_s + I_noise_s + I_stim_s + I_ext + I_syn + {ic_constant}) / ms : 1
        # Dendrite voltage
        dVd/dt = (1000.0/{cm}) * (-I_Ca - I_KCa - I_leak_d - I_coup_d + I_stim_d + {ic_constant_d}) / ms : 1

        # --- Soma currents (mA/cm2) ---
        I_Na = {p["soma_gmax_Na"]} * m_inf_pr**2 * h_pr * (Vs - E_Na) : 1
        I_K = {p["soma_gmax_K"]} * n_pr * (Vs - E_K) : 1
        I_leak_s = {p["soma_g_pas"]} * (Vs - ({e_pas})) : 1

        # --- Dendrite currents (mA/cm2) ---
        I_Ca = {p["dend_gmax_Ca"]} * s_pr**2 * r_pr * ghk_d : 1
        I_KCa = {p["dend_gmax_KCa"]} * c_pr * chi_d * (Vd - E_K) : 1
        I_leak_d = {p["dend_g_pas"]} * (Vd - ({e_pas})) : 1

        # --- Coupling (mA/cm2) ---
        I_coup_s = {gc_factor} * (Vs - Vd) : 1
        I_coup_d = {gc_factor} * (Vd - Vs) : 1

        # --- GHK for Ca ---
        ghk_d = -({fN}) * (1.0 - (Ca_d / {cao}) * exp(Vd / {fN})) * efun_d : 1
        efun_d = int(abs(Vd / {fN}) < 1e-4) * (1.0 - Vd / {fN} / 2.0) + int(abs(Vd / {fN}) >= 1e-4) * ((Vd / {fN}) / (exp(Vd / {fN}) - 1.0 + 1e-20)) : 1

        # --- Na gating (HH alpha/beta with Q10) ---
        m_inf_pr = am_pr / (am_pr + bm_pr) : 1
        am_pr = {tcorr} * 0.32 * linoid_am : 1
        bm_pr = {tcorr} * 0.28 * linoid_bm : 1
        linoid_am = int(abs((-46.9 - Vs) / 4.0) < 1e-6) * 4.0 * (1.0 - (-46.9 - Vs) / 4.0 / 2.0) + int(abs((-46.9 - Vs) / 4.0) >= 1e-6) * ((-46.9 - Vs) / (exp((-46.9 - Vs) / 4.0) - 1.0 + 1e-20)) : 1
        linoid_bm = int(abs((Vs + 19.9) / 5.0) < 1e-6) * 5.0 * (1.0 - (Vs + 19.9) / 5.0 / 2.0) + int(abs((Vs + 19.9) / 5.0) >= 1e-6) * ((Vs + 19.9) / (exp((Vs + 19.9) / 5.0) - 1.0 + 1e-20)) : 1

        dh_pr/dt = ({tcorr} * 0.128 * exp((-43.0 - Vs) / 18.0) - h_pr * ({tcorr} * 0.128 * exp((-43.0 - Vs) / 18.0) + {tcorr} * 4.0 / (1.0 + exp((-20.0 - Vs) / 5.0)))) / ms : 1

        # --- K gating ---
        dn_pr/dt = (n_inf_pr - n_pr) / tau_n_pr / ms : 1
        an_pr = {tcorr} * 0.016 * linoid_an : 1
        bn_pr = {tcorr} * 0.25 * exp(-1.0 - 0.025 * Vs) : 1
        linoid_an = int(abs((-24.9 - Vs) / 5.0) < 1e-6) * 5.0 * (1.0 - (-24.9 - Vs) / 5.0 / 2.0) + int(abs((-24.9 - Vs) / 5.0) >= 1e-6) * ((-24.9 - Vs) / (exp((-24.9 - Vs) / 5.0) - 1.0 + 1e-20)) : 1
        n_inf_pr = an_pr / (an_pr + bn_pr + 1e-20) : 1
        tau_n_pr = 1.0 / (an_pr + bn_pr + 1e-20) : 1

        # --- Ca channel gating (s^2 * r) ---
        ds_pr/dt = ({tcorr} * 5.0 / (1.0 + exp(0.1 * (5.0 - Vd))) - s_pr * ({tcorr} * 5.0 / (1.0 + exp(0.1 * (5.0 - Vd))) + {tcorr} * 0.2 * xs_pr / (1.0 - exp(-xs_pr) + 1e-20))) / ms : 1
        xs_pr = -0.2 * (Vd + 8.9) + 1e-10 : 1
        dr_pr/dt = ({tcorr} * 0.1673 * exp(-0.03035 * (Vd + 38.5)) - r_pr * ({tcorr} * 0.1673 * exp(-0.03035 * (Vd + 38.5)) + {tcorr} * 0.5 / (1.0 + exp(0.3 * (8.9 - Vd))))) / ms : 1

        # --- KCa gating (c * chi_d) ---
        dc_pr/dt = (c_inf_pr - c_pr) / tau_c_pr / ms : 1
        c_inf_pr = clip(1.0 / (1.0 + exp(-(10.1 + Vd) / 0.1016)), 1e-20, 1.0)**0.00925 : 1
        tau_c_pr = 3.627 * exp(0.03704 * Vd) / {tcorr} : 1
        chi_d = clip(1.073 * sin(0.003453 * Ca_d + 0.08095) + 0.08408 * sin(0.01634 * Ca_d - 2.34) + 0.01811 * sin(0.0348 * Ca_d - 0.9918), 0, inf) : 1

        # --- Ca dynamics ---
        dCa_d/dt = (clip(-{p["dend_d_Caconc"]} * 10.0 * (I_Ca - I_Ca_d_rest), 0, inf) - {p["dend_beta_Caconc"]} * (Ca_d - 1e-5)) / ms : 1
        I_Ca_d_rest : 1

        # --- Reversal potentials ---
        E_Na : 1
        E_K : 1

        # --- Extracellular stimulus as current density (mA/cm2) ---
        V_ext = stim(t, i + {offset}) / mV : 1
        I_stim_s = {p["soma_g_pas"]} * V_ext : 1
        I_stim_d = {p["dend_g_pas"]} * V_ext : 1

        # --- Synaptic conductances (dual-exponential, in uS) ---
        dA_ampa/dt = -A_ampa / (tau_rise_ampa * ms) : 1
        dB_ampa/dt = -B_ampa / (tau_decay_ampa * ms) : 1
        dA_nmda/dt = -A_nmda / (tau_rise_nmda * ms) : 1
        dB_nmda/dt = -B_nmda / (tau_decay_nmda * ms) : 1
        dA_gaba_a/dt = -A_gaba_a / (tau_rise_gaba_a * ms) : 1
        dB_gaba_a/dt = -B_gaba_a / (tau_decay_gaba_a * ms) : 1
        dA_gaba_b/dt = -A_gaba_b / (tau_rise_gaba_b * ms) : 1
        dB_gaba_b/dt = -B_gaba_b / (tau_decay_gaba_b * ms) : 1

        g_ampa = B_ampa - A_ampa : 1
        g_nmda = B_nmda - A_nmda : 1
        g_gaba_a = B_gaba_a - A_gaba_a : 1
        g_gaba_b = B_gaba_b - A_gaba_b : 1

        # NMDA Mg2+ block
        mgblock = 1.0 / (1.0 + exp(nmda_gamma * -(Vs + nmda_vshift)) * (nmda_mg / nmda_Kd)) : 1
        nmda_mg : 1
        nmda_Kd : 1
        nmda_gamma : 1
        nmda_vshift : 1

        tau_rise_ampa : 1
        tau_decay_ampa : 1
        tau_rise_nmda : 1
        tau_decay_nmda : 1
        tau_rise_gaba_a : 1
        tau_decay_gaba_a : 1
        tau_rise_gaba_b : 1
        tau_decay_gaba_b : 1

        e_ampa : 1
        e_nmda : 1
        e_gaba_a : 1
        e_gaba_b : 1

        I_syn_ampa = g_ampa * (Vs - e_ampa) * {syn_factor} : 1
        I_syn_nmda = g_nmda * mgblock * (Vs - e_nmda) * {syn_factor} : 1
        I_syn_gaba_a = g_gaba_a * (Vs - e_gaba_a) * {syn_factor} : 1
        I_syn_gaba_b = g_gaba_b * (Vs - e_gaba_b) * {syn_factor} : 1
        I_syn = -(I_syn_ampa + I_syn_nmda + I_syn_gaba_a + I_syn_gaba_b) : 1

        # --- STDP learning signal (excitatory rule) ---
        exc_ltd = int(Vs > theta_ltd_exc) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltd_exc) * log(slope_exc), -500, 500)))) : 1
        exc_ltp = int(Vs > theta_ltp_exc) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltp_exc) * log(slope_exc), -500, 500)))) : 1
        sig_sat_exc = 2.0 / (1.0 + exp(clip(-(-A_ltd_exc * exc_ltd + A_ltp_exc * 2.0 * exc_ltp) / (learning_tau_exc + 1e-20) * log(slope_exc), -500, 500))) - 1.0 : 1
        dlearning_w_exc/dt = -learning_w_exc / (4.0 * ms) + plasticity_on * sig_sat_exc / (125.0 * ms) : 1
        dlearn_int_exc/dt = learning_w_exc / ms : 1

        # --- STDP learning signal (inhibitory rule) ---
        inh_ltd = int(Vs < theta_ltd_inh) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltd_inh) * log(slope_inh), -500, 500)))) : 1
        inh_ltp = int(Vs < theta_ltp_inh) * (1.0 / (1.0 + exp(clip(-(Vs - half_ltp_inh) * log(slope_inh), -500, 500)))) : 1
        sig_sat_inh = 2.0 / (1.0 + exp(clip(-(-A_ltd_inh * inh_ltd + A_ltp_inh * 2.0 * inh_ltp) / (learning_tau_inh + 1e-20) * log(slope_inh), -500, 500))) - 1.0 : 1
        dlearning_w_inh/dt = -learning_w_inh / ms + plasticity_on * sig_sat_inh / (125.0 * ms) : 1
        dlearn_int_inh/dt = learning_w_inh / ms : 1

        plasticity_on : 1
        theta_ltp_exc : 1
        theta_ltd_exc : 1
        half_ltp_exc : 1
        half_ltd_exc : 1
        slope_exc : 1
        A_ltp_exc : 1
        A_ltd_exc : 1
        learning_tau_exc : 1
        theta_ltp_inh : 1
        theta_ltd_inh : 1
        half_ltp_inh : 1
        half_ltd_inh : 1
        slope_inh : 1
        A_ltp_inh : 1
        A_ltd_inh : 1
        learning_tau_inh : 1

        # --- Noise as current density (mA/cm2) ---
        I_noise_s = (g_noise_e * (Vs - 0) + g_noise_i * (Vs - (-75))) * (-1e-6 / {area_soma * 1e-8}) : 1

        # --- Noise conductances (updated externally via run_regularly) ---
        g_noise_e : 1
        g_noise_i : 1
        g_e0 : 1
        g_i0 : 1
        tau_e : 1
        tau_i : 1
        amp_e : 1
        amp_i : 1

        I : amp
        I_ext = I/amp * {1000.0 / (area_soma * 1e-8)} : 1
        noise_amplitude : 1

        # --- Total membrane current per compartment (mA/cm2, positive outward) ---
        I_memb_s = I_Na + I_K + I_leak_s + I_coup_s : 1
        I_memb_d = I_Ca + I_KCa + I_leak_d + I_coup_d : 1

        # v alias for voltage monitoring (Vs is dimensionless in mV, convert to volt)
        v = Vs * mV : volt
        """

    def brian2_population_group(self, population_name, n, offset, coordinates, prng):
        import brian2 as b2
        import math as _m

        if population_name == "EXC":
            p = self.params("BoothRinzelKiehn-MN")

            # Compute ic_constant dynamically from equilibrium condition
            v_rest = p["V_rest"]
            e_pas = p["e_pas"]
            celsius = 6.3
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                if abs(nu) < 1e-4:
                    ef = 1.0 - nu / 2.0
                else:
                    ef = nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            # Gating at rest
            m_inf = 1.0 / (1.0 + _m.exp(-(v_rest + 35.0) / 7.8))
            h_rest = 1.0 / (1.0 + _m.exp((v_rest + 55.0) / 7.0))
            n_rest = 1.0 / (1.0 + _m.exp(-(v_rest + 28.0) / 15.0))
            mnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            hnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))

            ghk_s = _ghk(v_rest, cai0, cao, fN)

            # Soma currents at rest
            I_Na = p["soma_gmax_Na"] * m_inf**3 * h_rest * (v_rest - 50.0)
            I_K = p["soma_gmax_K"] * n_rest**4 * (v_rest - (-77.0))
            I_KCa = p["soma_gmax_KCa"] * (cai0 / (cai0 + 0.0005)) * (v_rest - (-77.0))
            I_CaN = p["soma_gmax_CaN"] * mnS_rest**2 * hnS_rest * ghk_s
            I_leak = p["soma_g_pas"] * (v_rest - e_pas)
            # ic_constant balances soma at rest (coupling=0, input=0)
            p = dict(p)  # copy so we can override
            p["ic_constant"] = I_Na + I_K + I_KCa + I_CaN + I_leak

            # Dendrite currents at rest
            ghk_d_N = _ghk(v_rest, cai0, cao, fN)
            ghk_d_L = _ghk(v_rest, cai0, cao, fL)
            mnD_rest = mnS_rest
            hnD_rest = hnS_rest
            ml_rest = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            I_KCa_d = p["dend_gmax_KCa"] * (cai0 / (cai0 + 0.0005)) * (v_rest - (-77.0))
            I_CaN_d = p["dend_gmax_CaN"] * mnD_rest**2 * hnD_rest * ghk_d_N
            I_CaL_d = p["dend_gmax_CaL"] * ml_rest * ghk_d_L
            I_leak_d = p["dend_g_pas"] * (v_rest - e_pas)
            p["ic_constant_d"] = I_KCa_d + I_CaN_d + I_CaL_d + I_leak_d

            equations = self._brk_equations(offset, p)

            _use_gsl = os.environ.get("LIVN_USE_LIBGSL", "0") == "1"
            _method = "gsl_rkf45" if _use_gsl else "euler"
            _dt = 0.025 if _use_gsl else 0.005

            population = b2.NeuronGroup(
                n,
                equations,
                threshold="Vs > %f" % p["V_threshold"],
                reset="",  # no artificial reset for biophysical model
                refractory=2 * b2.ms,
                method=_method,
                name=population_name,
                dt=_dt * b2.ms,
            )

            # Initial conditions
            v_rest = p["V_rest"]
            population.Vs = v_rest
            population.Vd = v_rest

            population.h_s = 1.0 / (1.0 + _m.exp((v_rest + 55.0) / 7.0))
            population.n_s = 1.0 / (1.0 + _m.exp(-(v_rest + 28.0) / 15.0))
            population.mnS = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            population.hnS = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            population.mnD = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            population.hnD = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            population.ml_d = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            population.Ca_s = 1e-5
            population.Ca_d = 1e-5

            # Compute resting Ca currents for Ca_conc dynamics
            celsius = 6.3
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            fL = ((25.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                if abs(nu) < 1e-4:
                    ef = 1.0 - nu / 2.0
                else:
                    ef = nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            ghk_s = _ghk(v_rest, cai0, cao, fN)
            ghk_d_N = _ghk(v_rest, cai0, cao, fN)
            ghk_d_L = _ghk(v_rest, cai0, cao, fL)

            mnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 30.0) / (-5.0)))
            hnS_rest = 1.0 / (1.0 + _m.exp((v_rest + 45.0) / 5.0))
            I_CaN_s_rest = p["soma_gmax_CaN"] * mnS_rest**2 * hnS_rest * ghk_s
            population.I_CaN_s_rest = I_CaN_s_rest

            mnD_rest = mnS_rest
            hnD_rest = hnS_rest
            ml_rest = 1.0 / (1.0 + _m.exp((v_rest + 40.0) / (-7.0)))
            I_Ca_d_rest = (
                p["dend_gmax_CaN"] * mnD_rest**2 * hnD_rest * ghk_d_N
                + p["dend_gmax_CaL"] * ml_rest * ghk_d_L
            )
            population.I_Ca_d_rest = I_Ca_d_rest

            # Reversal potentials
            population.E_Na = 50.0
            population.E_K = -77.0

            _diam = p["global_diam"]
            _Ltot = p["Ltotal"]
            _pp = p["pp"]
            population.add_attribute("area_soma_cm2")
            population.add_attribute("area_dend_cm2")
            population.area_soma_cm2 = _m.pi * _diam * (_pp * _Ltot) * 1e-8
            population.area_dend_cm2 = _m.pi * _diam * ((1 - _pp) * _Ltot) * 1e-8

        else:
            # INH = Pinsky-Rinzel
            p = self.params("PinskyRinzel-PVBC")

            # Compute ic_constant from equilibrium condition
            v_rest = p["V_rest"]
            e_pas = p["e_pas"]
            celsius = 6.3
            tcorr = 3.0 ** ((celsius - 36.0) / 10.0)

            # Na gating at rest
            am = tcorr * 0.32 * _linoid(-46.9 - v_rest, 4.0)
            bm = tcorr * 0.28 * _linoid(v_rest + 19.9, 5.0)
            m_inf_pr = am / (am + bm)
            ah = tcorr * 0.128 * _m.exp((-43.0 - v_rest) / 18.0)
            bh = tcorr * 4.0 / (1.0 + _m.exp((-20.0 - v_rest) / 5.0))
            h_rest = ah / (ah + bh)

            # K gating at rest
            an = tcorr * 0.016 * _linoid(-24.9 - v_rest, 5.0)
            bn = tcorr * 0.25 * _m.exp(-1.0 - 0.025 * v_rest)
            n_rest = an / (an + bn)

            I_Na = p["soma_gmax_Na"] * m_inf_pr**2 * h_rest * (v_rest - 50.0)
            I_K = p["soma_gmax_K"] * n_rest * (v_rest - (-77.0))
            I_leak = p["soma_g_pas"] * (v_rest - e_pas)
            p = dict(p)
            p["ic_constant"] = I_Na + I_K + I_leak

            # Dendrite currents at rest
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk_pr(v, ci, co, f):
                nu = v / f
                if abs(nu) < 1e-4:
                    ef = 1.0 - nu / 2.0
                else:
                    ef = nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            a_s = tcorr * 5.0 / (1.0 + _m.exp(0.1 * (5.0 - v_rest)))
            xs = -0.2 * (v_rest + 8.9)
            if abs(xs) < 1e-10:
                xs = 1e-10
            b_s = tcorr * 0.2 * xs / (1.0 - _m.exp(-xs))
            s_rest = a_s / (a_s + b_s)

            ar = tcorr * 0.1673 * _m.exp(-0.03035 * (v_rest + 38.5))
            br = tcorr * 0.5 / (1.0 + _m.exp(0.3 * (8.9 - v_rest)))
            r_rest = ar / (ar + br)

            x_expit = (10.1 + v_rest) / 0.1016
            if x_expit < 0:
                expit_val = _m.exp(x_expit) / (1.0 + _m.exp(x_expit))
            else:
                expit_val = 1.0 / (1.0 + _m.exp(-x_expit))
            c_rest = max(expit_val, 1e-20) ** 0.00925

            ghk_rest = _ghk_pr(v_rest, cai0, cao, fN)
            chi_rest = (
                1.073 * _m.sin(0.003453 * cai0 + 0.08095)
                + 0.08408 * _m.sin(0.01634 * cai0 - 2.34)
                + 0.01811 * _m.sin(0.0348 * cai0 - 0.9918)
            )

            I_Ca_d = p["dend_gmax_Ca"] * s_rest**2 * r_rest * ghk_rest
            I_KCa_d = p["dend_gmax_KCa"] * c_rest * chi_rest * (v_rest - (-77.0))
            I_leak_d = p["dend_g_pas"] * (v_rest - e_pas)
            p["ic_constant_d"] = I_Ca_d + I_KCa_d + I_leak_d

            equations = self._prn_equations(offset, p)

            _use_gsl = os.environ.get("LIVN_USE_LIBGSL", "0") == "1"
            _method = "gsl_rkf45" if _use_gsl else "euler"
            _dt = 0.025 if _use_gsl else 0.005

            population = b2.NeuronGroup(
                n,
                equations,
                threshold="Vs > %f" % p["V_threshold"],
                reset="",
                refractory=2 * b2.ms,
                method=_method,
                name=population_name,
                dt=_dt * b2.ms,
            )

            v_rest = p["V_rest"]
            population.Vs = v_rest
            population.Vd = v_rest

            # PR gating initial conditions
            celsius = 6.3
            tcorr = 3.0 ** ((celsius - 36.0) / 10.0)

            # Na: m_inf, h
            am = tcorr * 0.32 * _linoid(-46.9 - v_rest, 4.0)
            bm = tcorr * 0.28 * _linoid(v_rest + 19.9, 5.0)
            population.h_pr = (tcorr * 0.128 * _m.exp((-43.0 - v_rest) / 18.0)) / (
                tcorr * 0.128 * _m.exp((-43.0 - v_rest) / 18.0)
                + tcorr * 4.0 / (1.0 + _m.exp((-20.0 - v_rest) / 5.0))
            )

            # K: n
            an = tcorr * 0.016 * _linoid(-24.9 - v_rest, 5.0)
            bn = tcorr * 0.25 * _m.exp(-1.0 - 0.025 * v_rest)
            population.n_pr = an / (an + bn)

            # Ca: s, r
            a_s = tcorr * 5.0 / (1.0 + _m.exp(0.1 * (5.0 - v_rest)))
            xs = -0.2 * (v_rest + 8.9)
            if abs(xs) < 1e-10:
                xs = 1e-10
            b_s = tcorr * 0.2 * xs / (1.0 - _m.exp(-xs))
            population.s_pr = a_s / (a_s + b_s)

            ar = tcorr * 0.1673 * _m.exp(-0.03035 * (v_rest + 38.5))
            br = tcorr * 0.5 / (1.0 + _m.exp(0.3 * (8.9 - v_rest)))
            population.r_pr = ar / (ar + br)

            # KCa: c
            x_expit = (10.1 + v_rest) / 0.1016
            if x_expit < 0:
                expit_val = _m.exp(x_expit) / (1.0 + _m.exp(x_expit))
            else:
                expit_val = 1.0 / (1.0 + _m.exp(-x_expit))
            population.c_pr = max(expit_val, 1e-20) ** 0.00925

            population.Ca_d = 1e-5

            # Resting Ca current
            fN = ((36.0 / 293.15) * (celsius + 273.15)) / 2.0
            cao = 2.0
            cai0 = 1e-5

            def _ghk(v, ci, co, f):
                nu = v / f
                if abs(nu) < 1e-4:
                    ef = 1.0 - nu / 2.0
                else:
                    ef = nu / (_m.exp(nu) - 1.0)
                return -f * (1.0 - (ci / co) * _m.exp(nu)) * ef

            s_rest = (
                float(population.s_pr[0])
                if hasattr(population.s_pr, "__getitem__")
                else population.s_pr
            )
            r_rest = (
                float(population.r_pr[0])
                if hasattr(population.r_pr, "__getitem__")
                else population.r_pr
            )
            if not isinstance(s_rest, (int, float)):
                s_rest = a_s / (a_s + b_s)
                r_rest = ar / (ar + br)
            population.I_Ca_d_rest = (
                p["dend_gmax_Ca"] * s_rest**2 * r_rest * _ghk(v_rest, cai0, cao, fN)
            )

            population.E_Na = 50.0
            population.E_K = -77.0

            _diam = p["global_diam"]
            _Ltot = p["Ltotal"]
            _pp = p["pp"]
            population.add_attribute("area_soma_cm2")
            population.add_attribute("area_dend_cm2")
            population.area_soma_cm2 = _m.pi * _diam * (_pp * _Ltot) * 1e-8
            population.area_dend_cm2 = _m.pi * _diam * ((1 - _pp) * _Ltot) * 1e-8

        # Common: noise init
        population.g_noise_e = 0.0
        population.g_noise_i = 0.0
        population.g_e0 = 0.0
        population.g_i0 = 0.0
        population.tau_e = 2.728
        population.tau_i = 10.49
        population.amp_e = 0.0
        population.amp_i = 0.0
        population.I = 0 * b2.amp
        population.noise_amplitude = 0.0

        # Synaptic conductance init
        population.A_ampa = 0.0
        population.B_ampa = 0.0
        population.A_nmda = 0.0
        population.B_nmda = 0.0
        population.A_gaba_a = 0.0
        population.B_gaba_a = 0.0
        population.A_gaba_b = 0.0
        population.B_gaba_b = 0.0

        # Default synaptic time constants (overridden per connection)
        population.tau_rise_ampa = 0.5
        population.tau_decay_ampa = 3.0
        population.tau_rise_nmda = 10.0
        population.tau_decay_nmda = 35.0
        population.tau_rise_gaba_a = 0.3
        population.tau_decay_gaba_a = 6.0
        population.tau_rise_gaba_b = 1.0
        population.tau_decay_gaba_b = 5.0

        # Default reversal potentials
        population.e_ampa = 0.0
        population.e_nmda = 0.0
        population.e_gaba_a = -60.0
        population.e_gaba_b = -90.0

        # NMDA Mg block defaults
        population.nmda_mg = 1.0
        population.nmda_Kd = 3.57
        population.nmda_gamma = 0.062
        population.nmda_vshift = 0.0

        # STDP init
        population.plasticity_on = 0.0
        population.learning_w_exc = 0.0
        population.learn_int_exc = 0.0
        population.learning_w_inh = 0.0
        population.learn_int_inh = 0.0

        # STDP parameters (defaults, overridden by enable_plasticity)
        population.theta_ltp_exc = -45.0
        population.theta_ltd_exc = -60.0
        population.half_ltp_exc = -40.0
        population.half_ltd_exc = -55.0
        population.slope_exc = 1.3
        population.A_ltp_exc = 1.0
        population.A_ltd_exc = 1.0
        population.learning_tau_exc = 20.0
        population.theta_ltp_inh = -77.0
        population.theta_ltd_inh = -70.0
        population.half_ltp_inh = -80.0
        population.half_ltd_inh = -73.0
        population.slope_inh = 1.2
        population.A_ltp_inh = 1.0
        population.A_ltd_inh = 1.0
        population.learning_tau_inh = 20.0

        return population

    def brian2_connection_synapse(self, pre_group, post_group):
        """Legacy single-synapse constructor (unused with conductance-based model)."""
        import brian2 as b2

        synapse = b2.Synapses(
            pre_group,
            post_group,
            """
            w : 1
            multiplier : 1
            distance : 1
            prefix : 1
            """,
            on_pre="I += prefix * w * multiplier * pA",
            dt=0.025 * b2.ms,
        )

        return synapse

    def brian2_mechanism_synapse(
        self, pre_group, post_group, mechanism_name, mechanism_params, synapse_type
    ):
        """Create a conductance-based synapse for a specific mechanism

        Parameters
        ----------
        pre_group : brian2.NeuronGroup
        post_group : brian2.NeuronGroup
        mechanism_name : str
            One of "AMPA", "NMDA", "GABA_A", "GABA_B"
        mechanism_params : dict
            Must contain: e, g_unit, tau_rise, tau_decay, weight
        synapse_type : str
            "excitatory" or "inhibitory"
        """
        import math

        import brian2 as b2

        tau_rise = mechanism_params["tau_rise"]
        tau_decay = mechanism_params["tau_decay"]

        # Compute normalization factor for dual-exponential
        if tau_decay > tau_rise and tau_rise > 0:
            tp = (
                (tau_rise * tau_decay)
                / (tau_decay - tau_rise)
                * math.log(tau_decay / tau_rise)
            )
            factor = 1.0 / (-math.exp(-tp / tau_rise) + math.exp(-tp / tau_decay))
        else:
            factor = 1.0

        g_unit = mechanism_params["g_unit"]
        mech_lower = mechanism_name.lower()

        # STDP-capable mechanisms: AMPA, NMDA, GABA_A
        has_stdp = mechanism_name in ("AMPA", "NMDA", "GABA_A")

        # Choose the learning integral variable based on synapse type
        if synapse_type == "excitatory":
            learn_int_var = "learn_int_exc"
        else:
            learn_int_var = "learn_int_inh"

        model_eqs = """
            w : 1
            multiplier : 1
            distance : 1
        """

        if has_stdp:
            model_eqs += """
            w_plastic : 1
            last_int : 1
            w_min : 1
            w_max : 1
        """

        if has_stdp:
            # STDP weight update happens before conductance delivery
            on_pre_code = f"""
            delta = {learn_int_var}_post - last_int
            last_int = {learn_int_var}_post
            w_plastic = clip(w_plastic + plasticity_on_post * delta * w_plastic, w_min, w_max)
            A_{mech_lower}_post += w * w_plastic * multiplier * {g_unit} * {factor}
            B_{mech_lower}_post += w * w_plastic * multiplier * {g_unit} * {factor}
            """
        else:
            on_pre_code = f"""
            A_{mech_lower}_post += w * multiplier * {g_unit} * {factor}
            B_{mech_lower}_post += w * multiplier * {g_unit} * {factor}
            """

        synapse = b2.Synapses(
            pre_group,
            post_group,
            model_eqs,
            on_pre=on_pre_code,
            dt=0.025 * b2.ms,
        )

        synapse._mechanism_name = mechanism_name
        synapse._mechanism_params = mechanism_params
        synapse._has_stdp = has_stdp
        synapse._factor = factor

        return synapse

    def brian2_noise_op(self, population_group, prng):
        """Ornstein-Uhlenbeck noise via run_regularly (Euler-Maruyama)

        The OU process is separated from the main ODE system so that the
        deterministic equations can use the GSL adaptive solver.  The noise
        conductances are updated every dt with an explicit Euler-Maruyama step.
        """
        # Euler-Maruyama update for OU process at each timestep
        # dg = -(g - g0)/tau * dt + amp * sqrt(2*dt/tau) * N(0,1)
        noise_update = population_group.run_regularly(
            """
            g_noise_e += -(g_noise_e - g_e0) / tau_e * (dt/ms) + amp_e * sqrt(2.0 * (dt/ms) / tau_e) * randn()
            g_noise_i += -(g_noise_i - g_i0) / tau_i * (dt/ms) + amp_i * sqrt(2.0 * (dt/ms) / tau_i) * randn()
            """,
            dt=population_group.clock.dt,
        )
        return noise_update

    def brian2_noise_configure(
        self,
        population_group,
        std_e=0.003,
        std_i=0.0066,
        g_e0=0.0121,
        g_i0=0.0573,
        tau_e=2.728,
        tau_i=10.49,
        **kwargs,
    ):
        """Configure Ornstein-Uhlenbeck noise for the two-compartment model.

        The NEURON Gfluct3 applies conductance noise to each compartment:
        - soma gets inhibitory noise only (g_i)
        - dendrite gets excitatory noise only (g_e)

        In brian2 we model this at the soma level for simplicity since the
        noise coupling enters as I_input_s which includes both g_noise_e and g_noise_i.
        The soma receives inhibitory fluctuations and the dendrite receives excitatory.
        Since our equations wire g_noise_e and g_noise_i into the soma input,
        we set: soma gets g_i (inhibitory), and we include excitatory through g_e.
        """
        population_group.g_e0 = g_e0
        population_group.g_i0 = g_i0
        population_group.tau_e = tau_e
        population_group.tau_i = tau_i
        population_group.amp_e = std_e
        population_group.amp_i = std_i

        population_group.g_noise_e = g_e0
        population_group.g_noise_i = g_i0

    def brian2_default_noise(self, system: str):
        return self.neuron_default_noise(system)

    def brian2_default_weights(self, system: str):
        neuron_weights = self.neuron_default_weights(system)
        brian2_weights = {}
        for k, v in neuron_weights.items():
            # Parse: PRE_POST-section-mechanism-weight
            parts = k.split("-")
            pre_post = parts[0]  # e.g., "EXC_EXC"
            if pre_post not in brian2_weights:
                brian2_weights[pre_post] = 0.0
            brian2_weights[pre_post] += v

        return brian2_weights


def _linoid(x, y):
    """Safe linoid function matching NEURON's linoid"""
    import math

    if abs(x / y) < 1e-6:
        return y * (1.0 - x / y / 2.0)
    else:
        return x / (math.exp(x / y) - 1.0)
