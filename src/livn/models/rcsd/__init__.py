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
    def stimulus_coordinates(
        self,
        neuron_coordinates: types.Float[types.Array, "n_coords ixyz=4"],
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
    ) -> types.Float[types.Array, "n_stim_coords ixyz=4"]:
        return self.stimulus_coordinates(neuron_coordinates)

    # neuron

    def params(self, name: str):
        return {
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
                "ic_constant": -0.015656504661833687,
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
            num_neurons=env.system.num_neurons,
            params=self.params("BoothRinzelKiehn-MN"),
            key=key,
        )
