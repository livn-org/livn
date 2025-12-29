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
            "AMPA": "LinExp2Syn",
            "NMDA": "LinExp2SynNMDA",
            "GABA_A": "LinExp2Syn",
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
        mechanism.on = 1 if (std_e > 0 or std_i > 0) else 0
        mechanism.std_e = std_e
        mechanism.std_i = std_i

        mechanism.g_e0 = g_e0
        mechanism.g_i0 = g_i0

        mechanism.tau_e = tau_e
        mechanism.tau_i = tau_i

        mechanism.E_e = E_e
        mechanism.E_i = E_i

    def neuron_default_noise(self, system: str, key: int = 0):
        return {
            "S1": [{}],
            "S2": [{}],
            "S3": [{}],
            "S4": [{}],
        }[system][key]

    def neuron_default_weights(self, system: str):
        return {
            "S1": {},
            "S2": {},
            "S3": {},
            "S4": {},
        }[system]

    # diffrax

    def diffrax_module(self, env, key):
        from livn.models.rcsd.diffrax.culture import MotoneuronCulture

        return MotoneuronCulture(
            num_neurons=env.system.num_neurons,
            params=self.params("BoothRinzelKiehn-MN"),
            key=key,
        )
