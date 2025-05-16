import os

from livn.types import Model


class ReducedCalciumSomaDendrite(Model):
    def neuron_template_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "templates")

    def neuron_mechanisms_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "mechanisms")

    def neuron_celltypes(self, celltypes):
        # optimized MOTONEURON
        param_dict = {
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
        }

        param_dict["ic_constant"] = -0.015656504661833687
        param_dict["V_rest"] = -57.4
        param_dict["V_threshold"] = -37.0

        celltypes["EXC"]["template class"] = "livn.models.rcsd.neuron.templates.BRK.BRK"
        celltypes["EXC"]["mechanism"] = {"BoothRinzelKiehn": param_dict}

        # optimized PVBC
        param_dict = {
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
        }
        param_dict["ic_constant"] = 0.013448839558146165
        param_dict["V_rest"] = -60.0
        param_dict["V_threshold"] = -37.0

        celltypes["INH"]["template class"] = "livn.models.rcsd.neuron.templates.PRN.PRN"
        celltypes["INH"]["mechanism"] = {"PinskyRinzel": param_dict}

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

    def neuron_noise_configure(self, population, mechanism, state, level):
        mechanism.on = 1 if level > 0 else 0

        if population == "EXC":
            mechanism.std_e = 0.0030 * level
            mechanism.std_i = 0
        else:
            mechanism.std_e = 0
            mechanism.std_i = 0.0066 * level

    def neuron_default_noise(self, system: str, key: int = 0):
        return {
            "S1": [
                {"exc": 4.693429946899414, "inh": 9.591781616210938},
                {"exc": 4.875756740570068, "inh": 9.722622871398926},
                {"exc": 2.6111457347869873, "inh": 9.635272979736328},
            ],
            "S2": [
                {"exc": 4.964247703552246, "inh": 9.183961868286133},
                {"exc": 9.897594451904297, "inh": 7.473959445953369},
                {"exc": 9.925007820129395, "inh": 7.525247573852539},
            ],
            "S3": [
                {"exc": 4.052318096160889, "inh": 7.568785667419434},
                {"exc": 4.675532341003418, "inh": 9.475052833557129},
                {"exc": 4.528378486633301, "inh": 0.1814909428358078},
            ],
            "S4": [
                {"exc": 6.436318874359131, "inh": 8.729865074157715},
                {"exc": 3.819654703140259, "inh": 9.882781028747559},
                {"exc": 6.6395416259765625, "inh": 8.888071060180664},
            ],
        }[system][key]

    def neuron_default_weights(self, system: str):
        return {
            "S1": {
                "EXC_EXC-apical-AMPA-weight": 1.5656249523162842,
                "EXC_EXC-apical-NMDA-weight": 3.4781250953674316,
                "EXC_EXC-basal-AMPA-weight": 2.409374952316284,
                "EXC_EXC-basal-NMDA-weight": 7.496874809265137,
                "EXC_EXC-soma-AMPA-weight": 4.921875,
                "EXC_EXC-soma-NMDA-weight": 3.965625047683716,
                "EXC_INH-ais-GABA_A-weight": 0.3031249940395355,
                "EXC_INH-apical-GABA_A-weight": 0.05312500149011612,
                "EXC_INH-basal-GABA_A-weight": 4.271874904632568,
                "EXC_INH-soma-GABA_A-weight": 6.040625095367432,
                "INH_EXC-apical-AMPA-weight": 4.621874809265137,
                "INH_EXC-basal-AMPA-weight": 2.796875,
                "INH_EXC-soma-AMPA-weight": 7.765625,
                "INH_INH-apical-GABA_A-weight": 1.553125023841858,
                "INH_INH-basal-GABA_A-weight": 0.484375,
                "INH_INH-soma-GABA_A-weight": 4.034375190734863,
            },
            "S2": {
                "EXC_EXC-apical-AMPA-weight": 3.2093749046325684,
                "EXC_EXC-apical-NMDA-weight": 0.078125,
                "EXC_EXC-basal-AMPA-weight": 5.440625190734863,
                "EXC_EXC-basal-NMDA-weight": 0.11562500149011612,
                "EXC_EXC-soma-AMPA-weight": 8.740625381469727,
                "EXC_EXC-soma-NMDA-weight": 5.896874904632568,
                "EXC_INH-ais-GABA_A-weight": 1.5593750476837158,
                "EXC_INH-apical-GABA_A-weight": 3.2906250953674316,
                "EXC_INH-basal-GABA_A-weight": 8.071874618530273,
                "EXC_INH-soma-GABA_A-weight": 1.009374976158142,
                "INH_EXC-apical-AMPA-weight": 0.015625,
                "INH_EXC-basal-AMPA-weight": 7.278124809265137,
                "INH_EXC-soma-AMPA-weight": 0.515625,
                "INH_INH-apical-GABA_A-weight": 4.578125,
                "INH_INH-basal-GABA_A-weight": 7.221875190734863,
                "INH_INH-soma-GABA_A-weight": 5.565625190734863,
            },
            "S3": {
                "EXC_EXC-apical-AMPA-weight": 3.7592809200286865,
                "EXC_EXC-apical-NMDA-weight": 0.0,
                "EXC_EXC-basal-AMPA-weight": 2.0468108654022217,
                "EXC_EXC-basal-NMDA-weight": 9.934853553771973,
                "EXC_EXC-soma-AMPA-weight": 3.1570630073547363,
                "EXC_EXC-soma-NMDA-weight": 0.8268722295761108,
                "EXC_INH-ais-GABA_A-weight": 0.0,
                "EXC_INH-apical-GABA_A-weight": 0.0,
                "EXC_INH-basal-GABA_A-weight": 0.3906591236591339,
                "EXC_INH-soma-GABA_A-weight": 0.6477571725845337,
                "INH_EXC-apical-AMPA-weight": 7.295553684234619,
                "INH_EXC-basal-AMPA-weight": 4.102510929107666,
                "INH_EXC-soma-AMPA-weight": 7.2068257331848145,
                "INH_INH-apical-GABA_A-weight": 7.237850666046143,
                "INH_INH-basal-GABA_A-weight": 0.0,
                "INH_INH-soma-GABA_A-weight": 5.702517509460449,
            },
            "S4": {
                "EXC_EXC-apical-AMPA-weight": 5.984375,
                "EXC_EXC-apical-NMDA-weight": 0.34062498807907104,
                "EXC_EXC-basal-AMPA-weight": 4.609375,
                "EXC_EXC-basal-NMDA-weight": 0.21562500298023224,
                "EXC_EXC-soma-AMPA-weight": 2.953125,
                "EXC_EXC-soma-NMDA-weight": 5.671875,
                "EXC_INH-ais-GABA_A-weight": 3.6031250953674316,
                "EXC_INH-apical-GABA_A-weight": 4.434374809265137,
                "EXC_INH-basal-GABA_A-weight": 1.178125023841858,
                "EXC_INH-soma-GABA_A-weight": 6.896874904632568,
                "INH_EXC-apical-AMPA-weight": 9.303125381469727,
                "INH_EXC-basal-AMPA-weight": 2.0843749046325684,
                "INH_EXC-soma-AMPA-weight": 2.6343750953674316,
                "INH_INH-apical-GABA_A-weight": 4.703125,
                "INH_INH-basal-GABA_A-weight": 0.34687501192092896,
                "INH_INH-soma-GABA_A-weight": 5.328125,
            },
        }[system]


class ReducedCalciumSomaDendriteIfluct(ReducedCalciumSomaDendrite):
    def neuron_noise_mechanism(self, section):
        from neuron import h

        noiseRandObj = h.Random()
        noiseRandObj.MCellRan4(1, 42)
        noiseRandObj.normal(0, 1)

        fluct = h.Ifluct1(section)

        fluct.m = 0  # [nA]
        fluct.s = 0  # [nA]
        fluct.tau = 0  # [ms]

        fluct.setRandObj(noiseRandObj)

        return fluct, noiseRandObj

    def neuron_noise_configure(self, kind, mechanism, state, level):
        if kind == "EXC":
            mechanism.m = 0.2
            mechanism.s = 0.1 * level
            mechanism.tau = 2.5
        else:
            mechanism.m = -0.1
            mechanism.s = 0.08 * level
            mechanism.tau = 1.5
