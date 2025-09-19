import os

from livn.types import Model


class PinskyRinzel(Model):
    def neuron_template_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "templates")

    def neuron_mechanisms_directory(self):
        return os.path.join(os.path.dirname(__file__), "neuron", "mechanisms")

    def neuron_celltypes(self, celltypes):
        pass  # use defaults

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
            "SatExp2Syn": {
                "mech_file": "sat_exp2syn.mod",
                "mech_params": ["sat", "dur_onset", "tau_offset", "e"],
                "netcon_params": {"weight": 0, "g_unit": 1},
                "netcon_state": {"onset": 2, "count": 3, "g0": 4, "t0": 5},
            },
        }

    def neuron_noise_mechanism(self, section):
        pass

    def neuron_noise_configure(
        self, population, mechanism, state, exc_level, inh_level
    ):
        pass

    def neuron_default_noise(self, system: str, key: int = 0):
        return {}

    def neuron_default_weights(self, system: str):
        return {
            "C5": {
                "AAC.BS.apical.GABA_A.weight": 15.351000785827637,
                "AAC.CA2.('apical', 'basal').AMPA.weight": 16.922027587890625,
                "AAC.CA2.('apical', 'basal').NMDA.weight": 10.989128112792969,
                "AAC.CA3.('apical', 'basal').AMPA.weight": 4.126145362854004,
                "AAC.CA3.('apical', 'basal').NMDA.weight": 0.10000000149011612,
                "AAC.CCKBC.('apical', 'soma').GABA_A.weight": 8.920540809631348,
                "AAC.EC.apical.AMPA.weight": 8.618342399597168,
                "AAC.EC.apical.NMDA.weight": 13.728931427001953,
                "AAC.IVY.apical.GABA_A.weight": 0.10000000149011612,
                "AAC.NGFC.apical.GABA_A.weight": 15.102134704589844,
                "AAC.NGFC.apical.GABA_B.weight": 15.246316909790039,
                "AAC.OLM.apical.GABA_A.weight": 6.890379905700684,
                "AAC.PVBC.('apical', 'basal', 'soma').GABA_A.weight": 0.13693206012248993,
                "AAC.PYR.('basal', 'soma').AMPA.weight": 18.49085807800293,
                "AAC.PYR.('basal', 'soma').NMDA.weight": 12.871319770812988,
                "AAC.SCA.apical.GABA_A.weight": 3.2796342372894287,
                "BS.BS.apical.GABA_A.weight": 20.0,
                "BS.CA2.('apical', 'basal').AMPA.weight": 3.521620273590088,
                "BS.CA3.('apical', 'basal').AMPA.weight": 9.639567375183105,
                "BS.CCKBC.('soma', 'apical').GABA_A.weight": 13.881185531616211,
                "BS.EC.apical.AMPA.weight": 0.10000000149011612,
                "BS.IS1.basal.GABA_A.weight": 3.2991855144500732,
                "BS.IS3.basal.GABA_A.weight": 16.52082061767578,
                "BS.IVY.apical.GABA_A.weight": 17.074373245239258,
                "BS.NGFC.apical.GABA_A.weight": 13.721741676330566,
                "BS.NGFC.apical.GABA_B.weight": 2.4268364906311035,
                "BS.OLM.apical.GABA_A.weight": 19.01692008972168,
                "BS.PVBC.('soma', 'basal').GABA_A.weight": 19.560476303100586,
                "BS.PYR.('soma', 'basal').AMPA.weight": 16.290584564208984,
                "BS.SCA.apical.GABA_A.weight": 4.563541412353516,
                "CCKBC.BS.apical.GABA_A.weight": 1.8320660591125488,
                "CCKBC.CA2.('apical', 'basal').AMPA.weight": 0.10000000149011612,
                "CCKBC.CA3.('apical', 'basal', 'soma').AMPA.weight": 0.10000000149011612,
                "CCKBC.CCKBC.('soma', 'apical', 'basal').GABA_A.weight": 1.9885337352752686,
                "CCKBC.EC.apical.AMPA.weight": 3.7254583835601807,
                "CCKBC.IS1.('apical', 'basal').GABA_A.weight": 0.40845781564712524,
                "CCKBC.IS2.apical.GABA_A.weight": 1.593267798423767,
                "CCKBC.IS3.basal.GABA_A.weight": 2.0,
                "CCKBC.IVY.apical.GABA_A.weight": 0.23504237830638885,
                "CCKBC.NGFC.apical.GABA_A.weight": 1.2329522371292114,
                "CCKBC.NGFC.apical.GABA_B.weight": 0.9300533533096313,
                "CCKBC.OLM.apical.GABA_A.weight": 1.0761815309524536,
                "CCKBC.PVBC.('soma', 'basal').GABA_A.weight": 0.24529221653938293,
                "CCKBC.PYR.basal.AMPA.weight": 6.418299198150635,
                "CCKBC.SCA.apical.GABA_A.weight": 1.0123575925827026,
                "IS1.BS.apical.GABA_A.weight": 16.63083267211914,
                "IS1.CA2.('apical', 'basal').AMPA.weight": 9.832049369812012,
                "IS1.CA3.('apical', 'basal').AMPA.weight": 3.223146915435791,
                "IS1.IS1.basal.GABA_A.weight": 19.554546356201172,
                "IS1.IS2.apical.GABA_A.weight": 8.704646110534668,
                "IS1.IVY.('apical', 'basal').GABA_A.weight": 13.808609008789062,
                "IS1.PVBC.soma.GABA_A.weight": 18.160898208618164,
                "IS1.PYR.('soma', 'basal').AMPA.weight": 15.482104301452637,
                "IS1.PYR.('soma', 'basal').NMDA.weight": 16.039966583251953,
                "IS1.SCA.('apical', 'basal').GABA_A.weight": 17.48859977722168,
                "IS2.CA3.('soma', 'apical', 'basal').AMPA.weight": 4.466995716094971,
                "IS2.EC.apical.AMPA.weight": 6.366391181945801,
                "IS2.IS1.('soma', 'apical').GABA_A.weight": 0.21135148406028748,
                "IS2.NGFC.('soma', 'apical', 'basal').GABA_A.weight": 0.19701561331748962,
                "IS2.SCA.basal.GABA_A.weight": 10.0,
                "IS3.BS.('apical', 'basal').GABA_A.weight": 9.683126449584961,
                "IS3.CA2.('soma', 'apical', 'basal').AMPA.weight": 5.522098541259766,
                "IS3.CA3.('soma', 'apical', 'basal').AMPA.weight": 3.0281879901885986,
                "IS3.EC.apical.AMPA.weight": 3.1364595890045166,
                "IS3.IS1.apical.GABA_A.weight": 7.694550037384033,
                "IS3.IS2.apical.GABA_A.weight": 0.8496452569961548,
                "IS3.IVY.('apical', 'basal').GABA_A.weight": 9.722854614257812,
                "IS3.NGFC.apical.GABA_A.weight": 6.503699779510498,
                "IS3.PVBC.soma.GABA_A.weight": 10.0,
                "IS3.PYR.basal.AMPA.weight": 6.509203910827637,
                "IS3.SCA.apical.GABA_A.weight": 0.7593812346458435,
                "IVY.BS.('apical', 'basal').GABA_A.weight": 20.0,
                "IVY.CA2.('apical', 'basal').AMPA.weight": 4.174244403839111,
                "IVY.CA2.('apical', 'basal').NMDA.weight": 14.688445091247559,
                "IVY.CA3.('soma', 'apical', 'basal').AMPA.weight": 15.561047554016113,
                "IVY.CA3.('soma', 'apical', 'basal').NMDA.weight": 3.4270358085632324,
                "IVY.CCKBC.('soma', 'apical', 'basal').GABA_A.weight": 5.565309047698975,
                "IVY.IS1.('apical', 'basal').GABA_A.weight": 18.481794357299805,
                "IVY.IS2.apical.GABA_A.weight": 20.0,
                "IVY.IVY.('apical', 'basal').GABA_A.weight": 20.0,
                "IVY.NGFC.('apical', 'basal').GABA_A.weight": 1.2288762331008911,
                "IVY.NGFC.('apical', 'basal').GABA_B.weight": 16.23963165283203,
                "IVY.PVBC.soma.GABA_A.weight": 12.943732261657715,
                "IVY.PYR.('apical', 'basal').AMPA.weight": 17.109193801879883,
                "IVY.PYR.('apical', 'basal').NMDA.weight": 2.7916951179504395,
                "IVY.SCA.('apical', 'basal').GABA_A.weight": 8.914420127868652,
                "NGFC.BS.basal.GABA_A.weight": 20.0,
                "NGFC.CA3.basal.AMPA.weight": 0.10000000149011612,
                "NGFC.CA3.basal.NMDA.weight": 0.10000000149011612,
                "NGFC.EC.('apical', 'basal').AMPA.weight": 14.061335563659668,
                "NGFC.EC.('apical', 'basal').NMDA.weight": 17.28775978088379,
                "NGFC.IVY.basal.GABA_A.weight": 12.436141967773438,
                "NGFC.NGFC.apical.GABA_A.weight": 18.407745361328125,
                "NGFC.NGFC.apical.GABA_B.weight": 12.886497497558594,
                "NGFC.OLM.apical.GABA_A.weight": 18.3082275390625,
                "NGFC.SCA.basal.GABA_A.weight": 15.717228889465332,
                "OLM.IS3.basal.GABA_A.weight": 14.881133079528809,
                "OLM.OLM.basal.GABA_A.weight": 7.002025604248047,
                "OLM.PYR.('soma', 'basal').AMPA.weight": 0.15471965074539185,
                "PVBC.BS.apical.GABA_A.weight": 6.697705268859863,
                "PVBC.CA2.('apical', 'basal').AMPA.weight": 17.30992317199707,
                "PVBC.CA2.('apical', 'basal').NMDA.weight": 1.9382027387619019,
                "PVBC.CA3.('apical', 'basal').AMPA.weight": 2.6853017807006836,
                "PVBC.CA3.('apical', 'basal').NMDA.weight": 8.313755989074707,
                "PVBC.CCKBC.('apical', 'soma').GABA_A.weight": 19.381195068359375,
                "PVBC.EC.apical.AMPA.weight": 5.5582780838012695,
                "PVBC.EC.apical.NMDA.weight": 9.10003662109375,
                "PVBC.IVY.apical.GABA_A.weight": 14.983209609985352,
                "PVBC.NGFC.apical.GABA_A.weight": 20.0,
                "PVBC.NGFC.apical.GABA_B.weight": 17.29940414428711,
                "PVBC.OLM.apical.GABA_A.weight": 12.258416175842285,
                "PVBC.PVBC.('apical', 'basal', 'soma').GABA_A.weight": 1.5871219635009766,
                "PVBC.PYR.('basal', 'soma').AMPA.weight": 5.206347942352295,
                "PVBC.PYR.('basal', 'soma').NMDA.weight": 15.080333709716797,
                "PVBC.SCA.apical.GABA_A.weight": 1.9071751832962036,
                "PYR.AAC.ais.GABA_A.weight": 0.43884018063545227,
                "PYR.BS.('apical', 'basal').GABA_A.weight": 0.4151322543621063,
                "PYR.CA2.basal.AMPA.weight": 5.132308006286621,
                "PYR.CA3.('apical', 'basal').AMPA.weight": 3.2369885444641113,
                "PYR.CCKBC.('apical', 'basal', 'soma').GABA_A.weight": 0.5565959215164185,
                "PYR.EC.('apical', 'basal').AMPA.weight": 1.032667636871338,
                "PYR.IVY.('apical', 'basal').GABA_A.weight": 5.895874977111816,
                "PYR.NGFC.apical.GABA_A.weight": 10.0,
                "PYR.NGFC.apical.GABA_B.weight": 1.0035855770111084,
                "PYR.OLM.('apical', 'basal').GABA_A.weight": 3.4616401195526123,
                "PYR.PVBC.('apical', 'basal', 'soma').GABA_A.weight": 6.660292625427246,
                "PYR.PYR.('soma', 'basal').AMPA.weight": 0.10000000149011612,
                "PYR.SCA.('apical', 'basal').GABA_A.weight": 0.10000000149011612,
                "SCA.BS.apical.GABA_A.weight": 8.610803604125977,
                "SCA.CA2.('apical', 'basal').AMPA.weight": 1.2676496505737305,
                "SCA.CA3.('apical', 'basal', 'soma').AMPA.weight": 7.827863693237305,
                "SCA.CCKBC.('soma', 'basal').GABA_A.weight": 0.10000000149011612,
                "SCA.EC.('apical', 'basal', 'soma').AMPA.weight": 4.144337177276611,
                "SCA.IS1.('basal', 'apical').GABA_A.weight": 1.3972240686416626,
                "SCA.IS2.apical.GABA_A.weight": 9.41629409790039,
                "SCA.IVY.apical.GABA_A.weight": 8.64366340637207,
                "SCA.NGFC.apical.GABA_A.weight": 5.4942474365234375,
                "SCA.NGFC.apical.GABA_B.weight": 1.657443642616272,
                "SCA.OLM.apical.GABA_A.weight": 1.8229607343673706,
                "SCA.PVBC.('basal', 'soma').GABA_A.weight": 6.164452075958252,
                "SCA.PYR.basal.AMPA.weight": 10.0,
                "SCA.SCA.apical.GABA_A.weight": 7.926729679107666,
            },
        }[system]
