import logging

from neuron import h

from . import axon as _axon
from . import resting as _resting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class V1In:
    """
    Single-compartment V1 spinal inhibitory interneuron model.

    Booth-Rinzel-Kiehn formalism and ion channels.

    Accepts a flat parameter dict (fixed + optimized combined), matching the
    convention used by dmosopt and neuron_utils.ic_constant_f.

    Passive properties (Rin, Cm) and Vthresh derive from Hoang et al. 2018.
    """

    def __init__(self, params=None):
        self.soma = h.Section(name="soma", cell=self)
        self.all = h.SectionList()
        self.sections = []

        self.set_default_parameters()
        if params is not None:
            self.set_parameters(params)

        self.geometry()
        self.biophys()

        self.all.append(self.soma)
        self.axon = _axon.attach(
            self, self.soma, self.axon_params, self.global_e_pas, self.soma_gmax_Na
        )
        for sec in self.axon:
            self.all.append(sec)
        self.sections = [self.soma, *self.axon]

    def set_default_parameters(self):
        # RC (Renshaw cell) defaults: geometry gives Cm ~15 pF, Rin ~1000 MOhm
        self.global_diam = 23.0  # um  sphere equivalent (L = diam)
        self.global_cm = 0.9  # uF/cm2
        self.global_e_pas = -65.0  # mV
        self.soma_g_pas = 6.0e-5  # S/cm2  -> Rin ~1000 MOhm
        self.soma_gmax_Na = 0.15  # S/cm2
        self.soma_vhalf_Na = -35.0
        self.soma_slope_Na = 7.8
        self.soma_gmax_K = 0.06  # S/cm2
        self.soma_gmax_KCa = 5.0e-5  # S/cm2  minimal for RC fast-spiking
        self.soma_gmax_CaN = 0.0  # S/cm2  0 for RC; optimized for IaIn
        self.soma_gmax_Ka = 0.0  # S/cm2  0 for RC; optimized for IaIn
        self.soma_f_Caconc = 0.004
        self.soma_alpha_Caconc = 1
        self.soma_kCa_Caconc = 8
        self.axon_params = _axon.axon_parameters(None)

    def set_parameters(self, params):
        self.axon_params = _axon.axon_parameters(params)
        self.global_diam = params.get("global_diam", self.global_diam)
        self.global_cm = params.get("global_cm", self.global_cm)
        self.global_e_pas = params.get("e_pas", self.global_e_pas)
        self.soma_g_pas = params.get("soma_g_pas", self.soma_g_pas)
        self.soma_gmax_Na = params.get("soma_gmax_Na", self.soma_gmax_Na)
        self.soma_vhalf_Na = params.get("soma_vhalf_Na", self.soma_vhalf_Na)
        self.soma_slope_Na = params.get("soma_slope_Na", self.soma_slope_Na)
        self.soma_gmax_K = params.get("soma_gmax_K", self.soma_gmax_K)
        self.soma_gmax_KCa = params.get("soma_gmax_KCa", self.soma_gmax_KCa)
        self.soma_gmax_CaN = params.get("soma_gmax_CaN", self.soma_gmax_CaN)
        self.soma_gmax_Ka = params.get("soma_gmax_Ka", self.soma_gmax_Ka)
        self.soma_f_Caconc = params.get("soma_f_Caconc", self.soma_f_Caconc)
        self.soma_alpha_Caconc = params.get("soma_alpha_Caconc", self.soma_alpha_Caconc)
        self.soma_kCa_Caconc = params.get("soma_kCa_Caconc", self.soma_kCa_Caconc)

    def geometry(self):
        self.soma.L = self.global_diam
        self.soma.diam = self.global_diam
        self.soma.nseg = 1
        self.soma.cm = self.global_cm

    def init_ic(self, v_init):
        h.finitialize(v_init)
        self.measure_ic()

    def measure_ic(self):
        _resting.pin(list(self.sections) or [self.soma])

    def biophys(self):
        sec = self.soma
        for mech in (
            "pas",
            "constant",
            "Na_conc",
            "K_conc",
            "Ca_conc",
            "Nas",
            "Kdr",
            "KCa",
            "CaN",
            "Ka_v1in",
        ):
            sec.insert(mech)

        sec.gmax_Nas = self.soma_gmax_Na
        sec.vhalf_Nas = self.soma_vhalf_Na
        sec.slope_Nas = self.soma_slope_Na
        sec.gmax_Kdr = self.soma_gmax_K
        sec.gmax_KCa = self.soma_gmax_KCa
        sec.gmax_CaN = self.soma_gmax_CaN
        sec.gmax_Ka_v1in = self.soma_gmax_Ka

        # f, alpha, kCa are RANGE parameters of livn's Ca_conc.mod (per section)
        sec.f_Ca_conc = self.soma_f_Caconc
        sec.alpha_Ca_conc = self.soma_alpha_Caconc
        sec.kCa_Ca_conc = self.soma_kCa_Caconc

        sec.g_pas = self.soma_g_pas
        sec.e_pas = self.global_e_pas

        self.configure_axon()
        _resting.insert_constant([self.soma, *(getattr(self, "axon", None) or [])])

    def configure_axon(self):
        if getattr(self, "axon", None):
            _axon.configure(
                self.axon, self.axon_params, self.global_e_pas, self.soma_gmax_Na
            )

    def position(self, x, y, z):
        x0 = getattr(self, "x", 0.0)
        y0 = getattr(self, "y", 0.0)
        z0 = getattr(self, "z", 0.0)
        for i in range(self.soma.n3d()):
            h.pt3dchange(
                i,
                x - x0 + self.soma.x3d(i),
                y - y0 + self.soma.y3d(i),
                z - z0 + self.soma.z3d(i),
                self.soma.diam3d(i),
                sec=self.soma,
            )
        self.x, self.y, self.z = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True
