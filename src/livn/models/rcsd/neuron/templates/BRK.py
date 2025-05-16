import numpy as np
from neuron import h


class BRK:
    def __init__(self, params=None):
        if params is not None:
            params = params["BoothRinzelKiehn"]

        # Create sections
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)

        # hack: use default attribute to trigger detection
        self.hillock = self.dend

        # Initialize position coordinates
        self.x = self.y = self.z = 0

        # Create section lists
        self.sections = h.SectionList()
        self.all = h.SectionList()

        if params is not None:
            self.set_parameters(params)
        else:
            self.set_default_parameters()

        self.init_topology()
        self.geometry()
        self.biophys()

        # Add sections to lists
        for sec in [self.soma, self.dend]:
            self.all.append(sec)
        self.sections = list(self.all)

    def set_default_parameters(self):
        self.pp = 0.5  # proportion of area taken up by soma
        self.Ltotal = 400 / np.pi  # total length of compartments
        self.gc = 10.5  # mS/cm2; Ra in ohm-cm

        self.global_e_pas = -60
        self.soma_g_pas = 0.0001
        self.soma_gmax_Na = 0.00030
        self.soma_gmax_K = 0.00010
        self.soma_gmax_KCa = 0.0005
        self.soma_gmax_CaN = 0.00010

        self.soma_f_Caconc = 0.004
        self.soma_alpha_Caconc = 1
        self.soma_kCa_Caconc = 8

        self.dend_g_pas = 0.0001
        self.dend_gmax_CaN = 0.00010
        self.dend_gmax_CaL = 0.00010
        self.dend_gmax_KCa = 0.00015

        self.dend_f_Caconc = 0.004
        self.dend_alpha_Caconc = 1
        self.dend_kCa_Caconc = 8

        self.global_cm = 3
        self.global_diam = 10  # Default value
        self.cm_ratio = 1

    def set_parameters(self, params):
        self.pp = params.get("pp")
        self.Ltotal = params.get("Ltotal")
        self.gc = params.get("gc")

        self.global_diam = params.get("global_diam")
        self.global_cm = params.get("global_cm")
        self.cm_ratio = params.get("cm_ratio", 1.0)

        self.global_e_pas = params.get("e_pas", -60)
        self.soma_g_pas = params.get("soma_g_pas")
        self.soma_gmax_Na = params.get("soma_gmax_Na")
        self.soma_gmax_K = params.get("soma_gmax_K")
        self.soma_gmax_KCa = params.get("soma_gmax_KCa")
        self.soma_gmax_CaN = params.get("soma_gmax_CaN")

        self.soma_f_Caconc = params.get("soma_f_Caconc")
        self.soma_alpha_Caconc = params.get("soma_alpha_Caconc")
        self.soma_kCa_Caconc = params.get("soma_kCa_Caconc")

        self.dend_g_pas = params.get("dend_g_pas")
        self.dend_gmax_CaN = params.get("dend_gmax_CaN")
        self.dend_gmax_CaL = params.get("dend_gmax_CaL")
        self.dend_gmax_KCa = params.get("dend_gmax_KCa")

        self.dend_f_Caconc = params.get("dend_f_Caconc")
        self.dend_alpha_Caconc = params.get("dend_alpha_Caconc")
        self.dend_kCa_Caconc = params.get("dend_kCa_Caconc")

    def lambda_f(self, section, freq):
        if section.n3d() < 2:
            return 1e5 * np.sqrt(
                section.diam / (4 * np.pi * freq * section.Ra * section.cm)
            )

        x1 = section.arc3d(0)
        d1 = section.diam3d(0)
        lam = 0

        for i in range(1, section.n3d()):
            x2 = section.arc3d(i)
            d2 = section.diam3d(i)
            lam += (x2 - x1) / np.sqrt(d1 + d2)
            x1, d1 = x2, d2

        lam *= np.sqrt(2) * 1e-5 * np.sqrt(4 * np.pi * freq * section.Ra * section.cm)
        return section.L / lam

    def init_topology(self):
        self.dend.connect(self.soma(1), 0)

    def geometry(self):
        self.init_dx()
        self.init_diam()
        self.init_nseg()

    def init_dx(self):
        self.soma.L = self.pp * self.Ltotal
        self.dend.L = (1 - self.pp) * self.Ltotal

    def init_diam(self):
        self.soma.diam = self.global_diam
        self.dend.diam = self.global_diam

    def init_nseg(self, freq=100, d_lambda=0.1):
        for sec in [self.soma, self.dend]:
            nseg = (
                int((sec.L / (d_lambda * self.lambda_f(sec, freq)) + 0.9) / 2) * 2 + 1
            )
            sec.nseg = nseg

    def init_ic(self, v_init):
        h.finitialize(v_init)
        self.soma.ic_constant = -(h.ina + h.ik + h.ica + h.i_pas)

    def biophys(self):
        # Set global parameters
        for sec in [self.soma, self.dend]:
            sec.Ra = 1
            sec.Ra = (
                1e-6 / (self.gc / self.pp * (h.area(0.5, sec=sec) * 1e-8) * 1e-3)
            ) / (2 * h.ri(0.5, sec=sec))
            sec.cm = self.global_cm

        # Soma-specific parameters
        self.soma.cm = self.global_cm * self.cm_ratio

        self.soma.insert("pas")
        self.soma.insert("constant")
        self.soma.insert("Na_conc")
        self.soma.insert("K_conc")
        self.soma.insert("Ca_conc")
        self.soma.insert("Kdr")
        self.soma.insert("Nas")
        self.soma.insert("CaN")
        self.soma.insert("KCa")
        self.soma.insert("extracellular")  # For stimulation

        self.soma.gmax_Nas = self.soma_gmax_Na
        self.soma.gmax_Kdr = self.soma_gmax_K
        self.soma.gmax_CaN = self.soma_gmax_CaN
        self.soma.gmax_KCa = self.soma_gmax_KCa

        self.soma.f_Ca_conc = self.soma_f_Caconc
        self.soma.alpha_Ca_conc = self.soma_alpha_Caconc
        self.soma.kCa_Ca_conc = self.soma_kCa_Caconc

        self.soma.g_pas = self.soma_g_pas
        self.soma.e_pas = self.global_e_pas

        # Dendrite-specific parameters
        self.dend.insert("pas")
        self.dend.insert("CaN")
        self.dend.insert("CaL")
        self.dend.insert("KCa")
        self.dend.insert("Ca_conc")
        self.dend.insert("K_conc")

        self.dend.f_Ca_conc = self.dend_f_Caconc
        self.dend.alpha_Ca_conc = self.dend_alpha_Caconc
        self.dend.kCa_Ca_conc = self.dend_kCa_Caconc

        self.dend.g_pas = self.dend_g_pas
        self.dend.e_pas = self.global_e_pas

        self.dend.gmax_CaN = self.dend_gmax_CaN
        self.dend.gmax_CaL = self.dend_gmax_CaL
        self.dend.gmax_KCa = self.dend_gmax_KCa

    def position(self, x, y, z):
        for sec in [self.soma, self.dend]:
            for i in range(sec.n3d()):
                h.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                    sec=sec,
                )
        self.x, self.y, self.z = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True

    def __repr__(self):
        return "BRK"
