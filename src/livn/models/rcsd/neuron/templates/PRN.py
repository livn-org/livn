import numpy as np
from neuron import h


class PRN:
    def __init__(self, params=None):
        if params is not None:
            params = params["PinskyRinzel"]

        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)

        # Create section lists
        self.sections = h.SectionList()
        self.all = h.SectionList()
        self.soma_list = h.SectionList()
        self.apical_list = h.SectionList()

        if params is not None:
            self.set_parameters(params)
        else:
            self.set_default_parameters()

        self.init_topology()
        self.geometry()
        self.biophys()

        # Add sections to lists
        self.soma_list.append(self.soma)
        self.apical_list.append(self.dend)
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
        self.soma_gmax_K = 0.00015

        self.dend_g_pas = 0.0001
        self.dend_gmax_Ca = 0.00010
        self.dend_gmax_KCa = 0.00015

        self.dend_d_Caconc = 13
        self.dend_beta_Caconc = 0.075

        self.global_cm = 3
        self.global_diam = 1
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

        self.dend_g_pas = params.get("dend_g_pas")
        self.dend_gmax_Ca = params.get("dend_gmax_Ca")
        self.dend_gmax_KCa = params.get("dend_gmax_KCa")
        self.dend_d_Caconc = params.get("dend_d_Caconc")
        self.dend_beta_Caconc = params.get("dend_beta_Caconc")

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
        self.soma.ic_constant = -(h.ina + h.ik + h.i_pas)

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
        self.soma.insert("Na_PR")
        self.soma.insert("K_PR")
        self.soma.insert("Na_conc_PR")
        self.soma.insert("K_conc_PR")

        self.soma.gmax_Na_PR = self.soma_gmax_Na
        self.soma.gmax_K_PR = self.soma_gmax_K
        self.soma.g_pas = self.soma_g_pas
        self.soma.e_pas = self.global_e_pas

        # Dendrite-specific parameters
        self.dend.insert("pas")
        self.dend.insert("Ca_PR")
        self.dend.insert("KCa_PR")
        self.dend.insert("Ca_conc_PR")
        self.dend.insert("K_conc_PR")

        self.dend.d_Ca_conc_PR = self.dend_d_Caconc
        self.dend.beta_Ca_conc_PR = self.dend_beta_Caconc
        self.dend.g_pas = self.dend_g_pas
        self.dend.e_pas = self.global_e_pas
        self.dend.gmax_Ca_PR = self.dend_gmax_Ca
        self.dend.gmax_KCa_PR = self.dend_gmax_KCa

    def position(self, x, y, z):
        xx = yy = zz = 0
        for sec in [self.soma, self.dend]:
            for i in range(sec.n3d()):
                pt3d = h.pt3dchange(
                    i,
                    x - xx + sec.x3d(i),
                    y - yy + sec.y3d(i),
                    z - zz + sec.z3d(i),
                    sec.diam3d(i),
                )
        xx, yy, zz = x, y, z

    def is_art(self):
        return False

    def is_reduced(self):
        return True
