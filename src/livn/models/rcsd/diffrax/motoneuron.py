import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Any, Mapping, Sequence, Optional, Tuple, Union


def array_to_dict(params: Sequence) -> Mapping[str, float]:
    """Convert an array of BRK parameters to a dict"""
    order = [
        "pp",
        "Ltotal",
        "gc",
        "global_diam",
        "global_cm",
        "soma_gmax_Na",
        "soma_gmax_K",
        "soma_gmax_CaN",
        "dend_gmax_CaN",
        "soma_gmax_KCa",
        "dend_gmax_KCa",
        "dend_gmax_CaL",
        "soma_g_pas",
        "e_pas",
        "soma_f_Caconc",
        "soma_alpha_Caconc",
        "soma_kCa_Caconc",
    ]
    mapping = {}
    for i, key in enumerate(order):
        if i < len(params):
            mapping[key] = float(params[i])
    return mapping


class BoothRinzelKiehn(eqx.Module):
    """
    Booth, Rinzel, Kiehn (1997) compartmental model of a vertebrate motoneuron
    """

    GNa: float
    GK_dr: float
    GCa_NS: float
    GCa_ND: float
    GK_CaS: float
    GK_CaD: float
    GCa_L: float
    gleak: float

    C: float
    gc: float
    p: float
    Kd: float
    f: float
    alpha: float
    kca: float

    Ra: float
    area_cm2: float

    Vhm: float
    Sm: float
    Vhh: float
    Sh: float

    Vhn: float
    Sn: float

    VhmN: float
    SmN: float
    TaumN: float
    VhhN: float
    ShN: float
    TauhN: float

    VhmL: float
    SmL: float
    TaumL: float

    ENa: float
    EK: float
    ECa: float
    Eleak: float

    nai: float
    nao: float
    ki: float
    ko: float

    Ltotal: float
    global_diam: float

    iCa_rest_S: float
    iCa_rest_D: float

    celsius: float
    cao: float

    R_const: float
    T_kelvin: float
    F_const: float

    solver: Union[diffrax.AbstractSolver, str] = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    E_syn_exc: float
    E_syn_inh: float

    input_mode: str = eqx.field(static=True)

    def set_default_parameters(self) -> None:
        # Conductances (mho/cm^2)
        self.GNa = 0.00030  # soma_gmax_Na
        self.GK_dr = 0.00010  # soma_gmax_K
        self.GCa_NS = 0.00010  # soma_gmax_CaN
        self.GCa_ND = 0.00010  # dend_gmax_CaN
        self.GK_CaS = 0.0005  # soma_gmax_KCa
        self.GK_CaD = 0.00015  # dend_gmax_KCa
        self.GCa_L = 0.00010  # dend_gmax_CaL
        self.gleak = 0.0001  # soma_g_pas and dend_g_pas

        self.C = 3.0  # uF/cm^2 - global_cm
        self.gc = 10.5  # coupling conductance (mS/cm^2)
        self.p = 0.5  # proportion of area taken up by soma (pp)

        # Calcium-activated K channel affinity.
        self.Kd = 0.0005  # mM
        self.f = 0.004  # percent free to bound Ca - soma_f_Caconc
        self.alpha = 1.0  # Ca removal scaling - soma_alpha_Caconc
        self.kca = 8.0  # Ca removal rate - soma_kCa_Caconc

        self.Ra = 293.7  # ohm-cm
        self.area_cm2 = 2000 * 1e-8  # 2000 um^2 converted to cm^2

        # Half Activation voltages (mV), Slopes (mV), Time Constants (ms)

        # From Nas.mod (Sodium channel)
        self.Vhm = -35.0  # minf = 1/(1+exp(-(v+35)/7.8))
        self.Sm = 7.8  # slope for m activation
        self.Vhh = -55.0  # hinf = 1/(1+exp((v+55)/7))
        self.Sh = 7.0  # slope for h inactivation

        # From Kdr.mod (Potassium delayed rectifier)
        self.Vhn = -28.0  # ninf = 1 / ((exp(-(v + 28)/15)) + 1)
        self.Sn = 15.0  # slope for n activation

        # From CaN.mod (N-type Calcium channel)
        self.VhmN = -30.0  # minf = 1 / (1 + exp((v+30)/-5))
        self.SmN = 5.0  # slope for CaN m activation
        self.TaumN = 4.0  # time constant for CaN m
        self.VhhN = -45.0  # hinf = 1 / (1 + exp((v+45)/5))
        self.ShN = 5.0  # slope for CaN h inactivation
        self.TauhN = 40.0  # time constant for CaN h

        # From CaL.mod (L-type Calcium channel)
        self.VhmL = -40.0  # minf = 1 / (1 + exp((v + 40)/-7))
        self.SmL = 7.0  # slope for CaL m activation
        self.TaumL = 60.0  # time constant for CaL m (mtau = 60)

        # Reversal potentials (mV)
        self.ENa = 50.0  # NEURON default for Na
        self.EK = -77.0  # NEURON default for K
        self.ECa = 132.5  # NEURON default for Ca
        self.Eleak = -60.0  # global_e_pas

        self.nai = 15.0  # mM
        self.nao = 145.0  # mM
        self.ki = 145.0  # mM
        self.ko = 5.0  # mM

        # Geometry parameters
        self.Ltotal = 400.0 / jnp.pi  # total length of compartments
        self.global_diam = 10.0  # diameter in um

        # Resting calcium currents (computed at initialization to reproduce Ca_conc.mod behavior)
        self.iCa_rest_S = 0.0
        self.iCa_rest_D = 0.0

        # Temperature-related constants for GHK formulation
        self.celsius = 6.3
        self.cao = 2.0  # mM external calcium

        # Physical constants
        self.R_const = 8.31441  # V*C / (Mol*K)
        self.T_kelvin = 309.15  # K (36 C)
        self.F_const = 96485.309  # Coulombs/mol

        # Solver (explicit 2nd order Heun for pmap compatibility with stiff systems)
        self.solver = "Heun"
        self.max_steps = 2_000_000  # per 1000 ms

        # Synaptic reversal potentials (mV)
        self.E_syn_exc = 0.0  # Excitatory (AMPA/NMDA)
        self.E_syn_inh = -75.0  # Inhibitory (GABA_A)

        self.input_mode = "conductance"

    def set_parameters(self, params: Mapping[str, Any]) -> None:
        """
        Set parameters from a dictionary.

        Parameters
        ----------
        params : dict
            - pp -> p
            - Ltotal -> Ltotal
            - gc -> gc
            - global_diam -> global_diam
            - global_cm -> C
            - e_pas, global_e_pas -> Eleak
            - soma_g_pas, dend_g_pas -> gleak
            - soma_gmax_Na -> GNa
            - soma_gmax_K -> GK_dr
            - soma_gmax_CaN -> GCa_NS
            - dend_gmax_CaN -> GCa_ND
            - soma_gmax_KCa -> GK_CaS
            - dend_gmax_KCa -> GK_CaD
            - dend_gmax_CaL -> GCa_L
            - soma_f_Caconc, dend_f_Caconc -> f
            - soma_alpha_Caconc, dend_alpha_Caconc -> alpha
            - soma_kCa_Caconc, dend_kCa_Caconc -> kca
        """
        get = params.get

        # Geometry and passive
        if get("pp") is not None:
            self.p = float(get("pp"))
        if get("Ltotal") is not None:
            self.Ltotal = float(get("Ltotal"))
        if get("gc") is not None:
            self.gc = float(get("gc"))
        if get("global_diam") is not None:
            self.global_diam = float(get("global_diam"))
        if get("global_cm") is not None:
            self.C = float(get("global_cm"))

        # Leak / reversal
        epas = get("e_pas", get("global_e_pas"))
        if epas is not None:
            self.Eleak = float(epas)

        # Conductances
        if get("soma_gmax_Na") is not None:
            self.GNa = float(get("soma_gmax_Na"))
        if get("soma_gmax_K") is not None:
            self.GK_dr = float(get("soma_gmax_K"))
        if get("soma_gmax_CaN") is not None:
            self.GCa_NS = float(get("soma_gmax_CaN"))
        if get("dend_gmax_CaN") is not None:
            self.GCa_ND = float(get("dend_gmax_CaN"))
        if get("soma_gmax_KCa") is not None:
            self.GK_CaS = float(get("soma_gmax_KCa"))
        if get("dend_gmax_KCa") is not None:
            self.GK_CaD = float(get("dend_gmax_KCa"))
        if get("dend_gmax_CaL") is not None:
            self.GCa_L = float(get("dend_gmax_CaL"))

        # Leak conductance
        soma_g_pas = get("soma_g_pas")
        dend_g_pas = get("dend_g_pas")
        if (
            soma_g_pas is not None
            and dend_g_pas is not None
            and soma_g_pas != dend_g_pas
        ):
            self.gleak = float(soma_g_pas)
        elif soma_g_pas is not None:
            self.gleak = float(soma_g_pas)
        elif dend_g_pas is not None:
            self.gleak = float(dend_g_pas)

        # Calcium removal
        f_val = get("soma_f_Caconc", get("dend_f_Caconc"))
        if f_val is not None:
            self.f = float(f_val)
        alpha_val = get("soma_alpha_Caconc", get("dend_alpha_Caconc"))
        if alpha_val is not None:
            self.alpha = float(alpha_val)
        kca_val = get("soma_kCa_Caconc", get("dend_kCa_Caconc"))
        if kca_val is not None:
            self.kca = float(kca_val)

        # Solver
        if get("solver") is not None:
            self.solver = get("solver")
        if get("max_steps") is not None:
            self.max_steps = int(get("max_steps"))

        # Synaptic reversal potentials
        if get("E_syn_exc") is not None:
            self.E_syn_exc = float(get("E_syn_exc"))
        if get("E_syn_inh") is not None:
            self.E_syn_inh = float(get("E_syn_inh"))

        # Input mode
        if get("input_mode") is not None:
            mode = get("input_mode")
            if mode not in ("current_density", "conductance", "current"):
                raise ValueError(
                    f"input_mode must be 'current_density', 'conductance', or 'current', got {mode}"
                )
            self.input_mode = mode

    def __init__(self, params: Optional[Any] = None, key=None):
        self.set_default_parameters()
        if params is not None:
            if isinstance(params, Mapping) and "BoothRinzelKiehn" in params:
                params = params["BoothRinzelKiehn"]

            if isinstance(params, (list, tuple)) or (
                hasattr(params, "ndim")
                and getattr(params, "ndim", 1) == 1
                and hasattr(params, "__len__")
            ):
                params = array_to_dict(params)

            if isinstance(params, Mapping):
                self.set_parameters(params)
            else:
                raise TypeError(
                    f"params must be a mapping/dict/array/None, not {type(params)}"
                )

        self.T_kelvin = self.celsius + 273.15
        self.ENa = self._nernst(self.nai, self.nao, 1.0)
        self.EK = self._nernst(self.ki, self.ko, 1.0)

    def _nernst(self, ci: float, co: float, valence: float) -> float:
        T_kelvin = self.celsius + 273.15
        return (
            1000.0
            * (self.R_const * T_kelvin)
            / (valence * self.F_const)
            * math.log(co / ci)
        )

    def _geometry(self) -> Tuple[Array, Array, Array, Array]:
        diam_cm = self.global_diam * 1e-4
        L_total_cm = self.Ltotal * 1e-4
        area_total = jnp.pi * diam_cm * L_total_cm
        area_s = self.p * area_total
        area_d = (1.0 - self.p) * area_total

        radius_cm = diam_cm / 2.0
        cross_area = jnp.pi * radius_cm**2
        L_s_cm = self.p * self.Ltotal * 1e-4
        L_d_cm = (1.0 - self.p) * self.Ltotal * 1e-4
        R_s_half = self.Ra * (L_s_cm / 2.0) / cross_area
        R_d_half = self.Ra * (L_d_cm / 2.0) / cross_area
        R_axial = R_s_half + R_d_half

        g_c_s = 1.0 / (R_axial * area_s)
        g_c_d = 1.0 / (R_axial * area_d)

        return area_s, area_d, g_c_s, g_c_d

    def _calculate_membrane_currents(self, t, y, args=None):
        """Calculate membrane currents for soma and dendrite.

        Args:
            t: time
            y: state vector
            args: (I_stim_array, t_array) for array stimulus
        """
        # area_s, area_d, g_c_s, g_c_d = self._geometry()

        Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD = y

        # Fast sodium activation (instantaneous)
        minf = 1.0 / (1.0 + jnp.exp(-(Vs + 35.0) / 7.8))

        # GHK current calculation
        def ghk(v, ci, co, factor):
            nu = v / factor
            efun = jnp.where(
                jnp.abs(nu) < 1e-4,
                1.0 - nu / 2.0,
                nu / (jnp.exp(nu) - 1.0),
            )
            return -factor * (1.0 - (ci / co) * jnp.exp(nu)) * efun

        fN = ((36.0 / 293.15) * (self.celsius + 273.15)) / 2.0
        fL = ((25.0 / 293.15) * (self.celsius + 273.15)) / 2.0

        ghk_soma = ghk(Vs, CaS, self.cao, fN)
        ghk_dend_N = ghk(Vd, CaD, self.cao, fN)
        ghk_dend_L = ghk(Vd, CaD, self.cao, fL)

        # Individual currents (mA/cm^2)
        INaS = self.GNa * minf**3 * h * (Vs - self.ENa)
        IKS = (self.GK_dr * n**4 + self.GK_CaS * CaS / (CaS + self.Kd)) * (Vs - self.EK)
        ICaS = self.GCa_NS * mnS**2 * hnS * ghk_soma
        IleakS = self.gleak * (Vs - self.Eleak)

        IKD = self.GK_CaD * CaD / (CaD + self.Kd) * (Vd - self.EK)
        ICaD_N = self.GCa_ND * mnD**2 * hnD * ghk_dend_N
        ICaD_L = self.GCa_L * ml * ghk_dend_L
        IleakD = self.gleak * (Vd - self.Eleak)

        # Total membrane current (mA/cm^2)
        # In NEURON, membrane current = sum of ionic currents (positive outward)
        # and does NOT include axial/coupling currents or stimulus
        I_mem_soma = INaS + IKS + ICaS + IleakS
        I_mem_dend = IKD + ICaD_N + ICaD_L + IleakD

        return I_mem_soma, I_mem_dend

    def __call__(self, t, y: Array, args) -> Array:
        """Vector field for the Booth-Rinzel-Kiehn model

        [Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD]

        Args:
            t: time
            y: state vector
            args: (I_stim_array, t_array) for array stimulus
        """

        area_s, area_d, g_c_s, g_c_d = self._geometry()

        Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD = y

        I_stim_array, t_array = args

        if self.input_mode == "current_density":
            # direct current density in mA/cm2
            if I_stim_array.ndim == 1:
                Iapp_density_soma = jnp.interp(t, t_array, I_stim_array)
                Iapp_density_dend = 0.0
            else:
                Iapp_density_soma = jnp.interp(t, t_array, I_stim_array[:, 0])
                Iapp_density_dend = jnp.interp(t, t_array, I_stim_array[:, 1])

        elif self.input_mode == "conductance":
            # synaptic conductance in uS (positive = excitatory, negative = inhibitory)
            if I_stim_array.ndim == 1:
                g_input = jnp.interp(t, t_array, I_stim_array)

                g_exc_soma = jnp.maximum(0.0, g_input)
                g_inh_soma = jnp.maximum(0.0, -g_input)

                I_exc_nA = g_exc_soma * (Vs - self.E_syn_exc) / 1000.0
                I_inh_nA = g_inh_soma * (Vs - self.E_syn_inh) / 1000.0
                Iapp_density_soma = -(I_exc_nA + I_inh_nA) * 1e-6 / area_s
                Iapp_density_dend = 0.0
            else:
                g_soma = jnp.interp(t, t_array, I_stim_array[:, 0])
                g_dend = jnp.interp(t, t_array, I_stim_array[:, 1])

                g_exc_soma = jnp.maximum(0.0, g_soma)
                g_inh_soma = jnp.maximum(0.0, -g_soma)
                g_exc_dend = jnp.maximum(0.0, g_dend)
                g_inh_dend = jnp.maximum(0.0, -g_dend)

                I_exc_soma_nA = g_exc_soma * (Vs - self.E_syn_exc) / 1000.0
                I_inh_soma_nA = g_inh_soma * (Vs - self.E_syn_inh) / 1000.0
                I_exc_dend_nA = g_exc_dend * (Vd - self.E_syn_exc) / 1000.0
                I_inh_dend_nA = g_inh_dend * (Vd - self.E_syn_inh) / 1000.0

                Iapp_density_soma = -(I_exc_soma_nA + I_inh_soma_nA) * 1e-6 / area_s
                Iapp_density_dend = -(I_exc_dend_nA + I_inh_dend_nA) * 1e-6 / area_d

        elif self.input_mode == "current":
            # direct current in nA
            if I_stim_array.ndim == 1:
                I_nA = jnp.interp(t, t_array, I_stim_array)
                Iapp_density_soma = I_nA * 1e-6 / area_s
                Iapp_density_dend = 0.0
            else:
                I_soma_nA = jnp.interp(t, t_array, I_stim_array[:, 0])
                I_dend_nA = jnp.interp(t, t_array, I_stim_array[:, 1])
                Iapp_density_soma = I_soma_nA * 1e-6 / area_s
                Iapp_density_dend = I_dend_nA * 1e-6 / area_d
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        Isoma = Iapp_density_soma
        Idend = Iapp_density_dend

        Tauh = 30.0 / (jnp.exp((Vs + 50.0) / 15.0) + jnp.exp(-(Vs + 50.0) / 16.0))
        Taun = 7.0 / (jnp.exp((Vs + 40.0) / 40.0) + jnp.exp(-(Vs + 40.0) / 50.0))

        minf = 1.0 / (1.0 + jnp.exp(-(Vs + 35.0) / 7.8))
        hinf = 1.0 / (1.0 + jnp.exp((Vs + 55.0) / 7.0))
        ninf = 1.0 / (1.0 + jnp.exp(-(Vs + 28.0) / 15.0))

        mnSinf = 1.0 / (1.0 + jnp.exp((Vs + 30.0) / -5.0))
        hnSinf = 1.0 / (1.0 + jnp.exp((Vs + 45.0) / 5.0))
        mnDinf = 1.0 / (1.0 + jnp.exp((Vd + 30.0) / -5.0))
        hnDinf = 1.0 / (1.0 + jnp.exp((Vd + 45.0) / 5.0))

        mlinf = 1.0 / (1.0 + jnp.exp((Vd + 40.0) / -7.0))

        def ghk(v, ci, co, factor):
            nu = v / factor
            efun = jnp.where(
                jnp.abs(nu) < 1e-4,
                1.0 - nu / 2.0,
                nu / (jnp.exp(nu) - 1.0),
            )
            return -factor * (1.0 - (ci / co) * jnp.exp(nu)) * efun

        fN = ((36.0 / 293.15) * (self.celsius + 273.15)) / 2.0
        fL = ((25.0 / 293.15) * (self.celsius + 273.15)) / 2.0

        ghk_soma = ghk(Vs, CaS, self.cao, fN)
        ghk_dend_N = ghk(Vd, CaD, self.cao, fN)
        ghk_dend_L = ghk(Vd, CaD, self.cao, fL)

        INaS = self.GNa * minf**3 * h * (Vs - self.ENa)
        IKS = (self.GK_dr * n**4 + self.GK_CaS * CaS / (CaS + self.Kd)) * (Vs - self.EK)
        ICaS = self.GCa_NS * mnS**2 * hnS * ghk_soma
        IleakS = self.gleak * (Vs - self.Eleak)

        IKD = self.GK_CaD * CaD / (CaD + self.Kd) * (Vd - self.EK)
        ICaD_N = self.GCa_ND * mnD**2 * hnD * ghk_dend_N
        ICaD_L = self.GCa_L * ml * ghk_dend_L
        IleakD = self.gleak * (Vd - self.Eleak)

        IcouplingS = g_c_s * (Vs - Vd)
        IcouplingD = g_c_d * (Vd - Vs)

        scale = 1000.0 / self.C
        dVs = scale * (Isoma - INaS - IKS - ICaS - IleakS - IcouplingS)
        dVd = scale * (Idend - IKD - ICaD_N - ICaD_L - IleakD - IcouplingD)

        dh = (hinf - h) / Tauh
        dn = (ninf - n) / Taun
        dmnS = (mnSinf - mnS) / self.TaumN
        dhnS = (hnSinf - hnS) / self.TauhN
        dmnD = (mnDinf - mnD) / self.TaumN
        dhnD = (hnDinf - hnD) / self.TauhN
        dml = (mlinf - ml) / self.TaumL

        ICaD_total = ICaD_N + ICaD_L
        channel_flow_S = jnp.maximum(0.0, -self.alpha * (ICaS - self.iCa_rest_S))
        channel_flow_D = jnp.maximum(0.0, -self.alpha * (ICaD_total - self.iCa_rest_D))

        cai0 = 1e-5
        dCaS = self.f * (channel_flow_S - self.kca * (CaS - cai0))
        dCaD = self.f * (channel_flow_D - self.kca * (CaD - cai0))

        return jnp.array([dVs, dVd, dh, dn, dmnS, dhnS, dmnD, dhnD, dml, dCaS, dCaD])

    def solve(
        self,
        t_dur: float,
        *,
        I_stim_array: Array,
        dt: float = 0.025,
    ):
        """
        Solve the Booth-Rinzel-Kiehn model using diffrax

        Parameters:
        -----------
        t_dur : float
            Simulation duration in ms
        I_stim_array : Array
            Input stimulus array. Interpretation depends on input_mode:

            - "current_density" (default): Current density in mA/cm2. Direct application to compartments.
            - "conductance": Synaptic conductance in microsiemens
                    Positive values = excitatory (→ E_syn_exc = 0 mV)
                    Negative values = inhibitory (→ E_syn_inh = -75 mV)
            - "current": Direct current injection in nA

            Shape can be either:
            - 1D array [n_time_points] for soma-only stimulation
            - 2D array [n_time_points, 2] for soma (column 0) and dendrite (column 1)
            where n_time_points = t_dur/dt + 1
        dt : float
            Time step in ms

        Returns:
        --------
        (time_array, soma_voltage, dendrite_voltage, soma_membrane_current, dendrite_membrane_current)
        """
        v_init = -60.0
        cai0 = 1e-5

        hinf_init = 1.0 / (1.0 + jnp.exp((v_init + 55.0) / 7.0))
        ninf_init = 1.0 / (1.0 + jnp.exp(-(v_init + 28.0) / 15.0))
        mn_init = 1.0 / (1.0 + jnp.exp((v_init + 30.0) / -5.0))
        hn_init = 1.0 / (1.0 + jnp.exp((v_init + 45.0) / 5.0))
        ml_init = 1.0 / (1.0 + jnp.exp((v_init + 40.0) / -7.0))

        y0 = jnp.array(
            [
                v_init,
                v_init,
                hinf_init,
                ninf_init,
                mn_init,
                hn_init,
                mn_init,
                hn_init,
                ml_init,
                cai0,
                cai0,
            ]
        )

        term = diffrax.ODETerm(self)

        if isinstance(self.solver, str):
            solver_map = {
                "Euler": diffrax.Euler,
                "Heun": diffrax.Heun,
                "Midpoint": diffrax.Midpoint,
                "Ralston": diffrax.Ralston,
                "Bosh3": diffrax.Bosh3,
                "Tsit5": diffrax.Tsit5,
                "Dopri5": diffrax.Dopri5,
                "Dopri8": diffrax.Dopri8,
                "Kvaerno3": diffrax.Kvaerno3,
                "Kvaerno4": diffrax.Kvaerno4,
                "Kvaerno5": diffrax.Kvaerno5,
                "ImplicitEuler": diffrax.ImplicitEuler,
            }
            solver_cls = solver_map.get(self.solver, diffrax.Kvaerno5)
            solver = solver_cls()
        else:
            solver = self.solver

        n_points = int(t_dur / dt) + 1
        t_array = jnp.linspace(0.0, t_dur, n_points)
        stimulus_args = (I_stim_array, t_array)

        saveat = diffrax.SaveAt(ts=t_array)

        # use constant step size for pmap compatibility
        stepsize_controller = diffrax.ConstantStepSize()

        scaled_max_steps = int(self.max_steps * (t_dur / 1000.0))
        scaled_max_steps = max(scaled_max_steps, 10000)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_dur,
            dt0=dt,
            y0=y0,
            args=stimulus_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=scaled_max_steps,
        )

        t_arr = solution.ts
        v_soma = solution.ys[:, 0]
        v_dend = solution.ys[:, 1]

        def calculate_currents_at_time(t, y):
            return self._calculate_membrane_currents(t, y, stimulus_args)

        i_mem_soma, i_mem_dend = jax.vmap(calculate_currents_at_time)(
            t_arr, solution.ys
        )

        return t_arr, v_soma, v_dend, i_mem_soma, i_mem_dend
