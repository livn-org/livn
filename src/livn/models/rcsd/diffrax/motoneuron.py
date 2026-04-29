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
    gleak: float  # soma g_pas (mho/cm^2)
    gleak_d: float  # dend g_pas (mho/cm^2)

    C: float
    cm_ratio: float  # soma cm = C * cm_ratio
    gc: float
    p: float
    Kd: float
    f: float
    alpha: float
    kca: float

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

    # Constant balancing currents (mA/cm^2) injected into soma/dend at every
    # timestep to mirror NEURON's constant mechanism (ic_constant,
    # ic_constant_d) used by the BRK template to pin the cell at the
    # tuned resting potential
    ic_constant: float
    ic_constant_d: float

    # Resting potential used to seed init_ic mirroring NEURON's
    # BRK template behaviour: cell.init_ic(V_rest) evaluates the
    # ionic currents at V_rest and pins ic_constant so the
    # quiescent membrane is balanced at that voltage
    V_rest: float

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

    # RhO3c 3-state opsin parameters (Nikolic et al. 2009)
    opsin_g0: float = eqx.field(static=True)  # pS
    opsin_E_rev: float = eqx.field(static=True)  # mV
    opsin_k_a: float = eqx.field(static=True)  # activation rate scaling (/ms)
    opsin_k_r: float = eqx.field(static=True)  # recovery rate scaling (/ms)
    opsin_Gd: float = eqx.field(static=True)  # deactivation rate (/ms)
    opsin_Gr0: float = eqx.field(static=True)  # dark recovery rate (/ms)
    opsin_p: float = eqx.field(static=True)  # Hill coefficient (activation)
    opsin_q: float = eqx.field(static=True)  # Hill coefficient (recovery)
    opsin_phi_m: float = eqx.field(static=True)  # half-max photon flux
    opsin_v0: float = eqx.field(static=True)  # voltage-dependence scale (mV)
    opsin_v1: float = eqx.field(static=True)  # voltage-dependence amplitude
    opsin_E_photon: float = eqx.field(static=True)  # energy per photon (mW·s)

    def set_default_parameters(self) -> None:
        # Conductances (mho/cm^2)
        self.GNa = 0.00030  # soma_gmax_Na
        self.GK_dr = 0.00010  # soma_gmax_K
        self.GCa_NS = 0.00010  # soma_gmax_CaN
        self.GCa_ND = 0.00010  # dend_gmax_CaN
        self.GK_CaS = 0.0005  # soma_gmax_KCa
        self.GK_CaD = 0.00015  # dend_gmax_KCa
        self.GCa_L = 0.00010  # dend_gmax_CaL
        self.gleak = 0.0001  # soma_g_pas
        self.gleak_d = 0.0001  # dend_g_pas

        self.C = 3.0  # uF/cm^2 - global_cm
        self.cm_ratio = 1.0  # NEURON BRK soma scale: soma.cm = C * cm_ratio
        self.gc = 10.5  # coupling conductance (mS/cm^2)
        self.p = 0.5  # proportion of area taken up by soma (pp)

        # Calcium-activated K channel affinity.
        self.Kd = 0.0005  # mM
        self.f = 0.004  # percent free to bound Ca - soma_f_Caconc
        self.alpha = 1.0  # Ca removal scaling - soma_alpha_Caconc
        self.kca = 8.0  # Ca removal rate - soma_kCa_Caconc

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

        # NEURON's ``constant.mod`` ``ic_constant`` (mA/cm^2) -- 0 by default.
        self.ic_constant = 0.0
        self.ic_constant_d = 0.0
        self.V_rest = -60.0

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

        # RhO3c 3-state opsin defaults (Nikolic et al. 2009)
        self.opsin_g0 = 1.0  # pS
        self.opsin_E_rev = 0.0  # mV
        self.opsin_k_a = 0.28  # /ms
        self.opsin_k_r = 0.28  # /ms
        self.opsin_Gd = 0.0909  # /ms
        self.opsin_Gr0 = 0.0002  # /ms
        self.opsin_p = 0.4
        self.opsin_q = 0.4
        self.opsin_phi_m = 1e16  # photons/s·mm²
        self.opsin_v0 = 43.0  # mV
        self.opsin_v1 = 17.1  # mV (auto-calc: (70+E)/(exp((70+E)/v0)-1))
        wavelength_nm = 473.0
        self.opsin_E_photon = 6.626e-34 * 3e8 / (wavelength_nm * 1e-9) * 1e3

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
        if get("cm_ratio") is not None:
            self.cm_ratio = float(get("cm_ratio"))

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

        soma_g_pas = get("soma_g_pas")
        dend_g_pas = get("dend_g_pas")
        if soma_g_pas is not None:
            self.gleak = float(soma_g_pas)
        if dend_g_pas is not None:
            self.gleak_d = float(dend_g_pas)
        elif soma_g_pas is not None:
            # Fall back to the soma value when only one was provided
            self.gleak_d = float(soma_g_pas)

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

        if get("ic_constant") is not None:
            self.ic_constant = float(get("ic_constant"))
        if get("ic_constant_d") is not None:
            self.ic_constant_d = float(get("ic_constant_d"))
        if get("V_rest") is not None:
            self.V_rest = float(get("V_rest"))

        # Input mode
        if get("input_mode") is not None:
            mode = get("input_mode")
            if mode not in ("current_density", "conductance", "current", "irradiance"):
                raise ValueError(
                    f"input_mode must be 'current_density', 'conductance', "
                    f"'current', or 'irradiance', got {mode}"
                )
            self.input_mode = mode

    def __init__(self, params: Optional[Any] = None, key=None):
        self.set_default_parameters()
        ic_provided_s = False
        ic_provided_d = False
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
                ic_provided_s = params.get("ic_constant") is not None
                ic_provided_d = params.get("ic_constant_d") is not None
                self.set_parameters(params)
            else:
                raise TypeError(
                    f"params must be a mapping/dict/array/None, not {type(params)}"
                )

        self.T_kelvin = self.celsius + 273.15
        self.ENa = self._nernst(self.nai, self.nao, 1.0)
        self.EK = self._nernst(self.ki, self.ko, 1.0)

        ic_s, _ic_d = self._init_ic_values(self.V_rest)
        if not ic_provided_s:
            self.ic_constant = float(ic_s)
        if not ic_provided_d:
            self.ic_constant_d = 0.0

    def _nernst(self, ci: float, co: float, valence: float) -> float:
        T_kelvin = self.celsius + 273.15
        return (
            1000.0
            * (self.R_const * T_kelvin)
            / (valence * self.F_const)
            * math.log(co / ci)
        )

    def _init_ic_values(self, V: float) -> Tuple[float, float]:
        cai0 = 1e-5

        minf = 1.0 / (1.0 + math.exp(-(V + 35.0) / 7.8))
        hinf = 1.0 / (1.0 + math.exp((V + 55.0) / 7.0))
        ninf = 1.0 / (1.0 + math.exp(-(V + 28.0) / 15.0))
        mnSinf = 1.0 / (1.0 + math.exp((V + 30.0) / -5.0))
        hnSinf = 1.0 / (1.0 + math.exp((V + 45.0) / 5.0))
        mlinf = 1.0 / (1.0 + math.exp((V + 40.0) / -7.0))

        fN = ((36.0 / 293.15) * (self.celsius + 273.15)) / 2.0
        fL = ((25.0 / 293.15) * (self.celsius + 273.15)) / 2.0

        def _ghk(v, ci, co, factor):
            nu = v / factor
            if abs(nu) < 1e-4:
                efun = 1.0 - nu / 2.0
            else:
                efun = nu / (math.exp(nu) - 1.0)
            return -factor * (1.0 - (ci / co) * math.exp(nu)) * efun

        ghk_s = _ghk(V, cai0, self.cao, fN)
        ghk_dN = _ghk(V, cai0, self.cao, fN)
        ghk_dL = _ghk(V, cai0, self.cao, fL)

        INaS = self.GNa * minf**3 * hinf * (V - self.ENa)
        IKS = (self.GK_dr * ninf**4 + self.GK_CaS * cai0 / (cai0 + self.Kd)) * (
            V - self.EK
        )
        ICaS = self.GCa_NS * mnSinf**2 * hnSinf * ghk_s
        IleakS = self.gleak * (V - self.Eleak)

        IKD = self.GK_CaD * cai0 / (cai0 + self.Kd) * (V - self.EK)
        ICaD_N = self.GCa_ND * mnSinf**2 * hnSinf * ghk_dN
        ICaD_L = self.GCa_L * mlinf * ghk_dL
        IleakD = self.gleak_d * (V - self.Eleak)

        ic_s = -(INaS + IKS + ICaS + IleakS)
        ic_d = -(IKD + ICaD_N + ICaD_L + IleakD)
        return ic_s, ic_d

    def _geometry(self) -> Tuple[Array, Array, Array, Array]:
        diam_cm = self.global_diam * 1e-4
        L_total_cm = self.Ltotal * 1e-4
        area_total = jnp.pi * diam_cm * L_total_cm
        area_s = self.p * area_total
        area_d = (1.0 - self.p) * area_total

        g_c_s = 2.0 * self.gc * (1.0 - self.p) / self.p * 1e-3
        g_c_d = 2.0 * self.gc * 1e-3

        return area_s, area_d, g_c_s, g_c_d

    def _opsin_derivatives(self, C, O, phi):  # noqa: E741
        """RhO3c 3-state Markov derivatives (Nikolic et al. 2009)"""
        D = 1.0 - C - O
        Hp = phi**self.opsin_p / (phi**self.opsin_p + self.opsin_phi_m**self.opsin_p)
        Hq = phi**self.opsin_q / (phi**self.opsin_q + self.opsin_phi_m**self.opsin_q)
        Ga = self.opsin_k_a * Hp
        Gr = self.opsin_Gr0 + self.opsin_k_r * Hq
        dC = Gr * D - Ga * C
        dO = Ga * C - self.opsin_Gd * O
        return dC, dO

    def _opsin_current(self, O, V):  # noqa: E741
        """Opsin photocurrent in nA, matching NEURON RhO3c.mod exactly.

        Uses g0 in pS, same formula and units as the .mod file:
            i = g0 * fphi * fv * (v - E) * 1e-6   [nA]
        """
        fphi = O
        fv = (
            self.opsin_v1
            * (1 - jnp.exp(-(V - self.opsin_E_rev) / self.opsin_v0))
            / (V - self.opsin_E_rev)
        )
        return self.opsin_g0 * fphi * fv * (V - self.opsin_E_rev) * 1e-6  # nA

    def _calculate_membrane_currents(self, t, y, args=None):
        """Calculate per-compartment transmembrane currents in microamperes.

        Mirrors NEURON's i_membrane_ (for cvode.use_fast_imem(1)),
        which is the total transmembrane current per segment:

            I_trans = I_ionic + I_cap          [mA/cm^2]
                    = I_ionic + C_m * dV/dt

        Args:
            t: time
            y: state vector (full, including any opsin states)
            args: (I_stim_array, t_array) for array stimulus

        Returns:
            (I_mem_soma_uA, I_mem_dend_uA) per-source currents matching
            the units expected by IO.potential_recording
        """
        area_s, area_d, _g_c_s, _g_c_d = self._geometry()

        Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD = y[:11]

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

        # Fast sodium activation (instantaneous)
        minf = 1.0 / (1.0 + jnp.exp(-(Vs + 35.0) / 7.8))

        INaS = self.GNa * minf**3 * h * (Vs - self.ENa)
        IKS = (self.GK_dr * n**4 + self.GK_CaS * CaS / (CaS + self.Kd)) * (Vs - self.EK)
        ICaS = self.GCa_NS * mnS**2 * hnS * ghk_soma
        IleakS = self.gleak * (Vs - self.Eleak)

        IKD = self.GK_CaD * CaD / (CaD + self.Kd) * (Vd - self.EK)
        ICaD_N = self.GCa_ND * mnD**2 * hnD * ghk_dend_N
        ICaD_L = self.GCa_L * ml * ghk_dend_L
        IleakD = self.gleak_d * (Vd - self.Eleak)

        I_ionic_soma = INaS + IKS + ICaS + IleakS
        I_ionic_dend = IKD + ICaD_N + ICaD_L + IleakD

        dy = self.__call__(t, y, args)
        dVs = dy[0]
        dVd = dy[1]
        Cs = self.C * self.cm_ratio
        Cd = self.C
        I_cap_soma = (Cs / 1000.0) * dVs  # mV/ms -> mA/cm^2
        I_cap_dend = (Cd / 1000.0) * dVd

        I_mem_soma = I_ionic_soma + I_cap_soma
        I_mem_dend = I_ionic_dend + I_cap_dend

        # Convert to per-source uA: area_*_cm2 * 1000 mA -> uA per cm2
        I_mem_soma_uA = I_mem_soma * area_s * 1000.0
        I_mem_dend_uA = I_mem_dend * area_d * 1000.0

        return I_mem_soma_uA, I_mem_dend_uA

    def __call__(self, t, y: Array, args) -> Array:
        """Vector field for the Booth-Rinzel-Kiehn model

        State vector: [Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD, ...opsin_states]

        When input_mode='irradiance', the state vector is extended with opsin
        Markov states (2 extra for RhO3c: C, O).

        Args:
            t: time
            y: state vector (11 neural + n_opsin_states)
            args: (I_stim_array, t_array) for array stimulus
        """

        area_s, area_d, g_c_s, g_c_d = self._geometry()

        # Split neural and opsin states
        y_neural = y[:11]
        Vs, Vd, h, n, mnS, hnS, mnD, hnD, ml, CaS, CaD = y_neural

        I_stim_array, t_array = args

        if self.input_mode == "irradiance":
            # I_stim_array is irradiance in mW/mm²
            Irr = jnp.interp(t, t_array, I_stim_array.ravel())
            phi = Irr / self.opsin_E_photon

            # Unpack RhO3c opsin states from extended state vector
            C = y[11]
            O = y[12]  # noqa: E741

            i_opsin_nA = self._opsin_current(O, Vs)
            Iapp_density_soma = -i_opsin_nA * 1e-6 / area_s  # nA → mA/cm²
            Iapp_density_dend = 0.0

        elif self.input_mode == "current_density":
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
        IleakD = self.gleak_d * (Vd - self.Eleak)

        IcouplingS = g_c_s * (Vs - Vd)
        IcouplingD = g_c_d * (Vd - Vs)

        # NEURON BRK template scales the soma capacitance by cm_ratio;
        # use separate scales for soma and dendrite to match
        scale_s = 1000.0 / (self.C * self.cm_ratio)
        scale_d = 1000.0 / self.C

        # ic_constant follows NEURON's outward-positive ionic-current
        # convention (it is registered as an electrode current in the same
        # sign as i_pas / i_ions), whereas Isoma/Idend follow the
        # inward-positive injected-current convention used by Iapp. Hence
        # the minus sign, matching the existing ``- INaS - ...`` terms
        dVs = scale_s * (
            Isoma - self.ic_constant - INaS - IKS - ICaS - IleakS - IcouplingS
        )
        dVd = scale_d * (
            Idend - self.ic_constant_d - IKD - ICaD_N - ICaD_L - IleakD - IcouplingD
        )

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

        dy_neural = jnp.array(
            [dVs, dVd, dh, dn, dmnS, dhnS, dmnD, dhnD, dml, dCaS, dCaD]
        )

        if self.input_mode == "irradiance":
            dC, dO = self._opsin_derivatives(C, O, phi)
            return jnp.concatenate([dy_neural, jnp.array([dC, dO])])

        return dy_neural

    def solve(
        self,
        t_dur: float,
        *,
        I_stim_array: Array,
        y0: Optional[Array] = None,
        dt: float = 0.025,
        dt_solver: Optional[float] = None,
        **kwargs,
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
            - "irradiance": Light intensity in mW/mm^2

            Shape can be either:
            - 1D array [n_time_points] for soma-only stimulation
            - 2D array [n_time_points, 2] for soma (column 0) and dendrite (column 1)
            where n_time_points = t_dur/dt + 1
        y0 : Array, optional
             Initial state vector; if None, defaults to resting state
        dt : float
            Time step in ms
        dt_solver : float, optional
             Internal time step for the solver; if None, defaults to dt

        Returns:
        --------
        (time_array, soma_voltage, dendrite_voltage, soma_membrane_current, dendrite_membrane_current, yT)
        """
        if dt_solver is None:
            dt_solver = dt

        if y0 is None:
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

        # Extend y0 with RhO3c opsin initial states for irradiance mode (C=1, O=0)
        if self.input_mode == "irradiance":
            y0 = jnp.concatenate([y0, jnp.array([1.0, 0.0])])

        term = diffrax.ODETerm(self)

        solver = self.solver
        if isinstance(solver, str):
            solver = getattr(diffrax, solver)()

        n_points = int(t_dur / dt) + 1
        t_array = jnp.linspace(0.0, t_dur, n_points)
        stimulus_args = (I_stim_array, t_array)

        saveat = diffrax.SaveAt(ts=t_array)

        # use constant step size for pmap compatibility
        stepsize_controller = diffrax.ConstantStepSize()

        scaled_max_steps = int(self.max_steps * (t_dur / 1000.0))
        scaled_max_steps = max(scaled_max_steps, 10000)

        kwargs.pop("unroll", None)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_dur,
            dt0=dt_solver,
            y0=y0,
            args=stimulus_args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=scaled_max_steps,
            **kwargs,
        )

        t_arr = solution.ts
        ys_neural = solution.ys[:, :11]
        v_soma = ys_neural[:, 0]
        v_dend = ys_neural[:, 1]

        def calculate_currents_at_time(t, y):
            return self._calculate_membrane_currents(t, y, stimulus_args)

        i_mem_soma, i_mem_dend = jax.vmap(calculate_currents_at_time)(
            t_arr, solution.ys
        )

        return t_arr, v_soma, v_dend, i_mem_soma, i_mem_dend, solution.ys[-1]
