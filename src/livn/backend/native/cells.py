from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from livn.backend.native import _lib as L
from livn.models.rcsd.neuron.templates import axon as _axon
from livn.types import Cell

SWC_SOMA = 1
SWC_AXON = 2

NEURON_DEFAULT_RA = 35.4
NEURON_DEFAULT_CM = 1.0
NAS_VHALF = -35.0
NAS_SLOPE = 7.8

AXON_MECHANISMS = L.M_PAS | L.M_CONSTANT | L.M_NA_CONC | L.M_K_CONC | L.M_NAS | L.M_KDR
BRK_SOMA_MECHANISMS = (
    L.M_PAS
    | L.M_CONSTANT
    | L.M_NA_CONC
    | L.M_K_CONC
    | L.M_CA_CONC
    | L.M_NAS
    | L.M_KDR
    | L.M_CAN
    | L.M_KCA
)
BRK_DEND_MECHANISMS = (
    L.M_PAS | L.M_CONSTANT | L.M_CA_CONC | L.M_K_CONC | L.M_CAN | L.M_CAL | L.M_KCA
)
V1IN_SOMA_MECHANISMS = (
    L.M_PAS
    | L.M_CONSTANT
    | L.M_NA_CONC
    | L.M_K_CONC
    | L.M_CA_CONC
    | L.M_NAS
    | L.M_KDR
    | L.M_KCA
    | L.M_CAN
    | L.M_KA_V1IN
)

# NEURON's suffixed range-variable names -> (parameter id, mechanism it needs)
PARAMETER_NAMES: dict[str, tuple[int, int]] = {
    "cm": (L.P_CM, 0),
    "g_pas": (L.P_G_PAS, L.M_PAS),
    "e_pas": (L.P_E_PAS, L.M_PAS),
    "ic_constant": (L.P_IC, L.M_CONSTANT),
    "d_Na_conc": (L.P_D_NA, L.M_NA_CONC),
    "beta_Na_conc": (L.P_BETA_NA, L.M_NA_CONC),
    "nai0_Na_conc": (L.P_NAI0, L.M_NA_CONC),
    "nao0_Na_conc": (L.P_NAO0, L.M_NA_CONC),
    "d_K_conc": (L.P_D_K, L.M_K_CONC),
    "beta_K_conc": (L.P_BETA_K, L.M_K_CONC),
    "ki0_K_conc": (L.P_KI0, L.M_K_CONC),
    "ko0_K_conc": (L.P_KO0, L.M_K_CONC),
    "f_Ca_conc": (L.P_F_CA, L.M_CA_CONC),
    "alpha_Ca_conc": (L.P_ALPHA_CA, L.M_CA_CONC),
    "kCa_Ca_conc": (L.P_KCA_CA, L.M_CA_CONC),
    "gmax_Nas": (L.P_GMAX_NAS, L.M_NAS),
    "vhalf_Nas": (L.P_VHALF_NAS, L.M_NAS),
    "slope_Nas": (L.P_SLOPE_NAS, L.M_NAS),
    "gmax_Kdr": (L.P_GMAX_KDR, L.M_KDR),
    "gmax_CaN": (L.P_GMAX_CAN, L.M_CAN),
    "gmax_CaL": (L.P_GMAX_CAL, L.M_CAL),
    "gmax_KCa": (L.P_GMAX_KCA, L.M_KCA),
    "gmax_Ka_v1in": (L.P_GMAX_KA, L.M_KA_V1IN),
}


@dataclass
class SectionSpec:
    name: str
    kind: int  # L.SEC_SOMA | L.SEC_DEND | L.SEC_AXON
    sec_type: str  # the weight-key section type: soma, dend or axon
    nseg: int
    L: float
    diam: float
    Ra: float
    cm: float
    mechanisms: int
    params: dict[int, float] = field(default_factory=dict)
    parent: int = -1
    parent_x: float = 0.0


def lambda_f(L_um: float, diam: float, Ra: float, cm: float, freq: float) -> float:
    """NEURON's lambda_f for a stylised section (no 3d points)."""
    return 1e5 * math.sqrt(diam / (4 * math.pi * freq * Ra * cm))


def d_lambda_nseg(
    L_um: float, diam: float, Ra: float, cm: float, freq=100, d_lambda=0.1
):
    lam = lambda_f(L_um, diam, Ra, cm, freq)
    return int((L_um / (d_lambda * lam) + 0.9) / 2) * 2 + 1


def segment_area(L_um: float, diam: float, nseg: int) -> float:
    """``h.area(0.5)``: the middle segment's membrane area in um2."""
    return math.pi * diam * (L_um / nseg)


def ri_at_half(L_um: float, diam: float, nseg: int, Ra: float) -> float:
    """``h.ri(0.5)``: NEURON's 1/NODERINV of the segment containing x = 0.5, MOhm."""
    half = 1e-2 * Ra * ((L_um / nseg) / 2.0) / (math.pi * diam * diam / 4.0)
    j = int(0.5 * nseg)
    if j >= nseg:
        j = nseg - 1
    return half if j == 0 else 2.0 * half


class _Template:
    """Shared plumbing: the last geometry each section had, for the rebuild path."""

    template_keys: tuple[str, ...] = ()
    axon_index: int = 1

    def __init__(self):
        self.axon_params = _axon.axon_parameters(None)
        self._sections: list[SectionSpec] = []

    @property
    def sections(self) -> list[SectionSpec]:
        return self._sections

    def section_types(self) -> list[str]:
        return [sec.sec_type for sec in self._sections]

    def has_template_param(self, name: str) -> bool:
        return name in self.template_keys or name in self.axon_params

    def set_template_param(self, name: str, value: float) -> None:
        if name in self.axon_params:
            self.axon_params[name] = value
        elif name in self.template_keys:
            setattr(self, name, value)
        else:
            raise KeyError(name)

    def _axon_sections(self, soma_index: int, e_pas: float, soma_gmax_na: float):
        n = int(self.axon_params.get("axon_segments", 0) or 0)
        if n <= 0:
            return []
        params = self.axon_params
        own = params.get("axon_e_pas")
        axon_e_pas = (
            float(own)
            if own is not None
            else _axon.balanced_e_pas(params, e_pas, soma_gmax_na)
        )
        gmax_na = _axon.sodium_density(params, soma_gmax_na)
        sections = []
        parent, x = soma_index, 0.0
        for index in range(n):
            sections.append(
                SectionSpec(
                    name=f"ais{index}",
                    kind=L.SEC_AXON,
                    sec_type="axon",
                    nseg=1,
                    L=float(params["axon_segment_um"]),
                    diam=float(params["axon_diam"]),
                    Ra=float(params["axon_Ra"]),
                    cm=float(params["axon_cm"]),
                    mechanisms=AXON_MECHANISMS,
                    params={
                        L.P_GMAX_NAS: gmax_na,
                        L.P_VHALF_NAS: NAS_VHALF,
                        L.P_SLOPE_NAS: NAS_SLOPE,
                        L.P_GMAX_KDR: float(params["axon_gmax_K"]),
                        L.P_G_PAS: float(params["axon_g_pas"]),
                        L.P_E_PAS: axon_e_pas,
                    },
                    parent=parent,
                    parent_x=x,
                )
            )
            parent, x = soma_index + 1 + index, 1.0
        return sections


class BRKTemplate(_Template):
    """``templates/BRK.py`` without NEURON."""

    template_keys = (
        "pp",
        "Ltotal",
        "gc",
        "global_diam",
        "global_cm",
        "cm_ratio",
        "global_e_pas",
        "soma_g_pas",
        "soma_gmax_Na",
        "soma_vhalf_Na",
        "soma_slope_Na",
        "soma_gmax_K",
        "soma_gmax_KCa",
        "soma_gmax_CaN",
        "soma_f_Caconc",
        "soma_alpha_Caconc",
        "soma_kCa_Caconc",
        "dend_g_pas",
        "dend_gmax_CaN",
        "dend_gmax_CaL",
        "dend_gmax_KCa",
        "dend_f_Caconc",
        "dend_alpha_Caconc",
        "dend_kCa_Caconc",
    )

    def __init__(self, params: dict | None = None):
        super().__init__()
        if params is not None:
            params = params.get("BoothRinzelKiehn", params)
        self.set_default_parameters()
        # the Ra/cm NEURON has on a fresh section, which the d_lambda rule sees
        self._geometry_ra = {"soma": NEURON_DEFAULT_RA, "dend": NEURON_DEFAULT_RA}
        self._geometry_cm = {"soma": NEURON_DEFAULT_CM, "dend": NEURON_DEFAULT_CM}
        self._nseg = {"soma": 1, "dend": 1}
        if params is not None:
            self.set_parameters(params)
        self.geometry()
        self.biophys()

    def set_default_parameters(self):
        self.pp = 0.5
        self.Ltotal = 400 / np.pi
        self.gc = 10.5
        self.global_e_pas = -60
        self.soma_g_pas = 0.0001
        self.soma_gmax_Na = 0.00030
        self.soma_vhalf_Na = NAS_VHALF
        self.soma_slope_Na = NAS_SLOPE
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
        self.global_diam = 10
        self.cm_ratio = 1
        self.axon_params = _axon.axon_parameters(None)

    def set_parameters(self, params):
        self.axon_params = _axon.axon_parameters(params)
        self.pp = params.get("pp", self.pp)
        self.Ltotal = params.get("Ltotal", self.Ltotal)
        self.gc = params.get("gc", self.gc)
        self.global_diam = params.get("global_diam", self.global_diam)
        self.global_cm = params.get("global_cm", self.global_cm)
        self.cm_ratio = params.get("cm_ratio", self.cm_ratio)
        self.global_e_pas = params.get("e_pas", -60)
        self.soma_g_pas = params.get("soma_g_pas", self.soma_g_pas)
        self.soma_gmax_Na = params.get("soma_gmax_Na", self.soma_gmax_Na)
        self.soma_vhalf_Na = params.get("soma_vhalf_Na", NAS_VHALF)
        self.soma_slope_Na = params.get("soma_slope_Na", NAS_SLOPE)
        self.soma_gmax_K = params.get("soma_gmax_K", self.soma_gmax_K)
        self.soma_gmax_KCa = params.get("soma_gmax_KCa", self.soma_gmax_KCa)
        self.soma_gmax_CaN = params.get("soma_gmax_CaN", self.soma_gmax_CaN)
        self.soma_f_Caconc = params.get("soma_f_Caconc", self.soma_f_Caconc)
        self.soma_alpha_Caconc = params.get("soma_alpha_Caconc", self.soma_alpha_Caconc)
        self.soma_kCa_Caconc = params.get("soma_kCa_Caconc", self.soma_kCa_Caconc)
        self.dend_g_pas = params.get("dend_g_pas", self.dend_g_pas)
        self.dend_gmax_CaN = params.get("dend_gmax_CaN", self.dend_gmax_CaN)
        self.dend_gmax_CaL = params.get("dend_gmax_CaL", self.dend_gmax_CaL)
        self.dend_gmax_KCa = params.get("dend_gmax_KCa", self.dend_gmax_KCa)
        self.dend_f_Caconc = params.get("dend_f_Caconc", self.dend_f_Caconc)
        self.dend_alpha_Caconc = params.get("dend_alpha_Caconc", self.dend_alpha_Caconc)
        self.dend_kCa_Caconc = params.get("dend_kCa_Caconc", self.dend_kCa_Caconc)

    def geometry(self):
        self._L = {"soma": self.pp * self.Ltotal, "dend": (1 - self.pp) * self.Ltotal}
        self._diam = {"soma": self.global_diam, "dend": self.global_diam}
        for name in ("soma", "dend"):
            self._nseg[name] = d_lambda_nseg(
                self._L[name],
                self._diam[name],
                self._geometry_ra[name],
                self._geometry_cm[name],
            )

    def _coupling_ra(self, name: str) -> float:
        """``biophys()``: Ra such that the half-section conductance is gc/pp per area."""
        L_um, diam, nseg = self._L[name], self._diam[name], self._nseg[name]
        area_cm2 = segment_area(L_um, diam, nseg) * 1e-8
        return (1e-6 / (self.gc / self.pp * area_cm2 * 1e-3)) / (
            2 * ri_at_half(L_um, diam, nseg, 1.0)
        )

    def biophys(self):
        ra = {name: self._coupling_ra(name) for name in ("soma", "dend")}
        cm = {"soma": self.global_cm * self.cm_ratio, "dend": self.global_cm}
        self._geometry_ra = dict(ra)
        self._geometry_cm = dict(cm)
        soma = SectionSpec(
            name="soma",
            kind=L.SEC_SOMA,
            sec_type="soma",
            nseg=self._nseg["soma"],
            L=self._L["soma"],
            diam=self._diam["soma"],
            Ra=ra["soma"],
            cm=cm["soma"],
            mechanisms=BRK_SOMA_MECHANISMS,
            params={
                L.P_GMAX_NAS: self.soma_gmax_Na,
                L.P_VHALF_NAS: self.soma_vhalf_Na,
                L.P_SLOPE_NAS: self.soma_slope_Na,
                L.P_GMAX_KDR: self.soma_gmax_K,
                L.P_GMAX_CAN: self.soma_gmax_CaN,
                L.P_GMAX_KCA: self.soma_gmax_KCa,
                L.P_F_CA: self.soma_f_Caconc,
                L.P_ALPHA_CA: self.soma_alpha_Caconc,
                L.P_KCA_CA: self.soma_kCa_Caconc,
                L.P_G_PAS: self.soma_g_pas,
                L.P_E_PAS: self.global_e_pas,
            },
            parent=-1,
        )
        dend = SectionSpec(
            name="dend",
            kind=L.SEC_DEND,
            sec_type="dend",
            nseg=self._nseg["dend"],
            L=self._L["dend"],
            diam=self._diam["dend"],
            Ra=ra["dend"],
            cm=cm["dend"],
            mechanisms=BRK_DEND_MECHANISMS,
            params={
                L.P_F_CA: self.dend_f_Caconc,
                L.P_ALPHA_CA: self.dend_alpha_Caconc,
                L.P_KCA_CA: self.dend_kCa_Caconc,
                L.P_G_PAS: self.dend_g_pas,
                L.P_E_PAS: self.global_e_pas,
                L.P_GMAX_CAN: self.dend_gmax_CaN,
                L.P_GMAX_CAL: self.dend_gmax_CaL,
                L.P_GMAX_KCA: self.dend_gmax_KCa,
            },
            parent=0,
            parent_x=1.0,
        )
        # the axon chain attaches to soma(0); its parent indices skip the dendrite
        axon = self._axon_sections(0, self.global_e_pas, self.soma_gmax_Na)
        for index, sec in enumerate(axon):
            sec.parent = 0 if index == 0 else 1 + index
        self._sections = [soma, dend, *axon]


class V1InTemplate(_Template):
    """``templates/V1In.py`` without NEURON."""

    template_keys = (
        "global_diam",
        "global_cm",
        "global_e_pas",
        "soma_g_pas",
        "soma_gmax_Na",
        "soma_vhalf_Na",
        "soma_slope_Na",
        "soma_gmax_K",
        "soma_gmax_KCa",
        "soma_gmax_CaN",
        "soma_gmax_Ka",
        "soma_f_Caconc",
        "soma_alpha_Caconc",
        "soma_kCa_Caconc",
    )

    def __init__(self, params: dict | None = None):
        super().__init__()
        self.set_default_parameters()
        if params is not None:
            self.set_parameters(params)
        self.geometry()
        self.biophys()

    def set_default_parameters(self):
        self.global_diam = 23.0
        self.global_cm = 0.9
        self.global_e_pas = -65.0
        self.soma_g_pas = 6.0e-5
        self.soma_gmax_Na = 0.15
        self.soma_vhalf_Na = -35.0
        self.soma_slope_Na = 7.8
        self.soma_gmax_K = 0.06
        self.soma_gmax_KCa = 5.0e-5
        self.soma_gmax_CaN = 0.0
        self.soma_gmax_Ka = 0.0
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
        # L = diam, one segment, cm set here; Ra stays NEURON's default
        pass

    def biophys(self):
        soma = SectionSpec(
            name="soma",
            kind=L.SEC_SOMA,
            sec_type="soma",
            nseg=1,
            L=float(self.global_diam),
            diam=float(self.global_diam),
            Ra=NEURON_DEFAULT_RA,
            cm=float(self.global_cm),
            mechanisms=V1IN_SOMA_MECHANISMS,
            params={
                L.P_GMAX_NAS: self.soma_gmax_Na,
                L.P_VHALF_NAS: self.soma_vhalf_Na,
                L.P_SLOPE_NAS: self.soma_slope_Na,
                L.P_GMAX_KDR: self.soma_gmax_K,
                L.P_GMAX_KCA: self.soma_gmax_KCa,
                L.P_GMAX_CAN: self.soma_gmax_CaN,
                L.P_GMAX_KA: self.soma_gmax_Ka,
                L.P_F_CA: self.soma_f_Caconc,
                L.P_ALPHA_CA: self.soma_alpha_Caconc,
                L.P_KCA_CA: self.soma_kCa_Caconc,
                L.P_G_PAS: self.soma_g_pas,
                L.P_E_PAS: self.global_e_pas,
            },
            parent=-1,
        )
        axon = self._axon_sections(0, self.global_e_pas, self.soma_gmax_Na)
        for index, sec in enumerate(axon):
            sec.parent = 0 if index == 0 else index
        self._sections = [soma, *axon]


TEMPLATES = {"BoothRinzelKiehn": BRKTemplate, "BRK": BRKTemplate, "V1In": V1InTemplate}


def push_section_params(lib, sim, section: int, spec: SectionSpec) -> None:
    for param, value in spec.params.items():
        L.check(lib.rcsd_section_set(sim, section, param, float(value)), lib)


class NativeCell(Cell):
    """Parameter handle over one cell in ``librcsd``.

    Parameters are addressed as ``"<section type>.<name>"`` under NEURON's
    suffixed names (``soma.g_pas``, ``dend.gmax_CaN``, ``axon.Ra``), and the
    template's own parameters (``global_diam``, ``axon_diam`` ...) rebuild the
    cell's geometry and biophysics the way the NEURON handle does.
    """

    def __init__(
        self,
        env,
        population: str,
        gid: int,
        index: int,
        template,
        sections,
        soma_type: str = "soma",
        dend_type: str = "dend",
    ):
        super().__init__(env, population, gid)
        self._index = int(index)
        self._template = template
        self._sections = list(sections)  # C section ids, in template order
        self._soma_type = soma_type
        self._dend_type = dend_type
        kinds = [spec.kind for spec in template.sections]
        self._soma_index = kinds.index(L.SEC_SOMA) if L.SEC_SOMA in kinds else 0
        self._dend_index = (
            kinds.index(L.SEC_DEND) if L.SEC_DEND in kinds else self._soma_index
        )

    @property
    def index(self) -> int:
        return self._index

    @property
    def template(self):
        return self._template

    @property
    def sections(self) -> list[int]:
        return self._sections

    @property
    def section_names(self) -> list[str]:
        return [sec.name for sec in self._template.sections]

    @property
    def threshold(self) -> float:
        return float(self._env._thresholds[self._index])

    def place(self, swc_type: int, loc: float) -> tuple[int, float]:
        """(section index, x) a synapse of SWC type lands on, as ReducedCell.place."""
        if swc_type == SWC_SOMA:
            return self._soma_index, 0.5
        if loc < 0.0:
            loc = 0.0
        elif loc > 1.0:
            loc = 1.0
        return self._dend_index, float(loc)

    def dest_sec_type(self, swc_type: int) -> str:
        if swc_type == SWC_SOMA:
            return self._soma_type
        if swc_type == SWC_AXON:
            return "axon"
        return self._dend_type

    def sections_by_type(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for spec, section in zip(self._template.sections, self._sections, strict=True):
            groups.setdefault(spec.sec_type, []).append(section)
        return groups

    def _lib(self):
        return self._env._lib, self._env._sim

    def get_params(self) -> dict[str, float]:
        lib, sim = self._lib()
        params: dict[str, float] = {}
        for sec_type, sections in self.sections_by_type().items():
            section = sections[0]
            spec = self._template.sections[self._sections.index(section)]
            params[f"{sec_type}.cm"] = float(lib.rcsd_section_get(sim, section, L.P_CM))
            params[f"{sec_type}.Ra"] = self._ra(section)
            for name, (param, needs) in PARAMETER_NAMES.items():
                if name == "cm":
                    continue
                if needs and not (spec.mechanisms & needs):
                    continue
                params[f"{sec_type}.{name}"] = float(
                    lib.rcsd_section_get(sim, section, param)
                )
        return params

    def _ra(self, section: int) -> float:
        lib, sim = self._lib()
        ra = L.c_double()
        L.check(lib.rcsd_section_info(sim, section, None, None, None, ra, None), lib)
        return float(ra.value)

    def set_params(self, params: dict[str, float]):
        lib, sim = self._lib()
        groups = self.sections_by_type()
        rebuild: dict[str, float] = {}
        for key, value in params.items():
            sec_type, _, name = key.partition(".")
            if not name:
                if self._template.has_template_param(key):
                    rebuild[key] = float(value)
                    continue
                raise KeyError(
                    f"{key!r} is not a cell parameter; expected "
                    f"'<section type>.<name>', e.g. 'soma.g_pas', or a template "
                    f"parameter such as 'global_diam'"
                )
            sections = groups.get(sec_type)
            if sections is None:
                raise self.unknown_param(key, self.get_params())
            if name == "Ra":
                for section in sections:
                    info = self._geometry(section)
                    L.check(
                        lib.rcsd_section_geometry(
                            sim, section, info[0], info[1], float(value)
                        ),
                        lib,
                    )
                continue
            found = PARAMETER_NAMES.get(name)
            if found is None:
                raise self.unknown_param(key, self.get_params())
            param, needs = found
            for section in sections:
                spec = self._template.sections[self._sections.index(section)]
                if needs and not (spec.mechanisms & needs):
                    raise self.unknown_param(key, self.get_params())
                L.check(lib.rcsd_section_set(sim, section, param, float(value)), lib)
        if rebuild:
            self._rebuild_template(rebuild)
        return self._env

    def _geometry(self, section: int) -> tuple[float, float, float]:
        lib, sim = self._lib()
        L_um, diam, ra = L.c_double(), L.c_double(), L.c_double()
        L.check(lib.rcsd_section_info(sim, section, None, L_um, diam, ra, None), lib)
        return float(L_um.value), float(diam.value), float(ra.value)

    def _rebuild_template(self, params: dict[str, float]) -> None:
        lib, sim = self._lib()
        template = self._template
        for name, value in params.items():
            template.set_template_param(name, value)
        template.geometry()
        template.biophys()
        for spec, section in zip(template.sections, self._sections, strict=True):
            nseg = L.c_int()
            L.check(
                lib.rcsd_section_info(sim, section, nseg, None, None, None, None), lib
            )
            if int(nseg.value) != spec.nseg:
                raise NotImplementedError(
                    f"{name!r} changes the number of segments of {spec.name} from "
                    f"{nseg.value} to {spec.nseg}, which the native backend cannot "
                    "do after init(); construct a new env"
                )
            L.check(
                lib.rcsd_section_geometry(sim, section, spec.L, spec.diam, spec.Ra), lib
            )
            L.check(lib.rcsd_section_set(sim, section, L.P_CM, spec.cm), lib)
            push_section_params(lib, sim, section, spec)


def build_cell(
    lib, sim, gid: int, population_code: int, template, threshold, v_hold, tref
):
    """Add one template cell to the sim and return (cell index, section ids)."""
    cell = L.check(
        lib.rcsd_add_cell(
            sim,
            int(gid),
            int(population_code),
            float(threshold),
            float(v_hold),
            float(tref),
        ),
        lib,
    )
    sections: list[int] = []
    for spec in template.sections:
        parent = -1 if spec.parent < 0 else sections[spec.parent]
        section = L.check(
            lib.rcsd_add_section(
                sim,
                cell,
                spec.kind,
                spec.nseg,
                float(spec.L),
                float(spec.diam),
                float(spec.Ra),
                float(spec.cm),
                spec.mechanisms,
                parent,
                float(spec.parent_x),
            ),
            lib,
        )
        push_section_params(lib, sim, section, spec)
        sections.append(section)
    return cell, sections
