from __future__ import annotations

import math
from bisect import bisect_right
from typing import Protocol, runtime_checkable

import numpy as np

from livn.types import Cell

# SWC type codes (Cannon et al. convention, matching the neuroh5 attributes).
SWC_SOMA = 1
SWC_AXON = 2
SWC_BASAL = 3
SWC_APICAL = 4


@runtime_checkable
class NeuronCell(Protocol):
    """Runtime contract every backend cell must satisfy."""

    sections: list  # ordered list of h.Section; index is canonical
    section_type: np.ndarray  # int8[len(sections)] SWC code per section
    threshold: float  # spike-detector threshold (mV)

    def place(self, swc_type: int, loc: float): ...  # -> h.Segment
    def dest_sec_type(self, swc_type: int) -> str: ...  # weight-key section name
    def spike_source(self): ...  # -> h.Segment the detector watches
    def position(self, x: float, y: float, z: float) -> None: ...
    def init_ic(self, v_rest: float) -> None: ...  # optional resting-current pin
    def resting_potential(self) -> float | None: ...
    def measure_ic(self) -> None: ...
    def axial_couplings(self) -> list: ...


def segment_at(sec, x: float):
    n = int(sec.nseg)
    index = min(n - 1, max(0, int(float(x) * n)))
    return sec((index + 0.5) / n)


def half_axial_resistance(seg) -> float:
    """Axial resistance (MOhm) of half a segment, along its own length."""
    sec = seg.sec
    half_length_cm = (float(sec.L) / int(sec.nseg)) * 1e-4 / 2.0
    radius_cm = float(seg.diam) * 1e-4 / 2.0
    if radius_cm <= 0.0:
        return 0.0
    return float(sec.Ra) * half_length_cm / (math.pi * radius_cm**2) * 1e-6


def axial_couplings(sections) -> list[tuple[int, object, int, object, float]]:
    by_name = {sec.name(): i for i, sec in enumerate(sections)}
    found = []
    for child_index, sec in enumerate(sections):
        parent = sec.parentseg()
        if parent is None:
            continue
        parent_index = by_name.get(parent.sec.name())
        if parent_index is None:
            continue  # parent is outside this cell
        child_seg = segment_at(sec, 0.0)
        parent_seg = segment_at(parent.sec, parent.x)
        resistance = half_axial_resistance(child_seg) + half_axial_resistance(
            parent_seg
        )
        if resistance <= 0.0:
            continue
        found.append((child_index, child_seg, parent_index, parent_seg, resistance))
    return found


class ReducedCell:
    """Adapter wrapping a reduced (few-section) template as a ``NeuronCell``.

    Classifies the template's sections by NEURON name where the section whose name
    contains ``soma`` is the soma and the first remaining section is the dendrite.
    ``place`` routes soma-type synapses to the soma and everything else to the
    dendrite, so rich-morphology placement data collapses cleanly onto two
    compartments. Destination section-type names (``soma_type``/``dend_type``)
    are declared explicitly by the model factory so weight keys can select on them
    """

    def __init__(
        self,
        template,
        threshold: float,
        v_rest: float | None = None,
        soma_type: str = "soma",
        dend_type: str = "dend",
        sec_types: dict[int, str] | None = None,
    ):
        self._template = template
        self.threshold = float(threshold)
        self._v_rest = v_rest
        self._soma_type = soma_type
        self._dend_type = dend_type
        self._sec_types = dict(sec_types) if sec_types else {}

        self.sections = list(template.sections)
        if not self.sections:
            raise ValueError("ReducedCell: template exposes no sections")

        axon = {id(sec) for sec in (getattr(template, "axon", None) or [])}

        types = np.empty(len(self.sections), dtype=np.int8)
        self._soma = None
        self._dend = None
        for i, sec in enumerate(self.sections):
            name = sec.name().split(".")[-1].lower()
            if "soma" in name:
                types[i] = SWC_SOMA
                if self._soma is None:
                    self._soma = sec
            elif id(sec) in axon:
                types[i] = SWC_AXON
            else:
                types[i] = SWC_BASAL
                if self._dend is None:
                    self._dend = sec
        if self._soma is None:
            self._soma = self.sections[0]
            types[0] = SWC_SOMA
        if self._dend is None:
            self._dend = self._soma
        self.section_type = types

    def place(self, swc_type: int, loc: float):
        if swc_type == SWC_SOMA:
            return self._soma(0.5)
        if loc < 0.0:
            loc = 0.0
        elif loc > 1.0:
            loc = 1.0
        return segment_at(self._dend, loc)

    def dest_sec_type(self, swc_type: int) -> str:
        if swc_type == SWC_SOMA:
            default = self._soma_type
        elif swc_type == SWC_AXON:
            default = _MORPH_SECTYPE_NAMES[SWC_AXON]
        else:
            default = self._dend_type
        return self._sec_types.get(swc_type, default)

    def spike_source(self):
        return self._soma(0.5)

    def position(self, x: float, y: float, z: float) -> None:
        self._template.position(x, y, z)

    def init_ic(self, v_rest: float | None = None) -> None:
        fn = getattr(self._template, "init_ic", None)
        if callable(fn):
            fn(self._v_rest if v_rest is None else v_rest)

    def resting_potential(self) -> float | None:
        """The potential this cell pins its resting current at."""
        return None if self._v_rest is None else float(self._v_rest)

    def axial_couplings(self):
        """Junctions between this cell's sections."""
        return axial_couplings(self.sections)

    def measure_ic(self) -> None:
        """Read this cell's currents and pin them, without initializing."""
        fn = getattr(self._template, "measure_ic", None)
        if callable(fn):
            fn()

    @property
    def template(self):
        return self._template


# SWC code -> weight-key section-type name for full-morphology cells.
_MORPH_SECTYPE_NAMES = {
    SWC_SOMA: "soma",
    SWC_AXON: "axon",
    SWC_BASAL: "basal",
    SWC_APICAL: "apical",
}

_CONFIG_SECTION_SWC = {
    "soma": SWC_SOMA,
    "axon": SWC_AXON,
    "basal": SWC_BASAL,
    "apical": SWC_APICAL,
    "dend": SWC_APICAL,
}


CONFIG_SECTION_NAMES = frozenset(_CONFIG_SECTION_SWC)


def config_section_swc(name: str) -> int:
    """The SWC code a graph config's section name places synapses at."""
    return _CONFIG_SECTION_SWC.get(str(name).lower(), SWC_APICAL)


class MorphologyCell:
    """Adapter for a full-morphology template as a ``NeuronCell``.

    Placement is morphology-independent as synapses are routed by ``swc_type``
    onto the group of sections of that type, and ``loc in [0, 1]`` selects a
    position along that group's cumulative arc length. This decouples wiring
    from any generator's exact section indexing so a cell rebuilt with different
    ``nseg`` or section splits still places synapses at the same relative
    dendritic position.
    """

    @classmethod
    def from_template(cls, template, threshold: float, v_rest: float | None = None):
        def collect(*attrs) -> list:
            out: list = []
            for attr in attrs:
                lst = getattr(template, attr, None)
                if lst is not None:
                    out.extend(list(lst))
            return out

        swc_sections = {
            SWC_SOMA: collect("soma_list"),
            SWC_APICAL: collect("apical_list"),
            SWC_BASAL: collect("basal_list"),
            SWC_AXON: collect("axon_list", "hillock_list", "ais_list"),
        }
        swc_sections = {k: v for k, v in swc_sections.items() if v}
        return cls(template, threshold, v_rest, swc_sections)

    def __init__(self, template, threshold, v_rest, swc_sections):
        self._template = template
        self.threshold = float(threshold)
        self._v_rest = v_rest
        self.sections = list(template.sections)

        sec_to_swc: dict[str, int] = {}
        for swc, secs in swc_sections.items():
            for s in secs:
                sec_to_swc[s.name()] = swc
        self.section_type = np.array(
            [sec_to_swc.get(s.name(), SWC_APICAL) for s in self.sections],
            dtype=np.int8,
        )

        self._groups: dict[int, tuple] = {}
        for swc, secs in swc_sections.items():
            if not secs:
                continue
            lengths = [float(s.L) for s in secs]
            starts, acc = [], 0.0
            for length in lengths:
                starts.append(acc)
                acc += length
            self._groups[swc] = (secs, starts, lengths, acc)

        self._soma = getattr(template, "soma", None) or self.sections[0]

    def place(self, swc_type: int, loc: float):
        grp = self._groups.get(swc_type)
        if grp is None or grp[3] <= 0.0:
            return self._soma(0.5)
        secs, starts, lengths, total = grp
        if loc < 0.0:
            loc = 0.0
        elif loc > 1.0:
            loc = 1.0
        target = loc * total
        i = bisect_right(starts, target) - 1
        if i < 0:
            i = 0
        elif i >= len(secs):
            i = len(secs) - 1
        seg_len = lengths[i]
        x = (target - starts[i]) / seg_len if seg_len > 0.0 else 0.5
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        return segment_at(secs[i], x)

    def dest_sec_type(self, swc_type: int) -> str:
        return _MORPH_SECTYPE_NAMES.get(swc_type, "apical")

    def spike_source(self):
        return self._soma(0.5)

    def position(self, x: float, y: float, z: float) -> None:
        fn = getattr(self._template, "position", None)
        if callable(fn):
            fn(x, y, z)

    def init_ic(self, v_rest: float | None = None) -> None:
        fn = getattr(self._template, "init_ic", None)
        if callable(fn):
            fn(self._v_rest if v_rest is None else v_rest)

    def resting_potential(self) -> float | None:
        """The potential this cell pins its resting current at."""
        return None if self._v_rest is None else float(self._v_rest)

    def axial_couplings(self):
        """Junctions between this cell's sections."""
        return axial_couplings(self.sections)

    def measure_ic(self) -> None:
        """Read this cell's currents and pin them, without initializing."""
        fn = getattr(self._template, "measure_ic", None)
        if callable(fn):
            fn()

    @property
    def template(self):
        return self._template


class CellHandle(Cell):
    """Per-cell parameter handle over a NEURON cell.

    Parameters are addressed as ``"<section type>.<name>"``, where the section
    type is the one the cell routes synapses by (``soma``, ``dend``, ``basal``,
    ...) and the name is a section attribute (``cm``, ``Ra``) or a mechanism
    parameter under its suffixed NEURON name (``g_pas``, ``gnabar_hh``)::

        env.cells[3].set_params({"soma.g_pas": 3e-5, "dend.cm": 1.5})

    Reads report the value at the middle of the section type's first section
    and writes are applied to every segment of every section of that type.

    Any other attribute access is forwarded to the underlying cell, so a handle
    can be used wherever the cell itself was.
    """

    _own = ("_env", "_population", "_gid", "_cell")

    def __init__(self, env, population: str, gid: int, cell):
        object.__setattr__(self, "_cell", cell)
        super().__init__(env, population, gid)

    @property
    def cell(self):
        """The wrapped ``NeuronCell``"""
        return self._cell

    def sections_by_type(self) -> dict[str, list]:
        """The cell's sections grouped under their section-type name"""
        cell = self._cell
        sections = list(getattr(cell, "sections", []) or [])
        types = getattr(cell, "section_type", None)
        name_of = getattr(cell, "dest_sec_type", None)

        groups: dict[str, list] = {}
        for i, sec in enumerate(sections):
            swc = int(types[i]) if types is not None and i < len(types) else SWC_SOMA
            if callable(name_of):
                sec_type = str(name_of(swc))
            else:
                sec_type = _MORPH_SECTYPE_NAMES.get(swc, "apical")
            groups.setdefault(sec_type, []).append(sec)
        return groups

    def get_params(self) -> dict[str, float]:
        from neuron import h

        params: dict[str, float] = {}
        for sec_type, sections in self.sections_by_type().items():
            sec = sections[0]
            seg = sec(0.5)
            params[f"{sec_type}.cm"] = float(seg.cm)
            params[f"{sec_type}.Ra"] = float(sec.Ra)

            # MechanismStandard(name, 1) enumerates exactly the mechanism's
            # parameters, leaving out its states and assigned variables, and
            # under the suffixed names a segment exposes them by (g_pas,
            # gnabar_hh)
            for mech in seg:
                standard = h.MechanismStandard(mech.name(), 1)
                for i in range(int(standard.count())):
                    ref = h.ref("")
                    if int(standard.name(ref, i)) != 1:
                        continue  # array parameters have no single value
                    name = ref[0]
                    try:
                        params[f"{sec_type}.{name}"] = float(getattr(seg, name))
                    except AttributeError:
                        continue  # GLOBAL parameters are not segment attributes
        return params

    def set_params(self, params: dict[str, float]):
        groups = self.sections_by_type()
        rebuild = {}
        for key, value in params.items():
            sec_type, _, name = key.partition(".")
            if not name:
                if hasattr(self._template, key) or key in getattr(
                    self._template, "axon_params", {}
                ):
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
            value = float(value)
            for sec in sections:
                if name == "Ra":
                    sec.Ra = value
                    continue
                for seg in sec:
                    setattr(seg, name, value)

        if rebuild:
            self._rebuild_template(rebuild)

        return self._env

    def _rebuild_template(self, params: dict[str, float]):
        template = self._template
        axon = getattr(template, "axon_params", None)
        for name, value in params.items():
            if axon is not None and name in axon:
                axon[name] = value
            else:
                setattr(template, name, value)
        for step in ("geometry", "biophys"):
            fn = getattr(template, step, None)
            if callable(fn):
                fn()

    def __getattr__(self, name):
        try:
            cell = object.__getattribute__(self, "_cell")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(cell, name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._cell, name, value)


def _accepts_gid(factory) -> bool:
    """Whether a cell factory takes the `gid=` keyword."""
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "gid" in parameters


class CellBuilder:
    """Builds rank-local cells for a population from a ``System``.

    Reduced cells are positioned from ``System.coordinate_array``.
    """

    def __init__(self, system, model, pc, comm):
        self.system = system
        self.model = model
        self.pc = pc
        self.comm = comm
        self._factories = model.neuron_cells()

    def build_local(
        self, population: str, selection: set[int] | None = None
    ) -> dict[int, NeuronCell]:
        if population not in self._factories:
            raise KeyError(f"model.neuron_cells() has no factory for {population!r}")
        factory = self._factories[population]

        nhost = int(self.pc.nhost())
        rank = int(self.pc.id())

        # position lookup where coordinates exist (reduced cells) while full-morphology
        # populations may have only Trees and no Generated Coordinates.
        coords = self.system.coordinate_array(population)  # [n, 4] = gid,x,y,z
        coord_by_gid = {int(r[0]): r[1:4] for r in coords} if len(coords) else {}

        if selection is not None:
            gids = sorted(int(g) for g in selection)
        elif coord_by_gid:
            gids = sorted(coord_by_gid.keys())
        else:
            raise RuntimeError(
                f"population {population!r} has no coordinates; a selection() is "
                "required to build it"
            )

        takes_gid = _accepts_gid(factory)

        cells: dict[int, NeuronCell] = {}
        for gid in gids:
            if gid % nhost != rank:
                continue
            cell = (
                factory(morphology=None, gid=gid)
                if takes_gid
                else factory(morphology=None)
            )
            xyz = coord_by_gid.get(gid)
            if xyz is not None:
                cell.position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            cells[gid] = cell
        return cells
