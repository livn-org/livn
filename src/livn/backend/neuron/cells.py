from __future__ import annotations

from bisect import bisect_right
from typing import Protocol, runtime_checkable

import numpy as np

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
    ):
        self._template = template
        self.threshold = float(threshold)
        self._v_rest = v_rest
        self._soma_type = soma_type
        self._dend_type = dend_type

        self.sections = list(template.sections)
        if not self.sections:
            raise ValueError("ReducedCell: template exposes no sections")

        types = np.empty(len(self.sections), dtype=np.int8)
        self._soma = None
        self._dend = None
        for i, sec in enumerate(self.sections):
            name = sec.name().split(".")[-1].lower()
            if "soma" in name:
                types[i] = SWC_SOMA
                if self._soma is None:
                    self._soma = sec
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
        return self._dend(loc)

    def dest_sec_type(self, swc_type: int) -> str:
        return self._soma_type if swc_type == SWC_SOMA else self._dend_type

    def spike_source(self):
        return self._soma(0.5)

    def position(self, x: float, y: float, z: float) -> None:
        self._template.position(x, y, z)

    def init_ic(self, v_rest: float | None = None) -> None:
        fn = getattr(self._template, "init_ic", None)
        if callable(fn):
            fn(self._v_rest if v_rest is None else v_rest)

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
        return secs[i](x)

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

    @property
    def template(self):
        return self._template


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

        cells: dict[int, NeuronCell] = {}
        for gid in gids:
            if gid % nhost != rank:
                continue
            cell = factory(morphology=None)
            xyz = coord_by_gid.get(gid)
            if xyz is not None:
                cell.position(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            cells[gid] = cell
        return cells
