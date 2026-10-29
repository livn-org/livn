from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from livn.backend.native import _lib as L
from livn.types import DEFAULT_CONDUCTION_VELOCITY, conduction_velocity

logger = logging.getLogger(__name__)

DEFAULT_VELOCITY = DEFAULT_CONDUCTION_VELOCITY

SWC_SOMA = 1
SWC_AXON = 2
SWC_BASAL = 3
SWC_APICAL = 4


@dataclass
class SynapseTable:
    """One row per point process (site) on a local cell."""

    post_gid: np.ndarray  # int32[S]
    swc_type: np.ndarray  # int8[S]
    dest_sectype: np.ndarray  # int8[S]
    mech_id: np.ndarray  # int16[S]
    # int8[S] the receptor (connections_config synapse class, e.g. GABA_B); unlike
    # the mechanism, it tells apart receptors that one mechanism implements
    receptor: np.ndarray
    syn_id: np.ndarray  # int64[S]

    @property
    def size(self) -> int:
        return len(self.post_gid)


@dataclass
class ConnectionTable:
    """One row per connection (edge x mechanism), in ``librcsd`` order."""

    pre_gid: np.ndarray  # int32[C]
    syn_row: np.ndarray  # int32[C]
    post_pop: np.ndarray  # int8[C]
    pre_pop: np.ndarray  # int8[C]
    mech_id: np.ndarray  # int16[C]
    receptor: np.ndarray  # int8[C]  receptor code, see SynapseTable.receptor
    swc_type: np.ndarray  # int8[C]
    dest_sectype: np.ndarray  # int8[C]
    weight: np.ndarray  # float64[C]
    delay: np.ndarray  # float64[C]  physical delay, floored at 2 dt inside the library
    wslot: np.ndarray  # int8[C]

    @property
    def size(self) -> int:
        return len(self.pre_gid)


def _codes(*columns) -> np.ndarray:
    """One int64 per row of `columns` that compares and orders as the rows do."""
    columns = [np.asarray(c, dtype=np.int64) for c in columns]
    if not len(columns[0]):
        return np.zeros(0, dtype=np.int64)
    lows = [int(c.min()) for c in columns]
    spans = [int(c.max()) - low + 1 for c, low in zip(columns, lows, strict=True)]
    if math.prod(spans) < 2**62:
        code = np.zeros(len(columns[0]), dtype=np.int64)
        for c, low, span in zip(columns, lows, spans, strict=True):
            code = code * span + (c - low)
        return code
    rows = np.stack(columns, axis=1)
    return np.unique(rows, axis=0, return_inverse=True)[1].reshape(-1)


def _first_seen(values) -> np.ndarray:
    """The distinct `values`, in the order they first appear."""
    unique, first = np.unique(values, return_index=True)
    return unique[np.argsort(first, kind="stable")]


def _edge_syn_ids_distances(projection, n: int):
    # imported here: livn.system imports livn.backend, which imports this module
    from livn.system import projection_attribute

    syn_ids = np.zeros(n, dtype=np.int64)
    distances = np.zeros(n, dtype=np.float64)
    if isinstance(projection, dict):
        found = projection_attribute(projection.get("Synapses"), "syn_id")
        if found is not None:
            syn_ids = np.asarray(found).astype(np.int64)
        found = projection_attribute(projection.get("Connections"), "distance")
        if found is not None:
            distances = np.asarray(found).astype(np.float64)
    return syn_ids, distances


class SynapseBuilder:
    """Builds the synapse and connection tables and wires them into the sim."""

    def __init__(
        self,
        env,
        selected_gids=None,
        simulated_pops=None,
    ):
        self.env = env
        self.system = env.system
        self.model = env.model
        self._selected_gids = (
            None if selected_gids is None else {int(g) for g in selected_gids}
        )
        self._simulated_pops = (
            set(simulated_pops) if simulated_pops is not None else None
        )
        self._microcircuit_inputs = bool(
            self.model.neuron_microcircuit_inputs()
            if hasattr(self.model, "neuron_microcircuit_inputs")
            else False
        )
        self._mech_map = self.model.neuron_synapse_mechanisms()
        self._rules = self.model.neuron_synapse_rules()
        self._ignored = (
            set(self.model.ignored_populations())
            if hasattr(self.model, "ignored_populations")
            else set()
        )
        self._pop_code: dict[str, int] = {}
        self._mech_code: dict[str, int] = {}
        self._receptor_code: dict[str, int] = {}
        self._sectype_code: dict[str, int] = {}
        self.input_indices: dict[int, int] = {}

    def _pop_id(self, name: str) -> int:
        return self._pop_code.setdefault(name, len(self._pop_code))

    def _mech_id(self, mech_name: str) -> int:
        return self._mech_code.setdefault(mech_name, len(self._mech_code))

    def _receptor_id(self, receptor: str) -> int:
        return self._receptor_code.setdefault(receptor, len(self._receptor_code))

    def _sectype_id(self, name: str) -> int:
        return self._sectype_code.setdefault(name, len(self._sectype_code))

    def _mechanisms_for(self, post: str, pre: str) -> dict:
        """Active ``{swc_type: {synapse_class: params}}`` for a projection."""
        syn_cfg = self.system.connections_config["synapses"][post][pre]
        blocks: dict = {}
        for key, mechs in (syn_cfg.get("mechanisms", {}) or {}).items():
            if not isinstance(mechs, dict):
                continue
            if key == "default":
                swc_type = None
            else:
                try:
                    swc_type = int(key)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{post}<-{pre} declares mechanisms under {key!r}; "
                        "expected 'default' or an SWC type number"
                    ) from None
            active = {}
            for cls, params in mechs.items():
                if params.get("tau_decay") is None:
                    continue
                active[cls] = params
            if active:
                blocks[swc_type] = active
        return blocks

    def _read_placement(self, population: str, local_gids: set[int]):
        """`(gid, syn_id, swc_type, loc)` of the local cells' sites, by `(gid, syn_id)`.

        A syn_id a cell lists twice keeps its last entry.
        """
        gids, syn_ids, swc_types, locs = [], [], [], []
        if local_gids:
            for gid, (sids, swc, loc) in self.system.placement(
                population, local_gids
            ).items():
                gids.append(np.full(len(sids), int(gid), dtype=np.int64))
                syn_ids.append(np.asarray(sids).astype(np.int64))
                swc_types.append(np.asarray(swc).astype(np.int64))
                locs.append(np.asarray(loc).astype(np.float64))
        if not gids:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, np.zeros(0)
        gid, syn_id = np.concatenate(gids), np.concatenate(syn_ids)
        swc_type, loc = np.concatenate(swc_types), np.concatenate(locs)
        order = np.lexsort((np.arange(len(gid)), syn_id, gid))
        gid, syn_id = gid[order], syn_id[order]
        last = np.ones(len(gid), dtype=bool)
        last[:-1] = (gid[1:] != gid[:-1]) | (syn_id[1:] != syn_id[:-1])
        keep = order[last]
        return gid[last], syn_id[last], swc_type[keep], loc[keep]

    def _mech_specs(self, active: dict) -> list:
        specs = []
        for cls_name, params in active.items():
            mech_name = self._mech_map.get(cls_name, cls_name)
            kind = L.SYNAPSE_KINDS.get(mech_name)
            if kind is None:
                raise ValueError(
                    f"the native backend has no synapse mechanism {mech_name!r} "
                    f"(for {cls_name}); it knows {sorted(L.SYNAPSE_KINDS)}"
                )
            rule = self._rules.get(mech_name, {})
            set_params = [
                (L.SP[p], float(params[p]))
                for p in rule.get("mech_params", [])
                if params.get(p) is not None and p in L.SP
            ]
            netcon_params = rule.get("netcon_params", {"weight": 0})
            wslot = int(netcon_params.get("weight", 0))
            w0 = [0.0] * L.NWEIGHT
            for pname, slot in netcon_params.items():
                if pname == "weight":
                    w0[int(slot)] = float(params.get("weight", 1.0))
                else:
                    v = params.get(pname)
                    if v is not None:
                        w0[int(slot)] = float(v)
            specs.append(
                (
                    mech_name,
                    kind,
                    set_params,
                    self._mech_id(mech_name),
                    wslot,
                    w0,
                    w0[wslot],
                    self._receptor_id(cls_name),
                )
            )
        return specs

    def build(self, cells_by_pop: dict[str, dict[int, object]]):
        lib, sim = self.env._lib, self.env._sim
        local = [
            (int(gid), cell)
            for cells in cells_by_pop.values()
            for gid, cell in cells.items()
        ]
        cell_gids = np.asarray([gid for gid, _ in local], dtype=np.int64)
        cell_indices = np.asarray([cell.index for _, cell in local], dtype=np.int64)
        order = np.argsort(cell_gids, kind="stable")
        cell_gids, cell_indices = cell_gids[order], cell_indices[order]
        connections_config = self.system.connections_config["synapses"]
        simulated = (
            self._simulated_pops
            if self._simulated_pops is not None
            else set(cells_by_pop.keys())
        )
        selected = (
            None
            if self._selected_gids is None
            else np.asarray(sorted(self._selected_gids), dtype=np.int64)
        )

        VEL = conduction_velocity(self.system)
        projections = []
        for post in cells_by_pop:
            if post in self._ignored:
                continue
            cells = cells_by_pop[post]
            post_id = self._pop_id(post)
            active_by_pre = {}
            for pre in connections_config.get(post, {}):
                if pre in self._ignored:
                    continue
                active = self._mechanisms_for(post, pre)
                if active:
                    active_by_pre[pre] = active
            if not active_by_pre or not cells:
                continue
            placement = self._read_placement(post, set(cells.keys()))

            for pre, active in active_by_pre.items():
                pre_id = self._pop_id(pre)
                is_input = pre not in simulated
                specs_by_swc = {
                    swc: self._mech_specs(mechs) for swc, mechs in active.items()
                }
                sel = None if (is_input or self._microcircuit_inputs) else selected
                found = self._projection(
                    pre,
                    post,
                    cells,
                    placement,
                    specs_by_swc,
                    sel,
                    is_input,
                    (cell_gids, cell_indices),
                )
                if found is not None:
                    found.update(post_pop=post_id, pre_pop=pre_id, cells=cells)
                    projections.append(found)

        for found in projections:
            self._connect_inputs(found)
            self._place(found)
        rows = self._expand(projections, VEL)
        n_conn = len(rows["source"])

        site_codes = _codes(rows["post_gid"], rows["syn_id"], rows["receptor"])
        unique, first, inverse = np.unique(
            site_codes, return_index=True, return_inverse=True
        )
        by_first = np.argsort(first, kind="stable")
        site_of = np.empty(len(unique), dtype=np.int64)
        site_of[by_first] = np.arange(len(unique))
        site_row = site_of[inverse.reshape(-1)] if n_conn else inverse
        made = first[by_first]
        n_sites = len(made)

        if n_sites:
            c_cell = L.as_int_array(rows["cell"][made])
            c_section = L.as_int_array(rows["section"][made])
            c_x = L.as_double_array(rows["x"][made])
            c_kind = L.as_int_array(rows["kind"][made])
            first_site = L.check(
                lib.rcsd_add_synapses(
                    sim,
                    n_sites,
                    L.int_ptr(c_cell),
                    L.int_ptr(c_section),
                    L.double_ptr(c_x),
                    L.int_ptr(c_kind),
                ),
                lib,
            )
            if first_site != 0:
                raise RuntimeError("synapse table and librcsd disagree on site order")
        if n_conn:
            source = L.as_int_array(rows["source"])
            site = L.as_int_array(site_row)
            delay = L.as_double_array(rows["delay"])
            w0 = L.as_double_array(rows["w0"])
            L.check(
                lib.rcsd_add_connections(
                    sim,
                    n_conn,
                    L.int_ptr(source),
                    L.int_ptr(site),
                    L.double_ptr(delay),
                    L.double_ptr(w0),
                ),
                lib,
            )
        if n_sites:
            sp = L.site_view(lib, sim, lib.rcsd_synapse_params(sim), L.SP_N)
            site_params = rows["set_params"][made]
            for spec in np.unique(site_params):
                at = np.flatnonzero(site_params == spec)
                for column, value in rows["set_params_table"][spec]:
                    sp[column, at] = value
            every = L.as_int_array(np.arange(n_sites))
            L.check(lib.rcsd_synapse_refresh_sites(sim, n_sites, L.int_ptr(every)), lib)

        syn = SynapseTable(
            post_gid=rows["post_gid"][made].astype(np.int32),
            swc_type=rows["swc_type"][made].astype(np.int8),
            dest_sectype=rows["dest_sectype"][made].astype(np.int8),
            mech_id=rows["mech_id"][made].astype(np.int16),
            receptor=rows["receptor"][made].astype(np.int8),
            syn_id=rows["syn_id"][made].astype(np.int64),
        )
        conn = ConnectionTable(
            pre_gid=rows["pre_gid"].astype(np.int32),
            syn_row=site_row.astype(np.int32),
            post_pop=rows["post_pop"].astype(np.int8),
            pre_pop=rows["pre_pop"].astype(np.int8),
            mech_id=rows["mech_id"].astype(np.int16),
            receptor=rows["receptor"].astype(np.int8),
            swc_type=rows["swc_type"].astype(np.int8),
            dest_sectype=rows["dest_sectype"].astype(np.int8),
            weight=rows["weight"].astype(np.float64),
            delay=rows["delay"].astype(np.float64),
            wslot=rows["wslot"].astype(np.int8),
        )
        return (
            syn,
            conn,
            dict(self._pop_code),
            dict(self._mech_code),
            dict(self._sectype_code),
            dict(self.input_indices),
            dict(self._receptor_code),
        )

    def _projection(
        self, pre, post, cells, placement, specs_by_swc, sel, is_input, local
    ):
        """The edges of one projection that become connections, as arrays."""
        post_gids, pre_gids, syn_ids, distances = [], [], [], []
        for post_gid, (pres, projection) in self.system.edges(
            pre, post, set(cells.keys())
        ):
            post_gid = int(post_gid)
            if post_gid not in cells:
                continue
            pres = np.asarray(pres).astype(np.int64)
            sids, dist = _edge_syn_ids_distances(projection, len(pres))
            post_gids.append(np.full(len(pres), post_gid, dtype=np.int64))
            pre_gids.append(pres)
            syn_ids.append(sids)
            distances.append(dist)
        if not post_gids:
            return None
        post_gid = np.concatenate(post_gids)
        pre_gid = np.concatenate(pre_gids)
        syn_id = np.concatenate(syn_ids)
        distance = np.concatenate(distances)

        keep = np.ones(len(pre_gid), dtype=bool)
        if sel is not None:
            keep &= np.isin(pre_gid, sel)

        p_gid, p_syn_id, p_swc, p_loc = placement
        codes = _codes(
            np.concatenate([p_gid, post_gid]), np.concatenate([p_syn_id, syn_id])
        )
        sites, wanted = codes[: len(p_gid)], codes[len(p_gid) :]
        at = np.minimum(np.searchsorted(sites, wanted), max(len(sites) - 1, 0))
        if len(sites):
            keep &= sites[at] == wanted
        else:
            keep[:] = False
        swc_type = p_swc[at] if len(sites) else np.zeros(len(keep), dtype=np.int64)
        loc = p_loc[at] if len(sites) else np.zeros(len(keep))

        default_specs = specs_by_swc.get(None)
        spec_lists, spec_group = [], np.full(len(keep), -1, dtype=np.int64)
        for swc in np.unique(swc_type[keep]).tolist():
            here = keep & (swc_type == swc)
            specs = specs_by_swc.get(swc, default_specs)
            if specs is None:
                keep &= ~here
                continue
            spec_group[here] = len(spec_lists)
            spec_lists.append(specs)

        cell_gids, cell_indices = local
        at = np.minimum(np.searchsorted(cell_gids, pre_gid), max(len(cell_gids) - 1, 0))
        simulated = cell_gids[at] == pre_gid if len(cell_gids) else np.zeros_like(keep)
        if not is_input and sel is not None:
            keep &= simulated
        source = np.where(simulated, cell_indices[at] if len(cell_gids) else 0, -1)

        return {
            "source": source[keep],
            "post_gid": post_gid[keep],
            "pre_gid": pre_gid[keep],
            "syn_id": syn_id[keep],
            "distance": distance[keep],
            "swc_type": swc_type[keep],
            "loc": loc[keep],
            "spec_group": spec_group[keep],
            "spec_lists": spec_lists,
        }

    def _connect_inputs(self, found: dict) -> None:
        """An input for each presynaptic cell not simulated here, on first sight."""
        source = found["source"]
        outside = source < 0
        if not outside.any():
            return
        gids = found["pre_gid"][outside]
        for gid in _first_seen(gids).tolist():
            self._input(gid)
        unique = np.unique(gids)
        index = np.asarray([self.input_indices[g] for g in unique.tolist()])
        source[outside] = -(index[np.searchsorted(unique, gids)] + 1)

    def _place(self, found: dict) -> None:
        """Section, position and destination section type of every connection."""
        post_gid, swc_type = found["post_gid"], found["swc_type"]
        cells = found["cells"]
        n = len(post_gid)
        section = np.zeros(n, dtype=np.int64)
        x = np.zeros(n)
        cell_index = np.zeros(n, dtype=np.int64)
        names: dict[str, int] = {}
        name_code = np.zeros(n, dtype=np.int64)
        # a post cell's edges arrive together
        starts = np.flatnonzero(np.r_[True, post_gid[1:] != post_gid[:-1]][:n])
        ends = [*starts[1:].tolist(), n] if n else []
        for lo, hi in zip(starts.tolist(), ends, strict=True):
            cell = cells[int(post_gid[lo])]
            section[lo:hi], x[lo:hi] = cell.place_many(
                swc_type[lo:hi], found["loc"][lo:hi]
            )
            cell_index[lo:hi] = cell.index
            for swc in np.unique(swc_type[lo:hi]).tolist():
                name = cell.dest_sec_type(swc)
                code = names.setdefault(name, len(names))
                name_code[lo:hi][swc_type[lo:hi] == swc] = code
        by_code = list(names)
        dest = np.zeros(len(names), dtype=np.int64)
        for code in _first_seen(name_code).tolist():
            dest[code] = self._sectype_id(by_code[code])
        found.update(
            section=section, x=x, cell=cell_index, dest_sectype=dest[name_code]
        )

    @staticmethod
    def _expand(projections: list[dict], velocity: float) -> dict:
        """One row per connection: each edge once for every mechanism of its site."""
        columns = (
            "post_gid",
            "pre_gid",
            "syn_id",
            "swc_type",
            "dest_sectype",
            "source",
            "cell",
            "section",
            "x",
            "kind",
            "mech_id",
            "receptor",
            "wslot",
            "weight",
            "w0",
            "set_params",
            "delay",
            "post_pop",
            "pre_pop",
        )
        parts: dict[str, list] = {c: [] for c in columns}
        set_params_table: list = []
        for found in projections:
            specs = [spec for group in found["spec_lists"] for spec in group]
            if not specs:
                continue
            sizes = np.asarray([len(g) for g in found["spec_lists"]], dtype=np.int64)
            base = np.cumsum(sizes) - sizes
            per_edge = sizes[found["spec_group"]]
            edge = np.repeat(np.arange(len(per_edge)), per_edge)
            offset = np.arange(len(edge)) - np.repeat(
                np.cumsum(per_edge) - per_edge, per_edge
            )
            spec = base[found["spec_group"]][edge] + offset

            # _, kind, set_params, mid, wslot, w0, wval, rid
            table = list(zip(*specs, strict=True))
            for name in (
                "post_gid",
                "pre_gid",
                "syn_id",
                "swc_type",
                "dest_sectype",
                "source",
                "cell",
                "section",
                "x",
            ):
                parts[name].append(found[name][edge])
            parts["delay"].append(found["distance"][edge] / velocity)
            parts["kind"].append(np.asarray(table[1], dtype=np.int64)[spec])
            parts["mech_id"].append(np.asarray(table[3], dtype=np.int64)[spec])
            parts["wslot"].append(np.asarray(table[4], dtype=np.int64)[spec])
            parts["w0"].append(
                np.asarray(table[5], dtype=np.float64).reshape(len(specs), -1)[spec]
            )
            parts["weight"].append(np.asarray(table[6], dtype=np.float64)[spec])
            parts["receptor"].append(np.asarray(table[7], dtype=np.int64)[spec])
            parts["set_params"].append(spec + len(set_params_table))
            set_params_table.extend(table[2])
            parts["post_pop"].append(
                np.full(len(edge), found["post_pop"], dtype=np.int64)
            )
            parts["pre_pop"].append(
                np.full(len(edge), found["pre_pop"], dtype=np.int64)
            )
        rows = {
            c: np.concatenate(v) if v else np.zeros(0, dtype=np.int64)
            for c, v in parts.items()
        }
        if not parts["w0"]:
            rows["w0"] = np.zeros((0, L.NWEIGHT))
        rows["set_params_table"] = set_params_table
        return rows

    def _input(self, gid: int) -> int:
        index = self.input_indices.get(gid)
        if index is None:
            lib, sim = self.env._lib, self.env._sim
            index = L.check(lib.rcsd_add_input(sim, int(gid)), lib)
            self.input_indices[gid] = index
        return index
