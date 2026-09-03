from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from livn.backend.native import _lib as L

logger = logging.getLogger(__name__)

DEFAULT_VELOCITY = 250.0  # um/ms

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
    swc_type: np.ndarray  # int8[C]
    dest_sectype: np.ndarray  # int8[C]
    weight: np.ndarray  # float64[C]
    delay: np.ndarray  # float64[C]  physical delay, floored at 2 dt inside the library
    wslot: np.ndarray  # int8[C]

    @property
    def size(self) -> int:
        return len(self.pre_gid)


@dataclass
class _Growable:
    post_gid: list = field(default_factory=list)
    swc_type: list = field(default_factory=list)
    dest_sectype: list = field(default_factory=list)
    mech_id: list = field(default_factory=list)
    syn_id: list = field(default_factory=list)
    site_params: list = field(default_factory=list)  # (site, [(column, value)])

    c_source: list = field(default_factory=list)
    c_site: list = field(default_factory=list)
    c_delay: list = field(default_factory=list)
    c_w0: list = field(default_factory=list)  # [4] per connection
    c_pre_gid: list = field(default_factory=list)
    c_syn_row: list = field(default_factory=list)
    c_post_pop: list = field(default_factory=list)
    c_pre_pop: list = field(default_factory=list)
    c_mech_id: list = field(default_factory=list)
    c_swc_type: list = field(default_factory=list)
    c_dest_sectype: list = field(default_factory=list)
    c_weight: list = field(default_factory=list)
    c_wslot: list = field(default_factory=list)


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
        self._sectype_code: dict[str, int] = {}
        self.input_indices: dict[int, int] = {}

    def _pop_id(self, name: str) -> int:
        return self._pop_code.setdefault(name, len(self._pop_code))

    def _mech_id(self, mech_name: str) -> int:
        return self._mech_code.setdefault(mech_name, len(self._mech_code))

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
        """gid -> {syn_id: (swc_type, loc)} for the local cells of a population."""
        out: dict[int, dict[int, tuple[int, float]]] = {}
        if not local_gids:
            return out
        endpoints = 0
        for gid, attrs in self.system.synapses(population, node_allocation=local_gids):
            syn_ids = np.asarray(attrs["syn_ids"])
            swc = np.asarray(attrs["swc_types"])
            locs = np.asarray(attrs["syn_locs"])
            endpoints += int(((locs <= 0.0) | (locs >= 1.0)).sum())
            out[int(gid)] = {
                int(syn_ids[i]): (int(swc[i]), float(locs[i]))
                for i in range(len(syn_ids))
            }
        if endpoints:
            logger.warning(
                "%s: %d synapse site(s) are recorded at section position 0 or 1, "
                "which cannot hold an ion mechanism; they were moved to the "
                "nearest segment centre",
                population,
                endpoints,
            )
        return out

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
                )
            )
        return specs

    def build(self, cells_by_pop: dict[str, dict[int, object]]):
        lib, sim = self.env._lib, self.env._sim
        g = _Growable()
        site_rows: dict[tuple[int, int, str], int] = {}
        cell_index = {
            int(gid): cell.index
            for cells in cells_by_pop.values()
            for gid, cell in cells.items()
        }
        connections_config = self.system.connections_config["synapses"]
        simulated = (
            self._simulated_pops
            if self._simulated_pops is not None
            else set(cells_by_pop.keys())
        )
        selected = self._selected_gids

        VEL = DEFAULT_VELOCITY
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
                default_specs = specs_by_swc.get(None)
                sel = None if (is_input or self._microcircuit_inputs) else selected

                for post_gid, (pre_gids, projection) in self.system.projection_array(
                    pre, post
                ):
                    post_gid = int(post_gid)
                    if post_gid not in cells:
                        continue
                    cell = cells[post_gid]
                    place = placement.get(post_gid, {})
                    pre_gids = np.asarray(pre_gids).astype(np.int64)
                    syn_ids, distances = _edge_syn_ids_distances(
                        projection, len(pre_gids)
                    )
                    dest_code: dict[int, int] = {}
                    for k in range(len(pre_gids)):
                        pre_gid = int(pre_gids[k])
                        if sel is not None and pre_gid not in sel:
                            continue
                        sid = int(syn_ids[k])
                        site = place.get(sid)
                        if site is None:
                            continue
                        swc_type, loc = site
                        specs = specs_by_swc.get(swc_type, default_specs)
                        if specs is None:
                            continue
                        source = cell_index.get(pre_gid)
                        if source is None:
                            if not is_input and sel is not None:
                                continue
                            source = -(self._input(pre_gid) + 1)
                        section, x = cell.place(swc_type, loc)
                        dsec = dest_code.get(swc_type)
                        if dsec is None:
                            dsec = self._sectype_id(cell.dest_sec_type(swc_type))
                            dest_code[swc_type] = dsec
                        phys = float(distances[k]) / VEL

                        for mech_name, kind, set_params, mid, wslot, w0, wval in specs:
                            key = (post_gid, sid, mech_name)
                            row = site_rows.get(key)
                            if row is None:
                                c_site = L.check(
                                    lib.rcsd_add_synapse(
                                        sim, cell.index, section, x, kind
                                    ),
                                    lib,
                                )
                                row = len(g.post_gid)
                                if c_site != row:
                                    raise RuntimeError(
                                        "synapse table and librcsd disagree on site order"
                                    )
                                site_rows[key] = row
                                g.post_gid.append(post_gid)
                                g.swc_type.append(swc_type)
                                g.dest_sectype.append(dsec)
                                g.mech_id.append(mid)
                                g.syn_id.append(sid)
                                g.site_params.append((row, set_params))
                            g.c_source.append(source)
                            g.c_site.append(row)
                            g.c_delay.append(phys)
                            g.c_w0.append(w0)
                            g.c_pre_gid.append(pre_gid)
                            g.c_syn_row.append(row)
                            g.c_post_pop.append(post_id)
                            g.c_pre_pop.append(pre_id)
                            g.c_mech_id.append(mid)
                            g.c_swc_type.append(swc_type)
                            g.c_dest_sectype.append(dsec)
                            g.c_weight.append(wval)
                            g.c_wslot.append(wslot)

        n_conn = len(g.c_source)
        if n_conn:
            source = L.as_int_array(g.c_source)
            site = L.as_int_array(g.c_site)
            delay = L.as_double_array(g.c_delay)
            w0 = L.as_double_array(np.asarray(g.c_w0, dtype=np.float64))
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
        n_sites = len(g.post_gid)
        if n_sites:
            sp = L.site_view(lib, sim, lib.rcsd_synapse_params(sim), L.SP_N)
            for row, set_params in g.site_params:
                for column, value in set_params:
                    sp[column, row] = value
            for row in range(n_sites):
                L.check(lib.rcsd_synapse_refresh(sim, row), lib)

        syn = SynapseTable(
            post_gid=np.asarray(g.post_gid, dtype=np.int32),
            swc_type=np.asarray(g.swc_type, dtype=np.int8),
            dest_sectype=np.asarray(g.dest_sectype, dtype=np.int8),
            mech_id=np.asarray(g.mech_id, dtype=np.int16),
            syn_id=np.asarray(g.syn_id, dtype=np.int64),
        )
        conn = ConnectionTable(
            pre_gid=np.asarray(g.c_pre_gid, dtype=np.int32),
            syn_row=np.asarray(g.c_syn_row, dtype=np.int32),
            post_pop=np.asarray(g.c_post_pop, dtype=np.int8),
            pre_pop=np.asarray(g.c_pre_pop, dtype=np.int8),
            mech_id=np.asarray(g.c_mech_id, dtype=np.int16),
            swc_type=np.asarray(g.c_swc_type, dtype=np.int8),
            dest_sectype=np.asarray(g.c_dest_sectype, dtype=np.int8),
            weight=np.asarray(g.c_weight, dtype=np.float64),
            delay=np.asarray(g.c_delay, dtype=np.float64),
            wslot=np.asarray(g.c_wslot, dtype=np.int8),
        )
        return (
            syn,
            conn,
            dict(self._pop_code),
            dict(self._mech_code),
            dict(self._sectype_code),
            dict(self.input_indices),
        )

    def _input(self, gid: int) -> int:
        index = self.input_indices.get(gid)
        if index is None:
            lib, sim = self.env._lib, self.env._sim
            index = L.check(lib.rcsd_add_input(sim, int(gid)), lib)
            self.input_indices[gid] = index
        return index
