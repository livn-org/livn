from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

DEFAULT_VELOCITY = 250.0  # um/ms
# NetCon delays are floored to 2*dt per NEURON fixed-step requirement but since dt
# is a run-time choice, ``ConnectionTable.delay`` stores the dt-independent
# physical delay and Env re-applies the 2*dt floor per run
DEFAULT_DT = 0.025


class ObjectStore(Protocol):
    def append(self, obj) -> int: ...
    def get(self, i: int): ...
    def __len__(self) -> int: ...
    def clear(self) -> None: ...


class PyListStore:
    """A Python list of HocObject wrappers."""

    def __init__(self):
        self._objs: list = []

    def append(self, obj) -> int:
        self._objs.append(obj)
        return len(self._objs) - 1

    def get(self, i: int):
        return self._objs[i]

    def __len__(self) -> int:
        return len(self._objs)

    def clear(self) -> None:
        self._objs.clear()


class HocListStore:
    """Large-scale store using ``h.List`` for C++ side memory

    Saves ~1 persistent Python wrapper per object where individual access mints a
    transient wrapper via ``List.o(i)``, which is fine because hot paths read
    the numpy columns, not the objects.
    """

    def __init__(self):
        from neuron import h

        self._h = h
        self._list = h.List()
        self._n = 0

    def append(self, obj) -> int:
        self._list.append(obj)
        idx = self._n
        self._n += 1
        return idx

    def get(self, i: int):
        return self._list.o(i)

    def __len__(self) -> int:
        return self._n

    def clear(self) -> None:
        # Drop the whole C++ container at once releasing every held object
        self._list = self._h.List()
        self._n = 0


def make_store(kind: str) -> ObjectStore:
    if kind == "python":
        return PyListStore()
    if kind == "hoc":
        return HocListStore()
    raise ValueError(f"unknown object store kind: {kind!r}")


@dataclass
class SynapseTable:
    """One row per point process instantiated on a local cell."""

    post_gid: np.ndarray  # int32[S]
    swc_type: np.ndarray  # int8[S]   (original placement type, for selection)
    dest_sectype: np.ndarray  # int8[S]  destination section-type code
    mech_id: np.ndarray  # int16[S]
    syn_id: np.ndarray  # int64[S]   neuroh5 synapse id
    store: ObjectStore  # point processes

    @property
    def size(self) -> int:
        return len(self.post_gid)


@dataclass
class ConnectionTable:
    """One row per NetCon (edge x mechanism)."""

    pre_gid: np.ndarray  # int32[C]
    syn_row: np.ndarray  # int32[C]  -> SynapseTable row
    post_pop: np.ndarray  # int8[C]
    pre_pop: np.ndarray  # int8[C]
    mech_id: np.ndarray  # int16[C]
    swc_type: np.ndarray  # int8[C]
    dest_sectype: np.ndarray  # int8[C]  destination section-type code
    weight: np.ndarray  # float64[C]  mirror of nc.weight[wslot]
    delay: (
        np.ndarray
    )  # float32[C]  physical (distance) delay; NetCon = max(delay, 2*dt)
    wslot: np.ndarray  # int8[C]
    store: ObjectStore  # NetCons

    @property
    def size(self) -> int:
        return len(self.pre_gid)


@dataclass
class _Growable:
    """Append-friendly list-backed column set, materialized to arrays at end."""

    post_gid: list = field(default_factory=list)
    swc_type: list = field(default_factory=list)
    dest_sectype: list = field(default_factory=list)
    mech_id: list = field(default_factory=list)
    syn_id: list = field(default_factory=list)

    c_pre_gid: list = field(default_factory=list)
    c_syn_row: list = field(default_factory=list)
    c_post_pop: list = field(default_factory=list)
    c_pre_pop: list = field(default_factory=list)
    c_mech_id: list = field(default_factory=list)
    c_swc_type: list = field(default_factory=list)
    c_dest_sectype: list = field(default_factory=list)
    c_weight: list = field(default_factory=list)
    c_delay: list = field(default_factory=list)
    c_wslot: list = field(default_factory=list)


class SynapseBuilder:
    """Builds the synapse/connection tables for the local rank.

    Parameters come from the model's synapse rules merged over the
    system's ``connections_config`` mechanisms. Mechanisms whose
    ``tau_decay`` is null are skipped as inactive.
    """

    def __init__(
        self,
        system,
        model,
        pc,
        comm,
        store: str = "python",
        selected_gids=None,
        simulated_pops=None,
        io_size: int = 1,
        auto_store_threshold: int = 200_000,
    ):
        self.system = system
        self.model = model
        self.pc = pc
        self.comm = comm
        self.store_kind = store
        # number of ranks that perform file I/O in the scoped scatter reads;
        # >1 parallelizes reading instead of funnelling through one rank
        self._io_size = max(1, int(io_size))
        # for store="auto", promote to the C++-side store once the wired synapse
        # count exceeds this (chosen after the read pass, when the count is known)
        self._auto_store_threshold = int(auto_store_threshold)
        # when set, only recurrent edges (from a simulated source) whose source
        # gid is selected are wired. edges from external input populations are
        # always wired so subselected networks keep their external drive
        self._selected_gids = selected_gids
        # populations built as biophysical cells with any other source population
        # treated as an external VecStim spike source
        self._simulated_pops = (
            set(simulated_pops) if simulated_pops is not None else None
        )
        # external input sources: gid -> VecStim (kept alive here)
        self.input_vecstims: dict[int, object] = {}
        self._input_ncs: list = []

        self._mech_map = model.neuron_synapse_mechanisms()  # AMPA -> LinExp2Syn
        self._rules = model.neuron_synapse_rules()  # LinExp2Syn -> {...}
        self._ignored = (
            set(model.ignored_populations())
            if hasattr(model, "ignored_populations")
            else set()
        )

        # categorical codes
        self._pop_code: dict[str, int] = {}
        self._mech_code: dict[str, int] = {}
        self._sectype_code: dict[str, int] = {}

    def _pop_id(self, name: str) -> int:
        return self._pop_code.setdefault(name, len(self._pop_code))

    def _mech_id(self, mech_name: str) -> int:
        return self._mech_code.setdefault(mech_name, len(self._mech_code))

    def _sectype_id(self, name: str) -> int:
        return self._sectype_code.setdefault(name, len(self._sectype_code))

    def _create_input_sources(self, h, gids) -> None:
        """Register VecStim spike sources this rank owns (``gid % nhost``).

        Each external input gid has exactly one owner rank so NEURON's parallel
        spike exchange has a single source per gid. Other ranks reach it through
        ``gid_connect``. The VecStim emits nothing until a train is played in
        (``Env.play_input_spikes`` / ``apply_stimulus_from_h5``).
        """
        nhost = int(self.pc.nhost())
        rank = int(self.pc.id())
        for gid in gids:
            gid = int(gid)
            if gid % nhost != rank or gid in self.input_vecstims:
                continue
            vs = h.VecStim()
            self.pc.set_gid2node(gid, rank)
            nc = h.NetCon(vs, None)
            self.pc.cell(gid, nc)
            self.input_vecstims[gid] = vs
            self._input_ncs.append(nc)

    def _mechanisms_for(self, post: str, pre: str) -> dict:
        """Active {synapse_class: params} for a (post, pre) projection."""
        syn_cfg = self.system.connections_config["synapses"][post][pre]
        mechs = syn_cfg.get("mechanisms", {}).get("default", {})
        active = {}
        for cls, params in mechs.items():
            if params.get("tau_decay") is None:
                continue
            active[cls] = params
        return active

    def build(self, cells_by_pop: dict[str, dict[int, object]]):
        from neuron import h

        g = _Growable()

        # map (post_gid, syn_id, mech_name) -> synapse table row (dedupe PPs)
        pp_rows: dict[tuple[int, int, str], int] = {}

        connections_config = self.system.connections_config["synapses"]
        simulated = (
            self._simulated_pops
            if self._simulated_pops is not None
            else set(cells_by_pop.keys())
        )

        # --- Pass 1: read edges, cache payloads, collect needed input gids ----
        # cached entry: (post_id, pre_id, is_input, active, cell, place, pre_gids,
        #                syn_ids, distances)
        cached: list = []
        needed_inputs: set[int] = set()
        # Iterate a rank-consistent population order and always issue the
        # collective reads, even for a rank that owns no cells here since
        # skipping would desync the collective scatter reads and deadlock.
        for post in cells_by_pop:
            if post in self._ignored:
                continue
            cells = cells_by_pop[post]
            post_id = self._pop_id(post)
            placement = self._read_placement(post, set(cells.keys()))

            pre_pops = list(connections_config.get(post, {}).keys())
            for pre in pre_pops:
                if pre in self._ignored:
                    continue
                active = self._mechanisms_for(post, pre)
                if not active:
                    continue
                pre_id = self._pop_id(pre)
                is_input = pre not in simulated

                for post_gid, (pre_gids, projection) in self._read_projection(
                    pre, post, set(cells.keys())
                ):
                    if post_gid not in cells:
                        continue
                    place = placement.get(post_gid, {})
                    pre_gids = np.asarray(pre_gids)
                    syn_ids, distances = _edge_syn_ids_distances(projection, pre_gids)
                    cached.append(
                        (
                            post_gid,
                            post_id,
                            pre_id,
                            is_input,
                            active,
                            cells[post_gid],
                            place,
                            pre_gids,
                            syn_ids,
                            distances,
                        )
                    )
                    if is_input:
                        for k in range(len(pre_gids)):
                            if int(syn_ids[k]) in place:
                                needed_inputs.add(int(pre_gids[k]))

        # --- Gather + create the input sources this rank owns -----------------
        if self.comm is not None and int(self.pc.nhost()) > 1:
            for part in self.comm.allgather(needed_inputs):
                needed_inputs |= part
        self._create_input_sources(h, needed_inputs)

        # --- Choose object store now that the synapse count is known ----------
        kind = self.store_kind
        if kind == "auto":
            est = sum(len(entry[7]) * len(entry[4]) for entry in cached)
            kind = "hoc" if est >= self._auto_store_threshold else "python"
        self.store_kind = kind
        syn_store = make_store(kind)
        nc_store = make_store(kind)

        # --- Pass 2: wire ------------------------------------------------------
        # Precompute per-mechanism specs once (constant per projection's `active`
        # dict) rather than per synapse. Bind hot attributes/methods to locals.
        spec_cache: dict[int, list] = {}
        gid_connect = self.pc.gid_connect
        VEL = DEFAULT_VELOCITY
        BUILD_FLOOR = 2 * DEFAULT_DT
        for (
            post_gid,
            post_id,
            pre_id,
            is_input,
            active,
            cell,
            place,
            pre_gids,
            syn_ids,
            distances,
        ) in cached:
            specs = spec_cache.get(id(active))
            if specs is None:
                specs = self._mech_specs(h, active)
                spec_cache[id(active)] = specs

            place_get = place.get
            cell_place = cell.place
            dest_code = {}  # swc_type -> dest_sectype code (per-cell tiny cache)
            cell_dest = cell.dest_sec_type
            sel = None if is_input else self._selected_gids
            pre_list = pre_gids.tolist()
            syn_list = syn_ids.tolist()
            dist_list = distances.tolist()

            for k in range(len(pre_list)):
                pre_gid = pre_list[k]
                if sel is not None and pre_gid not in sel:
                    continue
                sid = syn_list[k]
                site = place_get(sid)
                if site is None:
                    continue
                swc_type, loc = site
                seg = cell_place(swc_type, loc)
                dsec = dest_code.get(swc_type)
                if dsec is None:
                    dsec = self._sectype_id(cell_dest(swc_type))
                    dest_code[swc_type] = dsec
                phys = dist_list[k] / VEL  # physical (distance) delay, dt-independent
                delay = phys if phys > BUILD_FLOOR else BUILD_FLOOR

                for mech_name, pp_cls, set_params, mid, wslot, w0_items, wval in specs:
                    key = (post_gid, sid, mech_name)
                    row = pp_rows.get(key)
                    if row is None:
                        pp = pp_cls(seg)
                        for pname, val in set_params:
                            setattr(pp, pname, val)
                        row = syn_store.append(pp)
                        pp_rows[key] = row
                        g.post_gid.append(post_gid)
                        g.swc_type.append(swc_type)
                        g.dest_sectype.append(dsec)
                        g.mech_id.append(mid)
                        g.syn_id.append(sid)
                    else:
                        pp = syn_store.get(row)

                    nc = gid_connect(pre_gid, pp)
                    nc.delay = delay
                    for slot, val in w0_items:
                        nc.weight[slot] = val
                    nc_store.append(nc)

                    g.c_pre_gid.append(pre_gid)
                    g.c_syn_row.append(row)
                    g.c_post_pop.append(post_id)
                    g.c_pre_pop.append(pre_id)
                    g.c_mech_id.append(mid)
                    g.c_swc_type.append(swc_type)
                    g.c_dest_sectype.append(dsec)
                    g.c_weight.append(wval)
                    g.c_delay.append(
                        phys
                    )  # physical delay; effective = max(phys, 2*dt)
                    g.c_wslot.append(wslot)

        syn = SynapseTable(
            post_gid=np.asarray(g.post_gid, dtype=np.int32),
            swc_type=np.asarray(g.swc_type, dtype=np.int8),
            dest_sectype=np.asarray(g.dest_sectype, dtype=np.int8),
            mech_id=np.asarray(g.mech_id, dtype=np.int16),
            syn_id=np.asarray(g.syn_id, dtype=np.int64),
            store=syn_store,
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
            delay=np.asarray(g.c_delay, dtype=np.float32),
            wslot=np.asarray(g.c_wslot, dtype=np.int8),
            store=nc_store,
        )
        return (
            syn,
            conn,
            dict(self._pop_code),
            dict(self._mech_code),
            dict(self._sectype_code),
            self.input_vecstims,
        )

    def _read_placement(self, population: str, local_gids: set[int]):
        """gid -> {syn_id: (swc_type, loc)} for local cells."""
        from neuroh5.io import scatter_read_cell_attribute_selection

        # Note: always call the collective scatter read even for an empty
        # selection, so a rank owning no cells of this population still
        # participates, otherwise ranks desync and deadlock.
        out: dict[int, dict[int, tuple[int, float]]] = {}

        it, info = scatter_read_cell_attribute_selection(
            self.system.files["cells"],
            population,
            sorted(local_gids),
            namespace="Synapse Attributes",
            mask={"syn_ids", "swc_types", "syn_locs"},
            comm=self.comm,
            io_size=self._io_size,
            return_type="tuple",
        )
        i_ids = info.get("syn_ids")
        i_swc = info.get("swc_types")
        i_loc = info.get("syn_locs")
        if i_ids is None:
            return out
        for gid, data in it:
            syn_ids = np.asarray(data[i_ids])
            swc = np.asarray(data[i_swc])
            locs = np.asarray(data[i_loc])
            out[int(gid)] = {
                int(syn_ids[i]): (int(swc[i]), float(locs[i]))
                for i in range(len(syn_ids))
            }
        return out

    def _read_projection(self, pre: str, post: str, local_gids: set[int]):
        """Yield (post_gid, (pre_gids, projection)) scoped to ``local_gids``.

        Uses ``scatter_read_graph`` with ``node_allocation`` so only edges onto
        local post cells are read and, unlike ``read_graph_selection``, does
        not abort for gids absent from a projection.
        """
        from neuroh5.io import scatter_read_graph

        filepath = self.system.files["connections"]
        graph, _ = scatter_read_graph(
            filepath,
            comm=self.comm,
            io_size=self._io_size,
            projections=[(pre, post)],
            namespaces=["Synapses", "Connections"],
            node_allocation=set(local_gids),
        )
        if post in graph and pre in graph[post]:
            yield from graph[post][pre]

    def _mech_specs(self, h, active: dict) -> list:
        """Precompute per-mechanism creation specs for a projection's ``active``.

        Returns one tuple per active mechanism:
        ``(mech_name, pp_class, [(param, value)...], mech_id, weight_slot,
        [(slot, value)...], tunable_weight_value)``
        """
        specs = []
        for cls_name, params in active.items():
            mech_name = self._mech_map.get(cls_name, cls_name)
            pp_cls = getattr(h, mech_name)
            rule = self._rules.get(mech_name, {})
            set_params = [
                (p, float(params[p]))
                for p in rule.get("mech_params", [])
                if params.get(p) is not None
            ]
            netcon_params = rule.get("netcon_params", {"weight": 0})
            wslot = int(netcon_params.get("weight", 0))
            w0: dict[int, float] = {}
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
                    pp_cls,
                    set_params,
                    self._mech_id(mech_name),
                    wslot,
                    list(w0.items()),
                    w0.get(wslot, 0.0),
                )
            )
        return specs


def _edge_syn_ids_distances(projection, pre_gids):
    """Extract per-edge (syn_id, distance) arrays from a projection payload."""
    n = len(pre_gids)
    syn_ids = np.zeros(n, dtype=np.int64)
    distances = np.zeros(n, dtype=np.float64)
    if isinstance(projection, dict):
        if "Synapses" in projection:
            s = projection["Synapses"]
            syn_ids = np.asarray(s[0] if isinstance(s, (list, tuple)) else s).astype(
                np.int64
            )
        if "Connections" in projection:
            c = projection["Connections"]
            distances = np.asarray(c[0] if isinstance(c, (list, tuple)) else c).astype(
                np.float64
            )
    return syn_ids, distances
