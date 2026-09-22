from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from livn.types import DEFAULT_CONDUCTION_VELOCITY, conduction_velocity

logger = logging.getLogger(__name__)

DEFAULT_VELOCITY = DEFAULT_CONDUCTION_VELOCITY
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
    """One row per receptor of a point process instantiated on a local cell.

    A point process whose rule is ``shared`` serves every synapse on its
    segment with the same kinetics, and one whose rule has ``channels`` carries
    several receptors with a row for each. Such rows reference the same object
    in ``store``, and a shared one's ``syn_id`` is -1 since no synapse owns it.
    """

    post_gid: np.ndarray  # int32[S]
    swc_type: np.ndarray  # int8[S]   (original placement type, for selection)
    dest_sectype: np.ndarray  # int8[S]  destination section-type code
    mech_id: np.ndarray  # int16[S]  the receptor's own mechanism
    # int8[S] the receptor (connections_config synapse class, e.g. GABA_B); unlike
    # the mechanism, it tells apart receptors that one mechanism implements
    receptor: np.ndarray
    syn_id: np.ndarray  # int64[S]   neuroh5 synapse id, -1 when shared
    store: ObjectStore  # point processes, one entry per row
    # int16[S] index into ``param_names``, the receptor's parameter names on the
    # point process; None, or an entry of None, keeps the mechanism's own names
    param_map: np.ndarray | None = None
    param_names: list = field(default_factory=lambda: [None])

    @property
    def size(self) -> int:
        return len(self.post_gid)

    def attribute(self, row: int, param: str) -> str | None:
        """The point process attribute holding ``param`` of the receptor in ``row``."""
        if self.param_map is None:
            return param
        names = self.param_names[int(self.param_map[row])]
        return param if names is None else names.get(param)


@dataclass
class ConnectionTable:
    """One row per connection and receptor (edge x mechanism).

    A NetCon drives every receptor of its point process, so the rows of a
    multi-receptor point process share one, which ``nc_row`` names.
    """

    pre_gid: np.ndarray  # int32[C]
    syn_row: np.ndarray  # int32[C]  -> SynapseTable row
    post_pop: np.ndarray  # int8[C]
    pre_pop: np.ndarray  # int8[C]
    mech_id: np.ndarray  # int16[C]
    receptor: np.ndarray  # int8[C]  receptor code, see SynapseTable.receptor
    swc_type: np.ndarray  # int8[C]
    dest_sectype: np.ndarray  # int8[C]  destination section-type code
    weight: np.ndarray  # float64[C]  mirror of nc.weight[wslot]
    delay: (
        np.ndarray
    )  # float32[C]  physical (distance) delay; NetCon = max(delay, 2*dt)
    wslot: np.ndarray  # int8[C]
    store: ObjectStore  # NetCons
    # int32[C] -> NetCon in ``store``; None when row i is NetCon i
    nc_row: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.pre_gid)

    def netcon(self, row: int):
        """The NetCon that carries connection ``row``."""
        return self.store.get(row if self.nc_row is None else int(self.nc_row[row]))

    def netcons(self) -> tuple[np.ndarray, np.ndarray]:
        """Each NetCon once: its index in ``store`` and a row it carries."""
        if self.nc_row is None:
            rows = np.arange(self.size)
            return rows, rows
        return np.unique(self.nc_row, return_index=True)


class _Columns:
    """Append-only typed columns that become numpy arrays without a copy.

    A list of Python ints and floats costs several times the bytes of the
    values it holds, and these columns have a row per NetCon.
    """

    def __init__(self, **typecodes: str):
        for name, code in typecodes.items():
            setattr(self, name, array(code))

    def asarray(self, name: str, dtype) -> np.ndarray:
        column = getattr(self, name)
        if not len(column):
            return np.empty(0, dtype=dtype)
        return np.frombuffer(column, dtype=dtype)


@dataclass(frozen=True, slots=True)
class _PointProcessPlan:
    """A point process a synapse needs, and the NetCon a connection makes onto it."""

    name: str  # NEURON mechanism
    pp_cls: object
    set_params: tuple  # ((attribute, value), ...)
    shared: bool
    receptors: tuple  # receptor codes, in row order
    # names (mechanism, parameters, receptors) in a shared point process's key
    token: int
    w0_items: tuple  # ((NetCon weight slot, value), ...)
    syn_rows: tuple  # ((mech_id, receptor, param_map), ...), a row per receptor
    # ((mech_id, receptor, weight slot, tunable weight), ...), per receptor
    conn_rows: tuple


def split_by_owner(gids: np.ndarray, hosts: np.ndarray) -> list[np.ndarray]:
    """``gids`` split by the comm rank of the host that owns each (``gid % nhost``).

    ``hosts[r]`` is the ParallelContext host of comm rank ``r``, which need not
    be ``r``; ``len(hosts)`` is the number of hosts.
    """
    comm_rank_of = np.empty(len(hosts), dtype=np.int64)
    comm_rank_of[hosts] = np.arange(len(hosts))
    destination = comm_rank_of[gids % len(hosts)]
    order = np.argsort(destination, kind="stable")
    counts = np.bincount(destination, minlength=len(hosts))
    return np.split(gids[order], np.cumsum(counts)[:-1])


_NO_PLACEMENT = (
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=np.float64),
)


class SynapseBuilder:
    """Builds the synapse/connection tables for the local rank.

    Parameters come from the model's synapse rules merged over the
    system's ``connections_config`` mechanisms, which a projection states
    either once (``default``) or per destination SWC type. Mechanisms whose
    ``tau_decay`` is null are skipped as inactive.

    A rule may let synapses share a point process. ``shared`` is for a
    mechanism whose state sums linearly over its events, so that one point
    process on a segment is the sum of one per synapse there; it must keep its
    per-connection quantities (weight, unitary conductance) on the NetCon.
    ``channels`` names a mechanism that carries several receptors behind one
    NetCon, used where one connection has all of them::

        "LinExp2SynAMPANMDA": {
            "shared": True,
            "channels": [
                {"mechanism": "LinExp2Syn", "mech_params": {"tau_rise": "tau_rise"}},
                {"mechanism": "LinExp2SynNMDA", "netcon_offset": 2,
                 "mech_params": {"tau_rise": "nmda_tau_rise"}},
            ],
        }

    Each channel's parameters are renamed through its ``mech_params`` and its
    NetCon slots shifted by ``netcon_offset``. Rows in the tables stay per
    receptor, so weights and parameters are addressed as before. ``share=False``
    ignores both and builds a point process per synapse and receptor.
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
        share: bool = True,
    ):
        self.system = system
        self.model = model
        self.pc = pc
        self.comm = comm
        self.store_kind = store

        if io_size is not None and hasattr(system, "io_size"):
            system.io_size = max(1, int(io_size))
        # for store="auto", promote to the C++-side store once the wired synapse
        # count exceeds this (chosen after the read pass, when the count is known)
        self._auto_store_threshold = int(auto_store_threshold)
        # when set, only recurrent edges (from a simulated source) whose source
        # gid is selected are wired. edges from external input populations are
        # always wired so subselected networks keep their external drive
        self._selected_gids = selected_gids
        self._microcircuit_inputs = bool(
            model.neuron_microcircuit_inputs()
            if hasattr(model, "neuron_microcircuit_inputs")
            else False
        )
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

        self._share = bool(share)
        self._composites = (
            [(name, rule) for name, rule in self._rules.items() if rule.get("channels")]
            if self._share
            else []
        )
        self._tokens: dict[tuple, int] = {}
        self._param_names: list = [None]
        self._param_map_ids: dict[tuple, int] = {}

        # categorical codes
        self._pop_code: dict[str, int] = {}
        self._mech_code: dict[str, int] = {}
        self._receptor_code: dict[str, int] = {}
        self._sectype_code: dict[str, int] = {}

    def _is_cell(self, gid: int) -> bool:
        """Whether ``gid`` is built as a cell rather than replayed from file."""
        return self._selected_gids is None or gid in self._selected_gids

    def _pop_id(self, name: str) -> int:
        return self._pop_code.setdefault(name, len(self._pop_code))

    def _mech_id(self, mech_name: str) -> int:
        return self._mech_code.setdefault(mech_name, len(self._mech_code))

    def _receptor_id(self, receptor: str) -> int:
        return self._receptor_code.setdefault(receptor, len(self._receptor_code))

    def _sectype_id(self, name: str) -> int:
        return self._sectype_code.setdefault(name, len(self._sectype_code))

    def _route_inputs(self, needed: np.ndarray) -> np.ndarray:
        """The input gids any rank needs that this rank owns (``gid % nhost``)."""
        if self.comm is None or int(self.pc.nhost()) == 1:
            return needed
        hosts = np.asarray(self.comm.allgather(int(self.pc.id())), dtype=np.int64)
        received = self.comm.alltoall(split_by_owner(needed, hosts))
        return np.unique(np.concatenate(received))

    def _create_input_sources(self, h, gids: np.ndarray) -> None:
        """Register VecStim spike sources this rank owns (``gid % nhost``).

        Each external input gid has exactly one owner rank so NEURON's parallel
        spike exchange has a single source per gid. Other ranks reach it through
        ``gid_connect``. The VecStim emits nothing until a train is played in
        (``Env.play_input_spikes`` / ``apply_stimulus_from_h5``).
        """
        nhost = int(self.pc.nhost())
        rank = int(self.pc.id())
        for gid in gids.tolist():
            if gid % nhost != rank or gid in self.input_vecstims:
                continue
            vs = h.VecStim()
            self.pc.set_gid2node(gid, rank)
            nc = h.NetCon(vs, None)
            self.pc.cell(gid, nc)
            self.input_vecstims[gid] = vs
            self._input_ncs.append(nc)

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

    def build(self, cells_by_pop: dict[str, dict[int, object]]):
        from neuron import h

        connections_config = self.system.connections_config["synapses"]
        simulated = (
            self._simulated_pops
            if self._simulated_pops is not None
            else set(cells_by_pop.keys())
        )

        selected_sorted = (
            None
            if self._selected_gids is None
            else np.array(sorted(self._selected_gids), dtype=np.int64)
        )

        # --- Pass 1: read edges, cache payloads, collect needed input gids ----
        # cached entry: (post_gid, post_id, pre_id, is_input, active, cell,
        #                placement, pre_gids, syn_ids, distances)
        cached: list = []
        needed: list[np.ndarray] = []
        # Iterate a rank-consistent population order and always issue the
        # collective reads, even for a rank that owns no cells here since
        # skipping would desync the collective scatter reads and deadlock.
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

            # ``connections_config`` is replicated on every rank, so deciding
            # here is rank-consistent and skips the collective placement read
            # entirely for an unconnected population (which has no H5 to read)
            if not active_by_pre:
                continue
            placement = self.system.placement(post, set(cells.keys()))

            for pre, active in active_by_pre.items():
                pre_id = self._pop_id(pre)
                is_input = pre not in simulated

                for post_gid, (pre_gids, projection) in self.system.edges(
                    pre, post, set(cells.keys())
                ):
                    if post_gid not in cells:
                        continue
                    place = placement.get(post_gid, _NO_PLACEMENT)
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
                    if is_input or self._microcircuit_inputs:
                        sources = np.asarray(pre_gids, dtype=np.int64)[
                            np.isin(np.asarray(syn_ids, dtype=np.int64), place[0])
                        ]
                        if not is_input and selected_sorted is not None:
                            # a source of a simulated population is external
                            # only where it is not itself a built cell
                            sources = sources[
                                np.isin(sources, selected_sorted, invert=True)
                            ]
                        needed.append(np.unique(sources))

        # --- Route + create the input sources this rank owns ------------------
        local = np.unique(np.concatenate(needed)) if needed else np.empty(0, np.int64)
        del needed
        self._create_input_sources(h, self._route_inputs(local))
        del local

        # --- Choose object store now that the synapse count is known ----------
        kind = self.store_kind
        if kind == "auto":
            est = sum(len(entry[7]) * len(entry[4]) for entry in cached)
            kind = "hoc" if est >= self._auto_store_threshold else "python"
        self.store_kind = kind
        syn_store = make_store(kind)
        nc_store = make_store(kind)

        # --- Pass 2: wire ------------------------------------------------------
        s = _Columns(
            post_gid="i",
            swc_type="b",
            dest_sectype="b",
            mech_id="h",
            receptor="b",
            syn_id="q",
            param_map="h",
        )
        c = _Columns(
            pre_gid="i",
            syn_row="i",
            post_pop="b",
            pre_pop="b",
            mech_id="h",
            receptor="b",
            swc_type="b",
            dest_sectype="b",
            weight="d",
            delay="f",
            wslot="b",
            nc_row="i",
        )
        # a point process's first synapse table row: per synapse, keyed by
        # (post_gid, syn_id, receptors), or when shared, by (section, segment
        # centre, mechanism, parameters and receptors, destination section type).
        # Receptors, not the mechanism: GABA_A and GABA_B may both be LinExp2Syn
        # on one synapse, and each needs its own kinetics
        pp_rows: dict[tuple, int] = {}
        shared_rows: dict[tuple, int] = {}

        # Precompute per-mechanism plans once (constant per projection's
        # `active` dict) rather than per synapse. Bind hot attributes to locals.
        plan_cache: dict[int, dict] = {}
        gid_connect = self.pc.gid_connect
        VEL = conduction_velocity(self.system)
        BUILD_FLOOR = 2 * DEFAULT_DT
        for (
            post_gid,
            post_id,
            pre_id,
            is_input,
            active,
            cell,
            (place_ids, place_swc, place_loc),
            pre_gids,
            syn_ids,
            distances,
        ) in cached:
            plans_by_swc = plan_cache.get(id(active))
            if plans_by_swc is None:
                plans_by_swc = {
                    swc: self._plan(h, self._mech_specs(h, mechs))
                    for swc, mechs in active.items()
                }
                plan_cache[id(active)] = plans_by_swc
            default_plans = plans_by_swc.get(None)

            if len(place_ids):
                at = np.searchsorted(place_ids, syn_ids)
                at[at == len(place_ids)] = 0
                found_list = (place_ids[at] == syn_ids).tolist()
                swc_list = place_swc[at].tolist()
                loc_list = place_loc[at].tolist()
            else:
                found_list = [False] * len(syn_ids)
                swc_list = loc_list = found_list

            cell_place = cell.place
            dest_code = {}  # swc_type -> dest_sectype code (per-cell tiny cache)
            cell_dest = cell.dest_sec_type
            sel = (
                None if (is_input or self._microcircuit_inputs) else self._selected_gids
            )
            pre_list = pre_gids.tolist()
            syn_list = syn_ids.tolist()
            dist_list = distances.tolist()

            for k in range(len(pre_list)):
                pre_gid = pre_list[k]
                if sel is not None and pre_gid not in sel:
                    continue
                if not found_list[k]:
                    continue
                sid = syn_list[k]
                swc_type = swc_list[k]
                plans = plans_by_swc.get(swc_type, default_plans)
                if plans is None:
                    continue  # no mechanism declared for this destination type
                seg = cell_place(swc_type, loc_list[k])
                dsec = dest_code.get(swc_type)
                if dsec is None:
                    dsec = self._sectype_id(cell_dest(swc_type))
                    dest_code[swc_type] = dsec
                phys = dist_list[k] / VEL  # physical (distance) delay, dt-independent
                delay = phys if phys > BUILD_FLOOR else BUILD_FLOOR

                for plan in plans:
                    if plan.shared:
                        key = (seg.sec, seg.x, plan.token, dsec)
                        rows = shared_rows
                    else:
                        key = (post_gid, sid, plan.receptors)
                        rows = pp_rows
                    row = rows.get(key)
                    if row is None:
                        pp = plan.pp_cls(seg)
                        for attribute, value in plan.set_params:
                            setattr(pp, attribute, value)
                        row = len(syn_store)
                        rows[key] = row
                        owner = -1 if plan.shared else sid
                        for mid, rid, param_map in plan.syn_rows:
                            syn_store.append(pp)
                            s.post_gid.append(post_gid)
                            s.swc_type.append(swc_type)
                            s.dest_sectype.append(dsec)
                            s.mech_id.append(mid)
                            s.receptor.append(rid)
                            s.syn_id.append(owner)
                            s.param_map.append(param_map)
                    else:
                        pp = syn_store.get(row)

                    nc = gid_connect(pre_gid, pp)
                    nc.delay = delay
                    for slot, value in plan.w0_items:
                        nc.weight[slot] = value
                    nc_index = nc_store.append(nc)

                    for offset, (mid, rid, wslot, wval) in enumerate(plan.conn_rows):
                        c.pre_gid.append(pre_gid)
                        c.syn_row.append(row + offset)
                        c.post_pop.append(post_id)
                        c.pre_pop.append(pre_id)
                        c.mech_id.append(mid)
                        c.receptor.append(rid)
                        c.swc_type.append(swc_type)
                        c.dest_sectype.append(dsec)
                        c.weight.append(wval)
                        # physical delay; effective = max(phys, 2*dt)
                        c.delay.append(phys)
                        c.wslot.append(wslot)
                        c.nc_row.append(nc_index)
        cached.clear()
        del pp_rows, shared_rows

        syn = SynapseTable(
            post_gid=s.asarray("post_gid", np.int32),
            swc_type=s.asarray("swc_type", np.int8),
            dest_sectype=s.asarray("dest_sectype", np.int8),
            mech_id=s.asarray("mech_id", np.int16),
            receptor=s.asarray("receptor", np.int8),
            syn_id=s.asarray("syn_id", np.int64),
            store=syn_store,
            param_map=(
                s.asarray("param_map", np.int16) if len(self._param_names) > 1 else None
            ),
            param_names=list(self._param_names),
        )
        conn = ConnectionTable(
            pre_gid=c.asarray("pre_gid", np.int32),
            syn_row=c.asarray("syn_row", np.int32),
            post_pop=c.asarray("post_pop", np.int8),
            pre_pop=c.asarray("pre_pop", np.int8),
            mech_id=c.asarray("mech_id", np.int16),
            receptor=c.asarray("receptor", np.int8),
            swc_type=c.asarray("swc_type", np.int8),
            dest_sectype=c.asarray("dest_sectype", np.int8),
            weight=c.asarray("weight", np.float64),
            delay=c.asarray("delay", np.float32),
            wslot=c.asarray("wslot", np.int8),
            store=nc_store,
            nc_row=(
                c.asarray("nc_row", np.int32)
                if len(c.nc_row) != len(nc_store)
                else None
            ),
        )
        return (
            syn,
            conn,
            dict(self._pop_code),
            dict(self._mech_code),
            dict(self._sectype_code),
            self.input_vecstims,
            dict(self._receptor_code),
        )

    def _mech_specs(self, h, active: dict) -> list:
        """Precompute per-mechanism creation specs for a projection's ``active``.

        Returns one tuple per active mechanism:
        ``(mech_name, pp_class, [(param, value)...], mech_id, weight_slot,
        [(slot, value)...], tunable_weight_value, shared, receptor_id)``
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
                    self._share and bool(rule.get("shared", False)),
                    self._receptor_id(cls_name),
                )
            )
        return specs

    def _plan(self, h, specs: list) -> list[_PointProcessPlan]:
        """The point processes one synapse with these mechanisms needs.

        Receptors a ``channels`` rule covers go into one of its point processes
        each time all of them are present; the rest get one of their own, in the
        order they are declared.
        """
        remaining = list(specs)
        plans = []
        for name, rule in self._composites:
            channels = rule["channels"]
            while True:
                pool = list(remaining)
                picked = []
                for channel in channels:
                    at = next(
                        (
                            i
                            for i, spec in enumerate(pool)
                            if spec[0] == channel["mechanism"]
                        ),
                        None,
                    )
                    if at is None:
                        break
                    picked.append((pool.pop(at), channel))
                if len(picked) < len(channels):
                    break
                remaining = pool
                plans.append(self._composite_plan(h, name, rule, picked))
        for spec in remaining:
            mech_name, pp_cls, set_params, mid, wslot, w0, wval, shared, rid = spec
            plans.append(
                _PointProcessPlan(
                    name=mech_name,
                    pp_cls=pp_cls,
                    set_params=tuple(set_params),
                    shared=shared,
                    receptors=(rid,),
                    token=self._token(mech_name, set_params, (rid,)),
                    w0_items=tuple(w0),
                    syn_rows=((mid, rid, 0),),
                    conn_rows=((mid, rid, wslot, wval),),
                )
            )
        return plans

    def _composite_plan(self, h, name: str, rule: dict, picked) -> _PointProcessPlan:
        set_params, w0_items, syn_rows, conn_rows, receptors = [], [], [], [], []
        for spec, channel in picked:
            mech_name, _, params, mid, wslot, w0, wval, _, rid = spec
            names = channel.get("mech_params", {})
            offset = int(channel.get("netcon_offset", 0))
            for param, value in params:
                attribute = names.get(param)
                if attribute is None:
                    raise ValueError(
                        f"{name} carries {mech_name} but its channel does not "
                        f"name an attribute for {mech_name}'s {param!r}; add it "
                        "to that channel's mech_params"
                    )
                set_params.append((attribute, value))
            w0_items.extend((slot + offset, value) for slot, value in w0)
            syn_rows.append((mid, rid, self._param_map_id(names)))
            conn_rows.append((mid, rid, wslot + offset, wval))
            receptors.append(rid)
        return _PointProcessPlan(
            name=name,
            pp_cls=getattr(h, name),
            set_params=tuple(set_params),
            shared=bool(rule.get("shared", False)),
            receptors=tuple(receptors),
            token=self._token(name, set_params, receptors),
            w0_items=tuple(w0_items),
            syn_rows=tuple(syn_rows),
            conn_rows=tuple(conn_rows),
        )

    def _token(self, name: str, set_params, receptors) -> int:
        key = (name, tuple(set_params), tuple(receptors))
        return self._tokens.setdefault(key, len(self._tokens))

    def _param_map_id(self, names: dict) -> int:
        key = tuple(sorted(names.items()))
        found = self._param_map_ids.get(key)
        if found is None:
            found = self._param_map_ids[key] = len(self._param_names)
            self._param_names.append(dict(names))
        return found


def _edge_syn_ids_distances(projection, pre_gids):
    """Extract per-edge (syn_id, distance) arrays from a projection payload."""
    from livn.system import projection_attribute

    n = len(pre_gids)
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
