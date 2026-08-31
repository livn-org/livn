import json
import math
import os
from typing import Literal

import pandas as pd
from machinable import Interface, get
from machinable.config import Field as ConfigField
from machinable.config import to_dict
from pydantic import BaseModel, ConfigDict

from livn.utils import ObjSpec, P, import_instance


def _pj(p):
    return json.dumps(p, indent=4, sort_keys=True)


def _terminal_width(default: int = 200) -> int:
    import shutil

    return shutil.get_terminal_size((default, 24)).columns


def _numeric(name) -> bool:
    try:
        float(str(name).strip())
    except (TypeError, ValueError):
        return False
    return True


def _constraint_value(value) -> float:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return float(value)


def decode_strict(target, model, space, raw: dict) -> dict:
    unknown = sorted(set(raw) - set(space))
    if unknown:
        raise ValueError(
            f"{unknown} are not in this target's search space, so they have no "
            "inverse transform and would be emitted at their encoded value. "
            "The target is probably built differently from the run that "
            f"produced them as this offers {sorted(space)}."
        )
    return target.decode_params(raw, model=model)


def retained_in_degree(system, selection) -> float | None:
    if not selection:
        return None

    import numpy as npn

    kernels = [
        (spec or {})["kernel"]
        for sources in (system.connections_config.get("synapses") or {}).values()
        for spec in (sources or {}).values()
        if (spec or {}).get("kernel", {}).get("sigma")
    ]
    if not kernels:
        return None
    widest = max(kernels, key=lambda k: float(k["sigma"]))
    sigma = float(widest["sigma"])
    shape = str(widest.get("kernel") or "exponential")

    rows = [
        system.coordinate_array(p)
        for p in system.populations
        if system.population_count(p)
    ]
    if not rows:
        return None
    coordinates = npn.vstack(rows)
    gids = coordinates[:, 0].astype(npn.int64)
    xy = coordinates[:, 1:3]

    kept = npn.zeros(len(gids), dtype=bool)
    wanted = {int(g) for v in selection.values() for g in v}
    for i, gid in enumerate(gids):
        kept[i] = int(gid) in wanted
    if kept.sum() < 2:
        return None

    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    if shape == "gaussian":
        weights = npn.exp(-d2 / (2.0 * sigma**2))
    else:  # exponential, matching the generator's own fallback
        weights = npn.exp(-npn.sqrt(d2) / sigma)
    npn.fill_diagonal(weights, 0.0)

    total = weights.sum(axis=1)
    inside = weights[:, kept].sum(axis=1)
    with npn.errstate(divide="ignore", invalid="ignore"):
        ratio = npn.where(total > 0, inside / total, 0.0)
    return float(ratio[kept].mean())


def electrode_coverage(system, mea, selection=None) -> float | None:
    """Fraction of electrodes with a cell inside their recording radius.

    An electrode that reaches no cell records nothing whatever the parameters
    are, so this is a ceiling on `active_fraction` that no tune can raise. The
    culture's own floor was measured over its own array and knows nothing about
    this one over this graph.
    """
    import numpy as npn

    electrodes = npn.asarray(mea.get("electrode_coordinates") or (), dtype=float)
    if not len(electrodes):
        return None

    rows = [
        system.coordinate_array(p)
        for p in system.populations
        if system.population_count(p)
    ]
    if not rows:
        return None
    coordinates = npn.vstack(rows)

    if selection:
        wanted = {int(g) for v in system.selection(selection).values() for g in v}
        keep = npn.array([int(g) in wanted for g in coordinates[:, 0]], dtype=bool)
        coordinates = coordinates[keep]
        if not len(coordinates):
            return None

    xyz = coordinates[:, 1:4].astype(float)
    reached = npn.zeros(len(electrodes), dtype=bool)
    radius = float(mea["output_radius"])
    for i, electrode in enumerate(electrodes):  # one row at a time; the cross
        d = npn.linalg.norm(xyz - electrode[1:4], axis=1)  # product does not fit
        reached[i] = bool((d <= radius).any())
    return float(reached.mean())


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def node_memory_gib() -> float:
    """Memory a node is assumed to have, in GiB.

    ``LIVN_WORKER_MEMORY_MAX`` states it for a cluster whose nodes are not this
    machine; otherwise this machine's own RAM, which is right for a local run
    and the only figure available without asking the scheduler.
    """
    stated = _env_float("LIVN_WORKER_MEMORY_MAX")
    if stated is not None:
        return stated
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def node_cores() -> int:
    """Ranks a node can run, from the scheduler's view of it if there is one."""
    for name in ("LIVN_CORES_PER_NODE", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(name):
            return int(os.environ[name])
    return os.cpu_count() or 1


def _divisors(n: int):
    """Divisors of ``n``, ascending."""
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def _outward(wanted: int):
    """0, -1, +1, -2, +2 ... far enough to reach any worker count from 1 up."""
    yield 0
    for step in range(1, wanted + 1):
        yield -step
        yield step


WORKER_SLACK = 64


def _layout(ranks_per_worker, ranks_per_node_max, workers_wanted, max_nodes):
    """``(nodes, ranks_per_node, workers, total)`` for one ranks-per-worker."""

    def search(reach):
        best = None
        for offset in _outward(reach):
            workers = workers_wanted + offset
            if workers < 1:
                continue
            total = workers * ranks_per_worker + 1
            for nodes in _divisors(total):
                if max_nodes is not None and nodes > max_nodes:
                    break  # divisors ascend, so every later one is over too
                if total // nodes <= ranks_per_node_max:
                    key = (nodes, abs(offset))
                    if best is None or key < best[0]:
                        best = (key, (nodes, total // nodes, workers, total))
                    break  # smallest node count for this worker count
        return best[1] if best else None

    found = search(max(1, workers_wanted // 10)) or search(
        min(workers_wanted, WORKER_SLACK)
    )
    if found is not None:
        return found

    if max_nodes is None:
        return None

    best = None
    for nodes in range(1, max_nodes + 1):
        if ranks_per_worker == 1:
            ranks_per_node = ranks_per_node_max
        else:
            if math.gcd(nodes, ranks_per_worker) != 1:
                continue  # no ranks_per_node can satisfy the congruence
            residue = pow(nodes, -1, ranks_per_worker)
            ranks_per_node = (
                residue
                + ((ranks_per_node_max - residue) // ranks_per_worker)
                * ranks_per_worker
            )
            if not 1 <= ranks_per_node <= ranks_per_node_max:
                continue
        total = nodes * ranks_per_node
        workers = (total - 1) // ranks_per_worker
        if workers >= 1 and (best is None or workers > best[2]):
            best = (nodes, ranks_per_node, workers, total)
    return best


# Ranks one worker may span
MAX_RANKS_PER_WORKER = 4096


def plan_execution(
    worker_memory,
    workers_wanted: int,
    node_gib: float | None = None,
    cores_per_node: int | None = None,
    max_nodes: int | None = None,
    headroom: float = 0.9,
    min_ranks_per_worker: int = 1,
) -> dict:
    """Ranks per worker, ranks per node and nodes for a run.

    ``worker_memory(ranks)`` is what one worker of that many ranks costs, in
    bytes. Two constraints shape the answer and neither is negotiable:

    - **A rank's share must fit its slice of the node.** Ranks per worker is the
      fewest that bring it under `node / cores`, since the network divides
      between a worker's ranks while the per-process floor does not, but never
      fewer than `min_ranks_per_worker`: memory is the right rule only for a
      target that scales sublinearly with ranks, and one that does better says
      so rather than being cut to a single rank it does not want.
    - **distwq wants `(total ranks - 1) % ranks_per_worker == 0`**, the odd rank
      being the controller, and it refuses to launch otherwise. Slurm lays out
      `nodes x ranks_per_node` uniformly, so the two have to be searched
      together rather than derived.

    Workers past `workers_wanted` (an epoch's samples) would idle, so the search
    aims at that count and prefers fewer nodes, then denser ones, on a tie.
    `max_nodes=1` is what a local mpi run wants, which has only this one.

    `headroom` is what a node is planned to, not filled to: the memory model
    behind `worker_memory` is fitted, and a node planned to the last byte swaps.
    """
    node_bytes = (node_gib if node_gib is not None else node_memory_gib()) * 1024**3
    usable = node_bytes * headroom
    cores = cores_per_node if cores_per_node is not None else node_cores()
    per_rank_budget = usable / cores

    def share(ranks: int) -> float:
        return worker_memory(ranks) / ranks

    high = 1
    while share(high) > per_rank_budget:
        nxt = high * 2
        if share(nxt) > share(high) * 0.999:  # converged, and still over
            raise ValueError(
                f"a rank costs {share(nxt) / 1024**3:.2f} GiB however many share "
                f"the network, over the {per_rank_budget / 1024**3:.2f} GiB a "
                f"core gets on a {node_bytes / 1024**3:.0f} GiB / {cores} core "
                "node; state a bigger LIVN_WORKER_MEMORY_MAX, fewer "
                "LIVN_CORES_PER_NODE, or a smaller selection"
            )
        high = nxt
    low = high // 2  # known not to fit (or 0 when one rank was enough)
    while low + 1 < high:
        mid = (low + high) // 2
        if share(mid) > per_rank_budget:
            low = mid
        else:
            high = mid
    ranks_per_worker = max(high, min_ranks_per_worker)

    cap = min(MAX_RANKS_PER_WORKER, (max_nodes or MAX_RANKS_PER_WORKER) * cores)
    if ranks_per_worker > cap:
        raise ValueError(
            f"one worker would need {ranks_per_worker} ranks to bring a rank's "
            f"share under the {per_rank_budget / 1024**3:.2f} GiB a core gets "
            f"on a {node_bytes / 1024**3:.0f} GiB / {cores} core node, past the "
            f"{cap} this sizes for; the selection is too big for this cluster"
        )

    # Ranks per worker is a free variable too, not just the minimum that fits: a
    # capped cluster often cannot tile the minimum into whole nodes but can tile
    # one or two more, at the cost of an extra floor each. Try upward from the
    # cheapest.
    best = None
    for candidate in range(ranks_per_worker, cap + 1):
        ranks_per_node_max = min(cores, int(usable // share(candidate)))
        if ranks_per_node_max < 1:
            continue
        best = _layout(candidate, ranks_per_node_max, workers_wanted, max_nodes)
        if best is not None:
            ranks_per_worker = candidate
            break
        if max_nodes is None:
            break  # unbounded always tiles at `nodes = total, ranks_per_node = 1`

    if best is None:
        raise ValueError(
            f"no layout of {'any number of' if max_nodes is None else max_nodes} "
            f"node(s) x at most {cores} ranks gives a whole number of workers "
            f"plus a controller; a worker needs "
            f"{worker_memory(ranks_per_worker) / 1024**3:.1f} GiB over "
            f"{ranks_per_worker} rank(s)"
        )

    per_rank = share(ranks_per_worker)
    ranks_per_node_max = min(cores, max(1, int(usable // per_rank)))
    nodes, ranks_per_node, workers, total = best
    return {
        "ranks_per_worker": ranks_per_worker,
        "ranks_per_node": ranks_per_node,
        "nodes": nodes,
        "workers": workers,
        "workers_wanted": workers_wanted,
        "total_ranks": total,
        "node_gib": node_bytes / 1024**3,
        "cores_per_node": cores,
        "worker_gib": worker_memory(ranks_per_worker) / 1024**3,
        "node_used_gib": ranks_per_node * per_rank / 1024**3,
        "headroom": headroom,
    }


class Tune(Interface):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        system: str | int | dict[str, int] = "./systems/graphs/EI"
        selection: str | None = None
        model: ObjSpec = "livn.models.rcsd.ReducedCalciumSomaDendrite"
        target: ObjSpec = "systems.targets.EI.Culture"
        trials: int = 1
        nprocs_per_worker: int = 1
        worker_memory_max: float | None = ConfigField(None, identifying=False)
        cores_per_node: int | None = ConfigField(None, identifying=False)
        max_nodes: int | None = ConfigField(None, identifying=False)
        autosize: bool = ConfigField(False, identifying=False)
        n_initial: int = 100
        population_size: int = 100
        num_generations: int = 10
        n_epochs: int = 10

        class SurrogateConfig(BaseModel):
            method_name: (
                str
                | Literal[
                    "gpr",
                    "egp",
                    "megp",
                    "mdgp",
                    "mdspp",
                    "vgp",
                    "svgp",
                    "spv",
                    "siv",
                    "crv",
                ]
                | None
            ) = None
            method_kwargs: dict = {}
            custom_training: str | None = "dmosopt.model_transformer.joint"
            custom_training_kwargs: dict | None = {}

        surrogate: SurrogateConfig = SurrogateConfig()

    def _restrict_electrodes(
        self, geometry: dict, selection, readout: str, system
    ) -> dict:
        if not selection:
            return geometry

        from livn.system import resolve

        document = resolve(system).selection_document(selection)
        bounds = (document.get("meta") or {}).get("bounds")
        if not bounds:
            raise ValueError(
                f"selection {selection!r} records no bounds, so the array cannot "
                "be restricted to it; it was not cut from a box"
            )
        (x0, y0), (x1, y1) = bounds
        inside = [p for p in geometry["pos"] if x0 <= p[1] <= x1 and y0 <= p[2] <= y1]

        if readout == "channels" and len(inside) < 2:
            raise ValueError(
                f"selection {selection!r} covers {len(inside)} of "
                f"{len(geometry['pos'])} electrodes, too few for a channel "
                "readout; use readout='neurons' or a larger selection"
            )

        return {**geometry, "pos": inside}

    @staticmethod
    def version_cell(config: str):
        from systems.targets.cells.SingleCell import SingleCellOptConfig

        parsed = SingleCellOptConfig.from_yaml(config)
        cfg = parsed.model_dump()
        return {
            "system": {parsed.Population: 1},
            "model": "livn.models.rcsd.ReducedCalciumSomaDendrite",
            "target": ["systems.targets.cells.SingleCell.SingleCell", {"config": cfg}],
        }

    @staticmethod
    def axis_cells(directory: str = "./systems/targets/cells"):
        from glob import glob

        from systems.targets.cells.SingleCell import SingleCellOptConfig

        return [
            Tune.version_cell(path)
            for path in sorted(glob(os.path.join(directory, "*.yaml")))
            if not SingleCellOptConfig.from_yaml(path).Retired
        ]

    def version_spontaneous_recording(
        self,
        file: str,
        condition: str = "spontaneous",
        readout: str = "channels",
        duration: float | None = None,
        warmup: float | None = None,
        selection: str | None = None,
        system: str | None = None,
        adaptation: bool = False,
        ignition: bool = False,
    ):
        from systems.targets.schema import FREE_RUNNING, read_target

        if condition not in FREE_RUNNING:
            raise ValueError(
                f"condition {condition!r} measures windows containing a "
                f"stimulus. Tune on {' or '.join(FREE_RUNNING)} instead."
            )

        block = read_target(file, condition)
        measured = block.ei_targets
        features = block.summary.features
        if duration is None:
            duration = float(block.window_ms)

        mea = None
        metadata = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(file))), "metadata.json"
        )
        if os.path.isfile(metadata):
            with open(metadata) as f:
                geometry = json.load(f)["geometry"]
            geometry = self._restrict_electrodes(
                geometry, selection, readout, system or self.config.system
            )
            mea = {
                "electrode_coordinates": [
                    [float(i), float(x), float(y), 5.0] for i, x, y in geometry["pos"]
                ],
                "input_radius": 50.0,
                "output_radius": 50.0,
            }
        elif readout == "channels":
            raise FileNotFoundError(
                f"readout='channels' needs the recording set's metadata.json, but "
                f"{file!r} has none two levels up (looked in {metadata!r}). Pass "
                "readout='neurons' to tune without an array."
            )

        widen_factor = 2.0
        widen = {
            "BRANCHING_RATIO_BAND": ("branching_ratio", "band"),
            "POP_TAU_BAND_MS": ("pop_autocorr_tau", "band"),
            "SYNCHRONY_BAND": ("mean_channel_correlation", "band"),
            "MAX_MEAN_RATE_HZ": ("mfr", "max"),
            "MIN_MEAN_RATE_HZ": ("mfr", "min"),
            "MAX_NEURON_RATE_HZ": ("max_neuron_firing_rate", "max"),
            "MAX_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "max"),
            "MIN_POP_RATE_PER_UNIT_HZ": ("pop_rate_per_unit_hz", "min"),
            "MAX_BURST_RATE_HZ": ("burst_rate", "max"),
            "MIN_BURST_RATE_HZ": ("burst_rate", "min"),
            "MAX_SYNC_PEAK": ("max_synchronous_peak", "max"),
            "MIN_SYNC_PEAK": ("max_synchronous_peak", "min"),
        }
        gates = dict(measured)
        for key, (feature, kind) in widen.items():
            spec = features.get(feature)
            if not spec:
                continue
            lo, hi = spec.q_lo, spec.q_hi
            if lo is None or hi is None:
                continue
            slack = (hi - lo) / 2.0 * (widen_factor - 1.0)
            if kind == "band":
                gates[key] = [float(lo - slack), float(hi + slack)]
            elif kind == "max":
                gates[key] = float(hi + slack)
            else:
                gates[key] = float(max(0.0, lo - slack))

        floor = gates.get("MIN_ACTIVE_FRACTION")
        if mea is not None and floor is not None:
            from livn.system import resolve

            reachable = electrode_coverage(
                resolve(system or self.config.system), mea, selection
            )
            if reachable is not None and reachable < 1.0:
                gates["MIN_ACTIVE_FRACTION"] = max(
                    0.0, reachable - (1.0 - float(floor))
                )
                print(
                    f"{1.0 - reachable:.1%} of the array reaches no cell within "
                    f"{mea['output_radius']:g} um, so active_fraction cannot "
                    f"exceed {reachable:.4f}; lowered MIN_ACTIVE_FRACTION "
                    f"{float(floor):.4f} -> {gates['MIN_ACTIVE_FRACTION']:.4f}"
                )

        overrides = dict(gates)
        overrides["targets"] = dict(measured["targets"])

        skip_constraints = ["avalanche_r2"]

        synchrony_detection_floor = 0.01
        sync_lo, sync_hi = measured.get("SYNCHRONY_BAND", (0.0, 1.0))
        if sync_lo <= 0.0 or sync_hi < synchrony_detection_floor:
            overrides["SYNCHRONY_BAND"] = [0.0, float(synchrony_detection_floor)]
        else:
            overrides["targets"]["mean_channel_correlation"] = float(
                (sync_lo + sync_hi) / 2.0
            )

        for feature, key in (("pop_autocorr_tau", "POP_TAU_BAND_MS"),):
            spec = features.get(feature)
            if (
                spec
                and spec.q_lo is not None
                and spec.min is not None
                and abs(spec.q_lo - spec.min) <= 1e-9
            ):
                # the lower edge is the decoder's own bin size, not the
                # culture: keep the upper bound and drop the floor
                band = overrides.get(key) or measured.get(key)
                if band is not None:
                    overrides[key] = [0.0, float(band[1])]

        options = {
            "overrides": overrides,
            "feature_bands": {
                name: (spec.q_lo, spec.q_hi)
                for name, spec in features.items()
                if spec is not None and spec.q_lo is not None
            },
            "readout": readout,
            "mea": mea,
            "skip_objectives": [],
            "skip_constraints": skip_constraints,
            "adaptation": bool(adaptation),
            "ignition": bool(ignition),
        }
        if duration is not None:
            options["duration"] = float(duration)
        if warmup is not None:
            options["warmup"] = float(warmup)

        composed = {"target": ["systems.targets.EI.Culture", options]}
        if selection is not None:
            composed["selection"] = selection
        if system is not None:
            composed["system"] = system
        return composed

    def version_culture_recording(
        self,
        file: str,
        resting: str = "interstimulus",
        evoked: str = "evoked",
        repeats: int = 2,
        trial_ms: float | None = None,
        **kwargs,
    ):
        from systems.targets.EI import Protocol
        from systems.targets.schema import read_target

        composed = self.version_spontaneous_recording(file, condition=resting, **kwargs)
        options = dict(composed["target"][1])

        block = read_target(file, evoked)
        if block.threshold is None:
            raise ValueError(
                f"{evoked!r} block of {file!r} has no threshold to fit; its "
                "recruitment curve was too short to read"
            )

        options["stimulus"] = Protocol.from_block(
            block,
            repeats=int(repeats),
            trial_ms=float(block.window_ms if trial_ms is None else trial_ms),
        ).model_dump()
        options["stimulus_threshold"] = block.threshold.model_dump()

        composed["target"] = ["systems.targets.EI.Culture", options]
        return composed

    def version_rcsd(self, short_term_depression: bool = False):
        options = {}
        if short_term_depression:
            options["short_term_depression"] = True
        return {"model": ["livn.models.rcsd.ReducedCalciumSomaDendrite", options]}

    def version_E_only(self, short_term_depression: bool = False):
        return {
            **self.version_rcsd(short_term_depression=bool(short_term_depression)),
            "system": "./systems/graphs/E",
            "n_initial": 25,
            "autosize": True,
        }

    def version_EI(self):
        return {
            **self.version_rcsd(short_term_depression=True),
            "system": "./systems/graphs/EI",
            "nprocs_per_worker": 2,
            "n_initial": 25,
            "autosize": True,
        }

    def version_motoneuron(self):
        return self.version_cell("systems/targets/cells/motoneuron.yaml")

    def version_renshaw_perry(self):
        return self.version_cell("systems/targets/cells/rc_v1in_perry.yaml")

    def version_renshaw_invitro(self):
        return self.version_cell("systems/targets/cells/rc_v1in_invitro.yaml")

    def version_renshaw(self):
        return self.version_renshaw_perry()

    def version_ca1(
        self,
        file: str = "./systems/graphs/CA1/tuning.json",
        problem: str = "miv",
        selection: str | None = None,
        worker_memory_max: float | None = None,
        cores_per_node: int | None = None,
        max_nodes: int | None = None,
    ):
        options = {"config": file, "problem": problem}
        if selection is not None:
            options["selection"] = selection

        return {
            "system": "./systems/graphs/CA1",
            "model": "livn.models.ca1.PinskyRinzel",
            "n_initial": 2,
            "n_epochs": 5,
            "autosize": True,
            "worker_memory_max": worker_memory_max,
            "cores_per_node": cores_per_node,
            "max_nodes": max_nodes,
            "target": ["systems.targets.CA1.CA1", options],
        }

    @staticmethod
    def _instantiate(system, model, target):
        target = import_instance(target)
        model = import_instance(model)

        if getattr(target, "system", None) is None and hasattr(target, "system"):
            target.system = system

        return target, model

    def _target_and_model(self):
        target, model = self._instantiate(
            self.config.system, self.config.model, self.config.target
        )
        return target, model

    def _plan(self, target=None, model=None) -> dict | None:
        if not self.config.autosize:
            return None

        if target is None or model is None:
            target, model = self._target_and_model()
        worker_memory = getattr(target, "worker_memory", None)
        if worker_memory is None:
            return None

        sopt = get("interface.sopt", self._sopt_config(target, model))

        selection = getattr(target, "selection_name", None) or self.config.selection
        plan = plan_execution(
            lambda ranks: worker_memory(self.config.system, ranks, selection),
            sopt.num_evals_per_epoch,
            node_gib=self.config.worker_memory_max,
            cores_per_node=self.config.cores_per_node,
            max_nodes=self.config.max_nodes,
            min_ranks_per_worker=getattr(target, "MIN_RANKS_PER_WORKER", 1),
        )
        plan["selection"] = selection
        plan["space"] = sopt.num_parameters
        plan["initial_evals"] = sopt.num_initial_samples
        plan["total_evals"] = sopt.num_evals_total
        plan["floor"] = getattr(target, "MIN_RANKS_PER_WORKER", 1)
        return plan

    def sizing(self):
        """Preview the execution layout without launching anything.

            livn systems tune '~ca1(selection="e3")' --sizing

        `LIVN_WORKER_MEMORY_MAX` (GiB per node) and `LIVN_CORES_PER_NODE`
        describe the cluster you mean; without them this machine is assumed.
        """
        target, model = self._target_and_model()
        plan = self._plan(target, model)
        if plan is None:
            why = (
                "the target cannot price its own network"
                if self.config.autosize
                else "autosize=False"
            )
            print(
                f"\n  autosize      off ({why}) -- nprocs_per_worker="
                f"{self.config.nprocs_per_worker}, and the rank count is "
                "whatever the caller passes\n"
            )
            return

        print(f"\n  system        {self.config.system}")
        print(f"  selection     {plan['selection'] or 'none'}")
        print(
            f"  node          {plan['node_gib']:.1f} GiB x "
            f"{plan['cores_per_node']} cores, planned to "
            f"{plan['headroom']:.0%}"
            + (
                ""
                if self.config.worker_memory_max
                or os.environ.get("LIVN_WORKER_MEMORY_MAX")
                else "  (this machine -- state LIVN_WORKER_MEMORY_MAX for a cluster)"
            )
        )
        print(
            f"  worker        {plan['worker_gib']:.1f} GiB over "
            f"{plan['ranks_per_worker']} rank(s)"
            + (
                f"  (target asks for at least {plan['floor']})"
                if plan["floor"] > 1
                else ""
            )
        )
        print(
            f"  layout        {plan['nodes']} node(s) x {plan['ranks_per_node']} "
            f"rank(s) = {plan['total_ranks']} ranks "
            f"({plan['node_used_gib']:.1f} GiB used per node)"
        )
        print(
            f"  workers       {plan['workers']} for "
            f"{plan['workers_wanted']} samples per epoch"
            + (
                ""
                if plan["workers"] >= plan["workers_wanted"]
                else "  -- epochs will queue"
            )
        )
        print(
            f"  evaluations   {plan['initial_evals']} in the first epoch "
            f"({plan['space']} dims x n_initial={self.config.n_initial}), then "
            f"{plan['workers_wanted']} per epoch x {self.config.n_epochs - 1} "
            f"= {plan['total_evals']} in all"
        )
        if plan["nodes"] == 1:
            print(
                "\n  launch with\n"
                f'    livn systems mpi tune ... **resources=\'{{"-n": '
                f"{plan['total_ranks']}}}' --launch\n"
            )
        else:
            print(
                "\n  launch with\n"
                "    livn systems slurm tune ... --launch\n"
                "  which takes --nodes and --ntasks-per-node from this plan. A "
                "local mpi run\n  has one node; pass max_nodes=1 to size for it.\n"
            )

        return

    def _sopt_config(self, target, model, layout: dict | None = None) -> dict:
        surrogate_config = {}
        for k, v in self.config.surrogate.items():
            surrogate_config["surrogate_" + k] = v

        return {
            "system": self.config.system,
            "dopt_params": {
                "space": target.search_space(model),
                "obj_fun_init_args": {
                    "model": self.config.model,
                    "target": self.config.target,
                    "trials": self.config.trials,
                    "selection": self.config.selection,
                },
                "n_epochs": self.config.n_epochs,
                "n_initial": self.config.n_initial,
                "population_size": self.config.population_size,
                "num_generations": self.config.num_generations,
                **surrogate_config,
            },
            **(layout or {}),
        }

    def launch(self):
        target, model = self._target_and_model()
        plan = self._plan(target, model)
        layout = (
            {
                "nprocs_per_worker": plan["ranks_per_worker"],
                "nodes": str(plan["nodes"]),
                "ranks": plan["ranks_per_node"],
            }
            if plan
            else {"nprocs_per_worker": self.config.nprocs_per_worker}
        )

        get("interface.sopt", self._sopt_config(target, model, layout)).launch()

        return self

    @staticmethod
    def _feature_bands(target) -> dict:
        bands = getattr(target, "feature_bands", None)
        if callable(bands):
            bands = bands()
        return dict(bands or {})

    def _ranked_best(self, optimization, target):
        best = optimization.get_best()
        if hasattr(target, "rank_solutions"):
            best = target.rank_solutions(best)
        return best

    def reference(self, populations=None, sample: int = 1500, write: bool = False):
        from livn.system import System

        target, _ = self._target_and_model()
        if not hasattr(target, "reference_targets"):
            print(f"{self.config.target} states no reference activity")
            return None

        measured = target.reference_targets(
            System(self.config.system),
            populations=(
                None
                if populations is None
                else [p.strip() for p in populations.split(",")]
            ),
            sample=sample,
        )

        if write:
            print("wrote", target.write_reference(measured))

        rows = []
        for pop, features in measured.items():
            row = {"population": pop, "cells": int(features.get("n_total", 0))}
            row.update(
                {
                    k: round(float(v), 4)
                    for k, v in features.items()
                    if k not in ("n_total", "n_active")
                }
            )
            rows.append(row)
        table = pd.DataFrame(rows)
        with pd.option_context(
            "display.max_columns", None, "display.width", _terminal_width()
        ):
            print(table.to_string(index=False))
        return table

    def summary(self, params=None, sort=True):
        optimization = self._optimization()
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self._target_and_model()
        best = self._ranked_best(optimization, target)
        f = best.get("f")
        if f is None or len(f) == 0:
            print("No solutions")
            return None

        bands = self._feature_bands(target)
        wanted = [p.strip() for p in (params or "").split(",") if p.strip()]
        space = target.search_space(model) if wanted else {}

        rows = []
        for i in range(len(f)):
            row = {"loc": i}
            n_in = 0
            for name in f.columns:
                value = float(f[name].iloc[i])
                lo_hi = bands.get(name)
                inside = lo_hi is not None and lo_hi[0] <= value <= lo_hi[1]
                n_in += bool(inside)
                row[name] = f"{value:.4g}{'*' if inside else ''}"
            if bands:
                row["in_band"] = f"{n_in}/{len(bands)}"
                row["_n"] = n_in
            if wanted:
                decoded = decode_strict(
                    target,
                    model,
                    space,
                    optimization.parameter_vector_to_dict(
                        list(map(float, best["x"].to_numpy()[i]))
                    ),
                )
                for name in wanted:
                    value = decoded.get(name)
                    row[name] = "--" if value is None else f"{float(value):.4g}"
            rows.append(row)

        table = pd.DataFrame(rows)
        if sort and "_n" in table:
            table = table.sort_values("_n", ascending=False)
        table = table.drop(columns=[c for c in ("_n",) if c in table])

        with pd.option_context(
            "display.max_columns", None, "display.width", _terminal_width()
        ):
            print(table.to_string(index=False))
        if bands:
            print("\n* = inside the measured band:")
            for name, (lo, hi) in bands.items():
                print(f"    {name:<24} {lo:>10.4g} - {hi:<10.4g}")
        else:
            print("\n(the target states no feature bands, so nothing is marked)")
        return table

    def freeze_selection(self, name: str, force: bool = False, **selection):
        from livn.system import resolve

        if _numeric(name):
            raise ValueError(
                f"selection name {name!r} reads as a number, and a number is a "
                "valid selection *spec* -- cast anywhere along the way it would "
                f"quietly select {name} cells instead of loading this "
                f"selection. Name it 'e{name}' or something else non-numeric."
            )

        system = resolve(self.config.system)
        spec = selection.pop("spec", None)
        resolved = system.selection(spec, **selection)
        if not resolved:
            raise ValueError(f"selection({spec!r}, {selection}) selects no cells")

        directory = system.local_directory("selection")
        target = os.path.join(directory, f"{name}.json")
        if os.path.isfile(target) and not force:
            raise FileExistsError(
                f"{target!r} already exists, and runs refer to selections by "
                "name; pass force=True to rebind it, or choose another name"
            )
        counts = {p: len(g) for p, g in resolved.items()}

        if not P.is_root():
            return target
        os.makedirs(directory, exist_ok=True)
        with open(target, "w") as f:
            json.dump(
                {
                    "gids": {p: [int(g) for g in v] for p, v in resolved.items()},
                    "meta": {
                        "spec": spec,
                        **{k: v for k, v in selection.items() if v is not None},
                        "counts": counts,
                        # provenance: gids are positions in the cell arrays,
                        # so a regenerated graph leaves them naming other cells
                        "graph": getattr(system._graph.architecture, "uuid", None),
                    },
                },
                f,
                indent=2,
            )
        print(f"froze {counts} as selection {name!r} -> {target}")
        return target

    def promote(
        self,
        group: str = "default",
        loc: int | None = None,
        selection: str | None = None,
        force: bool = False,
        front: str | None = None,
    ):
        from livn.system import resolve

        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        loc = int(loc)

        if front:
            with open(front) as f:
                document = json.load(f)
            solutions = {int(s["loc"]): s for s in document["solutions"]}
            if loc not in solutions:
                raise ValueError(
                    f"no solution loc={loc} on this front; it has "
                    f"{', '.join(str(k) for k in sorted(solutions))}"
                )
            solution = solutions[loc]
            if selection is None:
                selection = document.get("selection")
            system = resolve(document["system"])
            target, model = self._instantiate(
                document["system"], document["model"], document["target"]
            )
            decoded = dict(solution["params"])
            meta = {
                "loc": loc,
                "ranked_by": "rank_solutions",
                "objectives": dict(solution.get("objectives") or {}),
                "features": dict(solution.get("features") or {}),
                "feasible": bool(solution.get("feasible")),
                "space": sorted(decoded),
                "source": document.get("source"),
                "target": document["target"],
                "model": document["model"],
            }
        else:
            if selection is None:
                selection = self.config.selection
            system = resolve(self.config.system)

            optimization = self._optimization()
            if not optimization.is_materialized() or not os.path.isfile(
                optimization.output_filepath
            ):
                print("No data yet")
                return None

            target, model = self._target_and_model()
            best = self._ranked_best(optimization, target)
            if best.get("f") is None or len(best["f"]) == 0:
                print("No solutions")
                return None

            decoded = decode_strict(
                target,
                model,
                target.search_space(model),
                optimization.parameter_vector_to_dict(
                    list(map(float, best["x"].to_numpy()[loc]))
                ),
            )

            constraints = best.get("c")
            meta = {
                "loc": loc,
                "ranked_by": "rank_solutions",
                "objectives": {k: float(v) for k, v in best["y"].iloc[loc].items()},
                "features": {k: float(v) for k, v in best["f"].iloc[loc].items()},
                "feasible": bool(
                    constraints is None or (constraints.iloc[loc] > 0).all()
                ),
                "space": sorted(self._recorded_space() or decoded),
                "source": optimization.output_filepath,
                "target": self.config.target,
                "model": self.config.model,
            }
        if selection:
            meta["selection"] = selection
            retained = retained_in_degree(system, system.selection(selection))
            if retained is not None:
                meta["retained_in_degree"] = round(retained, 4)

        directory = system.local_directory("params")
        filename = f"{selection or 'default'}.json"
        path = os.path.join(directory, filename)

        document = {}
        if os.path.isfile(path):
            with open(path) as f:
                document = json.load(f)

        key = model.params_key()
        groups = document.setdefault(key, {})
        if group in groups and not force:
            raise FileExistsError(
                f"{path!r} already has a {group!r} group for {key}, and runs "
                "refer to it by name; pass force=True to rebind it"
            )
        groups[group] = {
            "params": {k: float(v) for k, v in decoded.items()},
            "meta": meta,
        }

        # one writer, as in `freeze_selection`
        if not P.is_root():
            return path
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(document, f, indent=2, sort_keys=True)
        print(
            f"promoted loc={loc} to {key}/{group} in {path}\n"
            f"  use with: env.apply_default_params(group='{group}')"
            + (f" and env.selection('{selection}')" if selection else "")
        )
        return path

    def _recorded_space(self) -> list[str] | None:
        """The search space the run actually used, as machinable stored it."""
        try:
            return list(self._optimization().config.dopt_params.space.keys())
        except (AttributeError, KeyError, IndexError, TypeError):
            return None

    def export(self, path: str | None = None, feasible_only: bool = False):
        import numpy as np

        optimization = self._optimization()
        if not optimization.is_materialized() or not os.path.isfile(
            optimization.output_filepath
        ):
            print("No data yet")
            return None

        target, model = self._target_and_model()

        h5 = optimization.load_h5()
        n_rows, n_evals, n_epochs = self._evaluation_counts(h5)

        best = self._ranked_best(optimization, target)
        print(f"Front: {len(best['x'])} solutions over {n_evals} evaluations")

        space = target.search_space(model)
        c = np.asarray(best["c"]) if best.get("c") is not None else None

        solutions = []
        for loc in range(len(best["x"])):
            if feasible_only and c is not None and (c[loc] <= 0).any():
                continue
            raw = optimization.parameter_vector_to_dict(
                list(map(float, best["x"].to_numpy()[loc]))
            )
            decoded = decode_strict(target, model, space, raw)
            solutions.append(
                {
                    "loc": loc,
                    "objectives": {k: float(v) for k, v in best["y"].iloc[loc].items()},
                    "features": {k: float(v) for k, v in best["f"].iloc[loc].items()},
                    "constraints": (
                        {k: float(v) for k, v in best["c"].iloc[loc].items()}
                        if best.get("c") is not None
                        else None
                    ),
                    "feasible": bool(c is None or (c[loc] > 0).all()),
                    "params": {k: float(v) for k, v in decoded.items()},
                }
            )

        document = {
            "evaluations": n_evals,
            "table_rows": n_rows,
            "epoch_entries": n_epochs,
            "truncated": n_evals != n_rows,
            "executions": len(list(self.interfaces)),
            "system": self.config.system,
            "selection": self.config.selection,
            "model": self.config.model,
            "target": self.config.target,
            "source": optimization.output_filepath,
            "feature_bands": {
                k: list(v) for k, v in self._feature_bands(target).items()
            },
            "solutions": solutions,
        }

        document = to_dict(document)
        if path is None:
            path = optimization.save_file("front.json", document)
        else:
            with open(path, "w") as f:
                json.dump(document, f, indent=2)
        n_feasible = sum(1 for s in solutions if s["feasible"])
        print(f"wrote {len(solutions)} solutions ({n_feasible} feasible) to {path}")
        return path

    @staticmethod
    def _evaluation_counts(h5) -> tuple:
        import numpy as np

        objectives = h5.get("objectives")
        if objectives is None:
            return 0, 0, len(h5["epochs"])
        values = objectives.to_numpy()
        usable = int(np.logical_not(np.any(np.isnan(values), axis=1)).sum())
        return len(values), usable, len(h5["epochs"])

    def _completed(self, optimization) -> int:
        try:
            if not optimization.is_materialized() or not os.path.isfile(
                optimization.output_filepath
            ):
                return -1
            return self._evaluation_counts(optimization.load_h5())[1]
        except Exception:
            return -1

    def _optimization(self):
        interfaces = list(self.interfaces)
        if len(interfaces) == 1:
            return interfaces[0]
        if not interfaces:
            raise ValueError("no optimization has been launched for this config")

        try:
            target, model = self._target_and_model()
            wanted = set(target.search_space(model))
        except Exception:
            wanted = None

        candidates = []
        for candidate in interfaces:
            try:
                space = set(candidate.config.dopt_params.space.keys())
            except Exception:
                space = set()
            candidates.append((candidate, space, self._completed(candidate)))

        agreeing = [c for c in candidates if wanted is None or c[1] == wanted]
        if wanted is not None and not agreeing:
            print(
                f"WARNING: none of the {len(candidates)} stored runs searches "
                "this target's space, so every front below describes a "
                "different problem than the one this target states. Reading "
                "the largest anyway; re-run rather than trust it."
            )

        pool = sorted(agreeing or candidates, key=lambda c: -c[2])
        chosen, _chosen_space, chosen_n = pool[0]

        print(
            f"NOTE: {len(candidates)} runs are stored under this config; "
            f"reading the one with {chosen_n} completed evaluations."
        )
        for other, space, n in candidates:
            if other is chosen:
                continue
            if wanted is not None and space != wanted:
                missing = sorted(wanted - space)
                extra = sorted(space - wanted)
                why = "different space"
                if missing:
                    why += f", missing {missing}"
                if extra:
                    why += f", also searches {extra}"
            else:
                why = "same space, fewer evaluations"
            print(f"        {getattr(other, 'output_filepath', '?')}")
            print(f"          {n} evaluations -- skipped: {why}")

        return chosen

    def inspect(self, loc=None, params=None, verbose=0):
        if loc is None:
            loc = int(os.environ.get("LOC", 0))
        optimization = self._optimization()
        print(f"System: {self.config.system}")
        if not optimization.is_materialized():
            print("No data yet (nothing launched for this config)")
            return

        if not os.path.isfile(optimization.output_filepath):
            print("No data yet")
            return

        h5 = optimization.load_h5()
        n_rows, n_evals, _n_epochs = self._evaluation_counts(h5)

        if n_evals != n_rows:
            print(f"  WARNING: the table has {n_rows} rows but only {n_evals} ")

        target, model = self._target_and_model()

        best = self._ranked_best(optimization, target)
        n_front = 0 if best.get("y") is None else len(best["y"])

        bands = self._feature_bands(target)
        if bands:
            print("\nPer-solution band membership (`--summary` for parameters):")
            counts = []
            for i in range(len(best["f"])):
                n_in = sum(
                    lo <= float(best["f"][name].iloc[i]) <= hi
                    for name, (lo, hi) in bands.items()
                    if name in best["f"].columns
                )
                counts.append((n_in, i))
            counts.sort(reverse=True)
            shown = ", ".join(f"loc={i} ({n}/{len(bands)})" for n, i in counts[:6])
            print(f"    best: {shown}")
            if counts and counts[0][1] != loc:
                print(
                    f"    note: loc={counts[0][1]} has {counts[0][0]}/{len(bands)} "
                    f"in band; the selected loc={loc} has "
                    f"{ {i: n for n, i in counts}[loc] }/{len(bands)}"
                )

        print(f"\nSelected solution (loc={loc}):")
        print("  y:", dict(best["y"].iloc[loc]))
        print("  f:", dict(best["f"].iloc[loc]))

        raw_params = optimization.parameter_vector_to_dict(
            list(map(float, best["x"].to_numpy()[loc]))
        )
        decoded = decode_strict(target, model, target.search_space(model), raw_params)

        groups = (
            target.describe_params(decoded)
            if hasattr(target, "describe_params")
            else {"params": decoded}
        )
        if verbose > 0:
            for name, group in groups.items():
                if group:
                    print(f"\n{name}:")
                    print(_pj(group))

        wfn = optimization.save_file("params.json", decoded)
        print("\nSaved to", wfn)

        print(f"Front: {n_front} solutions over {n_evals} evaluations")

        with pd.option_context(
            "display.max_columns", None, "display.width", _terminal_width()
        ):
            print("\nFeatures (f):")
            print(best["f"])
            print("\nObjectives (y):")
            print(best["y"])
            if best.get("c") is not None:
                print("\nConstraints (c):")
                if (best["c"] > 0).all(axis=None):
                    print("All constraints satisfied")
                else:
                    print(best["c"])
                    import numpy as np

                    c = np.asarray(best["c"])
                    infeasible = int((c <= 0).any(axis=1).sum())
                    if infeasible:
                        print(
                            f"{infeasible} of {len(c)} solutions on this front violate at "
                            "least one constraint, so this is an infeasible front."
                        )

        print(
            "Epochs",
            h5["epochs"][-1],
            " Evals ",
            n_evals,
            " n_i: ",
            optimization.num_initial_samples,
        )
        print("Cached:", optimization.cached())

        print(optimization.output_filepath)

        print(optimization.execution.output_filepath())
