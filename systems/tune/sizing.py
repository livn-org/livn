import math
import os


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def node_memory_gib() -> float:
    """Memory a node is assumed to have, in GiB."""
    stated = _env_float("LIVN_WORKER_MEMORY_MAX")
    if stated is not None:
        return stated
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def min_ranks_per_worker(stated: int = 1) -> int:
    """Ranks one worker spans at least."""
    given = os.environ.get("LIVN_MIN_RANKS_PER_WORKER")
    return max(1, int(given)) if given not in (None, "") else max(1, int(stated))


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

    # within 10% of the wanted worker count, then within 64 of it
    found = search(max(1, workers_wanted // 10)) or search(min(workers_wanted, 64))
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
    """Ranks per worker, ranks per node and nodes for a run."""
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
