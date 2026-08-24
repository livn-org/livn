import faulthandler
import json
import sys
import time
from hashlib import blake2b
from pathlib import Path

from testing.collectives import tracer

ASYMMETRY_EXIT_CODE = 88


class CollectiveAsymmetry(AssertionError):
    pass


def verify(nodeid: str, trace: dict, *, arrival_timeout: float = 15.0, report_dir=None):
    from mpi4py import MPI

    with tracer.paused():
        if not _everyone_arrived(nodeid, trace, arrival_timeout, report_dir):
            return

        digests = {key: _digest(records) for key, records in trace.items()}
        gathered = MPI.COMM_WORLD.allgather(digests)
        if _agree(gathered):
            return

        everything = MPI.COMM_WORLD.gather(trace, root=0)
        message = _diff(everything) if MPI.COMM_WORLD.rank == 0 else None
        message = MPI.COMM_WORLD.bcast(message, root=0)

    if message:
        _dump(report_dir, nodeid, trace, reason="divergence")
        raise CollectiveAsymmetry(f"{nodeid}\n{message}")
    return


def _everyone_arrived(nodeid, trace, timeout, report_dir) -> bool:
    from mpi4py import MPI

    request = MPI.COMM_WORLD.Ibarrier()
    deadline = time.monotonic() + timeout
    while not request.Test():
        if time.monotonic() > deadline:
            _dump(report_dir, nodeid, trace, reason="peers-never-arrived")
            print(
                f"[rank {MPI.COMM_WORLD.rank}] waited {timeout:.0f}s at the end of "
                f"{nodeid} and the other ranks never got here",
                file=sys.stderr,
                flush=True,
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            MPI.COMM_WORLD.Abort(ASYMMETRY_EXIT_CODE)
        time.sleep(0.005)
    return True


def _digest(records) -> str:
    canonical = "\n".join(f"{r.op}|{r.site}|{r.root}" for r in records)
    return blake2b(canonical.encode(), digest_size=16).hexdigest()


def _agree(gathered: list) -> bool:
    keys = {key for per_rank in gathered for key in per_rank}
    for key in keys:
        members, _ordinal = key
        seen = {
            per_rank.get(key)
            for rank, per_rank in enumerate(gathered)
            if rank in members
        }
        if len(seen) > 1:
            return False
    return True


def _diff(everything: list) -> str | None:
    keys = {key for per_rank in everything for key in per_rank}

    for key in sorted(keys, key=lambda k: (len(k[0]), k)):
        members = [r for r in key[0] if r < len(everything)]
        sequences = {rank: everything[rank].get(key, []) for rank in members}
        longest = max((len(v) for v in sequences.values()), default=0)

        for index in range(longest):
            signatures = {
                rank: (
                    sequences[rank][index].signature()
                    if index < len(sequences[rank])
                    else None
                )
                for rank in members
            }
            if len(set(signatures.values())) <= 1:
                continue
            return _render(key, index, signatures, sequences, members)
    return None


def _render(key, index, signatures, sequences, members) -> str:
    grouped: dict = {}
    for rank, signature in signatures.items():
        grouped.setdefault(signature, []).append(rank)

    lines = [
        f"communicator over world ranks {tuple(key[0])}"
        + (f" (#{key[1]})" if key[1] else "")
        + f" diverged at index {index}",
        "",
    ]
    for signature, ranks in sorted(grouped.items(), key=lambda kv: kv[1]):
        who = "rank" + ("s" if len(ranks) > 1 else "")
        who = f"  {who} {', '.join(map(str, ranks))}"
        if signature is None:
            last = sequences[ranks[0]]
            trailer = (
                f" (last: {last[-1].op} at {last[-1].site})"
                if last
                else " (no collectives at all)"
            )
            lines.append(f"{who}: nothing further on this communicator{trailer}")
            continue

        op, root = signature
        at_root = f" root={root}" if root is not None else ""
        sites = sorted({sequences[rank][index].site for rank in ranks})
        where = " at " + (sites[0] if len(sites) == 1 else " / ".join(sites))
        lines.append(f"{who}: {op}{where}{at_root}")

    reference = sequences[members[0]]
    if index:
        lines += ["", "  agreed up to here:"]
        for offset in range(max(0, index - 3), index):
            if offset < len(reference):
                record = reference[offset]
                lines.append(f"    {offset:>4}  {record.op} at {record.site}")

    return "\n".join(lines)


def _dump(report_dir, nodeid, trace, reason: str) -> None:
    if not report_dir:
        return
    from mpi4py import MPI

    payload = {
        "nodeid": nodeid,
        "reason": reason,
        "rank": MPI.COMM_WORLD.rank,
        "by_comm": {
            f"{list(key[0])}#{key[1]}": [list(record) for record in records]
            for key, records in trace.items()
        },
    }
    try:
        path = Path(report_dir) / f"divergence-{MPI.COMM_WORLD.rank}.json"
        path.write_text(json.dumps(payload, indent=2))
    except OSError:  # pragma: no cover
        pass
