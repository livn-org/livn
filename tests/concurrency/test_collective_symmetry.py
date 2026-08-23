import pytest

from livn.backend import backend

pytestmark = [
    pytest.mark.skipif(backend() == "", reason="needs a backend for livn.utils.P"),
    pytest.mark.needs("mpi"),
]


TIMEOUT = 60


@pytest.fixture(autouse=True)
def _tracer_installed():
    from testing.collectives import tracer
    from testing.paths import REPO_ROOT

    tracer.install(roots=(str(REPO_ROOT),))
    tracer.watch(("livn",))
    yield


@pytest.mark.mpiexec(n=2, timeout=TIMEOUT, symmetry=False, isolated=True)
def test_a_divergence_is_detected_and_says_where():
    from mpi4py import MPI

    from testing.collectives import CollectiveAsymmetry, tracer, verify

    comm = MPI.COMM_WORLD
    tracer.start_test("synthetic")

    comm.Barrier()
    if comm.rank == 0:
        tracer.record_synthetic("bcast", "src/livn/system.py:1")

    with pytest.raises(CollectiveAsymmetry) as raised:
        verify("synthetic", tracer.finish_test(), arrival_timeout=TIMEOUT)

    message = str(raised.value)
    assert "diverged at index" in message
    assert "src/livn/system.py:1" in message
    assert "rank 0" in message and "rank 1" in message


@pytest.mark.mpiexec(n=2, timeout=TIMEOUT, symmetry=False, isolated=True)
def test_matching_collectives_are_not_reported():
    from mpi4py import MPI

    from testing.collectives import tracer, verify

    comm = MPI.COMM_WORLD
    tracer.start_test("symmetric")

    comm.Barrier()
    comm.bcast(None, root=0)
    comm.allgather(comm.rank)

    assert verify("symmetric", tracer.finish_test(), arrival_timeout=TIMEOUT) is None


@pytest.mark.mpiexec(n=4, timeout=TIMEOUT, symmetry=False, isolated=True)
def test_ranks_on_different_communicators_are_not_a_divergence():
    from mpi4py import MPI

    from testing.collectives import tracer, verify

    world = MPI.COMM_WORLD
    tracer.start_test("split")

    half = world.Split(world.rank // 2, world.rank)
    try:
        half.Barrier()
        if world.rank // 2 == 0:
            half.bcast(None, root=0)
            half.allgather(world.rank)

        assert verify("split", tracer.finish_test(), arrival_timeout=TIMEOUT) is None
    finally:
        half.Free()


@pytest.mark.mpiexec(n=2, timeout=TIMEOUT, symmetry=False, isolated=True)
def test_point_to_point_traffic_is_not_compared():
    from mpi4py import MPI

    from testing.collectives import tracer, verify

    comm = MPI.COMM_WORLD
    tracer.start_test("p2p")

    if comm.rank == 0:
        comm.send({"hello": True}, dest=1)
    else:
        comm.recv(source=0)

    assert verify("p2p", tracer.finish_test(), arrival_timeout=TIMEOUT) is None


@pytest.mark.mpiexec(n=2, timeout=TIMEOUT, symmetry=False, isolated=True)
def test_the_tracer_sees_livn_s_own_collectives():
    from mpi4py import MPI

    from livn.system import System
    from testing import livn_test_system
    from testing.collectives import tracer

    comm = MPI.COMM_WORLD
    tracer.watch(("livn",))
    tracer.start_test("sees-livn")

    system = System(livn_test_system())
    assert system.populations

    trace = tracer.finish_test()
    assert trace, "the tracer recorded nothing at all"

    everything = [record for records in trace.values() for record in records]
    operations = [record.op for record in everything]

    assert "bcast" in operations, (
        f"no broadcast was seen, only {sorted(set(operations))}"
    )
    assert "Split" in operations, (
        "the metadata read splits a communicator before broadcasting, and that "
        f"split is a collective too: {sorted(set(operations))}"
    )

    from livn.system import _H5_BACKEND

    if _H5_BACKEND == "neuroh5" and comm.rank == 0:
        opaque = [record.op for record in tracer.opaque_trace()]
        assert any(op.startswith("neuroh5.") for op in opaque), (
            "a collective that happens inside C is invisible unless entry into "
            f"it is recorded, and it was not: {sorted(set(opaque))}"
        )

    sites = {record.site for record in everything}
    assert any(site.startswith("src/livn/system.py:") for site in sites), (
        f"sites are not repo-relative, so ranks would spell them differently: {sites}"
    )

    for members, _ordinal in trace:
        assert set(members) <= set(range(comm.size)), (
            f"communicator key {members} names ranks outside this world of {comm.size}"
        )
