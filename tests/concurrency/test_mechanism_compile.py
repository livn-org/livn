import os
import shutil
import tempfile
from pathlib import Path

import pytest

from livn.backend import backend

pytestmark = [
    pytest.mark.skipif(
        backend() != "neuron", reason="mechanism compilation is NEURON's"
    ),
    pytest.mark.needs("mpi"),
]

TIMEOUT = 90

GOOD_MOD = """
NEURON {
    SUFFIX livn_scratch_probe
    RANGE g
}

PARAMETER { g = 0 }
"""

BROKEN_MOD = """
NEURON {
    SUFFIX livn_scratch_broken
    RANGE g
}

PARAMETER { g = 0 }

BREAKPOINT {
    this_function_does_not_exist(g)
}
"""


def scratch_mechanisms(name: str, contents: str, comm) -> str:
    directory = Path(tempfile.gettempdir()) / f"livn-scratch-{name}"

    if comm.Get_rank() == 0:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        (directory / f"{name}.mod").write_text(contents)
    comm.Barrier()

    return str(directory)


def env_on(comm, mechanisms_directory, **kwargs):
    from livn.env import Env
    from livn.models.rcsd import ReducedCalciumSomaDendrite

    model = ReducedCalciumSomaDendrite()
    model.neuron_mechanisms_directory = lambda: mechanisms_directory

    return Env(2, model=model, comm=comm, **kwargs)


@pytest.mark.mpiexec(timeout=TIMEOUT, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [2, 4])
def test_a_broken_mod_file_fails_every_rank_instead_of_hanging(mpiexec_n):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size == mpiexec_n

    directory = scratch_mechanisms("broken", BROKEN_MOD, comm)

    with pytest.raises(Exception) as raised:
        env_on(comm, directory)

    message = str(raised.value)
    if comm.rank == 0:
        assert "nrnivmodl" in message or "non-zero exit" in message.lower()
    else:
        assert directory in message
        assert "rank 0" in message


@pytest.mark.mpiexec(timeout=TIMEOUT, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [2])
def test_a_missing_mechanism_directory_fails_every_rank(mpiexec_n):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size == mpiexec_n

    absent = str(Path(tempfile.gettempdir()) / "livn-scratch-does-not-exist")
    if comm.rank == 0:
        shutil.rmtree(absent, ignore_errors=True)
    comm.Barrier()

    with pytest.raises(Exception) as raised:
        env_on(comm, absent)

    assert absent in str(raised.value)


@pytest.mark.mpiexec(timeout=TIMEOUT, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [2])
def test_every_rank_agrees_on_what_was_built(mpiexec_n):
    from mpi4py import MPI

    from livn.backend.neuron import mechanisms

    comm = MPI.COMM_WORLD
    assert comm.size == mpiexec_n

    directory = scratch_mechanisms("agree", GOOD_MOD, comm)
    env_on(comm, directory)

    compiled = mechanisms.compile_mechanisms(directory)
    assert len(set(comm.allgather(compiled))) == 1, "ranks built different digests"

    published = list(Path(directory, "compiled").iterdir())
    assert len(set(comm.allgather(len(published)))) == 1
    assert len(published) == 1, f"more than one build was published: {published}"


@pytest.mark.slow
@pytest.mark.mpiexec(timeout=300, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [2])
def test_a_cold_cache_and_a_warm_one_reach_the_same_place(mpiexec_n):
    from mpi4py import MPI

    from livn.backend.neuron import mechanisms

    comm = MPI.COMM_WORLD
    assert comm.size == mpiexec_n

    directory = scratch_mechanisms("cold", GOOD_MOD, comm)

    if comm.rank == 0:
        shutil.rmtree(os.path.join(directory, "compiled"), ignore_errors=True)
    comm.Barrier()

    env_on(comm, directory)
    cold = mechanisms.compile_mechanisms(directory)

    env_on(comm, directory)
    warm = mechanisms.compile_mechanisms(directory)

    assert cold == warm
    assert len(set(comm.allgather(cold))) == 1
