import glob
import os
import shutil

import pytest

from livn.backend import backend
from livn.models.rcsd import ReducedCalciumSomaDendrite

_HERE = os.path.dirname(os.path.abspath(__file__))
_GRAPHS = os.path.join(os.path.dirname(_HERE), "systems", "graphs")

# case -> (model import path, system graph dir, selection or None)
_CASES = {
    "rcsd": ("livn.models.rcsd.ReducedCalciumSomaDendrite", "EI1", None),
    "ca1": ("livn.models.ca1.PinskyRinzel", "CA1", {"PYR": 3}),
}


def test_model():
    ReducedCalciumSomaDendrite()


@pytest.mark.skipif(
    backend() != "neuron", reason="mechanism (re)compilation is neuron-specific"
)
@pytest.mark.mpiexec(timeout=600)
@pytest.mark.parametrize("mpiexec_n", [1, 2])
@pytest.mark.parametrize("case", list(_CASES))
def test_recompile_and_smoke(case, mpiexec_n):
    """Recompile a model's NEURON mechanisms from scratch, then simulate 100 ms.

    Each case runs in its own mpiexec subprocess so the two models never load
    their overlapping mechanism SUFFIXes (VecStim, LinExp2Syn, ...) into the
    same process, and so the cold-cache MPI compile path (rank-0 build +
    barrier + atomic publish) is exercised. Skipped when the graph is absent.
    """
    from mpi4py import MPI

    from livn.env import Env
    from livn.utils import import_instance

    model_spec, graph, selection = _CASES[case]
    system = os.path.join(_GRAPHS, graph)
    if not os.path.isdir(system) or not glob.glob(os.path.join(system, "*.h5")):
        pytest.skip(f"system graph '{graph}' not present")

    comm = MPI.COMM_WORLD
    model = import_instance(model_spec)

    # Force a from-scratch recompile: rank 0 wipes the cache, all ranks wait.
    compiled = os.path.join(model.neuron_mechanisms_directory(), "compiled")
    if comm.Get_rank() == 0 and os.path.isdir(compiled):
        shutil.rmtree(compiled)
    comm.Barrier()

    env = Env(system, model=model, comm=comm)
    if selection is not None:
        env.selection(selection)
    env.init()

    # the wipe above forced an actual (re)build
    assert os.path.isdir(compiled), "mechanisms were not recompiled"

    env.record_spikes()
    result = env.run(100.0, root_only=False)

    assert isinstance(result, tuple) and len(result) == 6
    assert env.t >= 100.0
