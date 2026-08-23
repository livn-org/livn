import glob
import os
import shutil

import pytest

from livn.backend import backend
from livn.models.rcsd import ReducedCalciumSomaDendrite
from testing.paths import GRAPHS

_GRAPHS = str(GRAPHS)

_CASES = {
    "rcsd": ("livn.models.rcsd.ReducedCalciumSomaDendrite", "EI", "e1"),
    "ca1": ("livn.models.ca1.PinskyRinzel", "CA1", {"PYR": 3}),
}


def test_model():
    ReducedCalciumSomaDendrite()


@pytest.mark.skipif(
    backend() != "neuron", reason="mechanism (re)compilation is neuron-specific"
)
@pytest.mark.slow
@pytest.mark.mpiexec(timeout=600, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [1])
@pytest.mark.parametrize("case", list(_CASES))
def test_recompile_and_smoke(case, mpiexec_n):
    from mpi4py import MPI

    from livn.env import Env
    from livn.utils import import_instance

    model_spec, graph, selection = _CASES[case]
    system = os.path.join(_GRAPHS, graph)
    if not os.path.isdir(system) or not glob.glob(os.path.join(system, "*.h5")):
        pytest.skip(f"system graph '{graph}' not present")

    comm = MPI.COMM_WORLD
    model = import_instance(model_spec)

    compiled = os.path.join(model.neuron_mechanisms_directory(), "compiled")
    if comm.Get_rank() == 0 and os.path.isdir(compiled):
        shutil.rmtree(compiled)
    comm.Barrier()

    env = Env(system, model=model, comm=comm)
    if selection is not None:
        env.selection(selection)
    env.init()

    assert os.path.isdir(compiled), "mechanisms were not recompiled"

    env.record_spikes()
    result = env.run(100.0, root_only=False)

    assert len(result) == 6
    it, tt, _iv, _v, _im, _mp = result
    assert it is result.spike_ids and tt is result.spike_times
    assert env.t >= 100.0
