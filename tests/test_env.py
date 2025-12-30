import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI

from livn.backend import backend
from livn.env import Env
from livn.io import MEA
from livn.models.izhikevich import Izhikevich
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.utils import P


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1, 4] if backend() != "brian2" else [1])
@pytest.mark.parametrize(
    "subworld",
    [False, True],
)
def test_env(mpiexec_n, subworld):
    if mpiexec_n == 1 and subworld:
        return

    assert MPI.COMM_WORLD.size == mpiexec_n

    comm = MPI.COMM_WORLD
    if subworld:
        color = comm.rank // 2
        subcomm = comm.Split(color, comm.rank)
        comm = subcomm

    tmp_data_path = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_data_path, exist_ok=True)

    reference = os.path.join(tmp_data_path, f"reference.p")

    system = os.environ["LIVN_TEST_SYSTEM"]
    io = MEA.from_json(os.path.join(system, "mea.json"))

    env = Env(
        system,
        model=ReducedCalciumSomaDendrite() if backend() == "neuron" else Izhikevich(),
        io=io,
        comm=comm,
        subworld_size=2 if subworld else None,
    )

    env.init()

    env.record_spikes()
    env.record_voltage()

    t_end = 250
    inputs = np.zeros([t_end, env.io.num_channels])
    for r in range(20):
        for c in range(env.io.num_channels):
            inputs[50 + r, c] = 750

    stimulus = env.cell_stimulus(inputs)
    assert np.any(stimulus > 0)

    it, t, *_ = env.run(t_end, stimulus=stimulus)

    cit, ct = env.channel_recording(it, t)

    if mpiexec_n == 1:
        # save the reference
        with open(reference, "wb") as f:
            pickle.dump((it, t, dict(cit), dict(ct)), f)

        return

    it, t, cit, ct = P.gather(it, t, cit, ct, comm=comm)

    if P.is_root(comm=comm):
        it, t, cit, ct = P.merge(it, t, cit, ct)

        with open(reference, "rb") as f:
            rit, rt, rcit, rct = pickle.load(f)

        np.testing.assert_allclose(np.sort(rt), np.sort(t))
        np.testing.assert_allclose(np.sort(rct[0]), np.sort(ct[0]))

    if hasattr(env, "pc"):
        env.pc.done()
