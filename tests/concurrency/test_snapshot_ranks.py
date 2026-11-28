import os

import numpy as np
import pytest

from livn.backend import backend
from testing import livn_test_env, livn_test_system

pytestmark = [pytest.mark.needs("mpi", "snapshot")]

TIMEOUT = int(os.environ.get("LIVN_TEST_TIMEOUT", 300))
NOISE = {
    "g_e0": 0.5,
    "g_i0": 0.3,
    "std_e": 0.15,
    "std_i": 0.1,
    "tau_e": 10.0,
    "tau_i": 10.0,
}


def _gathered(comm, run):
    parts = comm.gather(
        (run.spike_ids, run.spike_times, run.voltage_ids, run.voltage), root=0
    )
    if comm.rank != 0:
        return None
    ids = np.concatenate([p[0] for p in parts])
    times = np.concatenate([p[1] for p in parts])
    order = np.lexsort((ids, times))
    traces = [p for p in parts if p[3] is not None and p[3].size]
    return (
        ids[order],
        times[order],
        np.concatenate([p[2] for p in traces]),
        np.concatenate([p[3] for p in traces]),
    )


def _identical(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize("mpiexec_n", [2, 4])
def test_each_rank_restores_its_part_of_the_network(mpiexec_n):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.size == mpiexec_n
    env = livn_test_env(comm=comm).init()
    try:
        env.apply_model_defaults()
        env.set_noise(NOISE)
        env.record_spikes().record_voltage(dt=0.1)
        env.run(80.0)
        snapshot = env.save_state()

        straight = _gathered(comm, env.run(60.0))
        env.run(25.0)
        env.clear()
        env.run(10.0)
        forked = _gathered(comm, env.restore_state(snapshot).run(60.0))
        other = snapshot.noise_stream + 1
        reseeded = _gathered(
            comm, env.restore_state(snapshot, noise_stream=other).run(60.0)
        )
        if comm.rank == 0:
            assert straight[0].size > 0, "a silent window would prove nothing"
            assert _identical(straight, forked)
            assert not _identical(straight, reseeded)
    finally:
        env.close()


@pytest.mark.skipif(backend() != "neuron", reason="DistributedEnv runs NEURON")
@pytest.mark.mpiexec(timeout=TIMEOUT, isolated=True)
@pytest.mark.parametrize("mpiexec_n", [2])
def test_a_distributed_env_forks_its_workers(mpiexec_n):
    from livn.decoding import GatherAndMerge
    from livn.env.distributed import DistributedEnv
    from livn.parallel import Layout

    env = DistributedEnv(livn_test_system(), seed=123, layout=Layout(ranks_per_env=1))
    env.init()
    if env.controller is not None:
        env.apply_model_defaults()
        env.set_noise(NOISE)

        def spikes(duration):
            decoding = GatherAndMerge(
                duration=duration, voltages=False, membrane_currents=False
            )
            ids, times, *_ = env(decoding)[0]
            return np.asarray(ids), np.asarray(times)

        spikes(80)
        snapshot = env.save_state()
        straight = spikes(60)
        spikes(30)
        env.restore_state(snapshot)
        forked = spikes(60)
        env.release_state(snapshot)
        assert straight[1].size > 0, "a silent window would prove nothing"
        np.testing.assert_array_equal(straight[0], forked[0])
        np.testing.assert_array_equal(straight[1], forked[1])
    env.shutdown()
