import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI

from livn.backend import backend
from livn.decoding import GatherAndMerge
from livn.env import Env
from livn.integrations.distwq import DistributedEnv
from livn.types import Encoding

pytestmark = pytest.mark.skipif(backend() != "neuron", reason="NEURON only")


T_END = 250
STIM_AMPLITUDE = 750


class ConstantChannelInput(Encoding):
    """Applies a 20 ms stimulus on all channels starting at t=50 ms."""

    def __call__(self, env, t_end, inputs):
        channel_inputs = np.zeros([t_end, env.io.num_channels])
        for r in range(20):
            for c in range(env.io.num_channels):
                channel_inputs[50 + r, c] = STIM_AMPLITUDE
        return env.cell_stimulus(channel_inputs)


def _reference_path():
    tmp = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp, exist_ok=True)
    return os.path.join(tmp, "distwq_reference.p")


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [1, 3])
def test_distributed_env_matches_standard(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = os.environ["LIVN_TEST_SYSTEM"]
    reference = _reference_path()

    if mpiexec_n == 1:
        env = Env(system, seed=123)
        env.init()
        env.record_spikes()

        inputs = np.zeros([T_END, env.io.num_channels])
        for r in range(20):
            for c in range(env.io.num_channels):
                inputs[50 + r, c] = STIM_AMPLITUDE

        stimulus = env.cell_stimulus(inputs)
        it, t, *_ = env.run(T_END, stimulus=stimulus)

        with open(reference, "wb") as f:
            pickle.dump((it, t), f)

        env.close()
        return

    env = DistributedEnv(system, seed=123, subworld_size=1)

    # attributes should be accessible immediately after construction
    assert env.system is not None
    assert env.model is not None
    assert env.io is not None
    assert env.io.num_channels > 0

    env.init()

    responses = env(
        GatherAndMerge(
            duration=T_END, spikes=True, voltages=False, membrane_currents=False
        ),
        inputs=[None],
        encoding=ConstantChannelInput(),
    )

    if responses is not None:
        it, t, *_ = responses[0]

        with open(reference, "rb") as f:
            rit, rt = pickle.load(f)

        np.testing.assert_allclose(
            np.sort(rt),
            np.sort(t),
            err_msg="Spike times differ between standard and distributed env",
        )
        np.testing.assert_array_equal(
            np.sort(rit, axis=None),
            np.sort(it, axis=None),
            err_msg="Spiking neuron IDs differ between standard and distributed env",
        )

    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_env_attribute_access(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=42, subworld_size=1)

    assert env.system is not None, "system should resolve lazily"
    assert env.system.uri == system
    assert env.io is not None
    assert env.model is not None
    num_channels_pre = env.io.num_channels
    assert num_channels_pre > 0

    assert env.system.cells_meta_data is not None
    assert len(env.system.populations) > 0
    assert env.system.num_neurons > 0

    env.init()

    assert env.io.num_channels == num_channels_pre
    assert env.system.uri == system
    assert env.system.num_neurons > 0

    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_env_multiple_inputs(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    env.init()
    env.record_spikes()

    responses = env(
        GatherAndMerge(
            duration=T_END, spikes=True, voltages=False, membrane_currents=False
        ),
        inputs=[None, None],
        encoding=ConstantChannelInput(),
    )

    if responses is not None:
        assert len(responses) == 2

        it0, t0, *_ = responses[0]
        it1, t1, *_ = responses[1]
        np.testing.assert_allclose(np.sort(t0), np.sort(t1))
        np.testing.assert_array_equal(np.sort(it0, axis=None), np.sort(it1, axis=None))

    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_env_subworld_size_gt_one(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=2)

    assert env.system is not None
    assert env.model is not None
    assert env.io is not None
    assert env.io.num_channels > 0

    env.init()

    responses = env(
        GatherAndMerge(
            duration=T_END, spikes=True, voltages=False, membrane_currents=False
        ),
        inputs=[None],
        encoding=ConstantChannelInput(),
    )

    if responses is not None:
        assert len(responses) == 1
        assert len(responses[0]) >= 2

    env.shutdown()
