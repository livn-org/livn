import os
import pickle

import numpy as np
import pytest

from livn.backend import backend
from livn.types import Encoding
from testing import (
    livn_test_mea,
    livn_test_selection,
    livn_test_system,
)

pytestmark = [
    pytest.mark.skipif(backend() != "neuron", reason="NEURON only"),
    pytest.mark.mpiexec(isolated=True),
]

T_END = 250
STIM_AMPLITUDE = 250.0

if backend() == "neuron":
    from livn.decoding import GatherAndMerge
    from livn.env import Env
    from livn.env.distributed import DistributedEnv
    from livn.parallel import Layout

    class ConstantChannelInput(Encoding):
        def __call__(self, env, t_end, inputs):
            channel_inputs = np.zeros([t_end, env.io.num_channels])
            for r in range(20):
                for c in range(env.io.num_channels):
                    channel_inputs[50 + r, c] = STIM_AMPLITUDE
            return env.cell_stimulus(channel_inputs)


REFERENCE_ENV = "LIVN_TEST_DISTRIBUTED_REFERENCE"


@pytest.fixture(scope="module")
def standalone_reference(tmp_path_factory):
    inherited = os.environ.get(REFERENCE_ENV)
    if inherited:
        with open(inherited, "rb") as f:
            return pickle.load(f)

    env = Env(livn_test_system(), io=livn_test_mea(), seed=123)
    selection = livn_test_selection()
    if selection:
        env.selection(selection)
    try:
        env.init()
        env.record_spikes()

        inputs = np.zeros([T_END, env.io.num_channels])
        inputs[50:70, :] = STIM_AMPLITUDE
        it, t, *_ = env.run(T_END, stimulus=env.cell_stimulus(inputs))
        reference = (np.asarray(it), np.asarray(t))
    finally:
        env.close()

    path = tmp_path_factory.mktemp("livn-distributed-reference") / "standalone.p"
    with open(path, "wb") as f:
        pickle.dump(reference, f)
    os.environ[REFERENCE_ENV] = str(path)

    return reference


def _distributed_env(**kwargs):
    env = DistributedEnv(livn_test_system(), io=livn_test_mea(), **kwargs)
    selection = livn_test_selection()
    if selection:
        env.selection(selection)
    return env


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_env_matches_standard(mpiexec_n, standalone_reference):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n

    env = _distributed_env(seed=123, layout=Layout(ranks_per_env=1))

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
        rit, rt = standalone_reference

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
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n

    system = livn_test_system()
    env = _distributed_env(seed=42, layout=Layout(ranks_per_env=1))

    assert env.system is not None, "system should resolve lazily"
    assert env.system.uri == system
    assert env.io is not None
    assert env.model is not None
    num_channels_pre = env.io.num_channels
    assert num_channels_pre > 0

    assert len(env.system.population_ranges) > 0
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
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n

    env = _distributed_env(seed=123, layout=Layout(ranks_per_env=1))
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
def test_distributed_env_ranks_per_env_gt_one(mpiexec_n):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n

    env = _distributed_env(seed=123, layout=Layout(ranks_per_env=2))

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


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=30)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_property_access_before_init_no_deadlock(mpiexec_n):
    system = livn_test_system()
    env = _distributed_env(seed=42, layout=Layout(ranks_per_env=1))

    assert env.system is not None
    assert env.io is not None
    assert env.io.num_channels > 0
    assert env.model is not None

    assert env.system.uri == system
    assert len(env.system.populations) > 0


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=30)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_property_access_after_init_no_deadlock(mpiexec_n):
    env = _distributed_env(seed=42, layout=Layout(ranks_per_env=1))
    env.init()

    assert env.io is not None
    assert env.io.num_channels > 0
    assert env.model is not None

    env.shutdown()


def _spec_with(cells: int, sigma: float) -> dict:
    import json

    with open("./testing/culture.json") as f:
        spec = json.load(f)["system"]
    kwargs = spec.setdefault("kwargs", {})
    kwargs["total_cells"] = cells
    kwargs.setdefault("connectivity", {})["sigma"] = sigma
    return spec


class _GentleDrive(Encoding):
    def __call__(self, env, t_end, inputs):
        channel_inputs = np.zeros([t_end, env.io.num_channels])
        channel_inputs[50:70, :] = 20.0
        return env.cell_stimulus(channel_inputs)


@pytest.mark.mpiexec(timeout=300)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_a_described_spec_serves_as_well_as_a_path(mpiexec_n):
    from livn.env.distributed import DistributedEnv
    from livn.parallel import Layout

    env = DistributedEnv(_spec_with(40, 200.0), layout=Layout(ranks_per_env=1))
    env.init()
    assert sum(env.system.population_counts.values()) == 40
    env.shutdown()


@pytest.mark.mpiexec(timeout=300)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_restructuring_swaps_the_network_on_every_worker(mpiexec_n):
    from livn.env.distributed import DistributedEnv
    from livn.parallel import Layout

    env = DistributedEnv(_spec_with(40, 200.0), layout=Layout(ranks_per_env=1))
    env.init()

    seen = {}
    for label, spec in (("before", None), ("after", _spec_with(24, 300.0))):
        if spec is not None:
            env.restructure(spec)
        responses = env(
            GatherAndMerge(
                duration=T_END, spikes=True, voltages=False, membrane_currents=False
            ),
            inputs=[None],
            encoding=_GentleDrive(),
        )
        if responses is not None:  # workers return from init() too
            _it, t, *_ = responses[0]
            seen[label] = (sum(env.system.population_counts.values()), len(t))

    if seen:
        assert seen["before"][0] == 40, seen
        assert seen["after"][0] == 24, "the controller's view follows the workers"
        assert seen["after"][1] != seen["before"][1], (
            f"the workers kept simulating the old network: {seen}"
        )
    env.shutdown()


def test_a_live_system_is_refused_because_a_worker_cannot_rebuild_it():
    from livn.env.distributed import _reconstructible

    for allowed in ("./testing/culture.json", 40, {"cls": "x", "kwargs": {}}):
        _reconstructible(allowed)

    with pytest.raises(ValueError, match="cannot be rebuilt on a worker"):
        _reconstructible(object())
