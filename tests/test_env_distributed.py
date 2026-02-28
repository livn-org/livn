import os
import pickle
import time

import numpy as np
import pytest
from mpi4py import MPI

from livn.backend import backend
from livn.decoding import GatherAndMerge
from livn.env import Env
from livn.env.distributed import DistributedEnv
from livn.types import Encoding

pytestmark = pytest.mark.skipif(backend() != "neuron", reason="NEURON only")


T_END = 250
STIM_AMPLITUDE = 750

# Distributed operations should complete within this factor of the
# single-rank baseline accounting for MPI overhead and CI variability
_OVERHEAD_FACTOR = 3.0
_HARD_TIMEOUT = 60

_BASELINE_FILE = os.path.join(os.path.dirname(__file__), "tmp", "perf_baseline.p")


class ConstantChannelInput(Encoding):
    """Applies a 20 ms stimulus on all channels starting at t=50 ms."""

    def __call__(self, env, t_end, inputs):
        channel_inputs = np.zeros([t_end, env.io.num_channels])
        for r in range(20):
            for c in range(env.io.num_channels):
                channel_inputs[50 + r, c] = STIM_AMPLITUDE
        return env.cell_stimulus(channel_inputs)


def _save_baseline(timings: dict):
    os.makedirs(os.path.dirname(_BASELINE_FILE), exist_ok=True)
    with open(_BASELINE_FILE, "wb") as f:
        pickle.dump(timings, f)


def _load_baseline() -> dict | None:
    if not os.path.isfile(_BASELINE_FILE):
        return None
    with open(_BASELINE_FILE, "rb") as f:
        return pickle.load(f)


def _timed(fn, *args, **kwargs):
    """Run *fn* and return (result, elapsed_seconds)."""
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return result, time.monotonic() - t0


def _max_allowed(baseline_seconds: float) -> float:
    return min(max(baseline_seconds * _OVERHEAD_FACTOR, 5.0), _HARD_TIMEOUT)


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [1, 3])
def test_distributed_env_matches_standard(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    system = os.environ["LIVN_TEST_SYSTEM"]

    tmp = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp, exist_ok=True)

    reference = os.path.join(tmp, "distributed_reference.p")

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



@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=30)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_property_access_before_init_no_deadlock(mpiexec_n):
    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=42, subworld_size=1)

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
    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=42, subworld_size=1)
    env.init()

    assert env.io is not None
    assert env.io.num_channels > 0
    assert env.model is not None

    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [1])
def test_collect_baseline_timings(mpiexec_n):
    system = os.environ["LIVN_TEST_SYSTEM"]

    env = Env(system, seed=123)

    _, t_init = _timed(env.init)
    _, t_record = _timed(env.record_spikes)

    try:
        _, t_defaults = _timed(env.apply_model_defaults)
    except Exception:
        t_defaults = None

    weights = {name: 1.0 for name in env.system.weight_names}
    try:
        _, t_weights = _timed(env.set_weights, weights)
    except Exception:
        t_weights = None

    inputs = np.zeros([T_END, env.io.num_channels])
    for r in range(20):
        for c in range(env.io.num_channels):
            inputs[50 + r, c] = STIM_AMPLITUDE
    stimulus = env.cell_stimulus(inputs)

    _, t_run = _timed(env.run, T_END, stimulus=stimulus)

    env.close()

    _save_baseline(
        {
            "init": t_init,
            "record_spikes": t_record,
            "apply_model_defaults": t_defaults,
            "set_weights": t_weights,
            "run": t_run,
        }
    )


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_init_performance(mpiexec_n):
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("Run test_collect_baseline_timings[1] first")

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    _, elapsed = _timed(env.init)

    allowed = _max_allowed(baseline["init"])
    assert elapsed < allowed, (
        f"init took {elapsed:.1f}s, baseline {baseline['init']:.1f}s, "
        f"allowed {allowed:.1f}s"
    )
    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_record_spikes_performance(mpiexec_n):
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("Run test_collect_baseline_timings[1] first")

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    env.init()

    _, elapsed = _timed(env.record_spikes)

    allowed = _max_allowed(baseline["record_spikes"])
    assert elapsed < allowed, (
        f"record_spikes took {elapsed:.1f}s, baseline {baseline['record_spikes']:.1f}s, "
        f"allowed {allowed:.1f}s"
    )
    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_apply_model_defaults_performance(mpiexec_n):
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("Run test_collect_baseline_timings[1] first")
    if baseline["apply_model_defaults"] is None:
        pytest.skip("apply_model_defaults not supported by this system")

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    env.init()

    try:
        _, elapsed = _timed(env.apply_model_defaults)
    except Exception:
        env.shutdown()
        pytest.skip("apply_model_defaults not supported by this system")

    allowed = _max_allowed(baseline["apply_model_defaults"])
    assert elapsed < allowed, (
        f"apply_model_defaults took {elapsed:.1f}s, "
        f"baseline {baseline['apply_model_defaults']:.1f}s, "
        f"allowed {allowed:.1f}s"
    )
    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_set_weights_performance(mpiexec_n):
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("Run test_collect_baseline_timings[1] first")
    if baseline["set_weights"] is None:
        pytest.skip("set_weights not supported by this system")

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    env.init()

    weights = {name: 1.0 for name in env.system.weight_names}
    try:
        _, elapsed = _timed(env.set_weights, weights)
    except Exception:
        env.shutdown()
        pytest.skip("set_weights not supported by this system")

    allowed = _max_allowed(baseline["set_weights"])
    assert elapsed < allowed, (
        f"set_weights took {elapsed:.1f}s, baseline {baseline['set_weights']:.1f}s, "
        f"allowed {allowed:.1f}s"
    )
    env.shutdown()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=120)
@pytest.mark.parametrize("mpiexec_n", [3])
def test_distributed_run_performance(mpiexec_n):
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("Run test_collect_baseline_timings[1] first")

    system = os.environ["LIVN_TEST_SYSTEM"]
    env = DistributedEnv(system, seed=123, subworld_size=1)
    env.init()
    env.record_spikes()
    try:
        env.apply_model_defaults()
    except Exception:
        pass  # not all systems support this

    _, elapsed = _timed(
        env,
        GatherAndMerge(
            duration=T_END, spikes=True, voltages=False, membrane_currents=False
        ),
        inputs=[None],
        encoding=ConstantChannelInput(),
    )

    allowed = _max_allowed(baseline["run"])
    assert elapsed < allowed, (
        f"run took {elapsed:.1f}s, baseline {baseline['run']:.1f}s, "
        f"allowed {allowed:.1f}s"
    )
    env.shutdown()
