import numpy as np
import pytest

from livn.backend import backend


@pytest.mark.skipif(
    backend() not in ("brian2", "diffrax", "neuron"),
    reason="requires a simulation backend",
)
@pytest.mark.parametrize("num_neurons", [1, 4])
def test_parallel_system_env_runs_without_a_graph(num_neurons):
    from livn.env import Env
    from livn.system import ParallelSystem

    env = Env(num_neurons)
    assert isinstance(env.system, ParallelSystem)

    env.init()
    env.record_spikes()
    env.record_voltage()

    _, _, _, v, _, _ = env.run(20.0)

    assert v is not None
    assert np.asarray(v).shape[0] % num_neurons == 0
    assert np.asarray(v).shape[0] >= num_neurons

    if hasattr(env, "close"):
        env.close()


@pytest.mark.skipif(
    backend() not in ("brian2", "neuron"),
    reason="requires a backend that reports per-cell recording ids",
)
def test_parallel_system_env_reports_explicit_gids():
    from livn.env import Env
    from livn.system import ParallelSystem

    gids = [100, 7, 42]
    system = ParallelSystem(
        3, coordinates=[[g, float(i), 0.0, 0.0] for i, g in enumerate(gids)]
    )

    env = Env(system).init()
    env.record_voltage()
    _, _, iv, _, _, _ = env.run(10.0)

    assert set(np.asarray(iv).tolist()) == set(gids)

    if hasattr(env, "close"):
        env.close()


@pytest.mark.needs("simulation")
def test_parallel_system_env_selection():
    from livn.env import Env

    env = Env(10)
    env.selection(3)
    env.init()

    assert sorted(env.cells["EXC"]) == [0, 1, 2]

    env.close()


@pytest.mark.needs("simulation")
def test_parallel_system_env_patch_selection():
    from livn.env import Env
    from livn.system import ParallelSystem

    system = ParallelSystem(
        10, coordinates=[[gid, (9 - gid) * 100.0, 0.0, 0.0] for gid in range(10)]
    )

    env = Env(system)
    env.selection(None, method="patch", bounds=[[0.0, -1.0], [250.0, 1.0]])
    env.init()

    assert sorted(env.cells["EXC"]) == [7, 8, 9]

    env.close()
