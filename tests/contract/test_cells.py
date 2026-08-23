import os

import numpy as np
import pytest

from livn.backend import backend
from livn.cells import CellRegistry
from livn.types import Cell
from livn.types import Env as EnvProtocol

TIMEOUT = int(os.environ.get("LIVN_TEST_TIMEOUT", 300))


class _StubEnv:
    def __init__(self, comm=None):
        self.cells = CellRegistry(self, comm=comm)


class _StubCell(Cell):
    def __init__(self, env, gid, population="EXC", **params):
        super().__init__(env, population, gid)
        self.params = dict(params)

    def get_params(self):
        return dict(self.params)

    def set_params(self, params):
        self.params.update({k: float(v) for k, v in params.items()})
        return self._env


def _stub_registry():
    env = _StubEnv()
    env.cells.add(
        "EXC", {0: _StubCell(env, 0, r=1.0), 2: _StubCell(env, 2, r=3.0)}
    ).add("INH", {1: _StubCell(env, 1, "INH", r=2.0, tau=5.0)})
    return env.cells


def test_registry_indexes_by_population_and_gid():
    registry = _stub_registry()

    assert list(registry) == ["EXC", "INH"]
    assert len(registry) == 2
    assert sorted(registry["EXC"]) == [0, 2]
    assert registry[1].population == "INH"
    assert registry["EXC"][2] is registry[2]

    assert registry.get("MISSING", {}) == {}
    assert len(registry.get("EXC", {})) == 2
    assert {p for p in registry.keys()} == {"EXC", "INH"}
    assert sum(len(cells) for cells in registry.values()) == 3
    assert [p for p, _ in registry.items()] == ["EXC", "INH"]

    np.testing.assert_array_equal(registry.gids, [0, 1, 2])

    with pytest.raises(KeyError):
        registry[7]
    with pytest.raises(KeyError):
        registry["MISSING"]

    registry.clear()
    assert len(registry) == 0
    assert not registry


def test_registry_get_params_pads_missing_with_nan():
    params = _stub_registry().get_params()

    np.testing.assert_allclose(params["r"], [1.0, 2.0, 3.0])
    assert np.isnan(params["tau"][0])
    np.testing.assert_allclose(params["tau"][1], 5.0)


def test_registry_set_params_broadcasts_and_distributes():
    registry = _stub_registry()

    registry.set_params({"r": 7.0})
    np.testing.assert_allclose(registry.get_params()["r"], [7.0, 7.0, 7.0])

    registry.set_params({"r": [1.0, 2.0, 3.0]})
    np.testing.assert_allclose(registry.get_params()["r"], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="4 values but there are 3 cells"):
        registry.set_params({"r": [1.0, 2.0, 3.0, 4.0]})


class _StubComm:
    def __init__(self, peer):
        self._peer = peer

    def Get_size(self):
        return 2

    def allgather(self, value):
        return [value, self._peer(value)]


def test_registry_spans_the_ranks_of_its_communicator():
    peers = {0: _StubCell(None, 0, r=0.0), 1: _StubCell(None, 1, r=0.0)}

    def peer(value):
        if isinstance(value, dict):
            return {gid: cell.get_params() for gid, cell in peers.items()}
        return np.array(sorted(peers), dtype=np.int64)

    env = _StubEnv(comm=_StubComm(peer))
    registry = env.cells
    registry.add("EXC", {2: _StubCell(env, 2, r=0.0)})

    np.testing.assert_array_equal(registry.gids, [0, 1, 2])
    np.testing.assert_array_equal(registry.local_gids, [2])

    registry.set_params({"r": [1.0, 2.0, 3.0]})

    assert registry[2].get_params()["r"] == 3.0
    np.testing.assert_allclose(registry.get_params()["r"], [0.0, 0.0, 3.0])


class _RecordingCells:
    def __init__(self, env):
        self.params = {}
        self.env = env

    def set_params(self, params):
        self.params.update(params)
        return self.env


class _RoutingEnv(EnvProtocol):
    def __init__(self, cells_env=None):
        self.weights = {}
        self.noise = {}
        self.cells = _RecordingCells(self if cells_env is None else cells_env)

    def set_weights(self, weights):
        self.weights.update(weights)
        return self

    def set_noise(self, noise):
        self.noise.update(noise)
        return self


def test_set_params_routes_by_prefix():
    env = _RoutingEnv()

    env.set_params(
        {
            "EXC_EXC-hillock-AMPA-weight": 0.5,
            "weight-EXC_INH-hillock-AMPA-weight": 0.25,
            "noise-g_e0": 1.0,
            "cells-soma.g_pas": 3e-5,
        }
    )

    assert env.weights == {
        "EXC_EXC-hillock-AMPA-weight": 0.5,
        "EXC_INH-hillock-AMPA-weight": 0.25,
    }
    assert env.noise == {"g_e0": 1.0}
    assert env.cells.params == {"soma.g_pas": 3e-5}


def test_set_params_without_cells_is_unchanged():
    env = _RoutingEnv()

    assert env.set_params({"noise-g_e0": 1.0}) is env
    assert env.cells.params == {}


def test_set_params_returns_the_env_a_functional_backend_hands_back():
    successor = _RoutingEnv()
    env = _RoutingEnv(cells_env=successor)

    assert env.set_params({"cells-soma.g_pas": 3e-5}) is successor


class _ToyCulture:
    def __new__(cls, num_neurons):
        import equinox as eqx
        import jax.numpy as jnp

        class Culture(eqx.Module):
            num_neurons: int = eqx.field(static=True)
            tau: jnp.ndarray
            gain: jnp.ndarray

            def run(
                self,
                input_current=None,
                noise=None,
                t0=0.0,
                t1=1.0,
                dt=0.1,
                y0=None,
                record=None,
                **kwargs,
            ):
                v = jnp.zeros((self.num_neurons,))
                trace = []
                for _ in range(int((t1 - t0) / dt)):
                    v = v + dt * (-v / self.tau + self.gain)
                    trace.append(v)
                voltage = jnp.stack(trace, axis=1)
                ids = jnp.arange(self.num_neurons)
                return None, None, ids, voltage, ids, voltage, v, {}

        return Culture(
            num_neurons=num_neurons,
            tau=jnp.full((num_neurons,), 10.0),
            gain=jnp.full((num_neurons,), 1.0),
        )


class _ToyModel:
    def diffrax_module(self, env, key=None):
        return _ToyCulture(len(env.active_gids()))

    def prepare_stimulus(self, stimulus):
        return stimulus

    def ignored_populations(self):
        return set()


def _make_env(n=3, **kwargs):
    from livn.env import Env

    if backend() == "diffrax":
        return Env(n, model=_ToyModel(), **kwargs).init()
    return Env(n, **kwargs).init()


def _a_param(params):
    assert params, "the backend's cells expose no parameters"
    return sorted(params)[0]


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_cells_registry_covers_the_simulated_cells():
    env = _make_env(3)

    np.testing.assert_array_equal(env.cells.gids, [0, 1, 2])
    assert sorted(env.cells["EXC"]) == [0, 1, 2]
    assert isinstance(env.cells[1], Cell)
    assert env.cells[1].gid == 1
    assert env.cells[1].population == "EXC"
    assert env.cells[1].env is env
    assert len(env.cells.get("EXC", {})) == 3
    assert len(env.cells.get("MISSING", {})) == 0

    env.close()


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_cell_params_round_trip():
    env = _make_env(3)

    params = env.cells.get_params()
    name = _a_param(params)
    assert params[name].shape == (3,)

    env = env.cells.set_params({name: 0.5})
    np.testing.assert_allclose(env.cells.get_params()[name], [0.5, 0.5, 0.5])

    env = env.cells.set_params({name: [0.25, 0.5, 0.75]})
    np.testing.assert_allclose(env.cells.get_params()[name], [0.25, 0.5, 0.75])

    assert np.allclose(float(env.cells[2].get_params()[name]), 0.75)

    env.close()


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_cell_setter_only_affects_that_cell():
    env = _make_env(3)

    name = _a_param(env.cells.get_params())
    env = env.cells.set_params({name: 1.0})
    env = env.cells[1].set_params({name: 2.0})

    np.testing.assert_allclose(env.cells.get_params()[name], [1.0, 2.0, 1.0])

    env.close()


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_cells_prefix_reaches_every_cell():
    env = _make_env(3)

    name = _a_param(env.cells.get_params())
    env = env.set_params({f"cells-{name}": 1.5})

    np.testing.assert_allclose(env.cells.get_params()[name], [1.5, 1.5, 1.5])

    env.close()


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_unknown_cell_param_raises():
    env = _make_env(3)

    with pytest.raises(KeyError):
        env.cells.set_params({"not-a-parameter": 1.0})

    env.close()


@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
def test_simulation_runs_after_setting_cell_params():
    env = _make_env(3)

    params = env.cells.get_params()
    name = _a_param(params)
    env = env.cells.set_params({name: params[name]})
    env.record_voltage()

    run = env.run(10.0)

    assert run.voltage is not None
    assert np.all(np.isfinite(np.asarray(run.voltage)))

    env.close()


@pytest.mark.skipif(
    backend() != "diffrax", reason="only the diffrax backend is differentiable"
)
def test_cell_params_are_differentiable():
    import jax
    import jax.numpy as jnp

    env = _make_env(4)
    env.record_voltage()
    theta = env.cells.get_params()
    assert set(theta) == {"gain", "tau"}

    def loss(theta):
        return jnp.sum(env.cells.set_params(theta).run(20.0, dt=0.5).voltage ** 2)

    gradients = jax.grad(loss)(theta)

    assert set(gradients) == set(theta)
    for name, gradient in gradients.items():
        gradient = np.asarray(gradient)
        assert gradient.shape == (4,), name
        assert np.all(np.isfinite(gradient)), name
        assert np.all(gradient != 0.0), name

    assert np.isfinite(float(jax.jit(loss)(theta)))


@pytest.mark.skipif(
    backend() != "diffrax", reason="only the diffrax backend is functional"
)
def test_diffrax_setters_leave_the_original_env_alone():
    env = _make_env(3)
    before = np.asarray(env.cells.get_params()["tau"])

    updated = env.cells.set_params({"tau": 99.0})
    single = env.cells[0].set_params({"tau": 42.0})

    assert updated is not env
    np.testing.assert_allclose(env.cells.get_params()["tau"], before)
    np.testing.assert_allclose(updated.cells.get_params()["tau"], [99.0, 99.0, 99.0])
    np.testing.assert_allclose(
        single.cells.get_params()["tau"], [42.0, before[1], before[2]]
    )


@pytest.mark.skipif(backend() != "neuron", reason="wraps a NEURON cell")
def test_neuron_cell_handle_stands_in_for_the_cell():
    from livn.env import Env

    env = Env(1).init()
    handle = env.cells[0]

    assert handle.sections is handle.cell.sections
    assert handle.threshold == handle.cell.threshold
    assert callable(handle.spike_source)

    handle._v_rest = -61.0
    assert handle.cell._v_rest == -61.0

    env.close()
