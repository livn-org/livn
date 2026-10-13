import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from livn.env import Env
from livn.io import MEA
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.stimulus import Stimulus
from livn.system import (
    CartesianParameterization,
    LogRadialParameterization,
    OrientedTrainableSystem,
    SphericalParameterization,
    TrainableSystem,
)

N, DT, DURATION = 3, 0.1, 4.0


def _parameterization(cls=LogRadialParameterization):
    xy = np.column_stack(
        [np.linspace(-30.0, 30.0, N), np.zeros(N)],
    ).astype(np.float32)
    return cls.from_lateral_depth(xy, depth=50.0)


def _io():
    return MEA(np.array([[0, -40.0, 0.0, 0.0], [1, 40.0, 0.0, 0.0]]))


def _env(system):
    env = Env(
        system=system,
        model=ReducedCalciumSomaDendrite(input_mode="current"),
        io=_io(),
    ).init()
    return env.record_membrane_current(dt=DT)


def _stimulus():
    return Stimulus.from_current(
        np.full((round(DURATION / DT), N), 0.05), dt=DT, gids=np.arange(N)
    )


def _loss(env):
    run = env.run(DURATION, stimulus=_stimulus(), dt=DT, dt_solver=0.05)
    return jnp.sum(env.potential_recording(run.current) ** 2)


def _grad(env):
    # the env also carries PRNG keys, which are integers
    return jax.grad(_loss, allow_int=True)(env)


def _finite_and_nonzero(x):
    x = np.asarray(x)
    assert np.all(np.isfinite(x)) and np.any(x != 0.0)


@pytest.mark.parametrize(
    "cls",
    [CartesianParameterization, LogRadialParameterization, SphericalParameterization],
)
def test_positions_differentiate_through_run_and_recording(cls):
    grad = _grad(_env(TrainableSystem(_parameterization(cls), n_neurons=N)))

    leaves = jax.tree_util.tree_leaves(grad.system.parameterization)
    assert leaves
    for leaf in leaves:
        _finite_and_nonzero(leaf)


def test_sigma_differentiates_through_run_and_recording():
    env = _env(TrainableSystem(_parameterization(), n_neurons=N))
    grad = _grad(env)

    _finite_and_nonzero(grad.io.medium.electrical.sigma_s_per_m)
    # the recording is 1/sigma, so dL/dsigma = -2 L / sigma for a squared potential
    expected = -2.0 * float(_loss(env)) / env.io.medium.electrical.sigma_s_per_m
    assert float(grad.io.medium.electrical.sigma_s_per_m) == pytest.approx(
        expected, rel=1e-3
    )


def test_the_trainable_env_round_trips_as_a_pytree():
    env = _env(TrainableSystem(_parameterization(), n_neurons=N))
    out = jax.tree_util.tree_map(lambda x: x, env)

    assert isinstance(out.system, TrainableSystem)
    assert np.array_equal(out.system.params, env.system.params)
    assert np.allclose(_loss(out), _loss(env))


def test_the_default_io_is_a_medium_carrying_mea():
    io = TrainableSystem(_parameterization(), n_neurons=N).default_io()

    assert isinstance(io, MEA)
    assert io.medium.electrical.sigma_s_per_m > 0


def test_orientation_differentiates_through_run_and_recording():
    system = OrientedTrainableSystem.from_parameterization(
        _parameterization(), n_neurons=N, orient_init=np.tile([1.0, 0.2, 0.5], (N, 1))
    )
    grad = _grad(_env(system))

    assert grad.system.orientation.raw.shape == (N, 3)
    _finite_and_nonzero(grad.system.orientation.raw)
    _finite_and_nonzero(jax.tree_util.tree_leaves(grad.system.base.parameterization)[0])


def test_orientation_moves_only_the_dendrite_rows():
    model = ReducedCalciumSomaDendrite(input_mode="current")
    base = TrainableSystem(_parameterization(), n_neurons=N)
    oriented = OrientedTrainableSystem.from_parameterization(
        _parameterization(),
        n_neurons=N,
        orient_init=np.tile([0.0, 0.0, 1.0], (N, 1)),
        dipole_length_um=100.0,
    )

    plain = np.asarray(base.transform_coordinates(model.recording_coordinates))
    turned = np.asarray(oriented.transform_coordinates(model.recording_coordinates))

    width = plain.shape[0] // N
    plain, turned = plain.reshape(N, width, 4), turned.reshape(N, width, 4)
    np.testing.assert_allclose(turned[:, 0], plain[:, 0])
    np.testing.assert_allclose(turned[:, 2:], plain[:, 2:])
    np.testing.assert_allclose(
        turned[:, 1, 1:] - turned[:, 0, 1:],
        np.tile([0.0, 0.0, 100.0], (N, 1)),
        atol=1e-4,
    )


def test_orientation_is_normalised_at_use_time():
    from livn.system import LearnableOrientation

    unit = LearnableOrientation(
        raw=jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0], [1.0, 0.0, 0.0]])
    ).unit
    np.testing.assert_allclose(jnp.linalg.norm(unit, axis=-1), 1.0, atol=1e-5)
