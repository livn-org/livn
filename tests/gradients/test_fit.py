import functools

import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax.numpy as jnp
import optax

from livn.backend.diffrax import Env
from livn.models.eventloop import SolverConfig
from livn.models.glif import GLIF
from livn.optimization.fit import fit
from livn.optimization.losses import param_prior, voltage_mse
from livn.optimization.train import Callback
from livn.stimulus import Stimulus

DURATION, DT = 20.0, 0.5
AMPLITUDE = 0.05


def _env(n):
    model = GLIF.leaky_integrate_and_fire(
        mechanism="hard", config=SolverConfig(max_rate_per_ms=1.0)
    )
    env = Env(n, model=model)
    env.init()
    return env.record_voltage(dt=DT)


def _stimulus(n):
    steps = round(DURATION / DT) + 1
    return Stimulus.from_current(np.full((steps, n), AMPLITUDE), dt=DT)


@pytest.fixture(scope="module")
def one():
    return _env(1), _stimulus(1)


@pytest.fixture(scope="module")
def many():
    return _env(8), _stimulus(8)


def _simulate(env, stimulus, params):
    return env.cells.set_params(params).run(DURATION, stimulus, dt=DT).voltage


def _cell(name):
    return lambda m: getattr(m.func.module, name)


def _fit(env, stimulus, target, init, *, lr=0.5, **kwargs):
    kwargs.setdefault("optimizer", optax.adam(lr))
    return fit(
        functools.partial(env.cells.set_params(init), round(DURATION), stimulus, dt=DT),
        lambda run, _: voltage_mse(run.voltage, target),
        params={name: _cell(name) for name in init},
        **kwargs,
    )


def test_fit_recovers_a_known_parameter(one):
    env, stimulus = one
    truth = -73.5
    target = _simulate(env, stimulus, {"E_L": truth})

    result = _fit(env, stimulus, target, {"E_L": -70.0}, steps=40)

    assert result.history["loss"][-1] < result.history["loss"][0] / 100
    assert float(np.squeeze(result.theta["E_L"])) == pytest.approx(truth, abs=0.1)


def test_fit_works_on_an_env_whose_cell_registry_is_still_cold():
    env = _env(1)
    stimulus = _stimulus(1)
    assert "_cells" not in env.__dict__

    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)

    result = _fit(env, stimulus, target, {"E_L": -70.0}, steps=2, lr=0.1)

    assert np.all(np.isfinite(result.history["loss"]))


def test_a_batched_fit_recovers_n_distinct_parameter_sets(many):
    env, stimulus = many
    n = env.num_cells
    assert n > 1

    truth = np.linspace(-75.0, -68.0, n)
    target = _simulate(env, stimulus, {"E_L": truth})

    result = _fit(env, stimulus, target, {"E_L": np.full(n, -71.5)}, steps=60)

    recovered = np.asarray(result.theta["E_L"])
    assert recovered.shape == (n,)
    assert result.history["loss"][-1] < result.history["loss"][0] / 100
    np.testing.assert_allclose(recovered, truth, atol=0.3)
    assert float(np.std(recovered)) == pytest.approx(float(np.std(truth)), rel=0.1)


SPIKING_AMPLITUDE = 0.5
DISTINCT = {"tau_m": np.asarray([11.0, 14.0])}


def _spiking_stimulus(n):
    steps = round(DURATION / DT) + 1
    return Stimulus.from_current(np.full((steps, n), SPIKING_AMPLITUDE), dt=DT)


def test_batched_cells_with_distinct_parameters_spike_differently():
    env, stimulus = _env(2), _spiking_stimulus(2)
    env = env.record_spikes()
    padded = np.asarray(
        env.cells.set_params(DISTINCT)
        .run(DURATION, stimulus, dt=DT)
        .spikes.padded.times
    )
    counts = [int(np.isfinite(row).sum()) for row in padded]
    assert all(c > 0 for c in counts), f"cells must actually spike, got {counts}"
    assert len(set(counts)) > 1, (
        f"cells must spike a different number of times, got {counts}"
    )


def test_a_batched_fit_differentiates_through_cells_that_spike_differently():
    env, stimulus = _env(2), _spiking_stimulus(2)
    steps = round(DURATION / DT) + 1
    target = np.full((2, steps), -70.0)

    result = _fit(env, stimulus, target, DISTINCT, steps=2, lr=0.1)

    assert np.all(np.isfinite(result.history["loss"]))


def test_fit_leaves_the_env_it_was_given_untouched(one):
    env, stimulus = one
    before = {k: np.asarray(v).copy() for k, v in env.cells.get_params().items()}

    _fit(
        env, stimulus, _simulate(env, stimulus, {"E_L": -71.0}), {"E_L": -70.0}, steps=2
    )

    after = env.cells.get_params()
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(after[name]), values, err_msg=name)


def test_fit_history_covers_every_step_plus_the_returned_parameters(one):
    env, stimulus = one
    steps = 3
    result = _fit(
        env,
        stimulus,
        _simulate(env, stimulus, {"E_L": -71.0}),
        {"E_L": -70.0},
        steps=steps,
        lr=0.1,
    )

    history = result.history
    assert history["loss"].shape == (steps + 1,)
    assert history["params"]["E_L"].shape[0] == steps + 1
    assert np.all(np.isfinite(history["loss"]))
    np.testing.assert_allclose(
        history["params"]["E_L"][-1], np.asarray(result.theta["E_L"])
    )
    assert result.state.step == steps


def test_the_fitted_model_carries_the_fitted_parameters(one):
    env, stimulus = one
    result = _fit(
        env, stimulus, _simulate(env, stimulus, {"E_L": -71.0}), {"E_L": -70.0}, steps=3
    )

    np.testing.assert_allclose(
        np.asarray(result.model.func.cells.get_params()["E_L"]),
        np.asarray(result.theta["E_L"]).ravel(),
        rtol=1e-6,
    )


def test_a_prior_pulls_the_fit_toward_its_reference(one):
    env, stimulus = one
    target = _simulate(env, stimulus, {"E_L": -73.5})
    reference = -68.0

    free = _fit(env, stimulus, target, {"E_L": -70.0}, steps=15)
    pulled = _fit(
        env,
        stimulus,
        target,
        {"E_L": -70.0},
        steps=15,
        prior=functools.partial(
            param_prior, reference={"E_L": jnp.asarray(reference)}, scale=10.0
        ),
    )

    assert abs(float(np.squeeze(pulled.theta["E_L"])) - reference) < abs(
        float(np.squeeze(free.theta["E_L"])) - reference
    )


def test_a_callback_sees_every_step(one):
    env, stimulus = one
    seen = []

    class Watch(Callback):
        def on_step_end(self, step, state, metrics):
            seen.append((step, metrics["loss"]))

    _fit(
        env,
        stimulus,
        _simulate(env, stimulus, {"E_L": -71.0}),
        {"E_L": -70.0},
        steps=3,
        lr=0.1,
        callbacks=[Watch()],
    )

    assert [step for step, _ in seen] == [0, 1, 2]
    assert all(np.isfinite(value) for _, value in seen)
