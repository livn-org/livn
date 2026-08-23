import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax.numpy as jnp  # noqa: E402

from livn.backend.diffrax import Env  # noqa: E402
from livn.models.eventloop import SolverConfig  # noqa: E402
from livn.models.glif import GLIF  # noqa: E402
from livn.stimulus import Stimulus  # noqa: E402

from optimization.fit import fit  # noqa: E402
from optimization.losses import voltage_mse  # noqa: E402

DURATION, DT = 20.0, 0.5
AMPLITUDE = 0.05


def _env(n):
    model = GLIF.leaky_integrate_and_fire(
        mechanism="hard", config=SolverConfig(max_rate=1.0)
    )
    env = Env(n, model=model)
    env.init()
    env.record_voltage(dt=DT)
    return env


def _stimulus(n):
    steps = int(round(DURATION / DT)) + 1
    return Stimulus.from_current(np.full((steps, n), AMPLITUDE), dt=DT)


@pytest.fixture(scope="module")
def one():
    return _env(1), _stimulus(1)


@pytest.fixture(scope="module")
def many():
    return _env(8), _stimulus(8)


def _simulate(env, stimulus, params):
    return env.cells.set_params(params).run(DURATION, stimulus, dt=DT).voltage


def _voltage_loss(run, target):
    return voltage_mse(run.voltage, target)


def test_fit_recovers_a_known_parameter(one):
    env, stimulus = one
    truth = -73.5
    target = _simulate(env, stimulus, {"E_L": truth})

    theta, history = fit(
        env,
        target,
        _voltage_loss,
        {"E_L": -70.0},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=40,
        learning_rate=0.5,
    )

    assert history["loss"][-1] < history["loss"][0] / 100
    assert float(theta["E_L"]) == pytest.approx(truth, abs=0.1)


def test_fit_works_on_an_env_whose_cell_registry_is_still_cold():
    env = _env(1)
    stimulus = _stimulus(1)
    assert "_cells" not in env.__dict__

    steps = int(round(DURATION / DT)) + 1
    target = np.full((1, steps), -70.0)

    _, history = fit(
        env,
        target,
        _voltage_loss,
        {"E_L": -70.0},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=2,
        learning_rate=0.1,
    )

    assert np.all(np.isfinite(history["loss"]))


def test_a_batched_fit_recovers_n_distinct_parameter_sets(many):
    env, stimulus = many
    n = env.num_cells
    assert n > 1

    truth = np.linspace(-75.0, -68.0, n)
    target = _simulate(env, stimulus, {"E_L": truth})

    theta, history = fit(
        env,
        target,
        _voltage_loss,
        {"E_L": np.full(n, -71.5)},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=60,
        learning_rate=0.5,
    )

    recovered = np.asarray(theta["E_L"])
    assert recovered.shape == (n,)
    assert history["loss"][-1] < history["loss"][0] / 100
    np.testing.assert_allclose(recovered, truth, atol=0.3)
    assert float(np.std(recovered)) == pytest.approx(float(np.std(truth)), rel=0.1)


SPIKING_AMPLITUDE = 0.5
DISTINCT = {"tau_m": np.asarray([11.0, 14.0])}


def _spiking_stimulus(n):
    steps = int(round(DURATION / DT)) + 1
    return Stimulus.from_current(np.full((steps, n), SPIKING_AMPLITUDE), dt=DT)


def test_batched_cells_with_distinct_parameters_spike_differently():
    env, stimulus = _env(2), _spiking_stimulus(2)
    env.record_spikes()
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
    steps = int(round(DURATION / DT)) + 1
    target = np.full((2, steps), -70.0)

    _, history = fit(
        env,
        target,
        _voltage_loss,
        DISTINCT,
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=2,
        learning_rate=0.1,
    )

    assert np.all(np.isfinite(history["loss"]))


def test_fit_leaves_the_env_it_was_given_untouched(one):
    env, stimulus = one
    before = {k: np.asarray(v).copy() for k, v in env.cells.get_params().items()}

    fit(
        env,
        _simulate(env, stimulus, {"E_L": -71.0}),
        _voltage_loss,
        {"E_L": -70.0},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=2,
        learning_rate=0.5,
    )

    after = env.cells.get_params()
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(after[name]), values, err_msg=name)


def test_fit_history_covers_every_step_plus_the_returned_parameters(one):
    env, stimulus = one
    steps = 3
    theta, history = fit(
        env,
        _simulate(env, stimulus, {"E_L": -71.0}),
        _voltage_loss,
        {"E_L": -70.0},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=steps,
        learning_rate=0.1,
    )

    assert history["loss"].shape == (steps + 1,)
    assert history["params"]["E_L"].shape[0] == steps + 1
    assert np.all(np.isfinite(history["loss"]))
    np.testing.assert_allclose(history["params"]["E_L"][-1], np.asarray(theta["E_L"]))


def test_a_prior_pulls_the_fit_toward_its_reference(one):
    env, stimulus = one
    target = _simulate(env, stimulus, {"E_L": -73.5})
    reference = -68.0
    common = dict(
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=15,
        learning_rate=0.5,
    )

    free, _ = fit(env, target, _voltage_loss, {"E_L": -70.0}, **common)
    pulled, _ = fit(
        env,
        target,
        _voltage_loss,
        {"E_L": -70.0},
        prior={"E_L": jnp.asarray(reference)},
        prior_weight=10.0,
        **common,
    )

    assert abs(float(pulled["E_L"]) - reference) < abs(float(free["E_L"]) - reference)


def test_a_callback_sees_every_step(one):
    env, stimulus = one
    seen = []
    fit(
        env,
        _simulate(env, stimulus, {"E_L": -71.0}),
        _voltage_loss,
        {"E_L": -70.0},
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=3,
        learning_rate=0.1,
        callback=lambda step, theta, value: seen.append((step, value)),
    )

    assert [step for step, _ in seen] == [0, 1, 2]
    assert all(np.isfinite(value) for _, value in seen)
