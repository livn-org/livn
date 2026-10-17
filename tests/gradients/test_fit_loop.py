import functools
import json

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
import optax

from livn.io.mea import _USES_JAX
from livn.optimization import losses
from livn.optimization.fit import fit, select
from livn.optimization.train import (
    Best,
    Callback,
    EarlyStopping,
    EqxCheckpointer,
    Every,
    JsonlLogger,
    NaNGuard,
    group_chain,
)


@jax.tree_util.register_pytree_node_class
class _Tree(dict):
    def tree_flatten(self):
        keys = tuple(sorted(self))
        return [self[k] for k in keys], keys

    @classmethod
    def tree_unflatten(cls, keys, children):
        return cls(zip(keys, children, strict=True))

    def __call__(self, batch=None, key=None):
        return dict(self)


def _quadratic(output, batch):
    return jnp.sum((output["x"] - batch) ** 2)


def _model(*x):
    return _Tree(x=jnp.asarray(x, dtype=jnp.float32))


def _x(result):
    return np.asarray(result.theta["x"])


def test_groups_train_at_their_own_rates_and_a_frozen_group_stays_put():
    model = _Tree(a=jnp.array([5.0]), b=jnp.array([5.0]), c=jnp.array([5.0]))

    def loss(out, batch):
        return sum(jnp.sum(out[k] ** 2) for k in out)

    mask_a = _Tree(a=True, b=False, c=False)
    mask_b = _Tree(a=False, b=True, c=False)
    params = {"fast": mask_a, "slow": mask_b}

    result = fit(
        model,
        loss,
        params=params,
        optimizer={"fast": optax.sgd(0.2), "slow": optax.sgd(0.01)},
        steps=5,
    )

    fast = float(result.theta["fast"]["a"][0])
    slow = float(result.theta["slow"]["b"][0])
    assert fast == pytest.approx(5.0 * (1 - 0.4) ** 5, rel=1e-5)
    assert slow == pytest.approx(5.0 * (1 - 0.02) ** 5, rel=1e-5)
    assert float(result.model["c"][0]) == 5.0
    assert float(result.model["a"][0]) == pytest.approx(fast)


def test_group_labels_must_match_the_optimizer_groups():
    with pytest.raises(ValueError, match="optimizer groups"):
        fit(
            _model(1.0),
            _quadratic,
            data=0.0,
            params={"a": lambda m: m["x"]},
            optimizer={"b": optax.sgd(0.1)},
            steps=1,
        )


def test_overlapping_groups_are_refused():
    mask = _Tree(x=True)
    with pytest.raises(ValueError, match="same leaf"):
        fit(
            _model(1.0),
            _quadratic,
            data=0.0,
            params={"a": mask, "b": mask},
            steps=1,
        )


def test_a_predicate_selects_leaves_and_the_rest_is_frozen():
    model = _Tree(x=jnp.array([4.0]), n=jnp.array([3]))

    def loss(out, batch):
        return jnp.sum(out["x"] ** 2)

    result = fit(model, loss, optimizer=optax.sgd(0.25), steps=2)
    assert float(result.theta["x"][0]) == pytest.approx(1.0)
    assert int(result.model["n"][0]) == 3


def test_group_chain_composes_with_groups():
    model = _Tree(a=jnp.array([5.0]), b=jnp.array([5.0]))
    result = fit(
        model,
        lambda out, batch: sum(jnp.sum(v**2) for v in out.values()),
        params={"a": lambda m: m["a"], "b": lambda m: m["b"]},
        optimizer={
            "a": group_chain(0.5, clip_norm=1.0),
            "b": group_chain(0.5, trust_region=0.1),
        },
        steps=4,
    )
    # adam steps ~lr per step; the trust region caps b's at 0.1
    assert float(result.theta["b"][0]) >= 5.0 - 4 * 0.1 - 1e-6
    assert float(result.theta["a"][0]) < 5.0 - 4 * 0.1


def _affine(w, batch, b=0.0):
    return w * batch + b


def test_a_partial_is_converted_to_a_pytree_and_trains_through_its_arguments():
    x = jnp.linspace(-1.0, 1.0, 8)
    target = 3.0 * x - 1.0
    result = fit(
        functools.partial(_affine, jnp.asarray(0.0), b=jnp.asarray(0.0)),
        lambda out, batch: jnp.mean((out - target) ** 2),
        data=x,
        optimizer=optax.adam(0.2),
        steps=150,
    )
    assert result.model.func is _affine
    assert float(result.model.args[0]) == pytest.approx(3.0, abs=0.05)
    assert float(result.model.keywords["b"]) == pytest.approx(-1.0, abs=0.05)


def test_a_key_is_passed_only_to_a_model_that_takes_one():
    seen = []

    def noisy(w, batch, key):
        seen.append(key is not None)
        return w * batch + 0.0 * jax.random.normal(key, ())

    kwargs = {"data": jnp.ones(3), "steps": 2, "key": jax.random.PRNGKey(0)}
    fit(
        functools.partial(noisy, jnp.asarray(1.0)), lambda o, b: jnp.sum(o**2), **kwargs
    )
    assert seen and all(seen)

    with pytest.raises(ValueError, match="takes no `key`"):
        fit(
            functools.partial(_affine, jnp.asarray(1.0)),
            lambda o, b: jnp.sum(o**2),
            **kwargs,
        )


def test_a_selector_must_return_leaves_of_the_model():
    with pytest.raises(ValueError, match="must return a leaf"):
        fit(
            _model(1.0),
            _quadratic,
            data=0.0,
            params={"a": lambda m: jnp.asarray(1.0)},
            steps=1,
        )


def test_a_groups_theta_is_what_its_selector_returns():
    model = _Tree(a=jnp.array([1.0]), b=jnp.array([2.0]), n=jnp.array([3]))
    result = fit(
        model,
        lambda out, batch: jnp.sum(out["a"] ** 2 + out["b"] ** 2),
        params={"pair": lambda m: (m["a"], m["b"], m["n"])},
        steps=1,
    )
    a, b, n = result.theta["pair"]
    assert a.shape == b.shape == (1,)
    assert int(n[0]) == 3, "a leaf that does not train is shown as it is"
    assert result.history["params"]["pair"][0].shape == (2, 1)


def test_transform_and_prior_agree_with_a_hand_computed_objective():
    theta0, reference, weight, scale, lr = 3.0, 1.5, 2.0, 0.5, 0.05
    target = jnp.asarray(1.0)

    def hand(theta):
        return (theta - target) ** 2 + scale * weight * (theta - reference) ** 2

    model = _Tree(p=jnp.asarray(theta0))
    kwargs = {
        "data": target,
        "transform": {"p": "log"},
        "prior": functools.partial(
            losses.param_prior,
            reference={"p": reference},
            weights={"p": weight},
            scale=scale,
        ),
        "optimizer": optax.sgd(lr),
    }

    result = fit(
        _Tree(p=jnp.asarray(theta0)),
        lambda out, batch: (out["p"] - batch) ** 2,
        steps=0,
        **kwargs,
    )
    assert float(result.history["loss"][0]) == pytest.approx(
        float(hand(theta0)), rel=1e-5
    )

    result = fit(model, lambda out, batch: (out["p"] - batch) ** 2, steps=1, **kwargs)
    g = float(jax.grad(hand)(theta0))
    expected = float(np.exp(np.log(theta0) - lr * g * theta0))
    assert float(result.theta["p"]) == pytest.approx(expected, rel=1e-5)
    assert float(result.history["params"]["p"][0]) == pytest.approx(theta0, rel=1e-6)


def test_a_callable_prior_sees_the_constrained_theta():
    result = fit(
        _model(2.0),
        _quadratic,
        data=0.0,
        transform={"x": "log"},
        prior=lambda theta: 0.5 * jnp.sum((theta["x"] - 5.0) ** 2),
        steps=0,
    )
    assert float(result.history["loss"][0]) == pytest.approx(4.0 + 4.5)


def test_a_transform_naming_nothing_fitted_is_an_error():
    with pytest.raises(KeyError, match="not fitted"):
        fit(
            _model(1.0),
            _quadratic,
            data=0.0,
            transform={"y": "log"},
            steps=1,
        )


def test_a_transform_applies_to_a_whole_group_by_label():
    model = _Tree(a=jnp.array([2.0]), b=jnp.array([2.0]))
    result = fit(
        model,
        lambda out, batch: sum(jnp.sum((v + 5.0) ** 2) for v in out.values()),
        params={"a": lambda m: m["a"], "b": lambda m: m["b"]},
        transform={"a": "log"},
        optimizer=optax.sgd(0.1),
        steps=30,
    )
    assert float(result.theta["a"][0]) > 0.0
    assert float(result.theta["b"][0]) < 0.0


def test_an_optimistix_minimiser_drives_the_loop_and_converges():
    optx = pytest.importorskip("optimistix")
    seen = []

    class Watch(Callback):
        def on_step_end(self, step, state, metrics):
            seen.append(metrics["loss"])

    result = fit(
        _model(3.0, -2.0),
        _quadratic,
        data=jnp.array([1.0, 1.0]),
        optimizer=optx.BFGS(rtol=1e-5, atol=1e-5),
        steps=50,
        callbacks=[Watch()],
    )
    np.testing.assert_allclose(_x(result), [1.0, 1.0], atol=1e-3)
    assert 0 < len(seen) < 50, "BFGS reports convergence, which ends the loop"
    assert np.all(np.diff(result.history["loss"]) <= 1e-6)


def test_an_optimistix_minimiser_refuses_changing_batches():
    optx = pytest.importorskip("optimistix")
    with pytest.raises(ValueError, match="full-batch"):
        fit(
            _model(1.0),
            _quadratic,
            data=lambda step: jnp.asarray([float(step)]),
            optimizer=optx.BFGS(rtol=1e-5, atol=1e-5),
            steps=3,
        )


def test_an_optimistix_minimiser_fits_a_simulation_parameter():
    optx = pytest.importorskip("optimistix")
    pytest.importorskip("diffrax")
    import test_fit as simulation

    env, stimulus = simulation._env(1), simulation._stimulus(1)
    target = simulation._simulate(env, stimulus, {"E_L": -73.5})
    result = simulation._fit(
        env,
        stimulus,
        target,
        {"E_L": -70.0},
        steps=30,
        optimizer=optx.BFGS(rtol=1e-4, atol=1e-4),
    )
    assert float(np.squeeze(result.theta["E_L"])) == pytest.approx(-73.5, abs=0.2)


@pytest.mark.parametrize("solver", ["adam", "bfgs"])
def test_a_batched_fit_equals_separate_fits(solver):
    if solver == "bfgs":
        optx = pytest.importorskip("optimistix")
        optimizer = optx.BFGS(rtol=1e-6, atol=1e-6)
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.2))

    init = np.array([[3.0, -2.0], [0.0, 0.0], [1.0, 4.0]], dtype=np.float32)
    targets = np.array([[1.0, 1.0], [2.0, -2.0], [-1.0, 0.0]], dtype=np.float32)

    together = fit(
        _Tree(x=jnp.asarray(init)),
        _quadratic,
        data=jnp.asarray(targets),
        optimizer=optimizer,
        steps=25,
        batched=True,
    )
    assert together.history["loss"].shape[0] == 3
    assert together.history["params"]["x"].shape[:1] == (3,)

    for i in range(3):
        alone = fit(
            _model(*init[i]),
            _quadratic,
            data=jnp.asarray(targets[i]),
            optimizer=optimizer,
            steps=25,
        )
        np.testing.assert_allclose(
            np.asarray(together.theta["x"])[i], _x(alone), atol=1e-5
        )
        n = alone.history["loss"].shape[0]
        np.testing.assert_allclose(
            together.history["loss"][i][:n], alone.history["loss"], atol=1e-4
        )


def test_a_batched_fit_gives_every_problem_its_own_simulation_target():
    pytest.importorskip("diffrax")
    import equinox as eqx
    import test_fit as simulation

    env, stimulus = simulation._env(1), simulation._stimulus(1)
    truth = np.array([-74.0, -71.0, -68.0])
    targets = np.stack([simulation._simulate(env, stimulus, {"E_L": t}) for t in truth])

    class Simulate(eqx.Module):
        """The batch is the target, which the simulation does not read."""

        env: object

        def __call__(self, batch):
            duration, dt = round(simulation.DURATION), simulation.DT
            return self.env(duration, stimulus, dt=dt)

    module = eqx.tree_at(
        lambda mod: mod.E_L, env.module, np.full((3, 1), -71.5, np.float32)
    )
    model = Simulate(env._with_module(module))
    result = fit(
        model,
        lambda run, target: jnp.mean((run.voltage - target) ** 2),
        data=jnp.asarray(targets),
        params={"E_L": lambda m: m.env.module.E_L},
        optimizer=optax.adam(0.5),
        steps=50,
        batched=True,
    )
    np.testing.assert_allclose(np.asarray(result.theta["E_L"]).ravel(), truth, atol=0.3)


def test_early_stopping_stops_and_restores_the_best_point():
    stopper = EarlyStopping(patience=3, restore_best=True)
    result = fit(
        _model(3.0),
        _quadratic,
        data=jnp.array([1.0]),
        optimizer=optax.sgd(0.5),
        steps=200,
        callbacks=[stopper],
    )
    assert result.state.step < 20
    assert stopper.stopped_step is not None
    assert float(_x(result)[0]) == pytest.approx(1.0, abs=1e-5)


def test_nan_guard_stops_on_a_non_finite_loss_and_saves_the_last_good_state(tmp_path):
    checkpoint = EqxCheckpointer(tmp_path, every=1000)
    guard = NaNGuard()

    def data(step):
        return jnp.array([jnp.nan if step == 5 else 1.0])

    result = fit(
        _model(3.0),
        _quadratic,
        data=data,
        optimizer=optax.sgd(0.1),
        steps=50,
        callbacks=[guard],
        checkpoint=checkpoint,
    )
    assert guard.tripped_at == 5
    assert result.state.step < 50
    assert checkpoint.latest() == 4
    saved = checkpoint.restore(
        checkpoint.latest(),
        (_model(0.0), optax.sgd(0.1).init(_model(0.0)), np.asarray(0), None),
    )
    assert np.all(np.isfinite(np.asarray(saved[0]["x"])))


def test_best_tracks_the_minimum_on_the_host():
    best = Best()
    result = fit(
        _model(3.0),
        _quadratic,
        data=jnp.array([1.0]),
        optimizer=optax.adam(1.4),
        steps=30,
        callbacks=[best],
    )
    losses = result.history["loss"][:-1]
    assert best.loss == pytest.approx(float(losses.min()))
    assert best.step == int(losses.argmin())
    recorded = float(np.asarray(result.history["params"]["x"])[best.step][0])
    assert float(np.asarray(best.theta["x"])[0]) == pytest.approx(recorded)


def test_best_tracks_each_problem_of_a_batched_fit():
    best = Best()
    fit(
        _Tree(x=jnp.array([[3.0], [8.0]])),
        _quadratic,
        data=jnp.array([[1.0], [1.0]]),
        optimizer=optax.adam(1.4),
        steps=20,
        batched=True,
        callbacks=[best],
    )
    assert best.loss.shape == (2,)
    assert best.step.shape == (2,)
    assert np.asarray(best.theta["x"]).shape == (2, 1)


def test_every_and_the_logger_see_each_step(tmp_path):
    calls = []
    log = tmp_path / "run.jsonl"
    fit(
        _model(3.0),
        lambda out, batch: (_quadratic(out, batch), {"half": 0.5}),
        data=jnp.array([1.0]),
        optimizer=optax.sgd(0.1),
        steps=6,
        callbacks=[
            Every(2, lambda step, state, metrics: calls.append(step)),
            JsonlLogger(log),
        ],
    )
    assert calls == [0, 2, 4]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(6))
    assert all(r["half"] == 0.5 and np.isfinite(r["loss"]) for r in rows)


def test_aux_reaches_the_history():
    result = fit(
        _model(3.0),
        lambda out, batch: (_quadratic(out, batch), {"x0": out["x"][0]}),
        data=jnp.array([1.0]),
        optimizer=optax.sgd(0.1),
        steps=4,
    )
    assert result.history["aux"]["x0"].shape == (5,)
    assert result.history["aux"]["x0"][0] == pytest.approx(3.0)


def test_the_checkpointer_round_trips_and_prunes(tmp_path):
    checkpoint = EqxCheckpointer(tmp_path, every=2, keep=2, document={"run": "a"})
    template = (_model(0.0), optax.adam(0.1).init(_model(0.0)), np.asarray(0), None)
    state = (_model(1.5), optax.adam(0.1).init(_model(1.5)), np.asarray(7), None)

    for step in (2, 4, 6):
        checkpoint.save(step, state)

    assert checkpoint.steps() == [4, 6]
    assert checkpoint.latest() == 6
    assert not list(tmp_path.glob("*.tmp"))
    restored = checkpoint.restore(6, template)
    np.testing.assert_array_equal(np.asarray(restored[0]["x"]), [1.5])
    assert int(restored[2]) == 7
    assert checkpoint.manifest()["document"] == {"run": "a"}
    assert checkpoint.manifest()["latest"] == 6


def test_resuming_gives_the_theta_of_an_uninterrupted_run(tmp_path):
    def data(step):
        return jnp.array([1.0, -1.0]) * (1.0 + 0.1 * step)

    def run(steps, checkpoint=None):
        return fit(
            _model(3.0, 2.0),
            _quadratic,
            data=data,
            optimizer=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.3)),
            steps=steps,
            checkpoint=checkpoint,
            key=jax.random.PRNGKey(0),
        )

    whole = run(14)

    first = run(9, EqxCheckpointer(tmp_path, every=4))
    assert EqxCheckpointer(tmp_path).latest() == 9, "the terminal state is saved too"
    assert first.state.step == 9

    resumed = run(14, str(tmp_path))
    assert resumed.state.step == 14
    np.testing.assert_allclose(_x(resumed), _x(whole), atol=1e-6)
    assert resumed.history["loss"].shape[0] == 14 - 9 + 1


def test_resuming_addresses_a_pipeline_by_step(tmp_path):
    started = []

    class Pipeline:
        def iterate(self, start_step=0, process=None, processes=None):
            started.append((start_step, process, processes))
            step = start_step
            while True:
                yield jnp.array([1.0 + 0.1 * step])
                step += 1

    def run(steps):
        return fit(
            _model(3.0),
            _quadratic,
            data=Pipeline(),
            optimizer=optax.adam(0.3),
            steps=steps,
            checkpoint=EqxCheckpointer(tmp_path, every=3),
        )

    whole = fit(
        _model(3.0),
        _quadratic,
        data=Pipeline(),
        optimizer=optax.adam(0.3),
        steps=8,
    )
    run(5)
    resumed = run(8)
    assert started[-1] == (5, None, None)
    np.testing.assert_allclose(_x(resumed), _x(whole), atol=1e-6)


def test_an_iterable_of_batches_is_consumed_one_per_step():
    batches = iter([jnp.array([float(i)]) for i in range(10)])
    seen = []
    fit(
        _model(0.0),
        lambda out, batch: (_quadratic(out, batch), {"batch": batch[0]}),
        data=batches,
        optimizer=optax.sgd(0.0),
        steps=4,
        callbacks=[Every(1, lambda s, st, m: seen.append(float(m["aux"]["batch"])))],
    )
    assert seen == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.skipif(not _USES_JAX, reason="jax backend only")
def test_default_params_train_python_float_leaves_of_a_medium():
    from livn.io import MEA

    rng = np.random.default_rng(0)
    electrodes = np.array([[0, 0.0, 0.0, 5.0], [1, 120.0, 0.0, 5.0]])
    cells = np.column_stack(
        [np.arange(8), rng.uniform(-150, 250, (8, 2)), np.full(8, 15.0)]
    )
    currents = jnp.asarray(rng.normal(size=(8, 32)), dtype=jnp.float32)

    start = MEA(electrodes, output_radius=500.0)
    distances = start.distances(cells)

    import equinox as eqx

    class Potential(eqx.Module):
        device: object

        def __call__(self, batch=None):
            return self.device.potential_recording(distances, currents)

    sigma0 = start.medium.electrical.sigma_s_per_m
    assert isinstance(sigma0, float)

    where = lambda m: m.device.medium.electrical.sigma_s_per_m  # noqa: E731
    model = Potential(start)
    wanted = eqx.tree_at(where, model, 0.5)
    target = wanted()

    def loss(output, batch):
        return jnp.mean((output - batch) ** 2) / jnp.mean(batch**2)

    for params in (None, select(model, where), {"medium": where}):
        result = fit(
            model,
            loss,
            data=target,
            params=params,
            optimizer=optax.adam(0.02),
            steps=150,
        )
        sigma = where(result.model)
        assert sigma == pytest.approx(0.5, rel=0.05), f"params={type(params)}"
        assert result.history["loss"][-1] < result.history["loss"][0] / 50
    assert where(model) == sigma0, "the model passed in is untouched"
