import functools

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")
pytest.importorskip("optimistix")

import jax.numpy as jnp
import optax
import optimistix as optx
from test_fit_loop import _model, _quadratic, _Tree

from livn.optimization import losses
from livn.optimization.fit import _prior_call, fit, stack
from livn.optimization.train import Best, History, NaNGuard


def _pair():
    return _Tree(
        theta={"a": jnp.array([2.0]), "b": jnp.array([3.0])}, c=jnp.array([9.0])
    )


def _pair_loss(out, batch):
    return jnp.sum((out["theta"]["a"] - 1.0) ** 2 + (out["theta"]["b"] + 1.0) ** 2)


def test_a_selector_returning_a_dict_makes_one_group_per_key():
    result = fit(
        _pair(),
        _pair_loss,
        params={"theta": lambda m: m["theta"]},
        optimizer={"a": optax.sgd(0.5), "b": optax.sgd(0.25)},
        steps=1,
    )
    assert set(result.theta) == {"a", "b"}
    np.testing.assert_allclose(np.asarray(result.theta["a"]), [1.0])
    np.testing.assert_allclose(np.asarray(result.theta["b"]), [3.0 - 0.25 * 8.0])
    assert float(result.model["c"][0]) == 9.0
    assert set(result.history["params"]) == {"a", "b"}


def test_a_selector_on_its_own_returning_a_dict_is_a_dict_of_groups():
    result = fit(
        _pair(),
        _pair_loss,
        params=lambda m: m["theta"],
        transform={"a": "log"},
        prior=lambda theta: 0.0 * jnp.sum(theta["a"]) + 0.0 * jnp.sum(theta["b"]),
        optimizer=optax.sgd(0.1),
        steps=3,
    )
    assert set(result.theta) == {"a", "b"}
    assert float(result.theta["a"][0]) > 0


def test_a_selector_returning_a_leaf_trains_just_that_leaf():
    result = fit(
        _pair(), _pair_loss, params=lambda m: m["c"], optimizer=optax.sgd(0.1), steps=2
    )
    assert float(result.model["c"][0]) == 9.0  # no gradient reaches it
    assert float(result.model["theta"]["a"][0]) == 2.0


def test_a_key_that_names_another_group_is_refused():
    with pytest.raises(ValueError, match="defined twice"):
        fit(
            _pair(),
            _pair_loss,
            params={"a": lambda m: m["c"], "theta": lambda m: m["theta"]},
            steps=1,
        )


def test_the_prior_signature_decides_whether_it_sees_the_batch():
    def one(theta): ...
    def second(theta, data): ...
    def named(theta, *, batch): ...
    def optional(theta, batch=None): ...
    def defaulted(theta, scale=1.0): ...

    assert _prior_call(one) is None
    assert _prior_call(second) == "positional"
    assert _prior_call(named) == "keyword"
    assert _prior_call(optional) == "keyword"
    assert _prior_call(defaulted) is None
    assert _prior_call(None) is None
    assert _prior_call(functools.partial(optional, batch=1)) == "keyword"
    # a partial that binds the reference leaves `weights` as the second parameter, not the batch
    ref = {"x": jnp.zeros(1)}
    assert _prior_call(functools.partial(losses.param_prior, reference=ref)) is None
    assert _prior_call(functools.partial(losses.param_prior, ref)) is None


@pytest.mark.parametrize("form", ["positional", "keyword"])
def test_every_problem_of_a_batch_is_pulled_to_its_own_reference(form):
    def positional(theta, batch):
        return jnp.sum((theta["x"] - batch["ref"]) ** 2)

    def keyword(theta, *, batch):
        return positional(theta, batch)

    prior = positional if form == "positional" else keyword
    data = {"t": jnp.array([[1.0], [1.0]]), "ref": jnp.array([[0.0], [4.0]])}

    def loss(out, batch):
        return jnp.sum((out["x"] - batch["t"]) ** 2)

    together = fit(
        _Tree(x=jnp.array([[3.0], [3.0]])),
        loss,
        data=data,
        prior=prior,
        optimizer=optax.sgd(0.25),
        steps=40,
        batched=True,
    )
    # the minimum of (x-1)^2 + (x-ref)^2 is their mean
    np.testing.assert_allclose(
        np.asarray(together.theta["x"]).ravel(), [0.5, 2.5], atol=1e-4
    )


def test_a_prior_without_the_batch_is_called_as_before():
    seen = []

    def prior(theta):
        seen.append(theta)
        return 0.0 * jnp.sum(theta["x"])

    fit(
        _model(1.0),
        _quadratic,
        data=jnp.array([0.0]),
        prior=prior,
        params={"x": lambda m: m["x"]},
        steps=1,
    )
    assert seen


def _segments(*values):
    return [
        _Tree(x=jnp.array([v], dtype=jnp.float32), fixed=jnp.array([7.0]))
        for v in values
    ]


def test_stack_gives_the_trained_leaves_a_leading_axis_and_shares_the_rest():
    stacked = stack(_segments(1.0, 2.0, 3.0), params=lambda m: m["x"])
    assert stacked["x"].shape == (3, 1)
    np.testing.assert_array_equal(np.asarray(stacked["x"]).ravel(), [1.0, 2.0, 3.0])
    assert stacked["fixed"].shape == (1,)


def test_stack_refuses_an_untrained_leaf_that_differs():
    models = _segments(1.0, 2.0)
    models[1] = _Tree(x=models[1]["x"], fixed=jnp.array([8.0]))
    with pytest.raises(ValueError, match="not trained"):
        stack(models, params=lambda m: m["x"])

    assert stack(models, params={"x": lambda m: m["x"], "f": lambda m: m["fixed"]})[
        "fixed"
    ].shape == (2, 1)


def test_stack_refuses_models_of_another_structure():
    with pytest.raises(ValueError, match="structured"):
        stack([_Tree(x=jnp.ones(1)), _Tree(y=jnp.ones(1))])


def test_stack_accepts_a_dict_selector_and_the_default():
    models = [
        _Tree(theta={"a": jnp.array([float(i)]), "b": jnp.array([2.0 * i])})
        for i in range(2)
    ]
    stacked = stack(models, params=lambda m: m["theta"])
    assert stacked["theta"]["a"].shape == (2, 1)
    assert stack(models)["theta"]["b"].shape == (2, 1)


def test_problem_slices_a_batched_fit_by_problem():
    best = Best()
    models = _segments(3.0, 5.0, 1.0)
    result = fit(
        stack(models, params=lambda m: m["x"]),
        _quadratic,
        data=jnp.array([[1.0], [1.0], [1.0]]),
        params=lambda m: m["x"],
        optimizer=optx.BFGS(rtol=1e-6, atol=1e-6),
        steps=30,
        batched=True,
        callbacks=[best],
    )
    assert result.best is best
    one = result.problem(2)  # started on the minimum
    assert one.history["loss"].shape == (result.history["loss"].shape[1],)
    assert float(np.asarray(one.theta["x"]).ravel()[0]) == pytest.approx(1.0, abs=1e-4)
    assert one.history["steps"] < result.history["steps"][0]
    assert one.history["params"]["x"].shape[0] == one.history["loss"].shape[0]
    assert one.best["loss"] == pytest.approx(float(best.loss[2]))
    assert float(one.model["fixed"][0]) == 7.0
    np.testing.assert_allclose(np.asarray(one.model["x"]), np.asarray(one.theta["x"]))


def test_problem_is_for_batched_fits():
    result = fit(_model(1.0), _quadratic, data=jnp.array([0.0]), steps=1)
    with pytest.raises(ValueError, match="batched"):
        result.problem(0)


def _kinked(out, batch):
    return jnp.sum(jnp.sqrt(out["x"] ** 2))  # the gradient at 0 is 0/0


@pytest.mark.parametrize("batched", [False, True])
def test_the_gradient_norm_leaves_nans_out_and_counts_them(batched):
    x = jnp.array([0.0, 2.0])
    result = fit(
        _Tree(x=jnp.stack([x, x]) if batched else x),
        _kinked,
        data=jnp.zeros((2, 2)) if batched else jnp.zeros(2),
        optimizer=optax.chain(optax.zero_nans(), optax.sgd(0.1)),
        steps=2,
        batched=batched,
    )
    norm, nans = result.history["grad_norm"], result.history["nan_grads"]
    np.testing.assert_allclose(norm, 1.0, rtol=1e-5)
    assert np.all(nans == 1)
    assert norm.shape == nans.shape == ((2, 2) if batched else (2,))


def test_an_infinite_gradient_still_makes_the_norm_infinite():
    result = fit(
        _Tree(x=jnp.array([0.0, 1.0])),
        lambda out, batch: jnp.sum(1.0 / out["x"]),
        optimizer=optax.chain(optax.zero_nans(), optax.sgd(0.0)),
        steps=1,
    )
    assert np.isinf(result.history["grad_norm"][0])
    assert result.history["nan_grads"][0] == 0


def test_the_optimistix_gradient_norm_is_reported_too():
    result = fit(
        _model(3.0, -2.0),
        _quadratic,
        data=jnp.array([1.0, 1.0]),
        optimizer=optx.BFGS(rtol=1e-5, atol=1e-5),
        steps=20,
    )
    assert result.history["grad_norm"].shape == result.history["nan_grads"].shape
    assert np.all(result.history["nan_grads"] == 0)
    assert np.all(np.isfinite(result.history["grad_norm"]))


def test_a_nan_gradient_does_not_stop_the_default_guard_but_is_counted():
    guard = NaNGuard()
    result = fit(
        _Tree(x=jnp.array([0.0, 2.0])),
        _kinked,
        data=jnp.zeros(2),
        optimizer=optax.chain(optax.zero_nans(), optax.sgd(0.1)),
        steps=5,
        callbacks=[guard],
    )
    assert guard.tripped_at is None and guard.nan_grad_steps == 5
    assert result.state.step == 5


def test_the_guard_can_be_told_to_stop_on_a_nan_gradient_or_to_ignore_the_gradient():
    stops = NaNGuard(on=("loss", "grad", "nan_grad"))
    fit(
        _Tree(x=jnp.array([0.0, 2.0])),
        _kinked,
        data=jnp.zeros(2),
        optimizer=optax.chain(optax.zero_nans(), optax.sgd(0.1)),
        steps=5,
        callbacks=[stops],
    )
    assert stops.tripped_at == 0 and "NaN" in stops.reason

    loss_only = NaNGuard(on=("loss",))
    fit(
        _Tree(x=jnp.array([0.0, 1.0])),
        lambda out, batch: jnp.sum(1.0 / out["x"]),
        optimizer=optax.sgd(0.0),
        steps=3,
        callbacks=[loss_only],
    )
    # the loss is inf from the first step
    assert loss_only.tripped_at == 0 and loss_only.reason.startswith("loss")


@pytest.mark.parametrize("solver", ["adam", "bfgs"])
def test_every_optimizer_reports_the_same_history_keys(solver):
    optimizer = optax.adam(0.3) if solver == "adam" else optx.BFGS(rtol=1e-5, atol=1e-5)
    result = fit(
        _model(3.0, -2.0),
        lambda out, batch: (_quadratic(out, batch), {"aux": jnp.sum(out["x"])}),
        data=jnp.array([1.0, 1.0]),
        optimizer=optimizer,
        steps=12,
    )
    history = result.history
    assert {"loss", "params", "aux", "grad_norm", "nan_grads", "steps"} <= set(history)
    assert history["loss"].shape[0] == history["steps"] + 1
    assert history["params"]["x"].shape == (history["steps"] + 1, 2)
    assert history["aux"]["aux"].shape == (history["steps"] + 1,)
    assert isinstance(history["steps"], int)
    assert ("result" in history) == (solver == "bfgs")


def test_the_termination_of_an_optimistix_fit_is_named():
    converged = fit(
        _model(3.0),
        _quadratic,
        data=jnp.array([1.0]),
        optimizer=optx.BFGS(rtol=1e-3, atol=1e-3),
        steps=50,
    )
    assert converged.history["steps"] < 50
    assert converged.history["result"] == str(optx.RESULTS.successful)

    cut = fit(
        _model(3.0),
        lambda out, batch: jnp.sum(50 * (out["x"] - batch) ** 2),
        data=jnp.array([1.0]),
        optimizer=optx.BFGS(rtol=1e-12, atol=1e-12),
        steps=2,
    )
    assert cut.history["steps"] == 2
    assert cut.history["result"] == str(optx.RESULTS.nonlinear_max_steps_reached)


def test_a_batched_optimistix_fit_reports_steps_and_termination_per_problem():
    result = fit(
        _Tree(x=jnp.array([[3.0], [1.0]])),
        _quadratic,
        data=jnp.array([[1.0], [1.0]]),
        optimizer=optx.BFGS(rtol=1e-6, atol=1e-6),
        steps=30,
        batched=True,
    )
    steps = result.history["steps"]
    assert steps.shape == (2,) and steps[1] < steps[0]
    assert len(result.history["result"]) == 2
    assert result.history["result"][1] == str(optx.RESULTS.successful)
    assert result.history["loss"].shape[0] == 2
    assert result.history["loss"][1, -1] == result.history["loss"][1, steps[1]]


def test_a_callback_stop_is_not_reported_as_running_out_of_steps():
    class Stop(History):
        def on_step_end(self, step, state, metrics):
            super().on_step_end(step, state, metrics)
            return step == 0

    result = fit(
        _model(3.0),
        lambda out, batch: jnp.sum(50 * (out["x"] - batch) ** 2),
        data=jnp.array([1.0]),
        optimizer=optx.BFGS(rtol=1e-12, atol=1e-12),
        steps=10,
        callbacks=[Stop()],
    )
    assert result.history["result"] == "stopped by a callback"


def _steep(out, batch):
    return jnp.sum(50.0 * (out["x"] - batch) ** 2)


def test_best_sees_every_point_the_solver_evaluated():
    best = Best()
    result = fit(
        _model(3.0, -2.0),
        _steep,
        data=jnp.array([1.0, 1.0]),
        optimizer=optx.BFGS(rtol=1e-6, atol=1e-6),
        steps=25,
        callbacks=[best],
    )
    losses_seen = result.history["loss"][:-1]
    assert np.any(np.diff(losses_seen) > 0)
    assert best.loss == pytest.approx(float(losses_seen.min()))
    assert best.step == int(losses_seen.argmin())
    at = np.asarray(result.history["params"]["x"])[best.step]
    np.testing.assert_allclose(np.asarray(best.theta["x"]), at)
    check = float(_steep({"x": best.theta["x"]}, jnp.array([1.0, 1.0])))
    assert check == pytest.approx(best.loss, rel=1e-4, abs=1e-9)


def test_best_tracks_each_problem_of_a_batched_optimistix_fit():
    best = Best()
    result = fit(
        _Tree(x=jnp.array([[3.0, -2.0], [0.0, 0.5]])),
        _steep,
        data=jnp.ones((2, 2)),
        optimizer=optx.BFGS(rtol=1e-6, atol=1e-6),
        steps=25,
        batched=True,
        callbacks=[best],
    )
    seen = result.history["loss"][:, :-1]
    np.testing.assert_allclose(best.loss, seen.min(axis=1), rtol=1e-6)
    np.testing.assert_array_equal(best.step, seen.argmin(axis=1))
