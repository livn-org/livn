import json
from itertools import pairwise

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import jax.numpy as jnp

from livn.optimization.fit import select
from livn.optimization.losses import param_prior
from livn.optimization.train import (
    Best,
    EarlyStopping,
    EqxCheckpointer,
    Every,
    FitState,
    History,
    JsonlLogger,
    NaNGuard,
    group_chain,
    initialize,
)


def _state(step=0, value=1.0):
    return FitState({"x": jnp.asarray([value])}, {"count": step}, step)


def _metrics(step, loss, value=None, **extra):
    raw = {"x": jnp.asarray([step if value is None else value], dtype=float)}
    return {
        "step": step,
        "loss": loss,
        "aux": {},
        "raw": raw,
        "batched": False,
        **extra,
    }


def _drive(callback, losses, **kwargs):
    callback.on_fit_start(_state())
    stopped = None
    for step, loss in enumerate(losses):
        if callback.on_step_end(step, _state(step + 1), _metrics(step, loss, **kwargs)):
            stopped = step
            break
    return stopped


def test_group_chain_holds_the_rate_without_a_schedule():
    tx = group_chain(0.1)
    params = {"x": jnp.asarray([1.0])}
    state = tx.init(params)
    updates, _ = tx.update({"x": jnp.asarray([5.0])}, state, params)
    # the first adam step is lr * sign(grad)
    assert float(updates["x"][0]) == pytest.approx(-0.1, rel=1e-4)


def test_group_chain_warms_up_from_zero():
    tx = group_chain(0.1, warmup=4)
    params = {"x": jnp.asarray([1.0])}
    state = tx.init(params)
    first, state = tx.update({"x": jnp.asarray([1.0])}, state, params)
    for _ in range(3):
        last, state = tx.update({"x": jnp.asarray([1.0])}, state, params)
    assert float(first["x"][0]) == pytest.approx(0.0, abs=1e-7)
    assert abs(float(last["x"][0])) == pytest.approx(0.075, rel=1e-3)


def test_group_chain_decays_to_zero():
    tx = group_chain(0.1, decay=5)
    params = {"x": jnp.asarray([1.0])}
    state = tx.init(params)
    sizes = []
    for _ in range(6):
        update, state = tx.update({"x": jnp.asarray([1.0])}, state, params)
        sizes.append(abs(float(update["x"][0])))
    assert sizes[0] == pytest.approx(0.1, rel=1e-3)
    assert sizes[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(a >= b - 1e-9 for a, b in pairwise(sizes))


def test_group_chain_drops_nans_and_clips_the_update():
    tx = group_chain(1.0, trust_region=0.05, clip_norm=1.0)
    params = {"x": jnp.asarray([1.0, 1.0])}
    state = tx.init(params)
    updates, _ = tx.update({"x": jnp.asarray([jnp.nan, 3.0])}, state, params)
    u = np.asarray(updates["x"])
    assert np.all(np.isfinite(u))
    assert np.all(np.abs(u) <= 0.05 + 1e-9)


def test_group_chain_adds_weight_decay():
    plain = group_chain(0.1)
    decayed = group_chain(0.1, weight_decay=0.5)
    params = {"x": jnp.asarray([2.0])}
    grads = {"x": jnp.asarray([1.0])}
    a, _ = plain.update(grads, plain.init(params), params)
    b, _ = decayed.update(grads, decayed.init(params), params)
    assert float(b["x"][0]) < float(a["x"][0])


def test_best_keeps_the_lowest_loss_and_its_point():
    best = Best()
    _drive(best, [5.0, 3.0, 4.0, 1.0, 2.0])
    assert best.loss == 1.0
    assert best.step == 3
    assert float(best.raw["x"][0]) == 3.0


def test_best_ignores_nans():
    best = Best()
    _drive(best, [2.0, float("nan"), 1.0])
    assert best.loss == 1.0 and best.step == 2


def test_best_tracks_each_problem_of_a_batch_separately():
    best = Best()
    best.on_fit_start(_state())
    for step, loss in enumerate([[3.0, 1.0], [2.0, 4.0], [5.0, 0.5]]):
        raw = {"x": jnp.asarray([[step * 10.0], [step * 10.0 + 1]])}
        metrics = {**_metrics(step, np.asarray(loss)), "raw": raw, "batched": True}
        best.on_step_end(step, _state(step + 1), metrics)
    np.testing.assert_array_equal(best.loss, [2.0, 0.5])
    np.testing.assert_array_equal(best.step, [1, 2])
    np.testing.assert_array_equal(np.asarray(best.raw["x"]), [[10.0], [21.0]])


def test_early_stopping_waits_out_its_patience():
    stopper = EarlyStopping(patience=3)
    assert _drive(stopper, [5.0, 4.0, 4.5, 4.4, 4.6, 4.7, 9.0]) == 4
    assert stopper.best == 4.0 and stopper.best_step == 1


def test_early_stopping_respects_min_delta():
    stopper = EarlyStopping(patience=2, min_delta=0.5)
    assert _drive(stopper, [5.0, 4.9, 4.8, 4.7, 4.6]) == 2


def test_early_stopping_respects_a_relative_threshold():
    # each step improves by 1%: enough for min_rel_delta=0.005, not for 0.05
    losses = [100.0 * 0.99**i for i in range(8)]
    assert _drive(EarlyStopping(patience=2, min_rel_delta=0.005), losses) is None
    assert _drive(EarlyStopping(patience=2, min_rel_delta=0.05), losses) == 2


def test_early_stopping_stops_once_the_target_is_reached():
    stopper = EarlyStopping(patience=100, stop_below=2.0)
    assert _drive(stopper, [5.0, 3.0, 1.5, 1.0]) == 2
    assert stopper.best == 1.5
    restored = stopper.on_fit_end(_state(3))
    assert float(restored.raw["x"][0]) == 2.0


def test_early_stopping_counts_a_run_that_never_improves():
    assert _drive(EarlyStopping(patience=3), [float("nan")] * 6) == 2


def test_early_stopping_reads_a_dotted_path_into_the_aux():
    stopper = EarlyStopping(monitor="aux.mse", patience=1, restore_best=False)
    stopper.on_fit_start(_state())
    stopped = None
    for step, mse in enumerate([4.0, 3.0, 3.5, 3.6]):
        metrics = {**_metrics(step, 0.0), "aux": {"mse": mse}}
        if stopper.on_step_end(step, _state(step + 1), metrics):
            stopped = step
            break
    assert stopped == 2 and stopper.best == 3.0


def test_best_reports_one_problem_of_a_batch():
    best = Best()
    best.on_fit_start(_state())
    for step, loss in enumerate([[3.0, 1.0], [2.0, 4.0]]):
        raw = {"x": jnp.asarray([[step * 10.0], [step * 10.0 + 1]])}
        metrics = {**_metrics(step, np.asarray(loss)), "raw": raw, "batched": True}
        best.on_step_end(step, _state(step + 1), metrics)
    one = best.problem(1)
    assert one["loss"] == 1.0 and one["step"] == 0
    assert float(one["theta"]["x"][0]) == 1.0


def test_early_stopping_restores_the_best_point_at_the_end():
    stopper = EarlyStopping(patience=2, restore_best=True)
    _drive(stopper, [5.0, 1.0, 2.0, 3.0])
    restored = stopper.on_fit_end(_state(4))
    assert float(restored.raw["x"][0]) == 1.0
    assert EarlyStopping(restore_best=False).on_fit_end(_state()) is None


def test_early_stopping_can_watch_an_aux_value():
    stopper = EarlyStopping(monitor="acc", patience=1)
    stopper.on_fit_start(_state())
    rows = [0.3, 0.4, 0.5]
    stopped = None
    for step, acc in enumerate(rows):
        metrics = {**_metrics(step, 0.0), "aux": {"acc": acc}}
        if stopper.on_step_end(step, _state(step + 1), metrics):
            stopped = step
            break
    assert stopped == 1


class _Saves:
    every = 10

    def __init__(self):
        self.saved = []

    def save(self, step, state):
        self.saved.append((step, state))


def test_nan_guard_stops_and_saves_the_state_from_two_steps_before():
    saves = _Saves()
    guard = NaNGuard(saves)
    guard.on_fit_start(_state())
    results = []
    for step, loss in enumerate([1.0, 1.0, 1.0, float("nan"), 1.0]):
        results.append(guard.on_step_end(step, _state(step + 1), _metrics(step, loss)))
        if results[-1]:
            break
    assert results == [None, None, None, True]
    assert guard.tripped_at == 3
    assert [s for s, _ in saves.saved] == [2]


def test_nan_guard_also_trips_on_the_gradient():
    guard = NaNGuard(_Saves())
    guard.on_fit_start(_state())
    assert guard.on_step_end(0, _state(1), _metrics(0, 1.0, grad_norm=float("inf")))


def _stepped(guard, rows):
    """Drive ``guard`` with ``(loss, grad_norm, nan_grads)`` rows; the step it stopped on."""
    guard.on_fit_start(None)
    for i, (loss, norm, nans) in enumerate(rows):
        metrics = _metrics(i, loss, grad_norm=norm, nan_grads=nans)
        if guard.on_step_end(i, _state(i + 1), metrics):
            return i
    return None


def test_the_default_guard_goes_on_through_a_nan_gradient_and_counts_it():
    guard = NaNGuard(_Saves())
    assert _stepped(guard, [(1.0, 1.0, 0), (1.0, 1.0, 3), (0.9, 1.0, 0)]) is None
    assert guard.nan_grad_steps == 1 and guard.tripped_at is None


def test_the_default_guard_stops_on_an_infinite_gradient_or_a_bad_loss():
    guard = NaNGuard(_Saves())
    assert _stepped(guard, [(1.0, 1.0, 0), (1.0, np.inf, 0), (1.0, 1.0, 0)]) == 1
    assert "gradient norm" in guard.reason
    guard = NaNGuard(_Saves())
    assert _stepped(guard, [(1.0, 1.0, 0), (np.nan, 1.0, 0)]) == 1
    assert "loss" in guard.reason


def test_the_guard_can_watch_only_what_it_is_told_to():
    rows = [(1.0, np.inf, 0), (np.inf, 1.0, 0)]
    assert _stepped(NaNGuard(on=("loss",)), rows) == 1
    assert _stepped(NaNGuard(on=("grad",)), rows) == 0
    assert _stepped(NaNGuard(on="grad"), [(np.nan, 1.0, 0)]) is None
    nan_grad = NaNGuard(on=("nan_grad",))
    assert _stepped(nan_grad, [(1.0, 1.0, 0), (1.0, 1.0, 2)]) == 1
    assert "NaN gradient" in nan_grad.reason
    with pytest.raises(ValueError, match="watches"):
        NaNGuard(on=("weights",))


def test_nan_guard_uses_the_fits_own_checkpointer_by_default():
    saves = _Saves()
    guard = NaNGuard()
    guard.on_fit_start(_state())
    for step, loss in enumerate([1.0, 1.0, float("nan")]):
        state = _state(step + 1).replace(checkpointer=saves)
        guard.on_step_end(step, state, _metrics(step, loss))
    assert len(saves.saved) == 1


def test_every_calls_on_multiples_and_can_stop():
    seen = []
    every = Every(3, lambda step, state, metrics: seen.append(step) or step >= 6)
    results = [every.on_step_end(s, _state(), _metrics(s, 0.0)) for s in range(8)]
    assert seen == [0, 3, 6]
    assert results[6] is True and results[1] is None


def test_history_stacks_steps_and_batches():
    history = History()
    history.on_fit_start(_state())
    for step in range(3):
        history.on_step_end(
            step,
            _state(step + 1),
            _metrics(step, float(step), grad_norm=0.5, aux={"a": step * 2.0}),
        )
    history.record({"x": jnp.asarray([9.0])}, 7.0, {"a": 8.0})
    result = history.result()
    np.testing.assert_array_equal(result["loss"], [0, 1, 2, 7])
    np.testing.assert_array_equal(result["aux"]["a"], [0, 2, 4, 8])
    assert result["params"]["x"].shape == (4, 1)

    batched = History()
    batched.batched = True
    batched.on_fit_start(_state())
    for _ in range(3):
        batched.record({"x": jnp.zeros((2, 1))}, np.array([1.0, 2.0]), {})
    assert batched.result()["loss"].shape == (2, 3)
    assert batched.result()["params"]["x"].shape == (2, 3, 1)


def test_history_counts_the_steps_each_problem_took():
    history = History()
    history.batched = True
    history.on_fit_start(_state())
    for step, stepped in enumerate([[True, True], [True, False], [True, False]]):
        history.on_step_end(
            step,
            _state(step + 1),
            {
                **_metrics(step, np.array([1.0, 2.0])),
                "raw": {"x": jnp.zeros((2, 1))},
                "batched": True,
                "stepped": np.array(stepped),
                "grad_norm": np.array([1.0, 1.0]),
                "nan_grads": np.array([0, 2]),
            },
        )
    history.termination = ["a", "b"]
    result = history.result()
    np.testing.assert_array_equal(result["steps"], [3, 1])
    np.testing.assert_array_equal(result["nan_grads"], [[0, 0, 0], [2, 2, 2]])
    assert result["result"] == ["a", "b"]

    alone = History()
    alone.on_fit_start(_state())
    for step in range(4):
        alone.on_step_end(step, _state(step + 1), _metrics(step, 1.0))
    assert alone.result()["steps"] == 4 and "result" not in alone.result()


def test_history_leaves_out_parameters_that_are_too_large():
    history = History(max_size=4)
    history.on_fit_start(_state())
    history.record({"big": jnp.zeros(10), "small": jnp.zeros(2)}, 0.0, {})
    result = history.result()
    assert "small" in result["params"]
    assert result["params"]["big"] is None


def test_the_logger_writes_one_json_line_per_step(tmp_path):
    path = tmp_path / "sub" / "log.jsonl"
    logger = JsonlLogger(path)
    logger.on_fit_start(_state())
    for step in range(3):
        logger.on_step_end(step, _state(step + 1), _metrics(step, 0.5 * step))
    logger.on_fit_end(_state(3))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[2]["loss"] == 1.0


def test_the_checkpointer_prunes_and_leaves_no_partial_files(tmp_path):
    checkpoint = EqxCheckpointer(tmp_path, every=5, keep=2, document={"seed": 3})
    payload = (
        {"x": jnp.asarray([1.0, 2.0])},
        {"count": np.asarray(4)},
        np.asarray(5),
        None,
    )
    for step in (5, 10, 15):
        checkpoint.save(step, payload)

    assert checkpoint.steps() == [10, 15]
    assert checkpoint.latest() == 15
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "manifest.json",
        "step_00000010.eqx",
        "step_00000015.eqx",
    ]
    assert checkpoint.manifest() == {
        "latest": 15,
        "steps": [10, 15],
        "document": {"seed": 3},
    }

    template = ({"x": jnp.zeros(2)}, {"count": np.asarray(0)}, np.asarray(0), None)
    restored = checkpoint.restore(15, template)
    np.testing.assert_array_equal(np.asarray(restored[0]["x"]), [1.0, 2.0])
    assert int(restored[1]["count"]) == 4 and int(restored[2]) == 5


def test_an_empty_checkpoint_directory_has_no_latest(tmp_path):
    assert EqxCheckpointer(tmp_path / "nothing").latest() is None


def test_a_fit_state_saves_what_it_holds(tmp_path):
    checkpoint = EqxCheckpointer(tmp_path)
    state = FitState({"x": jnp.asarray([1.0])}, {"n": np.asarray(2)}, 7, None)
    checkpoint.save(7, state)
    restored = checkpoint.restore(
        7, ({"x": jnp.zeros(1)}, {"n": np.asarray(0)}, np.asarray(0), None)
    )
    assert int(restored[2]) == 7


def test_initialize_is_a_no_op_for_a_single_process(monkeypatch):
    import jax

    calls = []
    monkeypatch.setattr(jax.distributed, "initialize", lambda **kw: calls.append(kw))
    monkeypatch.delenv("SLURM_NTASKS", raising=False)

    class Solo:
        def Get_size(self):
            return 1

    initialize()
    initialize(Solo())
    assert calls == []


def test_initialize_hands_an_mpi_job_to_jaxs_mpi4py_detection(monkeypatch):
    import jax

    calls = []
    monkeypatch.setattr(jax.distributed, "initialize", lambda **kw: calls.append(kw))

    class Four:
        def Get_size(self):
            return 4

    initialize(Four())
    assert calls == [{"cluster_detection_method": "mpi4py"}]


def test_initialize_under_slurm_lets_jax_detect_the_cluster(monkeypatch):
    import jax

    calls = []
    monkeypatch.setattr(jax.distributed, "initialize", lambda **kw: calls.append(kw))
    monkeypatch.setenv("SLURM_NTASKS", "8")
    initialize()
    assert calls == [{}]


def test_prior_is_a_weighted_squared_pull_on_the_named_entries():
    theta = {"a": jnp.asarray([3.0, 1.0]), "b": jnp.asarray([100.0])}
    value = param_prior(theta, {"a": 1.0}, weights={"a": 3.0}, scale=0.5)
    assert float(value) == pytest.approx(0.5 * 3.0 * 4.0)
    with pytest.raises(KeyError, match="not being fitted"):
        param_prior(theta, {"zzz": 1.0})


def test_prior_over_an_array_pytree_takes_scalar_or_tree_weights():
    theta = (jnp.asarray([2.0]), jnp.asarray([5.0]))
    ref = (jnp.asarray([1.0]), jnp.asarray([1.0]))
    assert float(param_prior(theta, ref)) == pytest.approx(1.0 + 16.0)
    assert float(param_prior(theta, ref, weights=(2.0, 0.5))) == pytest.approx(
        2.0 + 8.0
    )


def test_select_marks_only_the_chosen_float_leaves():
    model = {"a": {"s": 0.3, "n": 4}, "b": jnp.asarray([1.0])}
    mask = select(model, lambda m: m["a"])
    assert mask == {"a": {"s": True, "n": False}, "b": False}
    assert select(model, lambda m: m["a"]["s"])["a"]["s"] is True
