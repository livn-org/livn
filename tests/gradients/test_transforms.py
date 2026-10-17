import functools

import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from livn.backend.diffrax import Env
from livn.models.eventloop import SolverConfig
from livn.models.glif import GLIF
from livn.optimization.fit import fit
from livn.optimization.losses import voltage_mse
from livn.optimization.transforms import (
    BIJECTORS,
    bijector_for,
    bounded,
    bounds_for,
    pack,
    unpack,
)
from livn.stimulus import Stimulus

DURATION, DT = 20.0, 0.5

BOUNDS = {
    "E_L": (-40.0, 40.0),
    "V_threshold_base": (1.0, 150.0),
    "tau_m": (0.5, 300.0),
    "g_L": (0.05, 500.0),
    "asc_r": (-2.0, 2.0),
}
DEFAULT = {
    "tau_m": "bounded",
    "g_L": "bounded",
    "E_L": "bounded",
    "V_threshold_base": "bounded",
    "asc_r": "bounded",
    "sigma": "log",
    "alpha": "log",
    "f_v": "logit",
    "a_v": "identity",
}


def _env(n=1):
    env = Env(
        n,
        model=GLIF.leaky_integrate_and_fire(
            mechanism="hard", config=SolverConfig(max_rate=1.0)
        ),
    )
    env.init()
    return env.record_voltage(dt=DT)


def _stimulus(n=1, amp=0.05):
    steps = round(DURATION / DT) + 1
    return Stimulus.from_current(np.full((steps, n), amp), dt=DT)


@pytest.mark.parametrize("name,kind", sorted(DEFAULT.items()))
def test_every_declared_parameter_round_trips(name, kind):
    if kind == "bounded":
        lo, hi = bounds_for(name, BOUNDS)
        x = lo + 0.37 * (hi - lo)
    else:
        x = {"log": 3.5, "logit": 0.42, "identity": -61.0}[kind]
    z = pack({name: x}, DEFAULT, BOUNDS)[name]
    assert float(unpack({name: z}, DEFAULT, BOUNDS)[name]) == pytest.approx(x, rel=1e-6)


@pytest.mark.parametrize("name", sorted(BOUNDS))
def test_bounded_parameters_cannot_leave_their_box(name):
    lo, hi = bounds_for(name, BOUNDS)
    for z in (-1e4, -50.0, 0.0, 50.0, 1e4):
        got = float(unpack({name: jnp.asarray(z)}, DEFAULT, BOUNDS)[name])
        assert lo <= got <= hi, f"{name} left ({lo}, {hi}) at z={z}: {got}"


def test_every_box_is_actually_used():
    unused = [n for n in BOUNDS if bijector_for(n, DEFAULT) != "bounded"]
    assert not unused, f"declared in BOUNDS but not bounded in DEFAULT: {unused}"
    missing = [
        n
        for n, k in DEFAULT.items()
        if k == "bounded" and bounds_for(n, BOUNDS) is None
    ]
    assert not missing, f"declared bounded in DEFAULT but no box in BOUNDS: {missing}"


def test_bounded_saturates_rather_than_wrapping():
    lo, hi = bounds_for("tau_m", BOUNDS)
    assert float(
        unpack({"tau_m": jnp.asarray(-1e4)}, DEFAULT, BOUNDS)["tau_m"]
    ) == pytest.approx(lo, abs=1e-9)
    assert float(
        unpack({"tau_m": jnp.asarray(+1e4)}, DEFAULT, BOUNDS)["tau_m"]
    ) == pytest.approx(hi, abs=1e-9)


def test_unclassified_parameters_are_unconstrained():
    assert bijector_for("something_nobody_classified", DEFAULT) == "identity"
    v = -7.25
    assert float(
        unpack(pack({"something_nobody_classified": v}, DEFAULT), DEFAULT)[
            "something_nobody_classified"
        ]
    ) == pytest.approx(v)


def test_log_space_cannot_produce_a_non_positive_value():
    for z in (-50.0, -1.0, 0.0, 25.0):
        assert float(unpack({"tau_m": z}, {"tau_m": "log"})["tau_m"]) > 0.0


def test_logit_space_stays_inside_the_unit_interval():
    for z in (-60.0, 0.0, 60.0):
        v = float(unpack({"f_v": z}, DEFAULT)["f_v"])
        assert 0.0 <= v <= 1.0


def test_bijectors_are_differentiable_with_finite_gradients():
    for kind, x in (("log", 3.5), ("logit", 0.42), ("identity", -61.0)):
        fwd, inv = BIJECTORS[kind]
        g = jax.grad(lambda z, inv=inv: jnp.sum(inv(z)))(
            jnp.asarray(fwd(jnp.asarray(x)))
        )
        assert np.isfinite(g) and abs(float(g)) > 0.0


def test_chain_rule_holds_through_the_transform():
    f = lambda t: jnp.sum(t["sigma"] ** 2 + 3.0 * t["sigma"])  # noqa: E731
    t0 = 12.0
    z0 = pack({"sigma": t0}, DEFAULT)
    g = float(jax.grad(lambda z: f(unpack(z, DEFAULT)))(z0)["sigma"])

    assert g == pytest.approx((2 * t0 + 3.0) * t0, rel=1e-6)

    eps = 1e-3
    hi = float(f(unpack({"sigma": z0["sigma"] + eps}, DEFAULT)))
    lo = float(f(unpack({"sigma": z0["sigma"] - eps}, DEFAULT)))
    assert g == pytest.approx((hi - lo) / (2 * eps), rel=1e-2)


def _fit(env, stimulus, target, init, *, lr, steps, **kwargs):
    import optax

    return fit(
        functools.partial(env.cells.set_params(init), round(DURATION), stimulus, dt=DT),
        lambda run, _: voltage_mse(run.voltage, target),
        params={
            name: (lambda m, name=name: getattr(m.func.module, name)) for name in init
        },
        optimizer=optax.adam(lr),
        steps=steps,
        **kwargs,
    )


TAU_M = {"tau_m": "log"}


def test_a_transformed_fit_never_leaves_the_domain():
    from livn.optimization.train import Callback

    env, stimulus = _env(), _stimulus()
    target = np.asarray(
        _env()
        .cells.set_params({"tau_m": jnp.asarray([3.0])})
        .run(DURATION, stimulus, dt=DT)
        .voltage
    )
    seen = []

    class Watch(Callback):
        def on_step_end(self, step, state, metrics):
            seen.append(float(np.squeeze(state.theta["tau_m"])))

    result = _fit(
        env,
        stimulus,
        target,
        {"tau_m": 12.0},
        transform=TAU_M,
        steps=15,
        lr=2.0,
        callbacks=[Watch()],
    )
    assert seen, "callback never fired"
    assert all(v > 0.0 for v in seen), (
        f"a non-positive tau_m reached the simulator: {seen}"
    )
    assert float(np.squeeze(result.theta["tau_m"])) > 0.0
    assert np.all(np.isfinite(result.history["loss"]))


def test_a_bounded_fit_never_leaves_its_box():
    env, stimulus = _env(), _stimulus()
    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)
    lo, hi = 5.0, 20.0

    for box in ((lo, hi), bounded(lo, hi)):
        result = _fit(
            env,
            stimulus,
            target,
            {"tau_m": 12.0},
            transform={"tau_m": box},
            steps=8,
            lr=3.0,
        )
        assert np.all(result.history["params"]["tau_m"] >= lo)
        assert np.all(result.history["params"]["tau_m"] <= hi)


def test_log_space_amplifies_the_step_size():
    stimulus = _stimulus()
    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)

    raw = _fit(_env(), stimulus, target, {"tau_m": 12.0}, steps=12, lr=1.0)
    tr = _fit(
        _env(), stimulus, target, {"tau_m": 12.0}, steps=12, lr=1.0, transform=TAU_M
    )

    assert float(np.squeeze(tr.theta["tau_m"])) > 10 * float(
        np.squeeze(raw.theta["tau_m"])
    ), (
        "log space is expected to move much further per step; if this stops holding the effective "
        f"step size changed (raw={float(np.squeeze(raw.theta['tau_m'])):.4g}, transformed={float(np.squeeze(tr.theta['tau_m'])):.4g})"
    )
    assert float(np.squeeze(tr.theta["tau_m"])) > 0.0


def test_history_and_result_are_reported_in_the_constrained_space():
    from livn.optimization.train import Callback

    env, stimulus = _env(), _stimulus()
    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)
    seen = []

    class Watch(Callback):
        def on_step_end(self, step, state, metrics):
            seen.append(float(np.squeeze(state.theta["tau_m"])))

    result = _fit(
        env,
        stimulus,
        target,
        {"tau_m": 12.0},
        transform=TAU_M,
        steps=3,
        lr=0.05,
        callbacks=[Watch()],
    )
    assert float(np.squeeze(result.theta["tau_m"])) == pytest.approx(12.0, rel=0.5)
    assert result.history["params"]["tau_m"][0, 0] == pytest.approx(12.0, rel=1e-3)
    assert all(s > 1.0 for s in seen), f"callback saw transformed values: {seen}"
