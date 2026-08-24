import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from livn.backend.diffrax import Env
from livn.models.eventloop import SolverConfig
from livn.models.glif import GLIF
from livn.stimulus import Stimulus
from optimization.fit import fit
from optimization.losses import voltage_mse
from optimization.transforms import (
    BIJECTORS,
    BOUNDS,
    DEFAULT,
    bijector_for,
    bounds_for,
    pack,
    unpack,
)

DURATION, DT = 20.0, 0.5


def _env(n=1):
    env = Env(
        n,
        model=GLIF.leaky_integrate_and_fire(
            mechanism="hard", config=SolverConfig(max_rate=1.0)
        ),
    )
    env.init()
    env.record_voltage(dt=DT)
    return env


def _stimulus(n=1, amp=0.05):
    steps = round(DURATION / DT) + 1
    return Stimulus.from_current(np.full((steps, n), amp), dt=DT)


@pytest.mark.parametrize("name,kind", sorted(DEFAULT.items()))
def test_every_declared_parameter_round_trips(name, kind):
    if kind == "bounded":
        lo, hi = bounds_for(name)
        x = lo + 0.37 * (hi - lo)
    else:
        x = {"log": 3.5, "logit": 0.42, "identity": -61.0}[kind]
    z = pack({name: x})[name]
    assert float(unpack({name: z})[name]) == pytest.approx(x, rel=1e-6)


@pytest.mark.parametrize("name", sorted(BOUNDS))
def test_bounded_parameters_cannot_leave_their_box(name):
    lo, hi = bounds_for(name)
    for z in (-1e4, -50.0, 0.0, 50.0, 1e4):
        got = float(unpack({name: jnp.asarray(z)})[name])
        assert lo <= got <= hi, f"{name} left ({lo}, {hi}) at z={z}: {got}"


def test_every_box_is_actually_used():
    unused = [n for n in BOUNDS if bijector_for(n) != "bounded"]
    assert not unused, f"declared in BOUNDS but not bounded in DEFAULT: {unused}"
    missing = [
        n for n, k in DEFAULT.items() if k == "bounded" and bounds_for(n) is None
    ]
    assert not missing, f"declared bounded in DEFAULT but no box in BOUNDS: {missing}"


def test_bounded_saturates_rather_than_wrapping():
    lo, hi = bounds_for("tau_m")
    assert float(unpack({"tau_m": jnp.asarray(-1e4)})["tau_m"]) == pytest.approx(
        lo, abs=1e-9
    )
    assert float(unpack({"tau_m": jnp.asarray(+1e4)})["tau_m"]) == pytest.approx(
        hi, abs=1e-9
    )


def test_unclassified_parameters_are_unconstrained():
    assert bijector_for("something_nobody_classified") == "identity"
    v = -7.25
    assert float(
        unpack(pack({"something_nobody_classified": v}))["something_nobody_classified"]
    ) == pytest.approx(v)


def test_log_space_cannot_produce_a_non_positive_value():
    for z in (-50.0, -1.0, 0.0, 25.0):
        assert float(unpack({"tau_m": z})["tau_m"]) > 0.0


def test_logit_space_stays_inside_the_unit_interval():
    for z in (-60.0, 0.0, 60.0):
        v = float(unpack({"f_v": z})["f_v"])
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
    z0 = pack({"sigma": t0})
    g = float(jax.grad(lambda z: f(unpack(z)))(z0)["sigma"])

    assert g == pytest.approx((2 * t0 + 3.0) * t0, rel=1e-6)

    eps = 1e-3
    hi = float(f(unpack({"sigma": z0["sigma"] + eps})))
    lo = float(f(unpack({"sigma": z0["sigma"] - eps})))
    assert g == pytest.approx((hi - lo) / (2 * eps), rel=1e-2)


def _voltage_loss(run, target):
    return voltage_mse(run.voltage, target)


def test_a_transformed_fit_never_leaves_the_domain():
    env, stimulus = _env(), _stimulus()
    target = np.asarray(
        _env()
        .cells.set_params({"tau_m": jnp.asarray([3.0])})
        .run(DURATION, stimulus, dt=DT)
        .voltage
    )
    seen = []
    theta, hist = fit(
        env,
        target,
        _voltage_loss,
        {"tau_m": 12.0},
        transform=True,
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=15,
        learning_rate=2.0,
        callback=lambda step, t, v: seen.append(float(t["tau_m"])),
    )
    assert seen, "callback never fired"
    assert all(v > 0.0 for v in seen), (
        f"a non-positive tau_m reached the simulator: {seen}"
    )
    assert float(theta["tau_m"]) > 0.0
    assert np.all(np.isfinite(hist["loss"]))


def test_log_space_amplifies_the_step_size():
    stimulus = _stimulus()
    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)
    common = {
        "duration": DURATION,
        "stimulus": stimulus,
        "dt": DT,
        "steps": 12,
        "learning_rate": 1.0,
    }

    raw, _ = fit(_env(), target, _voltage_loss, {"tau_m": 12.0}, **common)
    tr, _ = fit(
        _env(), target, _voltage_loss, {"tau_m": 12.0}, transform=True, **common
    )

    assert float(tr["tau_m"]) > 10 * float(raw["tau_m"]), (
        "log space is expected to move much further per step; if this stops holding the effective "
        f"step size changed (raw={float(raw['tau_m']):.4g}, transformed={float(tr['tau_m']):.4g})"
    )
    assert float(tr["tau_m"]) > 0.0


def test_history_and_result_are_reported_in_the_constrained_space():
    env, stimulus = _env(), _stimulus()
    steps = round(DURATION / DT) + 1
    target = np.full((1, steps), -70.0)
    seen = []
    theta, hist = fit(
        env,
        target,
        _voltage_loss,
        {"tau_m": 12.0},
        transform=True,
        duration=DURATION,
        stimulus=stimulus,
        dt=DT,
        steps=3,
        learning_rate=0.05,
        callback=lambda step, t, v: seen.append(float(t["tau_m"])),
    )
    assert float(theta["tau_m"]) == pytest.approx(12.0, rel=0.5)
    assert hist["params"]["tau_m"][0] == pytest.approx(12.0, rel=1e-3)
    assert all(s > 1.0 for s in seen), f"callback saw transformed values: {seen}"
