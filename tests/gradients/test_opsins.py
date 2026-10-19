import numpy as np
import pytest

from livn.backend import backend

pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from livn.models.eventloop import SolverConfig
from livn.models.glif import GLIF, GlifNeurons, cell_populations
from livn.models.opsins.chr2 import CHR2_DEFAULTS, chr2_current, chr2_derivatives
from livn.models.opsins.glif import ChR2H134R

DT = 0.1


def _pulse(steps, n, on=(100, 300), irradiance=5.0):
    light = np.zeros((steps + 1, n), dtype=np.float32)
    light[on[0] : on[1]] = irradiance
    return jnp.asarray(light)


def _cell(g_nS, n=1, **kwargs):
    return GlifNeurons(
        n,
        level=3,
        mechanisms=(ChR2H134R(g_nS, **kwargs),),
        config=SolverConfig(max_rate_per_ms=1.0),
    )


def test_the_mechanism_is_the_shared_kinetics():
    opsin = ChR2H134R(np.array([2.0, 5.0]), celsius=30.0)
    states = jnp.array([[0.2, 0.1, 0.6, 0.3], [0.05, 0.02, 0.9, 0.8]])
    v = jnp.array([-65.0, -30.0])
    current, dstates = opsin.drift(
        0.0, v, states, {"irradiance": jnp.array([1.0, 3.0])}
    )

    P = {k: x for k, x in CHR2_DEFAULTS.items() if k != "gmax"}
    for c in range(2):
        np.testing.assert_allclose(
            current[c], -chr2_current(states[c], v[c], opsin.g_nS[c], P), rtol=1e-5
        )
        np.testing.assert_allclose(
            dstates[c],
            chr2_derivatives(states[c], v[c], [1.0, 3.0][c], P, 30.0),
            rtol=1e-5,
        )
    assert opsin.states == ("O1", "O2", "C1", "p")
    assert opsin.inputs == ("irradiance",)
    np.testing.assert_array_equal(
        opsin.initial(3), np.tile([0.0, 0.0, 1.0, 0.0], (3, 1))
    )


def test_light_depolarizes_and_drives_spikes_only_with_an_opsin():
    duration = 60.0
    steps = round(duration / DT)
    light = _pulse(steps, 2)

    cell = _cell(np.array([0.0, 40.0]), n=2)
    solution = cell.solve(None, 0.0, duration, DT, inputs={"irradiance": light})

    v = np.asarray(solution.v)
    assert int(solution.num_spikes[0]) == 0
    assert int(solution.num_spikes[1]) > 0
    # the dark cell sits at rest; the lit one leaves it during the pulse (and resets to
    # rest at each spike, so its maximum is what shows it)
    assert np.ptp(v[0]) < 1e-3
    assert v[1, 100:300].max() > v[1, 50] + 1.0


def test_the_photocurrent_desensitizes_under_steady_light():
    # a cell that cannot fire (threshold far away) under a long pulse: the current peaks
    # and relaxes towards a steady state, as Williams et al. measure
    duration = 600.0
    steps = round(duration / DT)
    cell = GlifNeurons(
        1,
        {"V_threshold_base": 1e3},
        level=1,
        mechanisms=(ChR2H134R(10.0),),
        config=SolverConfig(max_rate_per_ms=0.1),
    )
    solution = cell.solve(
        None,
        0.0,
        duration,
        DT,
        inputs={"irradiance": _pulse(steps, 1, on=(100, 5100))},
        record={"voltage", "O1", "O2"},
    )
    v = np.asarray(solution.v)[0]
    depolarization = v - v[50]
    peak = depolarization[100:1500].max()
    steady = depolarization[4500:5000].mean()
    assert peak > 1.0
    assert 0.2 * peak < steady < 0.9 * peak


def test_temperature_speeds_the_kinetics():
    duration = 40.0
    steps = round(duration / DT)

    def rise(celsius):
        cell = GlifNeurons(
            1,
            {"V_threshold_base": 1e3},
            level=1,
            mechanisms=(ChR2H134R(10.0, celsius=celsius),),
            config=SolverConfig(max_rate_per_ms=0.1),
        )
        v = np.asarray(
            cell.solve(
                None, 0.0, duration, DT, inputs={"irradiance": _pulse(steps, 1)}
            ).v
        )[0]
        return v[300] - v[50]

    # deactivation is faster when warm, so after the pulse the warm cell is closer to rest
    assert rise(36.0) != pytest.approx(rise(22.0))


def test_the_conductance_carries_a_gradient():
    duration = 40.0
    steps = round(duration / DT)
    light = _pulse(steps, 1)

    def loss(g):
        solution = _cell(g).solve(None, 0.0, duration, DT, inputs={"irradiance": light})
        return jnp.mean(solution.v)

    gradient = float(jax.grad(loss)(jnp.asarray(5.0)))
    assert np.isfinite(gradient) and gradient > 0.0


_env = pytest.mark.skipif(
    backend() != "diffrax", reason="GLIF runs on the diffrax backend"
)


class OptoGLIF(GLIF):
    def diffrax_mechanisms(self, env):
        exc = cell_populations(env) == "EXC"
        return (ChR2H134R(jnp.asarray(np.where(exc, 40.0, 0.0))),)


@_env
def test_an_override_puts_the_opsin_into_one_population():
    from livn.env import Env
    from livn.stimulus import Stimulus
    from livn.system import Monolayer

    system = Monolayer(
        total_cells=12,
        area_kwargs={"x_range": (0.0, 1400.0), "y_range": (0.0, 1400.0)},
        connectivity={"sigma": 300.0, "mean_degree": {"default": 0.0}},
        seed=4242,
    )
    env = Env(system, model=OptoGLIF(level=3)).init().record_spikes()
    gids = np.asarray(env.simulated_gids()).astype(int)
    duration = 60.0
    light = np.asarray(_pulse(round(duration / DT) - 1, len(gids)))

    run = env.run(duration, Stimulus.from_irradiance(light, dt=DT, gids=gids), dt=DT)

    exc = cell_populations(env) == "EXC"
    fired = np.zeros(len(gids), bool)
    fired[np.searchsorted(gids, np.asarray(run.spike_ids).astype(int))] = True
    assert exc.any() and (~exc).any()
    assert fired[exc].all()
    assert not fired[~exc].any()
