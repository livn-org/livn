import numpy as np
import pytest

pytest.importorskip("diffrax")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx

from livn.models.eventloop import SolverConfig
from livn.models.glif import GlifMechanism, GlifNeurons
from livn.models.opsins.glif import ChR2H134R

DURATION, DT = 120.0, 0.1
STEPS = round(DURATION / DT) + 1


class Depression(GlifMechanism):
    states: tuple[str, ...] = eqx.field(static=True, default=("x",), kw_only=True)
    U: float = 0.4
    tau_rec: float = 80.0

    def initial(self, n_cells):
        return jnp.ones((n_cells, 1))

    def drift(self, t, v, states, inputs):
        return jnp.zeros_like(v), (1.0 - states) / self.tau_rec

    def on_spike(self, fired, states):
        return states * (1.0 - self.U)

    def presynaptic(self, states):
        return states[:, 0]


def _weights(n=4, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 150.0, (n, n)).astype(np.float32)
    np.fill_diagonal(w, 0.0)
    w[0, 1:] = 400.0
    return w


def _current(n=4):
    return jnp.zeros((STEPS, n)).at[:, 0].set(0.3).at[:, 1:].set(0.12)


def _light(n=4):
    light = np.zeros((STEPS, n), np.float32)
    light[300:900] = 3.0
    return jnp.asarray(light)


CONFIGURATIONS = {
    "hard": {"mechanism": "hard"},
    "escape": {"mechanism": "escape"},
    "escape+depression": {"mechanism": "escape", "mechanisms": (Depression(),)},
    "escape+depression+chr2": {
        "mechanism": "escape",
        "mechanisms": (Depression(), ChR2H134R(20.0)),
    },
}


def _module(network, tau_syn, **kwargs):
    return GlifNeurons(
        network.shape[-1],
        {"sigma": 1.0, "tau_syn": tau_syn},
        network=network,
        config=SolverConfig(max_rate_per_ms=1.0, dt_solver=0.05),
        **kwargs,
    )


def _solve(module, chr2):
    inputs = {"irradiance": _light(module.n_cells)} if chr2 else None
    return module.solve(
        _current(module.n_cells),
        0.0,
        DURATION,
        DT,
        key=jr.PRNGKey(3),
        inputs=inputs,
        record={"spikes", "voltage", "i_syn"},
    )


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_one_channel_of_a_stack_is_the_plain_network(name):
    w = _weights()
    plain = _module(w, 7.0, **CONFIGURATIONS[name])
    stacked = _module(w[None], [7.0], **CONFIGURATIONS[name])
    assert stacked.layout == plain.layout and stacked.network.ndim == 2

    a, b = _solve(plain, "chr2" in name), _solve(stacked, "chr2" in name)
    assert np.isfinite(np.asarray(a.spike_times)).sum() > 10
    for x, y in [(a.spike_times, b.spike_times), (a.ys, b.ys), (a.yT, b.yT)]:
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_a_silent_second_channel_leaves_the_network_bit_identical(name):
    w = _weights()
    plain = _module(w, 7.0, **CONFIGURATIONS[name])
    two = _module(np.stack([w, np.zeros_like(w)]), [7.0, 300.0], **CONFIGURATIONS[name])
    assert two.layout.channels == 2 and two.layout.size == plain.layout.size + 1

    a, b = _solve(plain, "chr2" in name), _solve(two, "chr2" in name)
    np.testing.assert_array_equal(np.asarray(a.spike_times), np.asarray(b.spike_times))
    np.testing.assert_array_equal(np.asarray(a.v), np.asarray(b.v))
    np.testing.assert_array_equal(np.asarray(a.i_syn), np.asarray(b.i_syn[..., :1]))
    assert not np.asarray(b.i_syn[..., 1]).any()


def test_two_channels_are_the_closed_form_sum_of_exponentials():
    # float64, so the residual is the Euler step's, not float32 accumulation
    with jax.enable_x64(True):
        _check_two_channel_closed_form()


def _check_two_channel_closed_form():
    # cell 0 fires once (its refractory period outlasts the run); cell 1 never fires
    tau, w = np.array([5.0, 500.0]), np.array([30.0, -12.0])
    tau_m, g_L, E_L = 10.0, 10.0, -70.0
    network = np.zeros((2, 2, 2))
    network[:, 0, 1] = w
    params = {
        "tau_syn": tau,
        "t_ref": np.array([1e4, 0.0]),
        "V_threshold_base": np.array([-55.0, 1e3]),
        "tau_m": tau_m,
        "g_L": g_L,
        "E_L": E_L,
    }
    module = GlifNeurons(
        2,
        params,
        level=1,
        network=network,
        config=SolverConfig(max_rate_per_ms=0.5, dt_solver=0.005),
    )
    current = jnp.zeros((STEPS, 2)).at[:, 0].set(0.3)
    solution = module.solve(
        current, 0.0, DURATION, DT, record={"spikes", "voltage", "i_syn"}
    )

    train = np.asarray(solution.spike_times[0])
    train = train[np.isfinite(train)]
    assert len(train) == 1
    assert not np.isfinite(np.asarray(solution.spike_times[1])).any()
    t_s = float(train[0])

    ts = np.asarray(solution.ts)
    after = ts > t_s + DT
    lag = ts[after, None] - t_s
    i_syn = np.asarray(solution.i_syn[1])
    assert not i_syn[ts < t_s - DT].any()
    # Euler's relative error on a decay is about lag * dt / (2 tau^2)
    expected = w * np.exp(-lag / tau)
    bound = 2.0 * np.abs(expected) * lag * 0.005 / (2 * tau**2) + 1e-6
    assert np.all(np.abs(i_syn[after] - expected) <= bound)

    # dV/dt = -(V - E_L)/tau_m + sum_c i_c / (g_L tau_m), from V = E_L at t_s
    shape = tau / (tau - tau_m) * (np.exp(-lag / tau) - np.exp(-lag / tau_m))
    v = E_L + (w / g_L * shape).sum(-1)
    np.testing.assert_allclose(np.asarray(solution.v[1])[after], v, rtol=0.0, atol=5e-4)


def test_channels_with_one_tau_are_one_channel_with_the_summed_weights():
    w = _weights()
    rng = np.random.default_rng(1)
    split = rng.uniform(0.0, 1.0, w.shape).astype(np.float32)
    parts = np.stack([w * split, w * (1.0 - split), np.zeros_like(w)])
    parts[2] = w - parts[0] - parts[1]  # the float32 remainder, so the sum is w

    one = _module(w, 7.0, mechanism="escape", mechanisms=(Depression(),))
    three = _module(parts, 7.0, mechanism="escape", mechanisms=(Depression(),))
    a, b = _solve(one, False), _solve(three, False)

    sa, sb = np.asarray(a.spike_times), np.asarray(b.spike_times)
    np.testing.assert_array_equal(np.isfinite(sa), np.isfinite(sb))
    assert np.isfinite(sa).sum() > 10
    np.testing.assert_allclose(sa[np.isfinite(sa)], sb[np.isfinite(sb)], atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(a.i_syn[..., 0]), np.asarray(b.i_syn).sum(-1), atol=1e-2
    )


def test_tau_syn_is_per_channel_and_may_be_per_cell():
    network = np.zeros((2, 3, 3))
    network[:, 0, 1] = 1.0
    assert _module(network, [5.0, 500.0]).tau_syn.shape == (3, 2)
    per_cell = np.array([[5.0, 400.0], [6.0, 500.0], [7.0, 600.0]])
    np.testing.assert_array_equal(_module(network, per_cell).tau_syn, per_cell)
    with pytest.raises(ValueError, match="channels"):
        _module(network, [5.0, 6.0, 7.0])
    with pytest.raises(ValueError, match="stack"):
        GlifNeurons(3, network=np.ones((2, 3, 2)))


def test_a_start_state_carries_every_channel():
    w = _weights()
    two = _module(np.stack([w, 0.5 * w]), [7.0, 300.0], mechanism="escape")
    first = two.solve(_current(), 0.0, 60.0, DT, key=jr.PRNGKey(0))
    lo, hi = two.layout.i_syn_columns[0], two.layout.i_syn_columns[-1] + 1
    assert first.yT.shape == (4, two.layout.size)
    assert np.asarray(first.yT[:, lo:hi]).any(axis=0).all()

    # restarting from the end state continues the slow channel's decay
    zero_drive = jnp.zeros((2, 4))
    resumed = two.solve(
        zero_drive, 60.0, 61.0, DT, y0=first.yT, record={"spikes", "i_syn"}
    )
    np.testing.assert_allclose(
        np.asarray(resumed.i_syn[:, 0, 1]), np.asarray(first.yT[:, hi - 1]), rtol=1e-6
    )
    with pytest.raises(ValueError, match="unconnected"):
        GlifNeurons(2).solve(None, 0.0, 1.0, DT, record={"i_syn"})


def test_gradients_through_a_multichannel_free_run_match_finite_differences():
    with jax.enable_x64(True):
        network = np.zeros((2, 2, 2))
        network[:, 0, 1] = [250.0, 120.0]
        config = SolverConfig(
            max_rate_per_ms=0.5,
            dt_solver=0.01,
            root_finder=optx.Newton(1e-10, 1e-10, optx.rms_norm),
        )
        base = GlifNeurons(
            2,
            {"tau_syn": [5.0, 200.0]},
            network=network,
            config=config,
        )
        duration = 40.0
        current = jnp.zeros((round(duration / DT) + 1, 2)).at[:, 0].set(0.3)
        current = current.at[:, 1].set(0.1)

        def loss(network, tau_syn):
            module = eqx.tree_at(
                lambda m: (m.network, m.tau_syn), base, (network, tau_syn)
            )
            spikes = module.solve(current, 0.0, duration, DT).spike_times[1]
            return jnp.sum(jnp.where(jnp.isfinite(spikes), spikes, 0.0))

        w0, tau0 = jnp.asarray(network), base.tau_syn
        assert (
            np.isfinite(
                np.asarray(base.solve(current, 0.0, duration, DT).spike_times[1])
            ).sum()
            >= 2
        )
        g_w, g_tau = jax.grad(loss, argnums=(0, 1))(w0, tau0)

        for c in range(2):
            step = 1e-3 * float(w0[c, 0, 1])
            plus = float(loss(w0.at[c, 0, 1].add(step), tau0))
            minus = float(loss(w0.at[c, 0, 1].add(-step), tau0))
            fd = (plus - minus) / (2 * step)
            assert float(g_w[c, 0, 1]) != 0.0
            np.testing.assert_allclose(float(g_w[c, 0, 1]), fd, rtol=1e-3)

            # finer steps resolve the loop's step-level kinks, not the slope
            step = 1e-3 * float(tau0[1, c])
            plus = float(loss(w0, tau0.at[1, c].add(step)))
            minus = float(loss(w0, tau0.at[1, c].add(-step)))
            fd = (plus - minus) / (2 * step)
            assert float(g_tau[1, c]) != 0.0
            np.testing.assert_allclose(float(g_tau[1, c]), fd, rtol=1e-2)
