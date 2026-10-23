import numpy as np
import pytest

from livn.backend import backend

pytest.importorskip("diffrax")

import equinox as eqx
import jax
import jax.numpy as jnp

from livn.models.eventloop import SolverConfig
from livn.models.glif import (
    GLIF,
    RECORDABLE_STATES,
    GlifMechanism,
    GlifNeurons,
    cell_populations,
)

DURATION, DT = 60.0, 0.1
STEPS = round(DURATION / DT) + 1


class Inert(GlifMechanism):
    states: tuple[str, ...] = eqx.field(static=True, default=("inert",), kw_only=True)

    def drift(self, t, v, states, inputs):
        return jnp.zeros_like(v), jnp.zeros_like(states)


class Constant(GlifMechanism):
    """A stateless current, in pA per cell"""

    amplitude: jax.Array

    def drift(self, t, v, states, inputs):
        return jnp.broadcast_to(self.amplitude, v.shape), states


class Charging(GlifMechanism):
    """q charges towards 1 with tau and injects g * q"""

    states: tuple[str, ...] = eqx.field(static=True, default=("q",), kw_only=True)
    g: jax.Array
    tau: float = 40.0

    def drift(self, t, v, states, inputs):
        q = states[:, 0]
        return self.g * q, ((1.0 - q) / self.tau)[:, None]


class Counter(GlifMechanism):
    states: tuple[str, ...] = eqx.field(static=True, default=("count",), kw_only=True)

    def drift(self, t, v, states, inputs):
        return jnp.zeros_like(v), jnp.zeros_like(states)

    def on_spike(self, fired, states):
        return states + 1.0


class Halving(GlifMechanism):
    def drift(self, t, v, states, inputs):
        return jnp.zeros_like(v), states

    def presynaptic(self, states):
        return jnp.full(states.shape[:1], 0.5)


class Light(GlifMechanism):
    """A current proportional to the irradiance, ``gain`` pA per mW/mm2"""

    inputs: tuple[str, ...] = eqx.field(
        static=True, default=("irradiance",), kw_only=True
    )
    gain: jax.Array

    def drift(self, t, v, states, inputs):
        return self.gain * inputs["irradiance"], states


class ThresholdIntegral(GlifMechanism):
    states: tuple[str, ...] = eqx.field(
        static=True, default=("int_theta_s", "int_theta_v", "int_shift"), kw_only=True
    )
    inputs: tuple[str, ...] = eqx.field(
        static=True, default=("theta_s", "theta_v", "threshold"), kw_only=True
    )
    base: jax.Array  # V_threshold_base, per cell

    def drift(self, t, v, states, inputs):
        shift = inputs["threshold"] - self.base
        rates = jnp.stack([inputs["theta_s"], inputs["theta_v"], shift], axis=-1)
        return jnp.zeros_like(v), rates


class ThresholdAtSpike(GlifMechanism):
    states: tuple[str, ...] = eqx.field(static=True, default=("crossed",), kw_only=True)
    inputs: tuple[str, ...] = eqx.field(
        static=True, default=("threshold",), kw_only=True
    )

    def drift(self, t, v, states, inputs):
        return jnp.zeros_like(v), jnp.zeros_like(states)

    def on_spike(self, fired, states, inputs):
        return inputs["threshold"][:, None]


def _rheobase_pA(neurons):
    return np.asarray((neurons.V_threshold_base - neurons.E_L) * neurons.g_L)


def _driven(neurons, amplitude=2.0, steps=STEPS):
    """A suprathreshold step current in nA"""
    current = amplitude * _rheobase_pA(neurons) / 1e3
    return jnp.asarray(np.tile(current, (steps, 1)))


def _chain(n=3, weight=800.0):
    connectivity = np.zeros((n, n), dtype=np.float32)
    for pre in range(n - 1):
        connectivity[pre, pre + 1] = weight
    return connectivity


def _network(mechanisms=(), weight=800.0, **kwargs):
    kwargs.setdefault("config", SolverConfig(max_rate_per_ms=1.0))
    return GlifNeurons(3, network=_chain(3, weight), mechanisms=mechanisms, **kwargs)


def _drive_first(n=3):
    return jnp.zeros((STEPS, n)).at[:, 0].set(0.3)


def _base(solution, size):
    return np.asarray(solution.yT)[..., :size]


_CASES = {
    "cells": lambda m: (GlifNeurons(3, mechanisms=m), {}),
    "escape": lambda m: (
        GlifNeurons(2, {"sigma": 2.0}, mechanism="escape", mechanisms=m),
        {},
    ),
    "batched": lambda m: (
        GlifNeurons(2, {"sigma": 2.0}, mechanism="escape", mechanisms=m),
        {"num_samples": 3},
    ),
    "diffusion": lambda m: (
        GlifNeurons(2, {"sigma_v": 1.0}, diffusion=True, mechanisms=m),
        {},
    ),
    "forced": lambda m: (
        GlifNeurons(2, {"t_ref": 2.0}, mechanisms=m),
        {"forced_spikes": np.array([[12.5, 30.0, np.inf], [20.0, np.inf, np.inf]])},
    ),
    "network": lambda m: (_network(m), {}),
    "network-escape": lambda m: (
        _network(m, params={"sigma": 2.0}, mechanism="escape"),
        {"num_samples": 2},
    ),
    "read-out": lambda m: (_network(m, read_out_neurons=[2]), {}),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_an_inert_mechanism_changes_nothing(case):
    plain, options = _CASES[case](())
    inert, _ = _CASES[case]((Inert(),))
    assert inert.layout.size == plain.layout.size + 1

    current = _drive_first() if plain.network is not None else _driven(plain)
    key = jax.random.PRNGKey(7)
    without = plain.solve(current, 0.0, DURATION, DT, key=key, **options)
    with_ = inert.solve(current, 0.0, DURATION, DT, key=key, **options)

    assert np.isfinite(np.asarray(without.spike_times)).sum() > 0
    np.testing.assert_array_equal(
        np.asarray(with_.spike_times), np.asarray(without.spike_times)
    )
    np.testing.assert_array_equal(np.asarray(with_.v), np.asarray(without.v))
    size = plain.layout.size
    if case == "forced":
        size += 1  # the forcing column, which the forced layout adds
    np.testing.assert_array_equal(_base(with_, size), _base(without, size))
    assert float(np.abs(np.asarray(with_.yT)[..., -1]).max()) == 0.0


@pytest.mark.parametrize("coupled", [False, True])
def test_a_constant_current_mechanism_is_the_same_current_as_an_input(coupled):
    if coupled:
        plain = _network()
        current = _drive_first()
        amplitude = jnp.asarray(np.asarray(current[0]) * 1e3)
        inserted = _network((Constant(amplitude=amplitude),))
    else:
        plain = GlifNeurons(2)
        current = _driven(plain)
        amplitude = jnp.asarray(np.asarray(current[0]) * 1e3)
        inserted = GlifNeurons(2, mechanisms=(Constant(amplitude=amplitude),))
    assert inserted.layout.size == plain.layout.size  # stateless

    given = plain.solve(current, 0.0, DURATION, DT)
    injected = inserted.solve(None, 0.0, DURATION, DT)

    spikes = np.asarray(given.spike_times)
    assert np.isfinite(spikes).sum() > 3
    np.testing.assert_allclose(np.asarray(injected.spike_times), spikes, atol=1e-3)
    np.testing.assert_allclose(np.asarray(injected.v), np.asarray(given.v), atol=1e-3)


@pytest.mark.parametrize("coupled", [False, True])
def test_a_mechanism_state_carries_through_the_reset(coupled):
    if coupled:
        neurons = _network((Charging(g=jnp.asarray([300.0, 0.0, 0.0])),))
        current = _drive_first()
    else:
        g = 3.0 * _rheobase_pA(GlifNeurons(2))
        neurons = GlifNeurons(2, mechanisms=(Charging(g=jnp.asarray(g)),))
        current = None

    solution = neurons.solve(current, 0.0, DURATION, DT)

    assert int(solution.num_spikes[0]) >= 3, "the cell has to reset to test this"
    q = np.asarray(solution.column(neurons.state_columns["q"]))
    # untouched by every reset and integrated across every cut: the free charge
    expected = 1.0 - np.exp(-np.asarray(solution.ts) / 40.0)
    np.testing.assert_allclose(q, np.broadcast_to(expected, q.shape), atol=2e-3)


def test_a_mechanism_state_is_recorded_by_name():
    neurons = GlifNeurons(1, mechanisms=(Charging(g=jnp.asarray([0.0])),))
    solution = neurons.solve(None, 0.0, 20.0, DT, record={"voltage", "q"})
    assert solution.columns == (0, neurons.state_columns["q"])

    _, _, _, _, _, _, _, states = neurons.run(None, t1=20.0, record={"spikes", "q"})
    np.testing.assert_allclose(
        np.asarray(states["q"][1])[0, -1], 1.0 - np.exp(-0.5), atol=1e-3
    )


@pytest.mark.parametrize("case", ["cells", "network", "forced"])
def test_on_spike_jumps_a_state_at_every_spike(case):
    options = {}
    if case == "network":
        neurons, current = _network((Counter(),)), _drive_first()
    else:
        neurons = GlifNeurons(2, {"t_ref": 2.0}, mechanisms=(Counter(),))
        current = _driven(neurons)
        if case == "forced":
            options["forced_spikes"] = np.array(
                [[12.5, 30.0, 47.25], [20.0, np.inf, np.inf]]
            )

    solution = neurons.solve(current, 0.0, DURATION, DT, **options)

    counts = np.asarray(solution.num_spikes)
    assert counts.sum() > 2
    np.testing.assert_array_equal(np.asarray(solution.yT)[..., -1], counts)
    if case == "forced":
        np.testing.assert_array_equal(counts, [3, 1])
    trace = np.asarray(solution.column(neurons.state_columns["count"]))
    assert np.all(np.diff(trace, axis=-1) >= 0)


def test_presynaptic_scaling_is_scaling_the_weights():
    current = _drive_first()
    halved = _network((Halving(),)).solve(current, 0.0, DURATION, DT)
    lighter = _network(weight=400.0).solve(current, 0.0, DURATION, DT)
    full = _network(weight=800.0).solve(current, 0.0, DURATION, DT)

    spikes = np.asarray(lighter.spike_times)
    assert np.isfinite(spikes[1]).sum() > 0, "the halved chain has to propagate"
    np.testing.assert_array_equal(np.asarray(halved.spike_times), spikes)
    np.testing.assert_array_equal(np.asarray(halved.v), np.asarray(lighter.v))
    assert not np.array_equal(np.asarray(full.spike_times), spikes)


@pytest.mark.parametrize("coupled", [False, True])
def test_gradients_reach_a_mechanism_parameter(coupled):
    if coupled:
        neurons = _network((Charging(g=jnp.asarray([100.0, 50.0, 50.0])),))
        current = _drive_first()
    else:
        g = 3.0 * _rheobase_pA(GlifNeurons(2))
        neurons = GlifNeurons(2, mechanisms=(Charging(g=jnp.asarray(g)),))
        current = None

    def loss(g):
        module = eqx.tree_at(lambda m: m.mechanisms[0].g, neurons, g)
        return jnp.mean(module.solve(current, 0.0, 40.0, DT).v)

    gradient = np.asarray(jax.grad(loss)(neurons.mechanisms[0].g))
    assert np.all(np.isfinite(gradient))
    assert np.abs(gradient).max() > 0.0


def test_a_batched_run_carries_the_mechanism_per_sample():
    g = 3.0 * _rheobase_pA(GlifNeurons(2))
    neurons = GlifNeurons(
        2,
        {"sigma": 2.0},
        mechanism="escape",
        mechanisms=(Charging(g=jnp.asarray(g)), Counter()),
    )
    solution = neurons.solve(None, 0.0, DURATION, DT, num_samples=3)

    q = np.asarray(solution.column(neurons.state_columns["q"]))
    assert q.shape == (3, 2, STEPS)
    np.testing.assert_allclose(q[0], q[2], atol=1e-4)  # deterministic, unlike spikes
    counts = np.asarray(solution.num_spikes)
    np.testing.assert_array_equal(np.asarray(solution.yT)[..., -1], counts)
    assert len({tuple(c) for c in counts.tolist()}) > 1


def test_the_mechanism_states_and_inputs_are_checked():
    # structure is static and per instance, so a mechanism can be given its names
    with pytest.raises(ValueError, match="extracellular"):
        GlifNeurons(1, mechanisms=(Inert(inputs=("extracellular",)),))
    with pytest.raises(ValueError, match="theta_s"):
        GlifNeurons(1, mechanisms=(Inert(states=("theta_s",)),))
    with pytest.raises(ValueError, match="named twice"):
        GlifNeurons(1, mechanisms=(Inert(), Inert()))
    with pytest.raises(TypeError):
        GlifNeurons(1, mechanisms=(object(),))

    lit = GlifNeurons(2, mechanisms=(Light(gain=jnp.ones(2)),))
    assert lit.input_mode == ("current", "irradiance")
    assert GlifNeurons(2).input_mode == "current"
    with pytest.raises(ValueError, match="conductance"):
        lit.solve(None, 0.0, 10.0, DT, inputs={"conductance": jnp.ones((10, 2))})
    with pytest.raises(ValueError, match="irradiance"):
        GlifNeurons(2).solve(None, 0.0, 10.0, DT, inputs={"irradiance": jnp.ones(2)})


class LitGLIF(GLIF):
    """The excitatory cells carry a light-gated current"""

    def diffrax_mechanisms(self, env):
        gain = np.where(cell_populations(env) == "EXC", 80.0, 0.0)
        return (Light(gain=jnp.asarray(gain)),)


class ChargedGLIF(GLIF):
    def diffrax_mechanisms(self, env):
        return (Charging(g=jnp.zeros(len(env.simulated_gids(everywhere=True)))),)

    def recordable_states(self):
        return (*RECORDABLE_STATES, *Charging.states)


_env = pytest.mark.skipif(
    backend() != "diffrax", reason="GLIF runs on the diffrax backend"
)


def _system():
    from livn.system import Monolayer

    return Monolayer(
        total_cells=12,
        area_kwargs={"x_range": (0.0, 1400.0), "y_range": (0.0, 1400.0)},
        connectivity={"sigma": 300.0, "mean_degree": {"default": 0.0}},
        seed=4242,
    )


def _light_and_current(gids, steps, n):
    light = np.zeros((steps, n), dtype=np.float32)
    light[100:300] = 1.0
    current = np.full((steps, n), 0.02, dtype=np.float32)
    return light, current


@_env
def test_an_override_addresses_populations():
    from livn.env import Env

    env = Env(_system(), model=LitGLIF(level=1)).init()
    populations = cell_populations(env)

    assert set(populations) == {"EXC", "INH"}
    for population, cells in env.cells.items():
        for cell in cells.values():
            assert populations[cell.index] == population
    gain = np.asarray(env.module.mechanisms[0].gain)
    np.testing.assert_array_equal(gain > 0, populations == "EXC")


@_env
def test_an_irradiance_reaches_the_mechanism_through_env_run():
    from livn.env import Env
    from livn.stimulus import Stimulus

    duration, dt = 40.0, 0.1
    lit = Env(_system(), model=LitGLIF(level=1)).init().record_voltage()
    plain = Env(_system(), model=GLIF(level=1)).init().record_voltage()
    assert lit.module.input_mode == ("current", "irradiance")

    gids = np.asarray(lit.simulated_gids()).astype(int)
    n, steps = len(gids), round(duration / dt)
    light, current = _light_and_current(gids, steps, n)
    gain = np.asarray(lit.module.mechanisms[0].gain)

    composed = Stimulus.from_current(
        current, dt=dt, gids=gids
    ) + Stimulus.from_irradiance(light, dt=dt, gids=gids)
    run = lit.run(duration, composed, dt=dt)

    # the same current, handed to the cell as one: gain pA per mW/mm2, in nA
    equivalent = current + light * gain[None, :] / 1e3
    reference = plain.run(duration, Stimulus.from_current(equivalent, dt=dt, gids=gids))

    v = np.asarray(run.voltage)
    np.testing.assert_allclose(v, np.asarray(reference.voltage), atol=1e-3)
    exc = gain > 0
    assert exc.any() and (~exc).any()
    lifted = v[:, 250] - v[:, 90]
    assert lifted[exc].min() > lifted[~exc].max() + 1.0

    alone = lit.run(duration, Stimulus.from_irradiance(light, dt=dt, gids=gids), dt=dt)
    assert (
        np.asarray(alone.voltage)[exc, 250].min()
        > np.asarray(alone.voltage)[~exc, 250].max()
    )


@_env
def test_a_mode_no_mechanism_reads_is_refused_by_the_env():
    from livn.env import Env
    from livn.stimulus import Stimulus

    duration, dt = 10.0, 0.1
    plain = Env(_system(), model=GLIF(level=1)).init()
    lit = Env(_system(), model=LitGLIF(level=1)).init()
    gids = np.asarray(plain.simulated_gids()).astype(int)
    n, steps = len(gids), round(duration / dt)
    light, current = _light_and_current(gids, steps, n)

    irradiance = Stimulus.from_irradiance(light, dt=dt, gids=gids)
    with pytest.raises(ValueError, match="irradiance"):
        plain.run(duration, irradiance, dt=dt)
    with pytest.raises(ValueError, match="irradiance"):
        plain.run(duration, Stimulus.from_current(current, dt=dt) + irradiance, dt=dt)
    with pytest.raises(ValueError, match="conductance"):
        lit.run(duration, Stimulus.from_conductance(current, dt=dt, gids=gids), dt=dt)
    with pytest.raises(ValueError, match="extracellular"):
        lit.run(duration, Stimulus.from_extracellular(current, dt=dt), dt=dt)


@_env
def test_a_mechanism_state_records_through_the_env():
    from livn.env import Env

    env = Env(_system(), model=ChargedGLIF(level=1)).init().record("q")
    run = env.run(20.0, dt=0.1)

    q = np.asarray(run["q"].values)
    assert q.shape == (env.num_cells, 201)
    np.testing.assert_allclose(q[:, -1], 1.0 - np.exp(-0.5), atol=1e-3)


def _threshold_case(coupled, level, mechanism_of):
    if coupled:
        plain = _network(level=level)
        return _network((mechanism_of(plain),), level=level), _drive_first()
    plain = GlifNeurons(2, level=level)
    neurons = GlifNeurons(2, level=level, mechanisms=(mechanism_of(plain),))
    return neurons, _driven(plain)


@pytest.mark.parametrize("level", [5, 4, 1])
@pytest.mark.parametrize("coupled", [False, True])
def test_a_mechanism_reads_the_cells_own_threshold(coupled, level):
    dt = 0.01
    neurons, current = _threshold_case(
        coupled, level, lambda plain: ThresholdIntegral(base=plain.V_threshold_base)
    )
    current = jnp.repeat(current, round(DT / dt), axis=0)
    solution = neurons.solve(current, 0.0, DURATION, dt, stimulus_dt=dt)
    assert int(solution.num_spikes[0]) >= 3, "the threshold has to jump to test this"

    ts = np.asarray(solution.ts)
    theta_s, theta_v = np.asarray(solution.theta_s), np.asarray(solution.theta_v)
    shift = (
        np.asarray(solution.threshold) - np.asarray(neurons.V_threshold_base)[:, None]
    )
    np.testing.assert_allclose(shift, theta_s + theta_v, atol=1e-5)
    if level < 5:
        assert np.abs(theta_v).max() == 0.0
    if level == 1:
        assert np.abs(theta_s).max() == 0.0
    else:
        assert theta_s.max() > 1.0

    jumps = np.asarray(solution.spike_times)
    if not coupled:
        jumps = jumps + np.asarray(neurons.t_ref)[:, None]  # the jump ends the cut
    near = np.abs(ts[None, :, None] - jumps[:, None, :]).min(axis=-1) < 0.25
    smooth = ~(near[:, 1:] | near[:, :-1])
    assert smooth.mean() > 0.8

    columns = neurons.state_columns
    for name, trace in [
        ("int_theta_s", theta_s),
        ("int_theta_v", theta_v),
        ("int_shift", theta_s + theta_v),
    ]:
        seen = np.asarray(solution.column(columns[name]))
        rate = np.diff(seen, axis=1) / np.diff(ts)
        expected = (trace[:, 1:] + trace[:, :-1]) / 2
        np.testing.assert_allclose(
            rate[smooth], expected[smooth], rtol=1e-2, atol=0.1, err_msg=name
        )


@pytest.mark.parametrize("level", [5, 1])
@pytest.mark.parametrize("coupled", [False, True])
def test_on_spike_reads_the_threshold_it_crossed(coupled, level):
    dt = 0.01
    neurons, current = _threshold_case(coupled, level, lambda plain: ThresholdAtSpike())
    current = jnp.repeat(current, round(DT / dt), axis=0)
    solution = neurons.solve(current, 0.0, DURATION, dt, stimulus_dt=dt)

    spikes = np.asarray(solution.spike_times)
    crossed = np.asarray(solution.yT)[..., neurons.state_columns["crossed"]]
    threshold = np.asarray(solution.threshold)
    ts = np.asarray(solution.ts)
    base = np.asarray(neurons.V_threshold_base)
    for cell in range(spikes.shape[0]):
        times = spikes[cell][np.isfinite(spikes[cell])]
        if len(times) == 0:
            assert crossed[cell] == 0.0
            continue
        # the hard threshold is crossed where V meets it, so read it just before
        before = np.interp(times[-1] - dt, ts, threshold[cell])
        np.testing.assert_allclose(crossed[cell], before, atol=0.05)
        if level == 1:
            assert crossed[cell] == pytest.approx(base[cell])
        elif len(times) > 1:
            assert crossed[cell] > base[cell] + 0.5
