import dataclasses
import json
import os

import numpy as np
import pytest

from livn.backend import backend

pytest.importorskip("diffrax")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from livn.models.eventloop import SolverConfig  # noqa: E402
from testing.paths import REPO_ROOT  # noqa: E402
from livn.models.glif import (  # noqa: E402
    ALL_PARAM_NAMES,
    DEFAULT_PARAMS,
    GLIF,
    LEVELS,
    LEVEL_ZEROED,
    LIF_LEVEL,
    LIF_PARAMS,
    MECHANISM_PARAM_NAMES,
    PARAM_NAMES,
    RECORDABLE_STATES,
    GlifNeuronJAX,
    GlifNeurons,
    from_neuron_config,
    from_vector,
    level_of_neuron_config,
    to_neuron_config,
    to_vector,
    trainable_param_names,
)

_LABELS = str(REPO_ROOT / "neurons" / "data" / "glif_labels.json")


def _cohort(limit=None):
    if not os.path.isfile(_LABELS):
        pytest.skip(
            "the Allen GLIF cohort (neurons/data/glif_labels.json) is not present"
        )
    labels = json.loads(open(_LABELS).read())
    out = []
    for cell, record in labels.items():
        for level, model in sorted(record["models"].items()):
            if model.get("params"):
                out.append((cell, int(level), model["params"]))
    return out if limit is None else out[:limit]


def _driven(neurons, amplitude=2.0, duration=60.0, dt=0.1):
    steps = int(round(duration / dt)) + 1
    distance = np.asarray(neurons.V_threshold_base - neurons.E_L)
    current = amplitude * distance * np.asarray(neurons.g_L) / 1e3
    return jnp.asarray(np.tile(current, (steps, 1)))


@pytest.mark.parametrize("level", LEVELS)
def test_a_level_only_zeroes_parameters(level):
    neurons = GlifNeurons(2, level=level)

    for name in PARAM_NAMES:
        value = np.asarray(getattr(neurons, name))
        if name in LEVEL_ZEROED[level]:
            np.testing.assert_array_equal(value, np.zeros(2), err_msg=name)
        else:
            np.testing.assert_allclose(value, DEFAULT_PARAMS[name], err_msg=name)

    assert set(trainable_param_names(level)) == set(PARAM_NAMES) - set(
        LEVEL_ZEROED[level]
    )


def test_levels_are_not_separate_implementations():
    lif = GLIF(level=1).diffrax_module(_FakeEnv(4))
    glif5 = GLIF(level=5).diffrax_module(_FakeEnv(4))

    assert jax.tree_util.tree_structure(lif) == jax.tree_util.tree_structure(glif5)
    static_lif = eqx.partition(lif, eqx.is_array)[1]
    static_glif5 = eqx.partition(glif5, eqx.is_array)[1]
    assert repr(static_lif) == repr(static_glif5)
    assert not any("level" in name for name in dir(lif) if not name.startswith("_"))


def test_the_leaky_integrator_is_a_point_in_the_space():
    model = GLIF.leaky_integrate_and_fire()
    assert model.level == LIF_LEVEL

    neurons = model.diffrax_module(_FakeEnv(1))
    for name, value in LIF_PARAMS.items():
        assert float(getattr(neurons, name)[0]) == value

    duration, dt = 60.0, 0.1
    solution = neurons.solve(
        _driven(neurons, duration=duration, dt=dt), 0.0, duration, dt
    )

    assert int(solution.num_spikes[0]) > 1
    assert float(jnp.max(jnp.abs(solution.theta_s))) == 0.0
    assert float(jnp.max(jnp.abs(solution.ascs))) == 0.0
    assert float(jnp.min(solution.v)) < LIF_PARAMS["E_L"] - 4.0


def test_a_level_restricts_the_dynamics_it_excludes():
    stimulus_dt, duration = 0.1, 60.0
    lif = GlifNeurons(1, level=1)
    solution = lif.solve(_driven(lif), 0.0, duration, stimulus_dt)

    assert float(jnp.max(jnp.abs(solution.theta_s))) == 0.0
    assert float(jnp.max(jnp.abs(solution.theta_v))) == 0.0
    assert float(jnp.max(jnp.abs(solution.ascs))) == 0.0
    assert int(solution.num_spikes[0]) > 1


class _FakeEnv:
    def __init__(self, n, connectivity=None):
        self._n = n
        self.system = _FakeSystem(
            np.zeros((n, n)) if connectivity is None else np.asarray(connectivity)
        )
        self._weights = None

    def active_gids(self):
        return np.arange(self._n)

    def simulated_gids(self, everywhere=False):
        return np.arange(self._n)


class _FakeSystem:
    def __init__(self, connectivity):
        self._connectivity = connectivity

    def connectivity_matrix(self, weights=None, seed=123, gids=None):
        if gids is None:
            return self._connectivity
        index = np.asarray(gids, dtype=int)
        return self._connectivity[np.ix_(index, index)]


def test_a_mechanism_only_masks_which_parameters_matter():
    hard = set(trainable_param_names(5, "hard"))
    escape = set(trainable_param_names(5, "escape"))

    assert escape - hard == set(MECHANISM_PARAM_NAMES)
    assert "sigma_v" in trainable_param_names(5, "hard", diffusion=True)
    assert "tau_syn" in trainable_param_names(5, "hard", coupled=True)
    assert "sigma_v" not in hard and "tau_syn" not in hard


def test_mechanisms_are_not_separate_implementations():
    hard = GLIF(mechanism="hard").diffrax_module(_FakeEnv(4))
    escape = GLIF(mechanism="escape").diffrax_module(_FakeEnv(4))

    assert [f.name for f in dataclasses.fields(hard)] == [
        f.name for f in dataclasses.fields(escape)
    ]
    assert set(hard.params()) == set(escape.params()) == set(ALL_PARAM_NAMES)
    assert escape.layout.size == hard.layout.size + 1
    assert hard.layout.s < 0 and escape.layout.s >= 0


def test_a_narrow_escape_converges_on_the_hard_threshold():
    duration, dt = 60.0, 0.1
    hard = GlifNeurons(1)
    stimulus = _driven(hard, duration=duration, dt=dt)

    expected = np.asarray(hard.solve(stimulus, 0.0, duration, dt).spike_times[0])
    expected = expected[np.isfinite(expected)]
    assert len(expected) > 3

    residuals = []
    for sigma in (1.0, 0.1, 0.02):
        escape = GlifNeurons(1, {"sigma": sigma}, mechanism="escape")
        got = np.asarray(
            escape.solve(stimulus, 0.0, duration, dt, key=jr.PRNGKey(0)).spike_times[0]
        )
        got = got[np.isfinite(got)]

        assert len(got) == len(expected), sigma
        drift = got - expected
        assert np.all(drift > -0.01), sigma
        residuals.append(float(np.max(drift)))

    assert residuals == sorted(residuals, reverse=True), residuals
    assert residuals[-1] < 0.2


def test_a_wide_escape_is_stochastic_and_the_draw_is_the_key():
    duration, dt = 60.0, 0.1
    escape = GlifNeurons(1, {"sigma": 2.0}, mechanism="escape")
    stimulus = _driven(escape, duration=duration, dt=dt)

    def first(seed):
        solution = escape.solve(stimulus, 0.0, duration, dt, key=jr.PRNGKey(seed))
        return float(solution.spike_times[0][0])

    assert first(0) != first(1)
    assert first(0) == first(0)


ESCAPE_PARAMS = ["sigma", "tau_s", "alpha"]


@pytest.fixture(scope="module")
def escape_gradients():
    duration, dt = 50.0, 0.1
    neurons = GlifNeurons(1, {"sigma": 1.0}, mechanism="escape")
    stimulus = _driven(neurons, duration=duration, dt=dt)

    def loss(values):
        module = eqx.tree_at(
            lambda m: [getattr(m, name) for name in ESCAPE_PARAMS],
            neurons,
            [values[name] for name in ESCAPE_PARAMS],
        )
        spikes = module.solve(
            stimulus, 0.0, duration, dt, key=jr.PRNGKey(0)
        ).spike_times
        return jnp.sum(jnp.where(jnp.isfinite(spikes), spikes, 0.0))

    start = {name: getattr(neurons, name) for name in ESCAPE_PARAMS}
    return loss, start, jax.grad(loss)(start)


@pytest.mark.parametrize("name", ESCAPE_PARAMS)
def test_the_escape_parameters_carry_gradients(name, escape_gradients):
    loss, start, gradients = escape_gradients
    gradient = float(gradients[name][0])

    step = 1e-3 * max(1.0, abs(float(start[name][0])))
    forward = dict(start, **{name: start[name] + step})
    backward = dict(start, **{name: start[name] - step})
    finite_difference = (float(loss(forward)) - float(loss(backward))) / (2 * step)

    assert gradient != 0.0, f"{name} carries no gradient"
    np.testing.assert_allclose(gradient, finite_difference, rtol=0.1)


def _chain(n=3, weight=800.0):
    connectivity = np.zeros((n, n), dtype=np.float32)
    for pre in range(n - 1):
        connectivity[pre, pre + 1] = weight
    return connectivity


def _chain_module(mechanism="hard", n=3, **kwargs):
    return GLIF(
        mechanism=mechanism, config=SolverConfig(max_rate=1.0), **kwargs
    ).diffrax_module(_FakeEnv(n, _chain(n)))


def test_an_unconnected_system_stays_off_the_shared_loop():
    assert GLIF().diffrax_module(_FakeEnv(4)).network is None
    assert GLIF().diffrax_module(_FakeEnv(4)).layout.size == 5

    coupled = _chain_module()
    assert coupled.network is not None
    assert coupled.layout.coupled


@pytest.mark.parametrize("mechanism", ["hard", "escape"])
def test_a_spike_propagates_along_the_chain(mechanism):
    module = _chain_module(mechanism)
    duration, dt = 60.0, 0.1
    stimulus = jnp.zeros((int(round(duration / dt)) + 1, 3)).at[:, 0].set(0.3)

    solution = module.solve(stimulus, 0.0, duration, dt, key=jr.PRNGKey(0))

    spikes = [
        np.asarray(row)[np.isfinite(np.asarray(row))] for row in solution.spike_times
    ]
    assert all(len(train) > 1 for train in spikes)
    assert spikes[0][0] < spikes[1][0] < spikes[2][0]


def test_the_shared_loop_reproduces_the_independent_one():
    duration, dt = 60.0, 0.1
    module = _chain_module()
    stimulus = jnp.zeros((int(round(duration / dt)) + 1, 3)).at[:, 0].set(0.3)

    coupled = np.asarray(module.solve(stimulus, 0.0, duration, dt).spike_times[0])

    alone = GlifNeurons(1, config=SolverConfig(max_rate=1.0))
    independent = np.asarray(
        alone.solve(stimulus[:, :1], 0.0, duration, dt).spike_times[0]
    )

    coupled = coupled[np.isfinite(coupled)]
    independent = independent[np.isfinite(independent)]
    assert len(coupled) > 3
    np.testing.assert_allclose(coupled, independent, atol=0.01)


def test_the_sign_of_a_weight_is_the_sign_of_its_effect():
    duration, dt = 60.0, 0.1
    stimulus = jnp.full((int(round(duration / dt)) + 1, 2), 0.3).at[:, 1].set(0.18)

    def spikes_of_cell_1(weight):
        connectivity = np.zeros((2, 2), dtype=np.float32)
        connectivity[0, 1] = weight
        module = GLIF(config=SolverConfig(max_rate=1.0)).diffrax_module(
            _FakeEnv(2, connectivity)
        )
        train = np.asarray(module.solve(stimulus, 0.0, duration, dt).spike_times[1])
        return train[np.isfinite(train)]

    alone = spikes_of_cell_1(0.0)
    excited = spikes_of_cell_1(200.0)
    inhibited = spikes_of_cell_1(-60.0)

    assert len(excited) > len(alone) > len(inhibited) > 0
    assert excited[0] < alone[0] < inhibited[0]


def test_read_out_neurons_receive_but_do_not_fire():
    module = _chain_module(read_out_neurons=[2])
    duration, dt = 60.0, 0.1
    stimulus = jnp.zeros((int(round(duration / dt)) + 1, 3)).at[:, 0].set(0.3)

    solution = module.solve(stimulus, 0.0, duration, dt)

    assert module.spike_cells == (0, 1)
    assert int(solution.num_spikes[2]) == 0
    assert float(jnp.max(solution.v[2])) > float(module.E_L[2]) + 1.0

    with pytest.raises(ValueError, match="network concept"):
        GlifNeurons(3, read_out_neurons=[2])


def test_a_network_run_is_differentiable_through_its_weights():
    duration, dt = 40.0, 0.1
    module = _chain_module()
    stimulus = jnp.zeros((int(round(duration / dt)) + 1, 3)).at[:, 0].set(0.3)

    def loss(network):
        coupled = eqx.tree_at(lambda m: m.network, module, network)
        return jnp.mean(coupled.solve(stimulus, 0.0, duration, dt).v[2] ** 2)

    gradient = np.asarray(jax.grad(loss)(module.network))

    assert np.all(np.isfinite(gradient))
    assert gradient[1, 2] != 0.0


def test_samples_are_independent_realisations_of_the_same_run():
    duration, dt = 60.0, 0.1
    module = GlifNeurons(2, {"sigma": 2.0}, mechanism="escape")
    stimulus = _driven(module, duration=duration, dt=dt)

    batched = module.solve(
        stimulus, 0.0, duration, dt, key=jr.PRNGKey(0), num_samples=4
    )

    assert batched.v.shape == (4, 2, int(round(duration / dt)) + 1)
    assert batched.yT.shape == (4, 2, module.layout.size)
    first = np.asarray(batched.spike_times[:, 0, 0])
    assert len(set(first.tolist())) > 1, "samples share a spike train"

    single = module.solve(stimulus, 0.0, duration, dt, key=jr.PRNGKey(0))
    assert single.v.shape == (2, int(round(duration / dt)) + 1)


def test_a_batched_run_keeps_its_sample_axis():
    duration, dt = 40.0, 0.1
    module = GlifNeurons(2, {"sigma": 2.0}, mechanism="escape")
    stimulus = _driven(module, duration=duration, dt=dt)

    rows, times, ids, voltage, _, _, yT, _ = module.run(
        stimulus,
        t0=0.0,
        t1=duration,
        dt=dt,
        key=jr.PRNGKey(0),
        num_samples=3,
    )

    steps = int(round(duration / dt)) + 1
    assert voltage.shape == (3, 2, steps)
    np.testing.assert_array_equal(np.asarray(ids), [0, 1])
    np.testing.assert_array_equal(np.asarray(rows), [0, 1])
    assert times.shape[:2] == (3, 2)
    assert int(jnp.sum(jnp.isfinite(times))) > 0


def test_a_batched_run_no_longer_has_to_refuse_the_ragged_form():
    module = GlifNeurons(2)
    stimulus = _driven(module, duration=20.0)

    rows, times, *_ = module.run(stimulus, t0=0.0, t1=20.0, dt=0.1, num_samples=2)

    assert times.ndim == 3
    np.testing.assert_array_equal(np.asarray(rows), [0, 1])


def test_a_batched_run_is_differentiable():
    duration, dt = 30.0, 0.1
    module = GlifNeurons(1, {"sigma": 2.0}, mechanism="escape")
    stimulus = _driven(module, duration=duration, dt=dt)

    def loss(tau_m):
        batched = eqx.tree_at(lambda m: m.tau_m, module, tau_m)
        return jnp.mean(
            batched.solve(
                stimulus, 0.0, duration, dt, key=jr.PRNGKey(0), num_samples=3
            ).v
            ** 2
        )

    gradient = np.asarray(jax.grad(loss)(module.tau_m))

    assert np.all(np.isfinite(gradient)) and np.all(gradient != 0.0)


def test_the_diffusion_term_moves_the_trace_and_only_when_asked():
    duration, dt = 40.0, 0.1
    clean = GlifNeurons(1)
    stimulus = _driven(clean, duration=duration, dt=dt)
    quiet = clean.solve(stimulus, 0.0, duration, dt)

    noisy = GlifNeurons(1, {"sigma_v": 1.0}, diffusion=True)
    loud = noisy.solve(stimulus, 0.0, duration, dt, key=jr.PRNGKey(1))

    assert float(jnp.max(jnp.abs(loud.v - quiet.v))) > 1.0
    unused = GlifNeurons(1, {"sigma_v": 1.0}).solve(stimulus, 0.0, duration, dt)
    np.testing.assert_allclose(np.asarray(unused.v), np.asarray(quiet.v))


def test_noise_reaches_the_module_through_run():
    duration, dt = 40.0, 0.1
    module = GlifNeurons(1)
    stimulus = _driven(module, duration=duration, dt=dt)

    def voltage(noise):
        return module.run(
            stimulus,
            noise=noise,
            t0=0.0,
            t1=duration,
            dt=dt,
            key=jr.PRNGKey(2),
            record={"voltage"},
        )[3]

    assert float(jnp.max(jnp.abs(voltage({"sigma_v": 1.0}) - voltage(None)))) > 1.0

    with pytest.raises(ValueError, match="sigma_v"):
        voltage({"level": 1.0})


def test_the_neuron_config_round_trip_is_the_identity_on_the_cohort():
    cohort = _cohort()
    assert len(cohort) > 100

    for cell, level, config in cohort:
        params = from_neuron_config(config)
        again = from_neuron_config(to_neuron_config(params, template=config))
        for name in PARAM_NAMES:
            np.testing.assert_allclose(
                again[name],
                params[name],
                rtol=1e-9,
                atol=0,
                err_msg=f"{cell} L{level} {name}",
            )


def test_the_round_trip_holds_without_a_template():
    for level in LEVELS:
        params = from_neuron_config(to_neuron_config(dict(DEFAULT_PARAMS), level=level))
        expected = GlifNeurons(1, level=level)
        unexpressible = set(LEVEL_ZEROED[level])
        if level in (1, 3):
            unexpressible.add("theta_decay_rate")
        for name in PARAM_NAMES:
            if name in unexpressible:
                continue
            tolerance = 0.5 * 5e-5 * 1e3 if name == "t_ref" else 0.0
            np.testing.assert_allclose(
                params[name],
                float(getattr(expected, name)[0]),
                rtol=1e-6,
                atol=tolerance,
                err_msg=name,
            )


def test_the_level_is_read_back_from_the_config():
    for cell, level, config in _cohort():
        assert level_of_neuron_config(config) == level, cell


def test_the_flat_vector_adapter_round_trips():
    assert from_vector(to_vector(DEFAULT_PARAMS)) == DEFAULT_PARAMS

    with pytest.raises(ValueError, match="canonical order"):
        from_vector([1.0, 2.0])


def test_a_population_is_built_from_configs():
    configs = [config for _, level, config in _cohort() if level == 5][:3]
    assert configs

    neurons = GlifNeurons.from_neuron_configs(configs)

    assert neurons.n_cells == len(configs)
    expected = [from_neuron_config(config)["tau_m"] for config in configs]
    np.testing.assert_allclose(np.asarray(neurons.tau_m), expected, rtol=1e-6)

    for written, original in zip(neurons.to_neuron_configs(configs), configs):
        for name, value in from_neuron_config(original).items():
            np.testing.assert_allclose(
                from_neuron_config(written)[name], value, rtol=1e-6, err_msg=name
            )


def test_the_output_lands_on_a_uniform_grid_at_the_requested_dt():
    neurons = GlifNeurons(2)
    duration, dt = 40.0, 0.25

    solution = neurons.solve(
        _driven(neurons, duration=duration, dt=dt), 0.0, duration, dt
    )

    ts = np.asarray(solution.ts)
    assert ts.shape == (int(round(duration / dt)) + 1,)
    np.testing.assert_allclose(np.diff(ts), dt, rtol=1e-5)
    assert solution.v.shape == (2, ts.shape[0])
    assert np.all(np.isfinite(np.asarray(solution.v)))


def test_the_refractory_period_holds_the_voltage():
    neurons = GlifNeurons(1, {"t_ref": 5.0})
    duration, dt = 60.0, 0.1

    solution = neurons.solve(
        _driven(neurons, duration=duration, dt=dt), 0.0, duration, dt
    )

    spikes = np.asarray(solution.spike_times[0])
    spikes = spikes[np.isfinite(spikes)]
    assert len(spikes) > 1
    np.testing.assert_array_less(4.9, np.diff(spikes))

    v = np.asarray(solution.v[0])
    ts = np.asarray(solution.ts)
    inside = (ts > spikes[0] + 0.5) & (ts < spikes[0] + 4.5)
    assert np.ptp(v[inside]) < 1e-3


def test_jit_and_vmap_over_cells():
    neurons = GlifNeurons(3)
    stimulus = _driven(neurons)

    @eqx.filter_jit
    def spikes(module, stimulus):
        return jnp.sum(jnp.isfinite(module.solve(stimulus, 0.0, 60.0, 0.1).spike_times))

    counted = int(spikes(neurons, stimulus))
    assert counted > 0

    single = GlifNeurons(1)
    per_cell = jax.vmap(
        lambda tau: jnp.sum(
            jnp.isfinite(
                eqx.tree_at(lambda m: m.tau_m, single, tau[None])
                .solve(stimulus[:, :1], 0.0, 60.0, 0.1)
                .spike_times
            )
        )
    )(jnp.array([5.0, 10.0]))
    assert per_cell.shape == (2,)


def test_a_run_past_the_spike_budget_fails_loudly():
    neurons = GlifNeurons(
        1, {"t_ref": 0.5}, config=SolverConfig(max_rate=0.1, points_per_segment=16)
    )
    stimulus = _driven(neurons, amplitude=6.0, duration=60.0)

    with pytest.raises(Exception, match="event budget exhausted"):
        jax.block_until_ready(neurons.solve(stimulus, 0.0, 60.0, 0.1).spike_times)

    lenient = eqx.tree_at(lambda m: m.tau_m, neurons, neurons.tau_m)
    lenient = GlifNeurons(
        1,
        {"t_ref": 0.5},
        config=SolverConfig(max_rate=0.1, points_per_segment=16, throw=False),
    )
    solution = lenient.solve(stimulus, 0.0, 60.0, 0.1)
    assert bool(solution.saturated[0])


def _spike_time_loss(config=None):
    neurons = GlifNeurons(1, config=config)
    stimulus = _driven(neurons, duration=50.0)

    def loss(params):
        module = eqx.tree_at(
            lambda m: [getattr(m, name) for name in PARAM_NAMES],
            neurons,
            [params[name] for name in PARAM_NAMES],
        )
        spikes = module.solve(stimulus, 0.0, 50.0, 0.1).spike_times
        return jnp.sum(jnp.where(jnp.isfinite(spikes), spikes, 0.0))

    return loss, neurons.params()


def test_every_parameter_the_configuration_trains_carries_a_spike_time_gradient():
    duration, dt = 50.0, 0.1
    neurons = GlifNeurons(1, {"sigma": 1.0}, mechanism="escape")
    stimulus = _driven(neurons, duration=duration, dt=dt)
    names = neurons.trainable_params()
    assert set(MECHANISM_PARAM_NAMES) <= set(names)

    def loss(params):
        module = eqx.tree_at(
            lambda m: [getattr(m, name) for name in names],
            neurons,
            [params[name] for name in names],
        )
        spikes = module.solve(
            stimulus, 0.0, duration, dt, key=jr.PRNGKey(0)
        ).spike_times
        return jnp.sum(jnp.where(jnp.isfinite(spikes), spikes, 0.0))

    start = {name: getattr(neurons, name) for name in names}
    gradients = jax.grad(loss)(start)

    silent = [
        name
        for name in names
        if not np.isfinite(float(gradients[name][0]))
        or float(gradients[name][0]) == 0.0
    ]
    assert not silent, f"no gradient reaches {silent}"


@pytest.fixture(scope="module")
def spike_time_gradients():
    loss, params = _spike_time_loss()
    return loss, params, jax.grad(loss)(params)


@pytest.mark.parametrize("name", ["tau_m", "g_L", "V_threshold_base", "t_ref", "f_v"])
def test_spike_time_gradients_are_nonzero_and_match_finite_differences(
    name, spike_time_gradients
):
    loss, params, gradients = spike_time_gradients
    gradient = float(gradients[name][0])

    step = 1e-3 * max(1.0, abs(float(params[name][0])))
    shifted = dict(params)
    shifted[name] = params[name] + step
    forward = float(loss(shifted))
    shifted[name] = params[name] - step
    backward = float(loss(shifted))
    finite_difference = (forward - backward) / (2 * step)

    assert gradient != 0.0, f"{name} carries no gradient"
    np.testing.assert_allclose(gradient, finite_difference, rtol=0.05)


def test_without_a_root_finder_spike_time_gradients_are_silently_zero():
    loss, params = _spike_time_loss(config=SolverConfig(root_finder=None))

    gradient = jax.grad(loss)(params)

    assert float(gradient["V_threshold_base"][0]) == 0.0


def test_the_voltage_trace_is_differentiable():
    neurons = GlifNeurons(2)
    stimulus = _driven(neurons, duration=30.0)

    def loss(tau_m):
        module = eqx.tree_at(lambda m: m.tau_m, neurons, tau_m)
        return jnp.mean(module.solve(stimulus, 0.0, 30.0, 0.1).v ** 2)

    gradient = np.asarray(jax.grad(loss)(neurons.tau_m))

    assert gradient.shape == (2,)
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient != 0.0)


def test_the_standalone_api_reports_a_dict():
    cell = GlifNeuronJAX(DEFAULT_PARAMS)
    hz = 10_000.0
    stimulus = np.full(600, 250.0)

    out = cell.run(stimulus, hz)

    assert set(out) >= {
        "times",
        "voltage",
        "threshold",
        "theta_s",
        "theta_v",
        "AScurrents",
        "spike_times",
        "n_spikes",
        "duration_s",
    }
    assert out["n_spikes"] == len(out["spike_times"])
    assert out["duration_s"] == pytest.approx(0.06)
    assert np.all(np.asarray(out["spike_times"]) < out["duration_s"])
    assert out["AScurrents"].shape == (out["times"].shape[0], 2)


def test_placing_events_on_step_boundaries_accumulates_lateness():
    cohort = _cohort(limit=6)
    step = 0.1
    slack = 0.02

    hz = 10_000.0
    found = SolverConfig(
        dt_solver=step,
        max_rate=1.0,
        max_segment_span=float("inf"),
        points_per_segment=50,
    )
    quantised = dataclasses.replace(found, root_finder=None)
    for cell, level, config in cohort:
        params = from_neuron_config(config)
        amplitude = 1.8 * params["V_threshold_base"] * params["g_L"]
        stimulus = np.zeros(1000)
        stimulus[100:900] = amplitude

        late = GlifNeuronJAX.from_dict(config, config=quantised).run(
            stimulus, hz, dt=0.1
        )
        exact = GlifNeuronJAX.from_dict(config, config=found).run(stimulus, hz, dt=0.1)

        late_spikes = np.asarray(late["spike_times"]) * 1e3
        exact_spikes = np.asarray(exact["spike_times"]) * 1e3
        assert len(late_spikes) == len(exact_spikes), f"{cell} L{level}"
        if len(late_spikes):
            drift = late_spikes - exact_spikes
            assert np.all(drift > -slack), f"{cell} L{level}"
            assert drift[0] < step + slack, f"{cell} L{level}"
            assert np.all(np.diff(drift) < step + slack), f"{cell} L{level}"


_env = pytest.mark.skipif(
    backend() != "diffrax", reason="GLIF runs on the diffrax backend"
)


def _make_env(n=2, **kwargs):
    from livn.env import Env

    return Env(n, model=GLIF(**kwargs)).init()


@_env
def test_a_run_reports_spikes_and_voltage():
    from livn.stimulus import Stimulus

    env = _make_env(2)
    env.record_spikes()
    env.record_voltage()
    duration, dt = 40.0, 0.1
    current = _driven(env.module, duration=duration, dt=dt)

    run = env.run(duration, Stimulus.from_current(current, dt=dt), dt=dt)

    assert run.voltage.shape == (2, int(round(duration / dt)) + 1)
    assert len(run.spike_times) > 0
    assert np.all(np.asarray(run.spike_times) <= duration)
    assert np.all(np.isin(np.asarray(run.spike_ids), [0, 1]))


@_env
def test_the_cells_registry_exposes_every_parameter():
    env = _make_env(3)

    params = env.cells.get_params()

    assert set(params) == set(ALL_PARAM_NAMES)
    assert params["tau_m"].shape == (3,)

    env = env.cells.set_params({"tau_m": [5.0, 10.0, 15.0]})
    np.testing.assert_allclose(env.cells.get_params()["tau_m"], [5.0, 10.0, 15.0])
    assert float(env.cells[1].get_params()["tau_m"]) == 10.0


@_env
def test_a_level_preset_reaches_the_env():
    env = _make_env(2, level=1)

    params = env.cells.get_params()

    for name in LEVEL_ZEROED[1]:
        np.testing.assert_array_equal(params[name], np.zeros(2))
    assert set(GLIF(level=1).trainable_params()) == set(PARAM_NAMES) - set(
        LEVEL_ZEROED[1]
    )


def _stimulated(env, duration=60.0, dt=0.1):
    from livn.stimulus import Stimulus

    current = _driven(env.module, duration=duration, dt=dt)
    return env.run(duration, Stimulus.from_current(current, dt=dt), dt=dt)


@_env
def test_the_model_contributes_its_states_to_recordable():
    env = _make_env(2)

    assert set(env.recordable()) == {
        "spikes",
        "voltage",
        "membrane_current",
        "threshold",
        "theta_s",
        "theta_v",
        "AScurrents",
    }

    with pytest.raises(TypeError, match="unexpected keyword argument 'dt'"):
        env.record("threshold", dt=0.5)

    assert GLIF(level=5).recordable_states() == RECORDABLE_STATES


@_env
@pytest.mark.parametrize("level", [2, 4, 5])
def test_theta_is_recoverable_for_the_levels_that_have_one(level):
    env = _make_env(1, level=level)
    env.record_spikes()
    env.record_voltage()
    env.record("threshold")
    env.record("theta_s")
    env.record("theta_v")

    run = _stimulated(env)

    threshold = np.asarray(run["threshold"].values)
    theta_s = np.asarray(run["theta_s"].values)
    theta_v = np.asarray(run["theta_v"].values)
    assert threshold.shape == run.voltage.shape

    assert len(run.spike_times) > 0
    assert float(np.max(theta_s)) > 0.0
    assert (float(np.max(np.abs(theta_v))) > 0.0) == (level == 5)

    np.testing.assert_allclose(
        threshold,
        float(env.cells.get_params()["V_threshold_base"][0]) + theta_s + theta_v,
        rtol=1e-6,
        atol=1e-6,
    )


@_env
def test_a_signal_that_was_not_recorded_allocates_no_buffer():
    env = _make_env(2)
    env.record_spikes()

    run = _stimulated(env)

    assert set(run.channels) == {"spikes"}
    assert run.voltage is None
    assert run["spikes"].times is not None

    solution = env.module.solve(t0=0.0, t1=10.0, dt=0.1, record={"spikes"})
    assert solution.ys.shape[-1] == 0
    assert solution.threshold is None

    voltage_only = env.module.solve(t0=0.0, t1=10.0, dt=0.1, record={"voltage"})
    assert voltage_only.ys.shape[-1] == 1
    with pytest.raises(AttributeError, match="was not sampled"):
        voltage_only.theta_s


@_env
def test_lambda_reconstructs_from_the_recorded_channels():
    sigma = 2.0

    def intensity(v, threshold):
        return jax.nn.softplus((v - threshold) / sigma)

    env = _make_env(2, level=5)
    env.record_voltage()
    env.record("threshold")
    env.record("theta_s")
    env.record("theta_v")

    run = _stimulated(env)

    reconstructed = intensity(run.voltage, run["threshold"].values)

    solution = env.module.solve(
        input_current=_driven(env.module, duration=60.0, dt=0.1),
        t0=0.0,
        t1=60.0,
        dt=0.1,
    )
    inside = intensity(solution.v, solution.threshold)

    np.testing.assert_allclose(
        np.asarray(reconstructed), np.asarray(inside), rtol=1e-5, atol=1e-6
    )

    np.testing.assert_allclose(
        np.asarray(run["threshold"].values),
        np.asarray(env.module.V_threshold_base)[:, None]
        + np.asarray(run["theta_s"].values)
        + np.asarray(run["theta_v"].values),
        rtol=1e-6,
        atol=1e-6,
    )


@_env
def test_the_reconstruction_is_differentiable():
    from livn.stimulus import Stimulus

    sigma = 2.0
    duration, dt = 30.0, 0.1
    env = _make_env(1, level=5)
    env.record_voltage()
    env.record("threshold")
    stimulus = Stimulus.from_current(
        _driven(env.module, duration=duration, dt=dt), dt=dt
    )

    def loss(params):
        run = env.cells.set_params(params).run(duration, stimulus, dt=dt)
        lam = jax.nn.softplus((run.voltage - run["threshold"].values) / sigma)
        return jnp.mean(lam)

    theta = {"V_threshold_base": env.cells.get_params()["V_threshold_base"]}
    gradients = jax.grad(loss)(theta)["V_threshold_base"]

    assert np.all(np.isfinite(np.asarray(gradients)))
    assert float(np.abs(np.asarray(gradients)).max()) > 0.0


def _predefined(name="S1"):
    from livn.system import predefined, resolve

    try:
        return resolve(predefined(name))
    except Exception as error:  # noqa: BLE001 - the system is an optional fixture
        pytest.skip(f"the predefined system {name} is not available ({error})")


@_env
@pytest.mark.parametrize("mechanism", ["hard", "escape"])
def test_a_run_through_env_on_a_connected_system(mechanism):
    from livn.env import Env
    from livn.stimulus import Stimulus

    system = _predefined()
    duration, dt = 40.0, 0.1
    env = Env(
        system,
        model=GLIF(
            level=1,
            mechanism=mechanism,
            config=SolverConfig(max_rate=6.0),
        ),
    )
    env = env.set_weights({"EXC_EXC": 400.0, "EXC_INH": 400.0, "INH_EXC": 400.0}).init()
    env.record_spikes()
    env.record_voltage()

    assert env.module.network is not None, "the system under test is not connected"

    n = env.num_cells
    current = np.full((int(round(duration / dt)) + 1, n), 0.35)
    run = env.run(duration, Stimulus.from_current(current, dt=dt), dt=dt)

    assert run.voltage.shape == (n, int(round(duration / dt)) + 1)
    assert len(run.spike_times) > 0
    assert np.all(np.asarray(run.spike_times) <= duration)
    assert set(np.asarray(run.spike_ids).tolist()) <= set(range(n))


def test_a_forced_run_spikes_exactly_when_told_and_never_on_its_own():
    neurons = GlifNeurons(1, {"t_ref": 2.0}, config=SolverConfig(max_rate=1.0))
    duration, dt = 60.0, 0.1
    stimulus = _driven(neurons, amplitude=3.0, duration=duration, dt=dt)

    free = neurons.solve(stimulus, 0.0, duration, dt)
    own = np.asarray(free.spike_times[0])
    assert np.isfinite(own).sum() > 5

    told = np.array([[12.5, 30.0, 47.25, np.inf, np.inf]])
    forced = neurons.solve(stimulus, 0.0, duration, dt, forced_spikes=told)
    got = np.asarray(forced.spike_times[0])
    got = got[np.isfinite(got)]
    np.testing.assert_allclose(got, [12.5, 30.0, 47.25], atol=0.02)

    v, ts = np.asarray(forced.v[0]), np.asarray(forced.ts)
    before = v[(ts > 11.0) & (ts < 12.4)]
    inside = v[(ts > 12.6) & (ts < 14.4)]
    assert before.max() > inside.max() + 1.0
    assert np.ptp(inside) < 1e-3


def test_a_forced_run_with_no_spikes_is_the_free_subthreshold_trajectory():
    neurons = GlifNeurons(1)
    duration, dt = 30.0, 0.1
    stimulus = _driven(neurons, amplitude=0.5, duration=duration, dt=dt)
    free = neurons.solve(stimulus, 0.0, duration, dt)
    forced = neurons.solve(
        stimulus, 0.0, duration, dt, forced_spikes=np.full((1, 2), np.inf)
    )
    np.testing.assert_allclose(np.asarray(forced.v), np.asarray(free.v), atol=1e-6)
    assert not np.isfinite(np.asarray(forced.spike_times)).any()


def test_a_forced_run_is_differentiable_and_batched_over_cells():
    neurons = GlifNeurons(2, {"t_ref": 2.0}, config=SolverConfig(max_rate=1.0))
    duration, dt = 40.0, 0.1
    stimulus = _driven(neurons, amplitude=3.0, duration=duration, dt=dt)
    told = jnp.array([[10.0, 20.0, jnp.inf], [15.0, jnp.inf, jnp.inf]])

    def voltage_at_end(g_L):
        module = eqx.tree_at(lambda m: m.g_L, neurons, g_L)
        sol = module.solve(stimulus, 0.0, duration, dt, forced_spikes=told)
        return jnp.sum(sol.v[:, -1])

    grad = jax.grad(voltage_at_end)(neurons.g_L)
    assert grad.shape == (2,)
    assert np.all(np.isfinite(np.asarray(grad)))
    sol = neurons.solve(stimulus, 0.0, duration, dt, forced_spikes=told)
    counts = np.isfinite(np.asarray(sol.spike_times)).sum(axis=-1)
    np.testing.assert_array_equal(counts, [2, 1])
