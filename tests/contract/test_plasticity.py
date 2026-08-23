from collections import defaultdict

import numpy as np
import pytest

from livn.backend import backend
from testing import livn_test_env

pytestmark = pytest.mark.needs("plasticity")

_is_neuron = backend() == "neuron"

LEARNING_MS = 200


def _plasticity_config():
    if _is_neuron:
        return {"A_ltp": 0.01, "A_ltd": 0.005}
    return {
        "A_ltp_exc": 0.01,
        "A_ltd_exc": 0.005,
        "A_ltp_inh": 0.01,
        "A_ltd_inh": 0.005,
    }


def _noise_config():
    if _is_neuron:
        return {
            "g_e0": 3.0,
            "g_i0": 1.0,
            "std_e": 1.0,
            "std_i": 0.5,
            "tau_e": 10.0,
            "tau_i": 10.0,
        }
    return {
        "g_e0": 0.5,
        "g_i0": 0.3,
        "std_e": 0.15,
        "std_i": 0.1,
        "tau_e": 10.0,
        "tau_i": 10.0,
    }


def _plastic_sites(env):
    if _is_neuron:
        for gid, syn_id, name, pp in env._iter_stdp_point_processes():
            yield f"{gid}/{syn_id}/{name}", pp
    else:
        for key, synapses in env._iter_stdp_synapses():
            yield str(key), synapses


def _plasticity_is_on(env) -> list[bool]:
    if _is_neuron:
        return [bool(pp.plasticity_on) for _, pp in _plastic_sites(env)]
    return [
        bool(flag)
        for population in env._populations.values()
        for flag in np.asarray(population.plasticity_on)
    ]


def _by_post_neuron(weights: dict) -> dict:
    grouped = defaultdict(list)
    for key, weight in weights.items():
        if _is_neuron:
            gid, _syn_id, _mech = key
            grouped[gid].append(weight)
        else:
            post, _pre, _mech, _i, j = key
            grouped[(post, j)].append(weight)
    return grouped


def _settle(env):
    if _is_neuron:
        from neuron import h

        h.finitialize(-75)


@pytest.fixture
def resting_env():
    env = livn_test_env().init()
    env.apply_model_defaults()
    yield env
    env.close()


@pytest.fixture
def learning_env():
    env = livn_test_env().init()
    env.apply_model_defaults()
    env.enable_plasticity(config=_plasticity_config())
    yield env
    env.close()


def test_the_backend_has_plastic_synapses_to_offer(resting_env):
    sites = list(_plastic_sites(resting_env))
    assert sites, (
        "the backend declares Capability.PLASTICITY but has no plastic synapses"
    )


def test_plasticity_is_off_until_it_is_asked_for(resting_env):
    flags = _plasticity_is_on(resting_env)
    assert flags, "nothing reported whether it was learning"
    assert not any(flags), (
        f"{sum(flags)} of {len(flags)} synapses were already learning"
    )


def test_weights_start_at_one(resting_env):
    _settle(resting_env)

    weights = resting_env.get_weights()
    assert weights, "no weights were reported"
    for key, value in weights.items():
        assert abs(value - 1.0) < 1e-6, f"{key} started at {value}, not 1.0"


def test_a_weight_is_addressed_by_the_synapse_it_belongs_to(resting_env):
    weights = resting_env.get_weights()
    key, value = next(iter(weights.items()))

    assert isinstance(value, float)
    if _is_neuron:
        gid, syn_id, mechanism = key
        assert isinstance(gid, int) and isinstance(syn_id, int)
        assert isinstance(mechanism, str)
    else:
        post, pre, mechanism, i, j = key
        assert isinstance(post, str) and isinstance(pre, str)
        assert isinstance(mechanism, str)
        assert isinstance(i, int) and isinstance(j, int)


def test_enabling_and_disabling_reaches_every_synapse():
    env = livn_test_env().init()
    try:
        env.apply_model_defaults()

        env.enable_plasticity()
        assert env._plasticity_enabled is True
        flags = _plasticity_is_on(env)
        assert flags and all(flags), (
            f"only {sum(flags)} of {len(flags)} synapses started learning"
        )

        env.disable_plasticity()
        assert env._plasticity_enabled is False
        flags = _plasticity_is_on(env)
        assert not any(flags), f"{sum(flags)} synapses were still learning"
    finally:
        env.close()


def test_a_custom_config_reaches_the_synapses():
    env = livn_test_env().init()
    try:
        env.apply_model_defaults()

        if _is_neuron:
            env.enable_plasticity(
                config={"A_ltp": 0.005, "A_ltd": 0.002, "theta_ltp": -40.0}
            )
            _label, pp = next(iter(_plastic_sites(env)))
            assert abs(pp.A_ltp - 0.005) < 1e-9
            assert abs(pp.A_ltd - 0.002) < 1e-9
            assert abs(pp.theta_ltp - (-40.0)) < 1e-9
        else:
            env.enable_plasticity(config={"A_ltp_exc": 0.005, "A_ltd_exc": 0.002})
            population = env._populations["EXC"]
            assert abs(float(population.A_ltp_exc[0]) - 0.005) < 1e-9
            assert abs(float(population.A_ltd_exc[0]) - 0.002) < 1e-9
    finally:
        env.close()


@pytest.mark.skipif(
    not _is_neuron, reason="per-population routing is addressed per point process"
)
def test_a_per_population_config_goes_to_the_right_population():
    from livn.models.rcsd import ReducedCalciumSomaDendrite

    env = livn_test_env(model=ReducedCalciumSomaDendrite()).init()
    try:
        env.apply_model_defaults()
        env.enable_plasticity(
            config={
                "EXC": {"A_ltp": 0.01, "A_ltd": 0.005},
                "INH": {"A_ltp": 0.02, "A_ltd": 0.008},
            }
        )

        seen = set()
        for _gid, _syn_id, name, pp in env._iter_stdp_point_processes():
            assert pp.plasticity_on == 1
            inhibitory = "Inh" in name
            expected = 0.02 if inhibitory else 0.01
            assert abs(pp.A_ltp - expected) < 1e-9, f"{name} learns at {pp.A_ltp}"
            seen.add("INH" if inhibitory else "EXC")

        assert seen == {"EXC", "INH"}, f"only reached {seen}"
    finally:
        env.close()


def test_weights_move_when_plasticity_is_on_and_not_when_it_is_off():

    def run_and_diff(plastic: bool):
        env = livn_test_env().init()
        try:
            env.apply_model_defaults()
            if plastic:
                env.enable_plasticity(config=_plasticity_config())
            _settle(env)

            before = env.get_weights()
            env.set_noise(_noise_config())
            env.run(LEARNING_MS)
            after = env.get_weights()

            shared = set(before) & set(after)
            assert shared, "no synapse survived the run to be compared"
            return {key: abs(before[key] - after[key]) for key in shared}
        finally:
            env.close()

    moved = run_and_diff(plastic=True)
    assert sum(delta > 1e-9 for delta in moved.values()) > 0, (
        f"no weight changed across {len(moved)} plastic synapses"
    )

    frozen = run_and_diff(plastic=False)
    for key, delta in frozen.items():
        assert delta < 1e-12, f"{key} moved by {delta} with plasticity off"


def test_recording_weights_samples_the_whole_run(learning_env):
    learning_env.record_weights(dt=1.0)

    learning_env.set_noise(_noise_config())
    learning_env.run(50)

    if _is_neuron:
        recordings = {
            key: np.asarray(vector.as_numpy())
            for key, vector in learning_env.w_recs.items()
        }
    else:
        recordings = {
            key: np.asarray(monitor.w_plastic[:])
            for key, monitor in learning_env._weight_monitors.items()
        }

    assert recordings, "recording was enabled but nothing was recorded"
    for key, samples in recordings.items():
        count = samples.shape[-1] if samples.ndim > 1 else samples.size
        assert count >= 40, f"{key} has {count} samples of an expected ~50"


def test_normalising_holds_each_neuron_s_total_and_its_ratios(learning_env):
    learning_env.set_noise(_noise_config())
    learning_env.run(LEARNING_MS)

    before = learning_env.get_weights()
    learning_env.normalize_weights()
    after = learning_env.get_weights()

    for neuron, weights in _by_post_neuron(after).items():
        assert abs(sum(weights) - len(weights)) < 0.01, (
            f"{neuron} sums to {sum(weights):.4f}, expected {len(weights)}"
        )

    grouped_before = _by_post_neuron(before)
    for neuron, weights_after in _by_post_neuron(after).items():
        weights_before = grouped_before[neuron]
        if len(weights_after) < 2 or min(weights_before[:2]) < 1e-9:
            continue
        ratio_before = weights_before[0] / weights_before[1]
        ratio_after = weights_after[0] / weights_after[1]
        assert abs(ratio_before - ratio_after) < 0.01, (
            f"{neuron}: ratio moved from {ratio_before:.4f} to {ratio_after:.4f}"
        )
        break


def test_normalising_to_a_target_respects_the_bounds():
    env = livn_test_env().init()
    try:
        env.apply_model_defaults()

        if _is_neuron:
            env.enable_plasticity(
                config={**_plasticity_config(), "w_max": 5.0, "w_min": 0.0001}
            )
        else:
            env.enable_plasticity(config=_plasticity_config())
            for _key, synapses in env._iter_stdp_synapses():
                if len(synapses) > 0:
                    synapses.w_min[:] = 0.0001
                    synapses.w_max[:] = 5.0

        env.set_noise(_noise_config())
        env.run(LEARNING_MS)
        env.normalize_weights(target=3.0)

        for neuron, weights in _by_post_neuron(env.get_weights()).items():
            assert abs(sum(weights) - 3.0) < 0.01, (
                f"{neuron} sums to {sum(weights):.4f}, expected 3.0"
            )

        for key, weight in env.get_weights().items():
            assert 0.0001 - 1e-9 <= weight <= 5.0 + 1e-9, (
                f"{key} normalised to {weight}, outside [0.0001, 5.0]"
            )
    finally:
        env.close()
