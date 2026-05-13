import os
from collections import defaultdict

import numpy as np
import pytest

from livn.backend import backend
from livn.env import Env

_is_neuron = backend() == "neuron"
_is_brian2 = backend() == "brian2"

pytestmark = [
    pytest.mark.skipif(
        not _is_neuron and not _is_brian2,
        reason="STDP tests require neuron or brian2 backend",
    ),
    pytest.mark.skipif(os.getenv("CI") == "true", reason="STDP tests unstable on CI"),
]

_neuron_only = pytest.mark.skipif(not _is_neuron, reason="neuron only")
_brian2_only = pytest.mark.skipif(not _is_brian2, reason="brian2 only")


def _plasticity_config():
    if _is_brian2:
        return {
            "A_ltp_exc": 0.01,
            "A_ltd_exc": 0.005,
            "A_ltp_inh": 0.01,
            "A_ltd_inh": 0.005,
        }
    return {"A_ltp": 0.01, "A_ltd": 0.005}


def _noise_config():
    if _is_brian2:
        return {
            "g_e0": 0.5,
            "g_i0": 0.3,
            "std_e": 0.15,
            "std_i": 0.1,
            "tau_e": 10.0,
            "tau_i": 10.0,
        }
    return {
        "g_e0": 3.0,
        "g_i0": 1.0,
        "std_e": 1.0,
        "std_i": 0.5,
        "tau_e": 10.0,
        "tau_i": 10.0,
    }


def _group_weights_by_neuron(weights):
    per_neuron = defaultdict(list)
    for key, w in weights.items():
        if _is_brian2:
            post, pre, mech, i_idx, j_idx = key
            per_neuron[(post, j_idx)].append(w)
        else:
            gid, syn_id, mech = key
            per_neuron[gid].append(w)
    return per_neuron


@_neuron_only
class TestNeuronMechanisms:
    def test_stdp_mechanisms_exist(self):
        from neuron import h

        env = Env("systems/graphs/EI1").init()
        try:
            sec = h.Section(name="test_mech_sec")
            pp_ampa = h.StdpLinExp2Syn(sec(0.5))
            pp_nmda = h.StdpLinExp2SynNMDA(sec(0.5))

            assert hasattr(pp_ampa, "plasticity_on")
            assert hasattr(pp_ampa, "w")
            assert hasattr(pp_ampa, "A_ltp")
            assert hasattr(pp_ampa, "A_ltd")
            assert hasattr(pp_ampa, "theta_ltp")
            assert hasattr(pp_ampa, "theta_ltd")
            assert hasattr(pp_ampa, "learning_w")
            assert hasattr(pp_ampa, "learn_int")
            assert hasattr(pp_ampa, "w_max")
            assert hasattr(pp_ampa, "w_min")

            assert hasattr(pp_nmda, "plasticity_on")
            assert hasattr(pp_nmda, "w")
            assert hasattr(pp_nmda, "pnmda")
            assert hasattr(pp_nmda, "mg")

            h.delete_section(sec=sec)
        finally:
            env.close()

    def test_stdp_default_off(self):
        from neuron import h

        env = Env("systems/graphs/EI1").init()
        try:
            sec = h.Section(name="test_default_sec")
            pp = h.StdpLinExp2Syn(sec(0.5))
            assert pp.plasticity_on == 0
            assert pp.w_init == 1.0
            h.finitialize(-65)
            assert pp.w == 1.0
            h.delete_section(sec=sec)
        finally:
            env.close()


@_brian2_only
class TestBrian2Variables:
    def test_conductance_variables_exist(self):
        env = Env("systems/graphs/EI1").init()
        try:
            pop = env._populations["EXC"]
            for var in (
                "A_ampa",
                "B_ampa",
                "A_nmda",
                "B_nmda",
                "A_gaba_a",
                "B_gaba_a",
                "A_gaba_b",
                "B_gaba_b",
            ):
                assert var in pop.variables, f"Missing variable {var}"
        finally:
            env.close()

    def test_stdp_variables_exist(self):
        env = Env("systems/graphs/EI1").init()
        try:
            pop = env._populations["EXC"]
            for var in (
                "plasticity_on",
                "learning_w_exc",
                "learn_int_exc",
                "learning_w_inh",
                "learn_int_inh",
            ):
                assert var in pop.variables, f"Missing variable {var}"
        finally:
            env.close()

    def test_plasticity_default_off(self):
        env = Env("systems/graphs/EI1").init()
        try:
            for pop in env._populations.values():
                assert np.all(np.array(pop.plasticity_on) == 0)
        finally:
            env.close()


@_brian2_only
class TestBrian2Synapses:
    def test_mechanism_synapses_created(self):
        from livn.models.rcsd import ReducedCalciumSomaDendrite

        env = Env(
            "systems/graphs/EI1",
            model=ReducedCalciumSomaDendrite(implicit_inhibition=False),
        ).init()
        try:
            ampa_keys = [k for k in env._synapses if len(k) == 3 and k[2] == "AMPA"]
            assert len(ampa_keys) > 0, "No AMPA synapses found"

            gaba_keys = [k for k in env._synapses if len(k) == 3 and k[2] == "GABA_A"]
            assert len(gaba_keys) > 0, "No GABA_A synapses found"
        finally:
            env.close()

    def test_stdp_synapses_have_w_plastic(self):
        env = Env("systems/graphs/EI1").init()
        try:
            for key, S in env._iter_stdp_synapses():
                assert "w_plastic" in S.variables, f"Missing w_plastic on {key}"
                assert "last_int" in S.variables, f"Missing last_int on {key}"
                assert "w_min" in S.variables, f"Missing w_min on {key}"
                assert "w_max" in S.variables, f"Missing w_max on {key}"
                break
        finally:
            env.close()

    def test_initial_w_plastic_is_one(self):
        env = Env("systems/graphs/EI1").init()
        try:
            for key, S in env._iter_stdp_synapses():
                if len(S) > 0:
                    w = np.array(S.w_plastic[:])
                    assert np.allclose(w, 1.0), (
                        f"Initial w_plastic not 1.0 for {key}: {w[:5]}"
                    )
                    break
        finally:
            env.close()


class TestEnablePlasticity:
    def test_plasticity_enabled_flag(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.enable_plasticity()
            assert env._plasticity_enabled is True

            env.disable_plasticity()
            assert env._plasticity_enabled is False
        finally:
            env.close()

    @_neuron_only
    def test_enable_sets_flag_neuron(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity()

            count = 0
            for gid, syn_id, name, pp in env._iter_stdp_point_processes():
                assert pp.plasticity_on == 1
                count += 1

            assert count > 0, "No STDP synapses found"
        finally:
            env.close()

    @_brian2_only
    def test_enable_sets_flag_brian2(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.enable_plasticity()

            for pop in env._populations.values():
                assert np.all(np.array(pop.plasticity_on) == 1)
        finally:
            env.close()

    @_neuron_only
    def test_disable_clears_flag_neuron(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity()
            env.disable_plasticity()

            for gid, syn_id, name, pp in env._iter_stdp_point_processes():
                assert pp.plasticity_on == 0
        finally:
            env.close()

    @_brian2_only
    def test_disable_clears_flag_brian2(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.enable_plasticity()
            env.disable_plasticity()

            for pop in env._populations.values():
                assert np.all(np.array(pop.plasticity_on) == 0)
        finally:
            env.close()

    @_neuron_only
    def test_custom_config_neuron(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()

            custom = {"A_ltp": 0.005, "A_ltd": 0.002, "theta_ltp": -40.0}
            env.enable_plasticity(config=custom)

            for gid, syn_id, name, pp in env._iter_stdp_point_processes():
                assert abs(pp.A_ltp - 0.005) < 1e-9
                assert abs(pp.A_ltd - 0.002) < 1e-9
                assert abs(pp.theta_ltp - (-40.0)) < 1e-9
                break
        finally:
            env.close()

    @_brian2_only
    def test_custom_config_brian2(self):
        env = Env("systems/graphs/EI1").init()
        try:
            custom = {"A_ltp_exc": 0.005, "A_ltd_exc": 0.002}
            env.enable_plasticity(config=custom)

            pop = env._populations["EXC"]
            assert abs(float(pop.A_ltp_exc[0]) - 0.005) < 1e-9
            assert abs(float(pop.A_ltd_exc[0]) - 0.002) < 1e-9
        finally:
            env.close()

    @_neuron_only
    def test_per_population_config(self):
        from livn.models.rcsd import ReducedCalciumSomaDendrite

        env = Env(
            "systems/graphs/EI1",
            model=ReducedCalciumSomaDendrite(implicit_inhibition=False),
        ).init()
        try:
            env.apply_model_defaults()

            config = {
                "EXC": {"A_ltp": 0.01, "A_ltd": 0.005},
                "INH": {"A_ltp": 0.02, "A_ltd": 0.008},
            }
            env.enable_plasticity(config=config)

            exc_seen = False
            inh_seen = False
            for gid, syn_id, name, pp in env._iter_stdp_point_processes():
                assert pp.plasticity_on == 1
                if "Inh" in name:
                    assert abs(pp.A_ltp - 0.02) < 1e-9
                    assert abs(pp.A_ltd - 0.008) < 1e-9
                    inh_seen = True
                else:
                    assert abs(pp.A_ltp - 0.01) < 1e-9
                    assert abs(pp.A_ltd - 0.005) < 1e-9
                    exc_seen = True

            assert exc_seen, "No excitatory STDP synapses found"
            assert inh_seen, "No inhibitory STDP synapses found"
        finally:
            env.close()

    @_neuron_only
    def test_enable_includes_inhibitory(self):
        from livn.models.rcsd import ReducedCalciumSomaDendrite

        env = Env(
            "systems/graphs/EI1",
            model=ReducedCalciumSomaDendrite(implicit_inhibition=False),
        ).init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity()

            mech_names = set()
            for gid, syn_id, name, pp in env._iter_stdp_point_processes():
                mech_names.add(name)

            assert "StdpLinExp2SynInh" in mech_names, (
                f"Inhibitory STDP mechanism not found, got: {mech_names}"
            )
        finally:
            env.close()

    @_brian2_only
    def test_stdp_synapses_found(self):
        env = Env("systems/graphs/EI1").init()
        try:
            stdp_keys = list(env._iter_stdp_synapses())
            assert len(stdp_keys) > 0, "No STDP synapses found"

            mech_names = {k[2] for k, S in env._iter_stdp_synapses()}
            assert "AMPA" in mech_names or "GABA_A" in mech_names, (
                f"Expected STDP-capable mechanisms, got: {mech_names}"
            )
        finally:
            env.close()


class TestGetWeights:
    @_neuron_only
    def test_get_weights_structure_neuron(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity()

            weights = env.get_weights()

            assert isinstance(weights, dict)
            assert len(weights) > 0

            for key, value in weights.items():
                gid, syn_id, mech_name = key
                assert isinstance(gid, int)
                assert isinstance(syn_id, int)
                assert isinstance(mech_name, str)
                assert isinstance(value, float)
                break
        finally:
            env.close()

    @_brian2_only
    def test_get_weights_structure_brian2(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.enable_plasticity()

            weights = env.get_weights()

            assert isinstance(weights, dict)
            assert len(weights) > 0

            for key, value in weights.items():
                post, pre, mech, i_idx, j_idx = key
                assert isinstance(post, str)
                assert isinstance(pre, str)
                assert isinstance(mech, str)
                assert isinstance(i_idx, int)
                assert isinstance(j_idx, int)
                assert isinstance(value, float)
                break
        finally:
            env.close()

    def test_initial_weights_are_one(self):
        env = Env("systems/graphs/EI1").init()
        try:
            if _is_neuron:
                env.apply_model_defaults()
            env.enable_plasticity()
            if _is_neuron:
                from neuron import h

                h.finitialize(-75)

            weights = env.get_weights()
            for key, value in weights.items():
                assert abs(value - 1.0) < 1e-6, f"Weight {key} = {value}, expected 1.0"
        finally:
            env.close()


class TestWeightDynamics:
    def test_weights_change_with_plasticity(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity(config=_plasticity_config())

            weights_before = env.get_weights()

            env.set_noise(_noise_config())
            env.run(200)

            weights_after = env.get_weights()

            changed = 0
            for key in weights_before:
                if key in weights_after:
                    if abs(weights_before[key] - weights_after[key]) > 1e-9:
                        changed += 1

            assert changed > 0, (
                f"No weights changed out of {len(weights_before)} STDP synapses"
            )
        finally:
            env.close()

    def test_weights_frozen_without_plasticity(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()

            if _is_neuron:
                from neuron import h

                h.finitialize(-75)

            weights_before = env.get_weights()

            env.set_noise(_noise_config())
            env.run(100)

            weights_after = env.get_weights()

            for key in weights_before:
                if key in weights_after:
                    assert abs(weights_before[key] - weights_after[key]) < 1e-12, (
                        f"Weight {key} changed without plasticity: "
                        f"{weights_before[key]} -> {weights_after[key]}"
                    )
        finally:
            env.close()


class TestRecordWeights:
    @_neuron_only
    def test_record_weights_neuron(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity(config={"A_ltp": 0.01, "A_ltd": 0.005})
            env.record_weights(dt=1.0)

            assert len(env.w_recs) > 0

            env.set_noise(_noise_config())
            env.run(50)

            for key, vec in env.w_recs.items():
                arr = np.array(vec.as_numpy())
                assert len(arr) > 0, f"Weight recording for {key} is empty"
                assert len(arr) >= 40
                break
        finally:
            env.close()

    @_brian2_only
    def test_record_weights_brian2(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.enable_plasticity(config={"A_ltp_exc": 0.01, "A_ltd_exc": 0.005})
            env.record_weights(dt=1.0)

            assert len(env._weight_monitors) > 0, "No weight monitors created"

            env.set_noise(_noise_config())
            env.run(50)

            for key, monitor in env._weight_monitors.items():
                arr = np.array(monitor.w_plastic[:])
                assert arr.size > 0, f"Weight recording for {key} is empty"
                # 50 ms at dt=1.0 ms -> ~50 samples
                assert arr.shape[1] >= 40, (
                    f"Expected >= 40 time points, got {arr.shape[1]}"
                )
                break
        finally:
            env.close()


class TestNormalizeWeights:
    def test_normalize_preserves_sum(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity(config=_plasticity_config())

            env.set_noise(_noise_config())
            env.run(200)

            env.normalize_weights()

            per_neuron = _group_weights_by_neuron(env.get_weights())

            for neuron_id, ws in per_neuron.items():
                target = float(len(ws))
                actual_sum = sum(ws)
                assert abs(actual_sum - target) < 0.01, (
                    f"Neuron {neuron_id}: sum={actual_sum:.4f}, expected={target:.1f}"
                )
        finally:
            env.close()

    def test_normalize_custom_target(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity(config=_plasticity_config())

            env.set_noise(_noise_config())
            env.run(200)

            custom_target = 3.0
            env.normalize_weights(target=custom_target)

            per_neuron = _group_weights_by_neuron(env.get_weights())

            for neuron_id, ws in per_neuron.items():
                actual_sum = sum(ws)
                assert abs(actual_sum - custom_target) < 0.01, (
                    f"Neuron {neuron_id}: sum={actual_sum:.4f}, "
                    f"expected={custom_target}"
                )
        finally:
            env.close()

    @_neuron_only
    def test_normalize_preserves_ratios(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()
            env.enable_plasticity(config={"A_ltp": 0.01, "A_ltd": 0.005})

            env.set_noise(_noise_config())
            env.run(200)

            weights_before = env.get_weights()
            env.normalize_weights()
            weights_after = env.get_weights()

            per_neuron_before = defaultdict(dict)
            per_neuron_after = defaultdict(dict)
            for (gid, syn_id, mech), w in weights_before.items():
                per_neuron_before[gid][(syn_id, mech)] = w
            for (gid, syn_id, mech), w in weights_after.items():
                per_neuron_after[gid][(syn_id, mech)] = w

            for gid in per_neuron_before:
                keys = list(per_neuron_before[gid].keys())
                if len(keys) < 2:
                    continue
                k0, k1 = keys[0], keys[1]
                wb0 = per_neuron_before[gid][k0]
                wb1 = per_neuron_before[gid][k1]
                wa0 = per_neuron_after[gid][k0]
                wa1 = per_neuron_after[gid][k1]
                if wb1 > 1e-9 and wa1 > 1e-9:
                    ratio_before = wb0 / wb1
                    ratio_after = wa0 / wa1
                    assert abs(ratio_before - ratio_after) < 0.01, (
                        f"Neuron {gid}: ratio changed from "
                        f"{ratio_before:.4f} to {ratio_after:.4f}"
                    )
                break
        finally:
            env.close()

    def test_normalize_respects_bounds(self):
        env = Env("systems/graphs/EI1").init()
        try:
            env.apply_model_defaults()

            if _is_brian2:
                env.enable_plasticity()
                for key, S in env._iter_stdp_synapses():
                    if len(S) > 0:
                        S.w_min[:] = 0.0001
                        S.w_max[:] = 5.0
            else:
                env.enable_plasticity(
                    config={
                        "A_ltp": 0.01,
                        "A_ltd": 0.005,
                        "w_max": 5.0,
                        "w_min": 0.0001,
                    }
                )

            env.set_noise(_noise_config())
            env.run(200)
            env.normalize_weights()

            for key, w in env.get_weights().items():
                assert w >= 0.0001 - 1e-9, f"{key}: w={w} below w_min"
                assert w <= 5.0 + 1e-9, f"{key}: w={w} above w_max"
        finally:
            env.close()
