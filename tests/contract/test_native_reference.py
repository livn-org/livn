"""The native backend against traces recorded from the NEURON backend."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from livn.backend import backend

pytestmark = pytest.mark.skipif(
    backend() != "native", reason="the traces are the native backend's contract"
)

DATA = os.path.join(os.path.dirname(__file__), "data")
NOISE = {
    "g_e0": 0.5,
    "g_i0": 0.3,
    "std_e": 0.15,
    "std_i": 0.1,
    "tau_e": 10.0,
    "tau_i": 10.0,
}


@pytest.mark.parametrize("population", ["EXC", "INH"])
def test_a_current_step_reproduces_the_neuron_trace(population):
    from livn.env import Env
    from livn.stimulus import Stimulus

    ref = np.load(os.path.join(DATA, f"native_{population.lower()}_current_step.npz"))
    dt = float(ref["dt"][0])
    duration = float(ref["duration_ms"][0])
    steps = round(duration / dt)
    current = np.zeros((steps, 1))
    current[round(float(ref["onset_ms"][0]) / dt) :, 0] = float(ref["amplitude_nA"][0])

    env = Env({population: 1}).init()
    try:
        env.record_spikes()
        env.record_voltage(dt=dt)
        env.record_membrane_current(dt=0.1)
        run = env.run(
            duration, stimulus=Stimulus.from_current(current, dt=dt, gids=np.array([0]))
        )
        assert list(run.voltage_sections) == list(ref["sections"])
        np.testing.assert_allclose(run.spike_times, ref["spikes"], atol=1e-6)
        v, v_ref = np.asarray(run.voltage), ref["v"]
        n = min(v.shape[1], v_ref.shape[1])
        assert np.abs(v[:, :n] - v_ref[:, :n]).max() < 2e-3
        i, i_ref = np.asarray(run.current), ref["current"]
        n = min(i.shape[1], i_ref.shape[1])
        assert np.abs(i[:, :n] - i_ref[:, :n]).max() < 1e-6
        # the pinned resting currents, per section
        cell = env.cells[0]
        pinned = [
            cell.get_params()[f"{t}.ic_constant"] for t in cell.template.section_types()
        ]
        np.testing.assert_allclose(pinned, ref["ic"], rtol=1e-9)
    finally:
        env.close()


def _test_graph_uuid(system: str) -> str | None:
    try:
        with open(os.path.join(system, "graph.json")) as fh:
            return str(json.load(fh)["architecture"]["uuid"])
    except (OSError, KeyError, ValueError):
        return None


def test_the_noisy_test_graph_reproduces_neurons_spikes():
    from livn.env import Env
    from testing import livn_test_system

    system = livn_test_system()
    ref = np.load(os.path.join(DATA, "native_test_graph_noise.npz"))
    if _test_graph_uuid(system) != str(ref["graph_uuid"][0]):
        pytest.skip("the reference was recorded on systems/graphs/test")

    env = Env(system, seed=123).init()
    try:
        env.apply_model_defaults()
        env.set_noise(NOISE)
        env.record_spikes()
        env.record_voltage(dt=0.1)
        run = env.run(float(ref["duration_ms"][0]))

        order = np.lexsort((run.spike_ids, run.spike_times))
        ref_order = np.lexsort((ref["spike_ids"], ref["spike_times"]))
        np.testing.assert_array_equal(
            np.asarray(run.spike_ids)[order], ref["spike_ids"][ref_order]
        )
        np.testing.assert_allclose(
            np.asarray(run.spike_times)[order], ref["spike_times"][ref_order], atol=1e-6
        )
        np.testing.assert_array_equal(run.voltage_ids, ref["voltage_ids"])
        v = np.asarray(run.voltage)[:, ::10]
        n = min(v.shape[1], ref["voltage"].shape[1])
        assert np.abs(v[:, :n] - ref["voltage"][:, :n]).max() < 5e-3
    finally:
        env.close()


def test_a_replayed_run_is_the_same_run():
    """Capability.REPLAYABLE_NOISE: clear(reseed=False) repeats, clear() does not."""
    from livn.env import Env
    from testing import livn_test_system

    env = Env(livn_test_system(), seed=7).init()
    try:
        env.apply_model_defaults()
        env.set_noise(NOISE)
        env.record_spikes()
        first = np.asarray(env.run(50.0).spike_times)
        env.clear(reseed=False)
        again = np.asarray(env.run(50.0).spike_times)
        np.testing.assert_array_equal(first, again)
        env.clear(reseed=True)
        fresh = np.asarray(env.run(50.0).spike_times)
        assert first.size and (
            fresh.size != first.size or not np.allclose(fresh, first)
        )
    finally:
        env.close()


def test_the_conductance_mode_is_refused_by_name():
    from livn.env import Env
    from livn.stimulus import Stimulus

    env = Env(1).init()
    try:
        with pytest.raises(ValueError, match="extracellular, current, current_density"):
            env.run(
                1.0,
                stimulus=Stimulus.from_conductance(
                    np.zeros((10, 1)), gids=np.array([0])
                ),
            )
    finally:
        env.close()


@pytest.mark.slow
def test_the_inner_loop_has_not_regressed():
    from livn.env import Env

    env = Env(200).init()
    try:
        env.record_spikes()
        started = time.perf_counter()
        env.run(1000.0)
        elapsed = time.perf_counter() - started
    finally:
        env.close()
    node_steps = 200 * 15 * 40_000
    per_node_step_ns = elapsed / node_steps * 1e9
    assert per_node_step_ns < 1000.0, (
        f"{per_node_step_ns:.0f} ns per node-step ({elapsed:.1f}s)"
    )
