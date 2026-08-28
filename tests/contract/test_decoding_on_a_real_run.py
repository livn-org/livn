import hashlib
import json
import os

import numpy as np
import pytest

from livn.backend import backend
from livn.decoding import (
    LFP,
    ActiveFraction,
    AvalancheAnalysis,
    MeanFiringRate,
    Pipe,
    Slice,
    Stability,
)
from testing import livn_test_env, livn_test_mea, livn_test_selection, livn_test_system

STIMULUS_AMPLITUDE = 1.5
RESPONSE_DURATION = 250


def _recording(it=None, tt=None, iv=None, vv=None, im=None, mp=None, dt=0.1):
    from livn.run import Run

    return (
        Run(duration=float(RESPONSE_DURATION))
        .add_spikes(it, tt)
        .add_voltage(iv, vv, dt=dt)
        .add_current(im, mp, dt=dt)
    )


def _graph_identity() -> str:
    path = os.path.join(str(livn_test_system()).rstrip("/"), "graph.json")
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError:
        return "no-graph"
    try:
        return str(json.loads(payload)["architecture"]["uuid"])
    except (ValueError, KeyError, TypeError):
        return hashlib.sha1(payload).hexdigest()[:12]


@pytest.fixture
def env_response(request):
    if not os.getenv("LIVN_BACKEND"):
        pytest.skip("no simulation backend selected")

    cache_key = "/".join(
        [
            "livn/env/response",
            backend(),
            os.path.basename(str(livn_test_system()).rstrip("/")),
            _graph_identity(),
            str(livn_test_selection() or "all"),
            f"{RESPONSE_DURATION}ms-{STIMULUS_AMPLITUDE}",
        ]
    )

    cache = getattr(request.config, "cache", None)

    response = cache.get(cache_key, None) if cache is not None else None
    if response is None:
        env = livn_test_env(io=livn_test_mea())
        env.init()

        env.record_spikes()
        env.record_voltage()
        env.record_membrane_current()

        t_end = RESPONSE_DURATION
        inputs = np.zeros([t_end, env.io.num_channels])
        inputs[50:70, :] = STIMULUS_AMPLITUDE

        try:
            stimulus = env.cell_stimulus(inputs)
            response = [np.asarray(r) for r in env.run(t_end, stimulus=stimulus)]
        finally:
            env.close()

        if cache is not None:
            cache.set(cache_key, [r.tolist() for r in response])

    return _recording(*[np.array(r) for r in response])


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_mean_firing_rate_integration(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()

    mfr = MeanFiringRate(duration=250)
    result = mfr(env_response, env)

    assert result is not None
    assert result["rate_hz"] >= 0
    assert result["total_spikes"] >= 0


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_active_fraction_integration(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()

    af = ActiveFraction(duration=250)
    result = af(env_response, env)

    assert result is not None
    assert 0 <= result["active_fraction"] <= 1


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_stability_integration(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()

    stability = Stability(duration=250, tail_window=100)
    result = stability(env_response, env)

    assert result is not None
    assert "is_stable" in result
    assert isinstance(result["is_stable"], bool)


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_lfp_integration(env_response):
    env = livn_test_env(io=livn_test_mea()).init()
    env.record_membrane_current()

    lfp = LFP(duration=250)
    result = lfp(env_response, env)

    assert result is not None
    assert "lfp" in result


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_pipeline_integration(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_membrane_current()

    pipeline = Pipe(
        duration=250,
        stages=[
            Slice(start=50, stop=200),
            MeanFiringRate(duration=150),
        ],
    )

    result = pipeline(env_response, env)

    assert result is not None
    assert "rate_hz" in result


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_avalanche_analysis_integration(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()

    aa = AvalancheAnalysis(duration=250, bin_width=4.0)
    result = aa(env_response, env)

    assert result is not None
    assert "n_avalanches" in result
    assert "branching_ratio" in result
    assert result["n_avalanches"] >= 0
    assert result["branching_ratio"] >= 0.0


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_slice_decoding(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    start = 100
    duration = 50
    recording_dt = 0.1

    _ii, tt, _iv, v, _im, mp = Slice(start=start, stop=start + duration)(env_response)
    _orig_ii, orig_tt, _orig_iv, orig_v, _orig_im, orig_m = env_response

    assert tt[tt < 0].shape[0] == 0
    assert tt[tt >= duration].shape[0] == 0
    expected_spikes = orig_tt[(orig_tt >= start) & (orig_tt < start + duration)].shape[
        0
    ]
    assert tt.shape[0] == expected_spikes

    expected_time_steps = int(duration / recording_dt)
    assert v.shape[0] == orig_v.shape[0]
    assert v.shape[1] == expected_time_steps

    assert mp.shape[0] == orig_m.shape[0]
    assert mp.shape[1] == expected_time_steps


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_slice_float_valid(env_response):
    env = livn_test_env()
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    recording_dt = 0.1
    start = 10.0
    duration = 5.0

    _ii, tt, _iv, v, _im, mp = Slice(start=start, stop=start + duration)(env_response)

    if len(tt) > 0:
        assert np.all(tt >= 0)
        assert np.all(tt < duration)

    expected_time_steps = int(duration / recording_dt)
    assert v.shape[1] == expected_time_steps
    assert mp.shape[1] == expected_time_steps

    start = 0.025
    duration = 10.0

    with pytest.raises(ValueError, match=r"does not align with.*recording dt"):
        Slice(start=start, stop=start + duration)(env_response)

    start = 10.0
    duration = 0.025

    with pytest.raises(ValueError, match=r"does not align with.*recording dt"):
        Slice(start=start, stop=start + duration)(env_response)
