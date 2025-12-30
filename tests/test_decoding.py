import os

import numpy as np
import pytest

from livn.env import Env
from livn.decoding import Slice
from livn.backend import backend


@pytest.fixture
def env_response(request):
    response = request.config.cache.get("livn/env/response", None)
    if response is None:
        env = Env(os.environ["LIVN_TEST_SYSTEM"])
        env.init()

        env.record_spikes()
        env.record_voltage()
        env.record_membrane_current()

        t_end = 250
        inputs = np.zeros([t_end, env.io.num_channels])
        for r in range(20):
            for c in range(env.io.num_channels):
                inputs[50 + r, c] = 750

        stimulus = env.cell_stimulus(inputs)
        response = env.run(t_end, stimulus=stimulus)

        request.config.cache.set("livn/env/response", [r.tolist() for r in response])

    return tuple([np.array(r) for r in response])


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
def test_slice_decoding(env_response):
    env = Env(os.environ["LIVN_TEST_SYSTEM"])
    if backend() == "brian2":
        env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    start = 100
    duration = 50
    recording_dt = 0.1

    ii, tt, iv, v, im, m = Slice(start=start, stop=start + duration)(env, *env_response)
    orig_ii, orig_tt, orig_iv, orig_v, orig_im, orig_m = env_response

    # spike times
    assert tt[tt < start].shape[0] == 0
    assert tt[tt >= start + duration].shape[0] == 0

    expected_time_steps = int(duration / recording_dt)
    assert v.shape[0] == orig_v.shape[0]
    assert v.shape[1] == expected_time_steps

    assert m.shape[0] == orig_m.shape[0]
    assert m.shape[1] == expected_time_steps
