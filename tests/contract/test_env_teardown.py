from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("livn")

from livn.backend import backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    backend() != "neuron", reason="teardown is asserted against NEURON's object model"
)


@pytest.fixture(autouse=True)
def settled_heap():
    gc.collect()
    yield
    gc.collect()


def _stimulated_env():
    from testing import livn_test_env, livn_test_mea

    env = livn_test_env(io=livn_test_mea()).init()
    env.record_spikes()

    dt, duration = 0.5, 10.0
    channel_inputs = np.zeros((int(round(duration / dt)), len(env.io.channel_ids)))
    channel_inputs[2, 0] = -50.0
    channel_inputs[3, 0] = 50.0
    stimulus = env.cell_stimulus(channel_inputs)
    stimulus.dt = dt

    env.run(duration, stimulus=stimulus)
    return env


def test_a_dropped_env_is_collected_rather_than_held_by_cvode():
    from neuron import h

    before = len(list(h.allsec()))

    env = _stimulated_env()
    assert env._stim_registered, "the stimulus did not register a cvode callback"
    ref = weakref.ref(env)

    del env
    gc.collect()

    assert ref() is None, "something is still holding the env after it was dropped"
    assert len(list(h.allsec())) == before, (
        "the dropped env left its sections behind; over a chain of envs they "
        "accumulate until a gid_clear() frees what their NetCons point at"
    )


def test_the_stale_registration_is_swept_on_the_next_init():
    from livn.backend.neuron import _REGISTERED_CALLBACKS
    from testing import livn_test_env

    env = _stimulated_env()
    assert any(c.alive for c in _REGISTERED_CALLBACKS)

    del env
    gc.collect()
    assert [c for c in _REGISTERED_CALLBACKS if not c.alive], (
        "expected the collected env's registration to still be listed"
    )

    successor = livn_test_env().init()

    assert all(c.alive for c in _REGISTERED_CALLBACKS), (
        "init() left callbacks registered for envs that no longer exist"
    )
    successor.close()


def test_close_releases_everything_too():
    from neuron import h

    before = len(list(h.allsec()))

    env = _stimulated_env()
    ref = weakref.ref(env)
    env.close()

    del env
    gc.collect()

    assert ref() is None
    assert len(list(h.allsec())) == before
