from __future__ import annotations

import contextlib

import numpy as np
import pytest

from testing import livn_test_env

_PI = 3.141592653589793
_H = (15.0**2 + 50.0**2) ** 0.5
_STIMULATION = 1000.0 * 3.5 / (4.0 * _PI * _H)

_OPEN_ENVS = []


@pytest.fixture(autouse=True)
def _close_envs():
    yield
    while _OPEN_ENVS:
        with contextlib.suppress(Exception):
            _OPEN_ENVS.pop().close()


def _env(io=None):
    pytest.importorskip("livn")
    env = livn_test_env(**({} if io is None else {"io": io}))
    env.init()
    _OPEN_ENVS.append(env)
    return env


def _expected_reach(env):
    coordinates = env.stimulus_coordinates()
    d = np.asarray(env.io.distances(coordinates))[:, -1]
    shape = _H / np.sqrt(d**2 + _H**2)
    return (_STIMULATION * shape).reshape(len(env.io.channel_ids), -1)


def _command(env, n_steps=30):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, (n_steps, len(env.io.channel_ids))).astype(np.float32)


def _check(env):
    reach = np.asarray(env.channel_reach())
    expected = _expected_reach(env)
    assert np.array_equal(reach, expected.astype(reach.dtype))

    command = _command(env)
    stimulus = env.cell_stimulus(command)
    values = np.asarray(
        stimulus.values if hasattr(stimulus, "values") else stimulus.array
    )
    assert np.all(np.isfinite(values)) and np.any(values != 0.0)

    coordinates = np.asarray(env.stimulus_coordinates())
    rows = {}
    for r, gid in enumerate(coordinates[:, 0].astype(int)):
        rows.setdefault(gid, []).append(r)
    gids = np.asarray(stimulus.gids).astype(int)
    sections = np.asarray(stimulus.sections).astype(int)
    for k in range(0, len(gids), max(1, len(gids) // 7)):
        column = rows[int(gids[k])][int(sections[k])]
        np.testing.assert_allclose(
            values[:, k], command @ expected[:, column], rtol=1e-4, atol=1e-6
        )


def test_the_default_device_stimulates_as_before():
    _check(_env())


def test_a_composition_with_a_stack_stimulates_as_before():
    from livn.io import MEA, ComposedIO, ElectrodeRC, Noise
    from livn.system import resolve
    from testing import livn_test_system

    xyz = np.asarray(resolve(livn_test_system()).neuron_coordinates)[:, 1:]
    mea = MEA(
        [[0, xyz[:, 0].min() - 100.0, xyz[:, 1].mean(), xyz[:, 2].mean()]],
        input_radius=1e9,
        output_radius=1e9,
    )
    env = _env(ComposedIO(None, [mea, ElectrodeRC(), Noise(3.0)]))
    _check(env)
