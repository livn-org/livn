import os
import pickle

import numpy as np
import pytest

from livn.backend import backend
from livn.utils import P
from testing import livn_test_env, livn_test_mea

TIMEOUT = int(os.environ.get("LIVN_TEST_TIMEOUT", 300))

_RANK_CASES = [
    (1, False),
    (4, False),
    pytest.param(4, True, marks=pytest.mark.mpiexec(isolated=True)),
]


def _create_env(comm, subworld):
    env = livn_test_env(
        io=livn_test_mea(),
        comm=comm,
        subworld_size=2 if subworld else None,
    )

    env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()

    return env


def _concat_optional(a, b):
    if a is None or getattr(a, "size", 0) == 0:
        return b
    if b is None or getattr(b, "size", 0) == 0:
        return a
    return np.concatenate([a, b])


def _concat_matrix(ids_a, data_a, ids_b, data_b):
    if data_a is None or getattr(data_a, "size", 0) == 0:
        return ids_b, data_b
    if data_b is None or getattr(data_b, "size", 0) == 0:
        return ids_a, data_a

    np.testing.assert_array_equal(ids_a, ids_b)
    return ids_a, np.concatenate([data_a, data_b], axis=1)


def _combine_run_outputs(first, second, offset=0.0):
    it_a, t_a, iv_a, v_a, im_a, currents_a = first
    it_b, t_b, iv_b, v_b, im_b, currents_b = second

    combined_it = _concat_optional(it_a, it_b)
    if t_b is not None and offset != 0.0:
        t_b = t_b + offset
    combined_t = _concat_optional(t_a, t_b)
    combined_iv, combined_v = _concat_matrix(iv_a, v_a, iv_b, v_b)
    combined_im, combined_currents = _concat_matrix(im_a, currents_a, im_b, currents_b)

    return (
        combined_it,
        combined_t,
        combined_iv,
        combined_v,
        combined_im,
        combined_currents,
    )


def _gather_and_merge(comm, *values):
    gathered = P.gather(*values, comm=comm)
    if P.is_root(comm=comm):
        return tuple(P.merge(item) for item in gathered)
    return None


STIM_AMPLITUDE = 250.0
"""uA. 750 induced ~1413 mV at the nearest section, outside the +/-1000 mV
the reduced model declares itself defined over (``stimulus_bounds``)."""


def _channel_inputs(env, t_end, amplitude=STIM_AMPLITUDE):
    inputs = np.zeros([t_end, env.io.num_channels])
    inputs[50:70, :] = amplitude
    return inputs


T_END = 250

REFERENCE_ENV = "LIVN_TEST_ENV_REFERENCE"


def _run_reference():
    env = _create_env(None, subworld=False)
    try:
        stimulus = env.cell_stimulus(_channel_inputs(env, T_END))
        it, t, *_ = env.run(T_END, stimulus=stimulus)
        cit, ct = env.channel_recording(it, t)
        return np.asarray(it), np.asarray(t), dict(cit), dict(ct)
    finally:
        env.close()


@pytest.fixture(scope="module")
def serial_reference(tmp_path_factory):
    inherited = os.environ.get(REFERENCE_ENV)
    if inherited:
        with open(inherited, "rb") as f:
            return pickle.load(f)

    reference = _run_reference()

    path = tmp_path_factory.mktemp("livn-env-reference") / "serial.p"
    with open(path, "wb") as f:
        pickle.dump(reference, f)
    os.environ[REFERENCE_ENV] = str(path)

    return reference


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize(
    ("mpiexec_n", "subworld"),
    [(1, False)] if backend() == "brian2" else _RANK_CASES,
)
def test_env(mpiexec_n, subworld, serial_reference):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n

    comm = MPI.COMM_WORLD
    if subworld:
        comm = comm.Split(comm.rank // 2, comm.rank)

    _rit, rt, _rcit, rct = serial_reference

    env = _create_env(comm, subworld)

    stimulus = env.cell_stimulus(_channel_inputs(env, T_END))
    assert np.any(np.asarray(stimulus.array) > 0)

    it, t, *_ = env.run(T_END, stimulus=stimulus)
    cit, ct = env.channel_recording(it, t)

    it, t, cit, ct = P.gather(it, t, cit, ct, comm=comm)

    if P.is_root(comm=comm):
        it, t, cit, ct = P.merge(it, t, cit, ct)

        np.testing.assert_allclose(np.sort(rt), np.sort(t))
        np.testing.assert_allclose(np.sort(rct[0]), np.sort(ct[0]))

    env.close()


def _continued_matches_single(comm, subworld, inputs, split):
    total = len(inputs)

    env_single = _create_env(comm, subworld)
    single_stimulus = env_single.cell_stimulus(inputs)
    single_outputs = env_single.run(total, stimulus=single_stimulus)
    single_potential = None
    if single_outputs[5] is not None:
        single_potential = P.reduce_sum(
            env_single.potential_recording(single_outputs[5], single_outputs[4]),
            all=True,
            comm=comm,
        )
    single_cit, single_ct = env_single.channel_recording(
        single_outputs[0], single_outputs[1]
    )
    single_arrays = _gather_and_merge(comm, *single_outputs[:4])
    single_channels = _gather_and_merge(comm, single_cit, single_ct)
    env_single.close()

    env = _create_env(comm, subworld)
    second_stimulus = env.cell_stimulus(inputs[split:, :])

    first_run = env.run(split)
    second_run = env.run(total - split, stimulus=second_stimulus)

    continued_outputs = _combine_run_outputs(first_run, second_run, offset=split)

    concatenated = first_run.concat(second_run)
    assert concatenated.t0 == first_run.t0
    assert concatenated.duration == total
    for combined, from_concat in zip(continued_outputs, concatenated, strict=False):
        if combined is None or from_concat is None:
            assert combined is from_concat
        else:
            np.testing.assert_allclose(
                np.asarray(combined), np.asarray(from_concat), rtol=1e-6, atol=1e-6
            )

    continued_potential = None
    if continued_outputs[5] is not None:
        continued_potential = P.reduce_sum(
            env.potential_recording(continued_outputs[5], continued_outputs[4]),
            all=True,
            comm=comm,
        )
    continued_cit, continued_ct = env.channel_recording(
        continued_outputs[0], continued_outputs[1]
    )
    continued_arrays = _gather_and_merge(comm, *continued_outputs[:4])
    continued_channels = _gather_and_merge(comm, continued_cit, continued_ct)

    if P.is_root(comm=comm):
        single_it, single_t, single_iv, single_v = single_arrays
        continued_it, continued_t, continued_iv, continued_v = continued_arrays
        single_cit_root, single_ct_root = single_channels
        continued_cit_root, continued_ct_root = continued_channels

        np.testing.assert_allclose(
            np.sort(single_t, axis=None), np.sort(continued_t, axis=None)
        )
        np.testing.assert_array_equal(
            np.sort(single_it, axis=None), np.sort(continued_it, axis=None)
        )

        if single_iv is not None:
            np.testing.assert_array_equal(single_iv, continued_iv)
        if single_v is not None:
            np.testing.assert_allclose(single_v, continued_v, rtol=1e-6, atol=1e-6)

        for label, single_side, continued_side in (
            ("channel ids", single_cit_root, continued_cit_root),
            ("channel times", single_ct_root, continued_ct_root),
        ):
            assert single_side.keys() == continued_side.keys(), label
            for key in single_side:
                np.testing.assert_allclose(
                    np.sort(single_side[key], axis=None),
                    np.sort(continued_side[key], axis=None),
                    err_msg=f"{label} for channel {key}",
                )

        if single_potential is not None and continued_potential is not None:
            np.testing.assert_allclose(
                single_potential, continued_potential, rtol=1e-5, atol=1e-5
            )

    env.close()


def _subcomm(comm, subworld):
    return comm.Split(comm.rank // 2, comm.rank) if subworld else comm


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
@pytest.mark.skipif(
    backend() == "diffrax",
    reason="diffrax continued runs produce different timestep counts",
)
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize(
    ("mpiexec_n", "subworld"),
    [(1, False)] if backend() == "brian2" else _RANK_CASES,
)
@pytest.mark.parametrize(
    "placement",
    ["tail", "middle"],
    ids=["stimulus-in-the-second-piece", "stimulus-spanning-the-split"],
)
def test_a_run_split_in_two_matches_the_whole_run(mpiexec_n, subworld, placement):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n
    comm = _subcomm(MPI.COMM_WORLD, subworld)

    env = _create_env(comm, subworld)
    channels = env.io.num_channels
    env.close()

    if placement == "tail":
        total, split = 30, 15
        inputs = np.zeros([total, channels])
        inputs[split:, :] = STIM_AMPLITUDE
    else:
        total, split = 90, 30
        inputs = np.zeros([total, channels])
        inputs[split : split + 10, :] = STIM_AMPLITUDE

    _continued_matches_single(comm, subworld, inputs, split)


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
@pytest.mark.skipif(
    backend() == "diffrax",
    reason="diffrax continued runs produce different timestep counts",
)
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize("mpiexec_n", [1])
def test_continuing_a_run_at_a_different_stimulus_dt_is_refused(mpiexec_n):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n
    comm = MPI.COMM_WORLD
    env = _create_env(comm, subworld=False)
    try:
        first = env.cell_stimulus(np.full((10, env.io.num_channels), 100.0))
        first.dt = 1.0
        env.run(10, stimulus=first)

        second = env.cell_stimulus(np.full((20, env.io.num_channels), 150.0))
        second.dt = 0.5

        with pytest.raises(ValueError, match="Stimulus dt mismatch"):
            env.run(10, stimulus=second)
    finally:
        env.close()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(
    backend() != "neuron", reason="the fluctuating-conductance mechanism is NEURON's"
)
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize("mpiexec_n", [1, 2])
def test_env_noise(mpiexec_n):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n
    comm = MPI.COMM_WORLD

    env = _create_env(comm, subworld=False)

    noise_params = {
        "std_e": 0.005,
        "std_i": 0.005,
        "g_e0": 0.01,
        "g_i0": 0.05,
    }

    env.set_noise(noise_params)

    pop_name = next(iter(env.cells.keys()))
    gid = next(iter(env.cells[pop_name].keys()))
    soma_key = f"{gid}-0"
    dend_key = f"{gid}-1"

    if soma_key in env._flucts:
        mech_soma, _ = env._flucts[soma_key]
        assert mech_soma.g_e0 == 0
        assert mech_soma.std_e == 0
        assert mech_soma.g_i0 == noise_params["g_i0"]
        assert mech_soma.std_i == noise_params["std_i"]
        assert mech_soma.on == 1

    if dend_key in env._flucts:
        mech_dend, _ = env._flucts[dend_key]
        assert mech_dend.g_e0 == noise_params["g_e0"]
        assert mech_dend.std_e == noise_params["std_e"]
        assert mech_dend.g_i0 == 0
        assert mech_dend.std_i == 0
        assert mech_dend.on == 1

    env.run(50)
    v_soma = env.v_recs[(gid, 0)].as_numpy()
    assert np.std(v_soma) > 1e-3

    env.close()


def _repeat_runs(env, duration, n, *, reseed):
    inputs = np.zeros([duration, env.io.num_channels])
    voltages = []
    for _ in range(n):
        stimulus = env.cell_stimulus(inputs)
        _, _, _, v, _, _ = env.run(duration, stimulus=stimulus)
        voltages.append(np.asarray(v).copy())
        env.clear(reseed=reseed)
    return voltages


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
@pytest.mark.needs("replayable_noise")
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize("mpiexec_n", [1])
def test_env_stochastic_variability(mpiexec_n):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n
    comm = MPI.COMM_WORLD

    duration = 50

    env = _create_env(comm, subworld=False)
    env.apply_model_defaults(noise=True)

    voltages = _repeat_runs(env, duration, 3, reseed=True)
    for i, later in enumerate(voltages[1:], start=1):
        assert not np.allclose(voltages[0], later), (
            f"run {i} reproduced run 0 exactly; the noise stream did not advance"
        )

    env.clear(reseed=False)
    repeated = _repeat_runs(env, duration, 2, reseed=False)
    np.testing.assert_allclose(
        repeated[0],
        repeated[1],
        err_msg="clear(reseed=False) should replay the same noise",
    )
    env.close()

    env = _create_env(comm, subworld=False)
    env.apply_model_defaults(noise=True)
    again = _repeat_runs(env, duration, 3, reseed=True)
    for i, (first, second) in enumerate(zip(voltages, again, strict=False)):
        np.testing.assert_allclose(
            first,
            second,
            err_msg=f"run {i} differed between two identically seeded envs",
        )

    env.close()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.skipif(backend() == "", reason="no simulation backend selected")
@pytest.mark.mpiexec(timeout=TIMEOUT)
@pytest.mark.parametrize("mpiexec_n", [1])
def test_env_deterministic_without_noise(mpiexec_n):
    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == mpiexec_n
    comm = MPI.COMM_WORLD

    env = _create_env(comm, subworld=False)

    duration = 50
    inputs = np.zeros([duration, env.io.num_channels])

    voltages = []
    for _ in range(2):
        stimulus = env.cell_stimulus(inputs)
        _, _, _iv, v, _, _ = env.run(duration, stimulus=stimulus)
        voltages.append(v.copy())
        env.clear()

    np.testing.assert_allclose(
        voltages[0],
        voltages[1],
        atol=1e-9,
        err_msg="Runs without noise should be deterministic",
    )

    env.close()
