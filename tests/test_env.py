import os
import pickle

import numpy as np
import pytest
from mpi4py import MPI

from livn.backend import backend
from livn.env import Env
from livn.io import MEA
from livn.models.izhikevich import Izhikevich
from livn.models.rcsd import ReducedCalciumSomaDendrite
from livn.stimulus import Stimulus
from livn.utils import P


def _create_env(comm, subworld):
    system = os.environ["LIVN_TEST_SYSTEM"]
    io = MEA.from_json(os.path.join(system, "mea.json"))

    env = Env(
        system,
        model=ReducedCalciumSomaDendrite() if backend() == "neuron" else Izhikevich(),
        io=io,
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


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1, 4] if backend() != "brian2" else [1])
@pytest.mark.parametrize(
    "subworld",
    [False, True],
)
def test_env(mpiexec_n, subworld):
    if mpiexec_n == 1 and subworld:
        return

    assert MPI.COMM_WORLD.size == mpiexec_n

    comm = MPI.COMM_WORLD
    if subworld:
        color = comm.rank // 2
        subcomm = comm.Split(color, comm.rank)
        comm = subcomm

    tmp_data_path = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_data_path, exist_ok=True)

    reference = os.path.join(tmp_data_path, f"reference.p")

    env = _create_env(comm, subworld)

    t_end = 250
    inputs = np.zeros([t_end, env.io.num_channels])
    for r in range(20):
        for c in range(env.io.num_channels):
            inputs[50 + r, c] = 750

    stimulus = env.cell_stimulus(inputs)
    assert np.any(stimulus > 0)

    it, t, *_ = env.run(t_end, stimulus=stimulus)

    cit, ct = env.channel_recording(it, t)

    if mpiexec_n == 1:
        # save the reference
        with open(reference, "wb") as f:
            pickle.dump((it, t, dict(cit), dict(ct)), f)

        return

    it, t, cit, ct = P.gather(it, t, cit, ct, comm=comm)

    if P.is_root(comm=comm):
        it, t, cit, ct = P.merge(it, t, cit, ct)

        with open(reference, "rb") as f:
            rit, rt, rcit, rct = pickle.load(f)

        np.testing.assert_allclose(np.sort(rt), np.sort(t))
        np.testing.assert_allclose(np.sort(rct[0]), np.sort(ct[0]))

    env.close()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1, 4] if backend() != "brian2" else [1])
@pytest.mark.parametrize(
    "subworld",
    [False, True],
)
def test_env_continued_runs(mpiexec_n, subworld):
    if mpiexec_n == 1 and subworld:
        return

    assert MPI.COMM_WORLD.size == mpiexec_n

    comm = MPI.COMM_WORLD
    if subworld:
        color = comm.rank // 2
        subcomm = comm.Split(color, comm.rank)
        comm = subcomm

    env_single = _create_env(comm, subworld)

    total_duration = 120
    split = 60

    inputs = np.zeros([total_duration, env_single.io.num_channels])
    inputs[split:, :] = 500.0

    single_stimulus = env_single.cell_stimulus(inputs)
    single_outputs = env_single.run(total_duration, stimulus=single_stimulus)
    single_potential = None
    if single_outputs[5] is not None:
        single_potential = P.reduce_sum(
            env_single.potential_recording(single_outputs[5]),
            all=True,
            comm=comm,
        )
    single_cit, single_ct = env_single.channel_recording(
        single_outputs[0], single_outputs[1]
    )

    single_arrays = _gather_and_merge(
        comm,
        single_outputs[0],
        single_outputs[1],
        single_outputs[2],
        single_outputs[3],
    )
    single_channels = _gather_and_merge(comm, single_cit, single_ct)

    env_single.close()

    env = _create_env(comm, subworld)

    second_inputs = inputs[split:, :]
    second_stimulus = env.cell_stimulus(second_inputs)

    first_run = env.run(split)
    second_run = env.run(total_duration - split, stimulus=second_stimulus)

    continued_outputs = _combine_run_outputs(first_run, second_run, offset=split)
    continued_potential = None
    if continued_outputs[5] is not None:
        continued_potential = P.reduce_sum(
            env.potential_recording(continued_outputs[5]),
            all=True,
            comm=comm,
        )
    continued_cit, continued_ct = env.channel_recording(
        continued_outputs[0], continued_outputs[1]
    )

    continued_arrays = _gather_and_merge(
        comm,
        continued_outputs[0],
        continued_outputs[1],
        continued_outputs[2],
        continued_outputs[3],
    )
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

        assert single_cit_root.keys() == continued_cit_root.keys()
        for key in single_cit_root:
            np.testing.assert_allclose(
                np.sort(single_cit_root[key], axis=None),
                np.sort(continued_cit_root[key], axis=None),
            )

        assert single_ct_root.keys() == continued_ct_root.keys()
        for key in single_ct_root:
            np.testing.assert_allclose(
                np.sort(single_ct_root[key], axis=None),
                np.sort(continued_ct_root[key], axis=None),
            )

        if single_potential is not None and continued_potential is not None:
            np.testing.assert_allclose(
                single_potential,
                continued_potential,
                rtol=1e-5,
                atol=1e-5,
            )

    env.close()


@pytest.mark.skipif(
    "LIVN_TEST_SYSTEM" not in os.environ, reason="LIVN_TEST_SYSTEM missing"
)
@pytest.mark.mpiexec(timeout=60)
@pytest.mark.parametrize("mpiexec_n", [1])
def test_env_continued_runs_stimulus_dt_mismatch(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

    comm = MPI.COMM_WORLD
    env = _create_env(comm, subworld=False)

    first_duration = 10
    first_inputs = np.full((first_duration, env.io.num_channels), 100.0)
    first_stimulus = Stimulus(env.cell_stimulus(first_inputs), dt=1.0)

    env.run(first_duration, stimulus=first_stimulus)

    second_duration = 10
    second_steps = second_duration * 2  # dt=0.5 ms
    second_inputs = np.full((second_steps, env.io.num_channels), 250.0)
    second_stimulus = Stimulus(env.cell_stimulus(second_inputs), dt=0.5)

    with pytest.raises(ValueError, match="Stimulus dt mismatch"):
        env.run(second_duration, stimulus=second_stimulus)

    env.close()

    # common application: no stimulus, then a short stimulus, then no stimulus again
    env_single = _create_env(comm, subworld=False)

    total_duration = 90
    stim_start = 30
    stim_length = 10

    mid_inputs = np.zeros((total_duration, env_single.io.num_channels))
    mid_inputs[stim_start : stim_start + stim_length, :] = 400.0

    single_stimulus = env_single.cell_stimulus(mid_inputs)
    single_outputs = env_single.run(total_duration, stimulus=single_stimulus)
    single_potential = None
    if single_outputs[5] is not None:
        single_potential = P.reduce_sum(
            env_single.potential_recording(single_outputs[5]), all=True, comm=comm
        )
    single_cit, single_ct = env_single.channel_recording(
        single_outputs[0], single_outputs[1]
    )

    single_arrays = _gather_and_merge(
        comm,
        single_outputs[0],
        single_outputs[1],
        single_outputs[2],
        single_outputs[3],
    )
    single_channels = _gather_and_merge(comm, single_cit, single_ct)

    env_single.close()

    env = _create_env(comm, subworld=False)

    first_run = env.run(stim_start)
    mid_stimulus = env.cell_stimulus(mid_inputs[stim_start:, :])
    second_run = env.run(stim_length, stimulus=mid_stimulus)
    third_run = env.run(total_duration - stim_start - stim_length)

    combined_outputs = _combine_run_outputs(first_run, second_run, offset=stim_start)
    combined_outputs = _combine_run_outputs(
        combined_outputs,
        third_run,
        offset=stim_start + stim_length,
    )

    continued_potential = None
    if combined_outputs[5] is not None:
        continued_potential = P.reduce_sum(
            env.potential_recording(combined_outputs[5]), all=True, comm=comm
        )
    continued_cit, continued_ct = env.channel_recording(
        combined_outputs[0], combined_outputs[1]
    )

    continued_arrays = _gather_and_merge(
        comm,
        combined_outputs[0],
        combined_outputs[1],
        combined_outputs[2],
        combined_outputs[3],
    )
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

        assert single_cit_root.keys() == continued_cit_root.keys()
        for key in single_cit_root:
            np.testing.assert_allclose(
                np.sort(single_cit_root[key], axis=None),
                np.sort(continued_cit_root[key], axis=None),
            )

        assert single_ct_root.keys() == continued_ct_root.keys()
        for key in single_ct_root:
            np.testing.assert_allclose(
                np.sort(single_ct_root[key], axis=None),
                np.sort(continued_ct_root[key], axis=None),
            )

        if single_potential is not None and continued_potential is not None:
            np.testing.assert_allclose(
                single_potential,
                continued_potential,
                rtol=1e-5,
                atol=1e-5,
            )

    env.close()
