from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap

import pytest
from testing import mpiexec as mpiexec_plugin
from testing.mpiexec import mpi_slot
from testing.paths import REPO_ROOT

REPO = str(REPO_ROOT)

MISALIGNING_RANKS = 3

_ARRAY = """
    mea = MEA.from_json({
        "electrode_coordinates": [
            [float(i), float(x), float(y), 5.0]
            for i, (x, y) in enumerate(
                (x, y)
                for x in range(0, 1500, 200)
                for y in range(0, 3100, 200)
            )
        ],
        "input_radius": 100.0,
        "output_radius": 100.0,
    })
"""

READOUT_PROBE = textwrap.dedent(
    """
    import numpy as np
    from mpi4py import MPI
    from livn.env import Env
    from livn.io import MEA
    from livn.models.rcsd import ReducedCalciumSomaDendrite as M
    from livn.utils import P
    """
    + _ARRAY
    + """
    env = Env("SYSTEM", model=M(), io=mea, comm=MPI.COMM_WORLD)
    env.init()

    gids = np.asarray(sorted(g for c in env.cells.values() for g in c), dtype=np.int64)
    ids = np.repeat(gids, 10)
    times = np.linspace(0.0, 1000.0, len(ids))

    _, per_channel = env.channel_recording(ids, times)
    captured = sum(len(v) for v in per_channel.values())

    total = P.reduce_sum(np.array(len(ids), dtype=np.int64), all=True)
    kept = P.reduce_sum(np.array(captured, dtype=np.int64), all=True)
    if P.is_root():
        print("RESULT", float(kept) / float(total), flush=True)
    """
)

STIMULUS_PROBE = textwrap.dedent(
    """
    import numpy as np
    from mpi4py import MPI
    from livn.env import Env
    from livn.io import MEA
    from livn.models.rcsd import ReducedCalciumSomaDendrite as M
    from livn.utils import P
    """
    + _ARRAY
    + """
    env = Env("SYSTEM", model=M(), io=mea, comm=MPI.COMM_WORLD)
    env.init()

    n_channels = len(env.io.channel_ids)
    channel_inputs = np.zeros((8, n_channels))
    channel_inputs[2:4, 0] = -1.0
    channel_inputs[4:6, 0] = 1.0

    stimulus = env.cell_stimulus(channel_inputs)

    # what this rank would actually deliver: the columns for its own cells
    labels = np.asarray(stimulus.gids)
    mine = env.simulated_gids()
    delivered = float(np.abs(stimulus.array[:, np.isin(labels, mine)]).sum())

    total = P.reduce_sum(np.array(delivered), all=True)
    if P.is_root():
        print("RESULT", float(total), flush=True)
    """
)


def _run(nranks: int, script: str) -> float:
    with mpi_slot(REPO):
        proc = subprocess.run(
            [
                *shlex.split(mpiexec_plugin.MPIEXEC),
                "-n",
                str(nranks),
                sys.executable,
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
            env={**os.environ, "LIVN_BACKEND": "neuron", "PYTHONPATH": REPO},
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return float(line.split()[1])
    raise AssertionError(
        f"probe produced no result on {nranks} ranks\n"
        f"stdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    )


@pytest.fixture(scope="module")
def system():
    pytest.importorskip("mpi4py")
    from livn.backend import backend

    if backend() != "neuron":
        pytest.skip("needs the neuron backend")

    path = os.path.join(REPO, "systems", "graphs", "EI")
    if not os.path.isfile(os.path.join(path, "graph.json")):
        pytest.skip("EI graph not generated")
    return path


def test_the_readout_does_not_lose_spikes_when_ranks_are_added(system):
    script = READOUT_PROBE.replace("SYSTEM", system)
    one = _run(1, script)
    many = _run(MISALIGNING_RANKS, script)

    assert one > 0.05, f"the probe captures almost nothing even serially ({one:.3%})"
    assert many == pytest.approx(one, rel=0.05), (
        f"1 rank captures {one:.2%} of spikes but {MISALIGNING_RANKS} ranks "
        f"capture {many:.2%}. The readout is dropping cells whose coordinates "
        "this rank did not read."
    )


def test_the_stimulus_reaches_the_same_cells_when_ranks_are_added(system):
    script = STIMULUS_PROBE.replace("SYSTEM", system)
    one = _run(1, script)
    many = _run(MISALIGNING_RANKS, script)

    assert one > 0, "the probe delivers no stimulus even serially"
    assert many == pytest.approx(one, rel=1e-9), (
        f"1 rank delivers {one:.6g} of stimulus but {MISALIGNING_RANKS} ranks "
        f"deliver {many:.6g}; the array and its gids describe different cells"
    )


def test_every_rank_can_resolve_every_cell_it_holds(system):
    anchor = "_, per_channel = env.channel_recording(ids, times)"
    script = READOUT_PROBE.replace("SYSTEM", system)
    assert anchor in script, "the probe changed; this substitution is a no-op"
    probe = script.replace(
        anchor,
        "coords = set(env.active_neuron_coordinates()[:, 0].astype(int).tolist())\n"
        "missing = sorted(set(gids.tolist()) - coords)\n"
        "assert not missing, (\n"
        '    f"rank {P.rank()} holds {len(missing)} of {len(gids)} cells it "\n'
        '    f"has no coordinates for, e.g. {missing[:5]}"\n'
        ")\n" + anchor,
    )
    assert probe != script
    assert _run(MISALIGNING_RANKS, probe) > 0.05


def test_the_read_partition_is_not_selectable(system):
    import inspect

    from livn.system import resolve

    signature = inspect.signature(type(resolve(system)).coordinate_array)
    assert "all" not in signature.parameters, (
        "coordinate_array still exposes the read partition"
    )
    signature = inspect.signature(type(resolve(system)).transform_coordinates)
    assert "all" not in signature.parameters


def test_a_stimulus_cannot_be_labelled_with_the_wrong_cells():
    import numpy as np

    from livn.stimulus import Stimulus

    array = np.zeros((10, 4))
    assert Stimulus(array, gids=np.arange(4)) is not None
    with pytest.raises(ValueError, match="4 channels but 8 gids"):
        Stimulus(array, gids=np.arange(8))
