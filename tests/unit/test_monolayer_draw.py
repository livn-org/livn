"""The connectivity a Monolayer draws is pinned: fits depend on the exact graph."""

import hashlib
import json

import numpy as np
import pytest

from livn.system import Monolayer, predefined_document
from livn.system import monolayer as M


def spec(name: str, **overrides) -> dict:
    with open(predefined_document(name)) as f:
        kwargs = json.load(f)["system"]["kwargs"]
    return {**kwargs, **overrides}


def graph_digest(system: Monolayer) -> str:
    """Every edge and synapse site, in the order the system hands them out."""
    digest = hashlib.sha256()
    for post, (start, count) in system.population_ranges.items():
        gids = range(start, start + count)
        for gid, (syn_ids, swc, loc) in sorted(system.placement(post, gids).items()):
            digest.update(np.int64(gid).tobytes())
            digest.update(np.asarray(syn_ids, dtype=np.int64).tobytes())
            digest.update(np.asarray(swc, dtype=np.int64).tobytes())
            digest.update(np.asarray(loc, dtype=np.float64).tobytes())
        for pre in system.population_ranges:
            for gid, (pres, projection) in system.edges(pre, post, gids):
                digest.update(f"{pre}->{post}:{gid}".encode())
                digest.update(np.asarray(pres, dtype=np.int64).tobytes())
                digest.update(projection["Synapses"]["syn_id"].tobytes())
                digest.update(projection["Connections"]["distance"].tobytes())
    return digest.hexdigest()[:16]


# drawn by the all-pairs implementation these replaced
GOLDEN = {
    "E3I-470": "706337913f3084ad",
    "E3I-470-cutoff": "5772189a2e72add9",
    "E3I-470-guard": "6180e00209825537",
    # 2049 post cells: a last chunk one column wide, whose kernel mass numpy
    # sums pairwise rather than row by row
    "odd-chunks": "161a7924927fcf5e",
}

CASES = {
    "E3I-470": lambda: spec("E3I-470"),
    "E3I-470-cutoff": lambda: spec(
        "E3I-470",
        connectivity={**spec("E3I-470")["connectivity"], "cutoff": 0.01},
    ),
    "E3I-470-guard": lambda: spec("E3I-470", boundary=150.0),
    "odd-chunks": lambda: spec(
        "E3I",
        total_cells=None,
        populations={
            "EXC": {"count": 2049, "synapse_type": "excitatory"},
            "INH": {"count": 1, "synapse_type": "inhibitory"},
        },
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_graph_is_pinned(name):
    assert graph_digest(Monolayer(**CASES[name]())) == GOLDEN[name]


@pytest.mark.parametrize(
    ("columns", "threads"), [(2, 1), (7, 3), (2048, 1), (1949, 2), (64, 8)]
)
def test_draw_does_not_depend_on_how_it_is_split(monkeypatch, columns, threads):
    kwargs = spec("E3I")
    n = 1950  # EXC cells, so 1949 columns leave a slice of one
    reference = Monolayer(**kwargs)._edge_columns("EXC", "EXC", np.arange(n))
    monkeypatch.setattr(M, "DRAW_SLICE_ELEMENTS", columns * n)
    monkeypatch.setattr(Monolayer, "_draw_threads", lambda self: threads)
    split = Monolayer(**kwargs)._edge_columns("EXC", "EXC", np.arange(n))
    assert reference.keys() == split.keys()
    for index, (selected, distances) in reference.items():
        np.testing.assert_array_equal(split[index][0], selected)
        np.testing.assert_array_equal(split[index][1], distances)


def test_a_subset_of_columns_is_drawn_as_in_the_whole():
    system = Monolayer(**spec("E3I-470"))
    start, count = system.population_ranges["EXC"]
    whole = dict(system.edges("EXC", "EXC", range(start, start + count)))
    for gid, (pres, projection) in system.edges(
        "EXC", "EXC", range(start, start + count, 7)
    ):
        np.testing.assert_array_equal(whole[gid][0], pres)
        np.testing.assert_array_equal(
            whole[gid][1]["Synapses"]["syn_id"], projection["Synapses"]["syn_id"]
        )
