from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

from testing import livn_test_env

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVN_BACKEND"), reason="no simulation backend selected"
)

SERIES = ("voltage", "current")


@pytest.fixture(scope="module")
def run():
    env = livn_test_env()
    env.init()
    env.record_spikes()
    env.record_voltage()
    env.record_membrane_current()
    yield env.run(20, dt=0.025)
    with contextlib.suppress(Exception):
        env.close()


@pytest.mark.parametrize("name", SERIES)
def test_a_repeated_gid_names_its_sections(run, name):
    if run.channel(name) is None:
        pytest.skip(f"this backend records no {name}")
    ids = np.asarray(run.ids(name)).astype(np.int64)
    sections = run.sections(name)

    if len(np.unique(ids)) == len(ids):
        return
    assert sections is not None, f"{name} repeats gids but names no sections"

    sections = np.asarray(sections)
    assert sections.shape == ids.shape
    pairs = list(zip(ids.tolist(), sections.tolist(), strict=True))
    assert len(set(pairs)) == len(pairs), f"two {name} rows name the same compartment"


def test_currents_and_voltage_name_a_compartment_alike(run):
    current, voltage = run.current_sections, run.voltage_sections
    if current is None or voltage is None:
        pytest.skip("this backend names the rows of at most one series")

    named: dict[int, set[str]] = {}
    for gid, section in zip(run.voltage_ids.tolist(), voltage.tolist(), strict=True):
        named.setdefault(int(gid), set()).add(str(section))
    for gid, section in zip(run.current_ids.tolist(), current.tolist(), strict=True):
        assert str(section) in named.get(int(gid), set()), (
            f"gid {gid}'s current row {section!r} names no voltage row"
        )
