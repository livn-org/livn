"""A projected recording stands in for the membrane currents only when it applies."""

from __future__ import annotations

import numpy as np
import pytest

from livn.io import MEA, DispersivePointSource, Medium
from livn.io.recording import ExtracellularPotential, gain_digest

DT = 0.1


def _setup(mea, n_sources=60, steps=80, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-100.0, 1500.0, size=(n_sources, 2))
    coordinates = np.column_stack(
        [np.arange(n_sources), xy, np.full(n_sources, 5.0)]
    ).astype(np.float64)
    currents = rng.normal(size=(n_sources, steps)).astype(np.float32)
    distances = mea.distances(coordinates)
    gain = np.asarray(mea.source_gain(distances), dtype=np.float64)
    return coordinates, currents, gain


def _call(mea, coordinates, **conditions):
    return np.asarray(
        ExtracellularPotential()(
            io=mea, neuron_coordinates=coordinates, dt=DT, **conditions
        )
    )


@pytest.mark.parametrize(
    "electrical", [None, DispersivePointSource()], ids=["point", "dispersive"]
)
def test_a_projection_finishes_to_what_the_currents_give(electrical):
    mea = MEA() if electrical is None else MEA(medium=Medium(electrical=electrical))
    coordinates, currents, gain = _setup(mea)
    assert mea.projects

    full = _call(mea, coordinates, membrane_currents_uA=currents)
    projected = _call(
        mea,
        coordinates,
        extracellular_uV=gain @ currents.astype(np.float64),
        extracellular_dt=DT,
        extracellular_gain=gain_digest(gain),
    )

    scale = np.abs(full).max()
    assert scale > 0
    # the currents path sums float32 values, the projection float64
    np.testing.assert_allclose(projected, full, rtol=1e-5, atol=1e-5 * scale)


def test_a_projection_is_preferred_when_both_apply():
    mea = MEA()
    coordinates, currents, gain = _setup(mea)
    marker = np.full((gain.shape[0], currents.shape[1]), 7.0)

    out = _call(
        mea,
        coordinates,
        membrane_currents_uA=currents,
        extracellular_uV=marker,
        extracellular_dt=DT,
        extracellular_gain=gain_digest(gain),
    )

    np.testing.assert_array_equal(out, marker)


@pytest.mark.parametrize(
    "stale",
    [
        {"extracellular_gain": "0" * 32},
        {"extracellular_dt": 0.5},
        {"extracellular_gain": None},
    ],
    ids=["other-gain", "other-dt", "no-digest"],
)
def test_a_projection_that_does_not_apply_gives_way_to_the_currents(stale):
    mea = MEA()
    coordinates, currents, gain = _setup(mea)
    projection = {
        "extracellular_uV": np.full((gain.shape[0], currents.shape[1]), 7.0),
        "extracellular_dt": DT,
        "extracellular_gain": gain_digest(gain),
        **stale,
    }

    out = _call(mea, coordinates, membrane_currents_uA=currents, **projection)
    np.testing.assert_allclose(
        out, _call(mea, coordinates, membrane_currents_uA=currents)
    )

    with pytest.raises(ValueError, match="does not apply"):
        _call(mea, coordinates, **projection)


def test_a_changed_medium_invalidates_the_projection():
    mea = MEA()
    coordinates, currents, gain = _setup(mea)
    other = mea.set_params({"medium-electrical-sigma_s_per_m": 0.6})

    with pytest.raises(ValueError, match="different gain"):
        _call(
            other,
            coordinates,
            extracellular_uV=gain @ currents.astype(np.float64),
            extracellular_dt=DT,
            extracellular_gain=gain_digest(gain),
        )


def test_a_per_pair_medium_cannot_be_projected():
    mea = MEA(medium=Medium(electrical=DispersivePointSource(per_pair=True)))
    coordinates, currents, gain = _setup(mea)
    assert not mea.projects

    with pytest.raises(ValueError, match="cannot be formed as a channel sum"):
        _call(
            mea,
            coordinates,
            extracellular_uV=gain @ currents.astype(np.float64),
            extracellular_dt=DT,
            extracellular_gain=gain_digest(gain),
        )


def test_neither_source_is_an_error_that_names_both():
    mea = MEA()
    coordinates, _, _ = _setup(mea)
    with pytest.raises(ValueError, match=r"membrane_current.*extracellular"):
        _call(mea, coordinates)
