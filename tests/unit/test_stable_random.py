import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "systems"))

generate_2d = pytest.importorskip(
    "generate_2d", reason="the generator needs neuroh5 and machinable"
)
StableRandom = generate_2d.StableRandom


GOLDEN = np.array(
    [0.84683114, 0.3904857, 0.9566024, 0.9521333, 0.55634326, 0.60935867],
    dtype=np.float32,
)


def test_stream_is_pinned():
    drawn = StableRandom(10921, "connectivity", "EXC", "EXC").random(6, at=0)
    np.testing.assert_array_equal(drawn, GOLDEN)


def test_uniform_is_pinned():
    drawn = StableRandom(10921, "coordinates", "EXC").uniform(-100.0, 1500.0, 4)
    np.testing.assert_array_equal(
        drawn,
        np.array([24.037933, 1075.9879, 857.3267, 349.50467], dtype=np.float32),
    )


def test_key_is_derived_from_the_name_not_the_order():
    a = StableRandom(1, "connectivity", "EXC", "INH")
    b = StableRandom(1, "connectivity", "INH", "EXC")
    assert not np.array_equal(a.random(4), b.random(4))
    assert np.array_equal(
        StableRandom(1, "connectivity", "EXC", "INH").random(4),
        StableRandom(1, "connectivity", "EXC", "INH").random(4),
    )


def test_at_is_position_addressed():
    """What makes the connectivity draw independent of CONNECTIVITY_CHUNK."""
    r = StableRandom(7, "connectivity", "EXC", "EXC")
    whole = np.concatenate([r.random(10, at=j * 10) for j in range(4)])
    r2 = StableRandom(7, "connectivity", "EXC", "EXC")
    in_two = np.concatenate(
        [
            np.concatenate([r2.random(10, at=j * 10) for j in range(lo, lo + 2)])
            for lo in (0, 2)
        ]
    )
    np.testing.assert_array_equal(whole, in_two)


def test_draws_are_in_range():
    drawn = StableRandom(3, "x").random(10_000)
    assert drawn.dtype == np.float32
    assert drawn.min() >= 0.0
    assert drawn.max() < 1.0
