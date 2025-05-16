import numpy as np
import pytest
from mpi4py import MPI

from livn.utils import P, merge, merge_array, merge_dict


def test_utils_merge_dict():
    m = merge_dict(
        [
            {1: np.array([1, 2]), 2: np.array([3, 4])},
            {1: np.array([5, 6]), 3: np.array([7, 8])},
        ]
    )

    assert np.array_equal(m[1], np.array([1, 2, 5, 6]))
    assert np.array_equal(m[2], np.array([3, 4]))
    assert np.array_equal(m[3], np.array([7, 8]))
    assert 4 not in m

    assert merge_dict([]) == {}

    single_dict = {1: np.array([1, 2])}
    assert merge_dict(single_dict) == single_dict


def test_utils_merge_array():
    arrays = [np.array([1, 2]), np.array([3, 4, 5])]
    merged = merge_array(arrays)
    assert np.array_equal(merged, np.array([1, 2, 3, 4, 5]))

    arrays_with_none = [np.array([1, 2]), None, np.array([]), np.array([3, 4])]
    merged = merge_array(arrays_with_none)
    assert np.array_equal(merged, np.array([1, 2, 3, 4]))

    assert len(merge_array([])) == 0
    assert len(merge_array([None, np.array([])])) == 0


def test_utils_merge():
    arrays = [np.array([1, 2]), np.array([3, 4])]
    result = merge(arrays)
    assert len(result) == 4
    assert np.array_equal(result, np.array([1, 2, 3, 4]))

    dicts = [{1: np.array([1, 2])}, {1: np.array([3, 4]), 2: np.array([5, 6])}]
    result = merge(dicts)
    assert len(result) == 2
    assert np.array_equal(result[1], np.array([1, 2, 3, 4]))
    assert np.array_equal(result[2], np.array([5, 6]))

    single_dict = {1: np.array([1, 2])}
    result = merge(single_dict)
    assert len(result) == 1
    assert np.array_equal(result[1], np.array([1, 2]))

    arrays = [np.array([1, 2]), np.array([3, 4])]
    dicts = [{1: np.array([5, 6])}, {1: np.array([7, 8]), 2: np.array([9, 10])}]
    result = merge(arrays, dicts)
    assert len(result) == 2
    assert np.array_equal(result[0], np.array([1, 2, 3, 4]))
    assert np.array_equal(result[1][1], np.array([5, 6, 7, 8]))
    assert np.array_equal(result[1][2], np.array([9, 10]))

    assert merge(None, None) == (None, None)


def test_utils_P(monkeypatch):
    im = __import__

    def mock_import(*args, **kwargs):
        if args[0] == "mpi4py":
            raise ImportError("No module named 'mpi4py'")

        return im(*args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    assert P.is_root()

    a = {"A": 1}
    b = {"B": 2}

    assert P.gather(a) == [a]
    assert P.merge(P.gather(b)) == b

    assert P.gather(a, b) == ([a], [b])
    gb, ga = P.gather(b, a)
    ma, mb = P.merge(ga, gb)
    assert ma == a
    assert mb == b

    assert P.broadcast(a) == a
    assert P.broadcast(b, a) == (b, a)


@pytest.mark.mpiexec(timeout=10)
@pytest.mark.parametrize(
    "mpiexec_n",
    [
        1,
        2,
        4,
    ],
)
def test_utils_P_parallel(mpiexec_n):
    rank = MPI.COMM_WORLD.Get_rank()

    assert P.is_root() is not bool(rank)

    a = [rank]
    b = {"X": [rank]}

    # single
    p = P.gather(a)
    if P.is_root():
        assert p == [[i] for i in range(mpiexec_n)]
    else:
        assert p is None

    if P.is_root():
        assert np.array_equal(P.merge(p), np.arange(mpiexec_n))

    # multiple
    gb, ga = P.gather(b, a)
    if P.is_root():
        ma, mb = P.merge(ga, gb)
        assert np.array_equal(ma, np.arange(mpiexec_n))
        assert np.array_equal(mb["X"], np.arange(mpiexec_n))
    else:
        assert ga is None
        assert gb is None

    p = P.gather(a, all=True)
    assert p == [[i] for i in range(mpiexec_n)]

    p = P.gather(b, all=True)
    assert p == [{"X": [i]} for i in range(mpiexec_n)]

    gb, ga = P.gather(b, a, all=True)
    assert ga == [[i] for i in range(mpiexec_n)]
    assert gb == [{"X": [i]} for i in range(mpiexec_n)]

    assert P.broadcast(a) == [0]
    bb, ba = P.broadcast(b, a)
    assert bb == {"X": [0]}
    assert ba == [0]
