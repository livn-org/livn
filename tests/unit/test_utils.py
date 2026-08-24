import numpy as np

from livn.utils import P, merge, merge_array, merge_dict

try:
    import mpi4py  # noqa: F401

    _has_mpi4py = True
except ImportError:
    _has_mpi4py = False


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


def test_utils_reduce_sum_no_mpi(monkeypatch):
    im = __import__

    def mock_import(*args, **kwargs):
        if args[0] == "mpi4py":
            raise ImportError("No module named 'mpi4py'")
        return im(*args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    arr = np.array([1, 2, 3])
    assert np.array_equal(P.reduce_sum(arr), arr)

    assert np.array_equal(
        P.reduce_sum([np.array([1, 2]), np.array([3, 4])]), np.array([4, 6])
    )

    d = {1: np.array([1, 2])}
    rd_local = P.reduce_sum(d)
    assert set(rd_local.keys()) == {1}
    assert np.array_equal(rd_local[1], np.array([1, 2]))

    ra, rd = P.reduce_sum(arr, d)
    assert np.array_equal(ra, arr)
    assert set(rd.keys()) == {1}
    assert np.array_equal(rd[1], np.array([1, 2]))
