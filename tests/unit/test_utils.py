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


def test_stable_hash_is_a_function_of_the_key_not_the_batch():
    import numpy as np

    from livn.utils import P

    keys = np.arange(500)
    full = P.stable_uniform(keys)

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(keys)[:120]
    assert np.array_equal(P.stable_uniform(shuffled), full[shuffled])

    assert np.array_equal(P.stable_uniform(keys[:10]), full[:10])
    assert float(P.stable_uniform(7)) == full[7]


def test_stable_uniform_covers_its_range_without_collapsing():
    import numpy as np

    from livn.utils import P

    keys = np.arange(4000)
    values = P.stable_uniform(keys)
    assert values.min() >= 0.0 and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.02
    assert len(set(P.stable_hash(keys).tolist())) == len(keys), "hash collided"

    counts, _ = np.histogram(values, bins=10, range=(0.0, 1.0))
    assert counts.min() > len(keys) / 10 * 0.8, f"clumped: {counts}"

    scaled = P.stable_uniform(keys, high=2.0 * np.pi)
    assert scaled.max() < 2.0 * np.pi
    assert np.allclose(scaled, values * 2.0 * np.pi)


def test_stable_hash_does_not_map_zero_to_zero():
    """Key 0 under seed 0 are both defaults and 0 is a real gid; the bare
    SplitMix64 finalizer would hand back exactly 0.0."""
    from livn.utils import P

    assert P.stable_hash(0) != 0
    assert 0.0 < float(P.stable_uniform(0)) < 1.0


def test_stable_hash_separates_neighbouring_keys_and_seeds():
    import numpy as np

    from livn.utils import P

    neighbours = P.stable_uniform(np.arange(8))
    assert np.abs(np.diff(neighbours)).min() > 0.01, "adjacent gids came out alike"
    assert not np.array_equal(neighbours, P.stable_uniform(np.arange(8), seed=1))


def test_stable_hash_summing_two_keys_is_the_wrong_way_to_compose():
    from livn.utils import P

    def compose(a, b):
        return int(P.stable_hash(b, seed=int(P.stable_hash(a))))

    assert compose(0, 1) != compose(1, 0)
    pairs = [(a, b) for a in range(12) for b in range(12)]
    assert len({compose(a, b) for a, b in pairs}) == len(pairs)


def test_stable_hash_does_not_warn_on_scalars():
    import warnings

    from livn.utils import P

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        P.stable_hash(0)
        P.stable_hash(2**63 + 5, seed=2**63 + 7)
        P.stable_uniform(11, seed=3)
