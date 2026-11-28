import gc

import pytest

from livn.types import Env, Snapshot


class _Stub(Env):
    def __init__(self):
        self.reseeded = []
        self.freed = []

    def reseed_noise(self, stream=None):
        self.reseeded.append(stream)
        return self

    def snapshot(self, stream=0):
        return Snapshot(self, 12.5, stream, {"x": 1}, release=self.freed.append)


def test_a_backend_without_snapshots_says_so():
    with pytest.raises(NotImplementedError, match="SNAPSHOT"):
        _Stub().save_state()
    with pytest.raises(NotImplementedError, match="SNAPSHOT"):
        _Stub().restore_state(None)


def test_a_snapshot_restores_only_into_the_env_that_saved_it():
    env, other = _Stub(), _Stub()
    snapshot = env.snapshot()
    assert env._snapshot_state(snapshot) == {"x": 1}
    with pytest.raises(ValueError, match="another env"):
        other._snapshot_state(snapshot)
    with pytest.raises(TypeError):
        env._snapshot_state({"x": 1})


def test_the_snapshots_own_stream_continues_it_and_any_other_reseeds():
    env = _Stub()
    snapshot = env.snapshot(stream=3)
    env._continue_noise(snapshot, None)
    env._continue_noise(snapshot, 3)
    assert env.reseeded == []
    env._continue_noise(snapshot, 4)
    assert env.reseeded == [4]


def test_closing_the_env_releases_its_snapshots_once():
    env = _Stub()
    snapshot = env.snapshot()
    env._release_snapshots()
    assert env.freed == [{"x": 1}]
    with pytest.raises(ValueError, match="released"):
        env._snapshot_state(snapshot)
    snapshot.release()
    del snapshot
    gc.collect()
    assert env.freed == [{"x": 1}]


def test_a_dropped_snapshot_frees_what_it_held():
    env = _Stub()
    env.snapshot()
    gc.collect()
    assert env.freed == [{"x": 1}]
