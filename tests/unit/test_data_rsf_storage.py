import json
import os
import pickle

import numpy as np
import pytest

pytest.importorskip("zarr", minversion="3")

import livn.data
from livn.data import Missing, rsf
from livn.data.rsf import RsfRecording, SamplesMissing

CHUNKS = (100, 2)  # 1000 frames in ten time chunks


def _chunk(path, i):
    return os.path.join(path, "traces.zarr", "traces_seg0", "c", str(i), "0")


@pytest.fixture
def session(tmp_path, physical_session):
    return physical_session(str(tmp_path / "s"), trace_chunks=CHUNKS)


def test_a_removed_chunk_raises_instead_of_reading_zeros(session):
    whole = RsfRecording(session).read(0, 0, 1000)
    for i in (4, 5):
        os.remove(_chunk(session, i))
    recording = RsfRecording(session)

    np.testing.assert_array_equal(recording.read(0, 0, 400), whole[:400])
    np.testing.assert_array_equal(recording.read(0, 600, 400), whole[600:])
    with pytest.raises(SamplesMissing, match=r"\[400, 600\)") as e:
        recording.read(0, 350, 200)
    assert isinstance(e.value, Missing)
    assert e.value.ranges == [(400, 600)]
    assert e.value.keys == [
        "traces.zarr/traces_seg0/c/4/0",
        "traces.zarr/traces_seg0/c/5/0",
    ]
    assert "fill_missing=True" in str(e.value)

    filled = RsfRecording(session, fill_missing=True).read(0, 350, 300)
    np.testing.assert_array_equal(filled[:50], whole[350:400])
    assert not filled[50:250].any()
    assert pickle.loads(
        pickle.dumps(RsfRecording(session, fill_missing=True))
    ).fill_missing


def test_a_chunk_is_checked_on_every_read_of_a_growing_store(session):
    recording = RsfRecording(session)
    recording.read(0, 700, 100)
    with open(_chunk(session, 7), "rb") as f:
        kept = f.read()
    os.remove(_chunk(session, 7))
    with pytest.raises(SamplesMissing):
        recording.read(0, 700, 100)
    with open(_chunk(session, 7), "wb") as f:
        f.write(kept)
    assert recording.read(0, 700, 100).any()


def test_an_offload_marker_names_the_copy_that_holds_the_chunks(session):
    key = "traces.zarr/traces_seg0/c/2/0"
    marker = {
        "store_id": "s1",
        "store_name": "archive",
        "store_url": "s3://bucket/lab/recordings/s",
        "recording_id": "0f6c1d2e3b4a59687766554433221100",
        "removed": {key: 400},
    }
    with open(os.path.join(session, ".acme-offloaded.json"), "w") as f:
        json.dump(marker, f)
    os.remove(_chunk(session, 2))
    with pytest.raises(SamplesMissing, match="removed here after upload") as e:
        RsfRecording(session).read(0, 250, 10)
    assert e.value.elsewhere == "s3://bucket/lab/recordings/s"


def test_the_writer_keeps_chunks_that_are_all_zeros(tmp_path):
    import zarr

    group = zarr.open_group(str(tmp_path / "z"), mode="w")
    rsf._array(group, "traces_seg0", np.zeros((20, 2), np.int16), chunks=(10, 2))
    for i in (0, 1):
        assert os.path.isfile(tmp_path / "z" / "traces_seg0" / "c" / str(i) / "0")


class _MemoryStorage:
    """A :class:`rsf.Storage` over a dict of key to bytes, standing in for an object store."""

    def __init__(self, files):
        self.files = dict(files)
        self.reads = 0

    def read(self, path):
        self.reads += 1
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    def exists(self, path):
        return path in self.files

    def list(self, path):
        prefix = f"{path.rstrip('/')}/" if path else ""
        return sorted(
            {k[len(prefix) :].split("/")[0] for k in self.files if k.startswith(prefix)}
        )


@pytest.fixture
def remote(session):
    """The session's files copied under ``lab/recordings/s`` of a :class:`_MemoryStorage`."""
    ext = os.path.join(session, "traces.zarr", "rsf", "ext", "acme")
    os.makedirs(os.path.join(ext, "closed-loop"))
    with open(os.path.join(ext, "closed-loop", "a.jsonl"), "w") as f:
        f.write('{"t": 1}\n{"t": 2}\n')
    with open(os.path.join(ext, "stats.json"), "w") as f:
        json.dump({"n": 3}, f)
    files = {}
    for folder, _, names in os.walk(session):
        for name in names:
            local = os.path.join(folder, name)
            key = os.path.relpath(local, session).replace(os.sep, "/")
            with open(local, "rb") as f:
                files[f"lab/recordings/s/{key}"] = f.read()
    return _MemoryStorage(files)


def test_the_local_storage_is_a_storage():
    assert isinstance(rsf.LocalStorage(), rsf.Storage)


def test_a_session_reads_through_a_storage(session, remote):
    assert isinstance(remote, rsf.Storage)
    local = RsfRecording(session)
    there = livn.data.open("lab/recordings/s", storage=remote)
    assert isinstance(there, RsfRecording)
    np.testing.assert_array_equal(there.read(0, 0, 1000), local.read(0, 0, 1000))
    assert there.label == local.label
    assert there.channel_ids.tolist() == local.channel_ids.tolist()
    again = pickle.loads(pickle.dumps(there))
    assert again.path == "lab/recordings/s"
    assert isinstance(again.storage, _MemoryStorage)

    assert there.ext_files("acme") == ["closed-loop/a.jsonl", "stats.json"]
    assert there.ext("acme", "closed-loop/a.jsonl") == [{"t": 1}, {"t": 2}]
    assert there.ext("acme", "stats.json") == {"n": 3}
    assert there.ext("acme", "absent.json") is None
    path = there.ext_path("acme", "stats.json")
    assert path == "lab/recordings/s/traces.zarr/rsf/ext/acme/stats.json"
    assert json.loads(there.storage.read(path)) == {"n": 3}
    with there.open_ext("acme", "closed-loop/a.jsonl") as f:
        assert f.read().startswith(b'{"t": 1}')
    with pytest.raises(ValueError):
        there.ext_path("acme", "../observe.json")


def test_removed_chunks_raise_through_a_storage(remote):
    there = RsfRecording("lab/recordings/s", storage=remote)
    whole = there.read(0, 0, 1000)
    del remote.files["lab/recordings/s/traces.zarr/traces_seg0/c/3/0"]
    with pytest.raises(SamplesMissing, match=r"\[300, 400\)") as e:
        there.read(0, 250, 100)
    assert e.value.keys == ["traces.zarr/traces_seg0/c/3/0"]
    np.testing.assert_array_equal(there.read(0, 400, 600), whole[400:])
    filled = RsfRecording("lab/recordings/s", storage=remote, fill_missing=True)
    assert not filled.read(0, 300, 100).any()


def test_extensions_read_locally_the_same(session, remote):
    local = RsfRecording(session)
    assert local.ext_files("acme") == ["closed-loop/a.jsonl", "stats.json"]
    assert local.ext("acme", "closed-loop/a.jsonl") == [{"t": 1}, {"t": 2}]
    with local.open_ext("acme", "stats.json", "r") as f:
        assert json.load(f) == {"n": 3}
