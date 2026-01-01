import sys
import importlib
import collections
import json
from typing import TYPE_CHECKING, Any, Optional, Union

from livn.types import Array

if TYPE_CHECKING:
    import numpy as np
    from mpi4py import MPI


def import_object_by_path(path):
    module_path, _, obj_name = path.rpartition(".")
    if module_path == "__main__" or module_path == "":
        module = sys.modules["__main__"]
    else:
        module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def lnp() -> "np":
    from livn.backend import backend

    if "ax" in backend():
        import jax.numpy as np
    else:
        import numpy as np

    return np


def merge_array(data):
    if data is None:
        return None
    np = lnp()
    filtered = [x for x in data if x is not None and len(x) > 0]
    return np.concatenate(filtered) if filtered else np.array([])


def sum_array(data):
    if data is None:
        return None
    np = lnp()
    filtered = [x for x in data if x is not None and getattr(x, "size", 0) > 0]
    if not filtered:
        return np.array([])

    try:
        stacked = np.stack(filtered, axis=0)
        return stacked.sum(axis=0)
    except Exception:
        # iterative fallback
        result = np.array(filtered[0], copy=True)
        for x in filtered[1:]:
            result = result + x
        return result


def merge_dict(data: list[dict[int, Array]]):
    np = lnp()
    merged_dict = {}

    if isinstance(data, collections.abc.Mapping):
        merged_dict = data
    else:
        for d in data:
            for k, v in d.items():
                if k in merged_dict:
                    merged_dict[k] = np.concatenate([merged_dict[k], v])
                else:
                    merged_dict[k] = v

    return merged_dict


def sum_dict(data: list[dict[int, Array]]):
    if isinstance(data, collections.abc.Mapping):
        return data

    reduced: dict[int, Array] = {}
    for d in data:
        for k, v in d.items():
            if k in reduced:
                reduced[k] = reduced[k] + v
            else:
                reduced[k] = v
    return reduced


def merge(*data):
    results = []
    for d in data:
        if isinstance(d, dict) or (
            isinstance(d, list) and all(isinstance(x, dict) for x in d)
        ):
            results.append(merge_dict(d if isinstance(d, list) else [d]))
        else:
            results.append(merge_array(d))

    if len(data) == 1:
        return results[0]

    return tuple(results)


def reduce_sum(
    *data,
    comm: Optional["MPI.Intracomm"] = None,
    root: int = 0,
    all: bool = False,
):
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = False

    import numpy as _np
    import builtins as _builtins

    def _mpi_sum(buf):
        if comm.Get_size() == 1:
            return buf

        flat = _np.ravel(buf)
        recv_flat = _np.empty_like(flat) if (all or comm.Get_rank() == root) else None
        if all:
            comm.Allreduce(flat, recv_flat, op=MPI.SUM)
            return recv_flat.reshape(buf.shape)
        else:
            comm.Reduce(flat, recv_flat, op=MPI.SUM, root=root)
            return recv_flat.reshape(buf.shape) if comm.Get_rank() == root else None

    results = []
    for d in data:
        if d is None:
            results.append(None)
            continue

        # dict reduce
        if isinstance(d, dict) or (
            isinstance(d, list) and _builtins.all(isinstance(x, dict) for x in d)
        ):
            results.append(sum_dict(d if isinstance(d, list) else [d]))
            continue

        # local list reduce
        if isinstance(d, (list, tuple)):
            results.append(sum_array(d))
            continue

        if MPI:
            if comm is None:
                comm = MPI.COMM_WORLD

            try:
                # TODO: we should fall back on mpi4jax if available
                sendbuf = _np.asarray(d)
            except Exception:
                # conversion failed (e.g. jax array is not numpy)
                if comm.Get_size() > 1:
                    # "failing" with None
                    reduced = None if not all else None
                else:
                    reduced = d
            else:
                if isinstance(sendbuf, _np.ndarray) and sendbuf.dtype == _np.dtype("O"):
                    try:
                        local = _np.array(d, dtype=_np.float64)
                    except Exception:
                        if comm.Get_size() > 1:
                            reduced = None if not all else None
                        else:
                            reduced = d
                    else:
                        reduced = _mpi_sum(local)
                else:
                    reduced = _mpi_sum(sendbuf)
        else:
            reduced = d

        results.append(reduced)

    if len(data) == 1:
        return results[0]

    return tuple(results)


class P:
    @staticmethod
    def rank(comm: Optional["MPI.Intracomm"] = None) -> int:
        try:
            from mpi4py import MPI

            if comm is None:
                comm = MPI.COMM_WORLD

            return comm.Get_rank()
        except ImportError:
            return 0

    @staticmethod
    def size(comm: Optional["MPI.Intracomm"] = None) -> int:
        try:
            from mpi4py import MPI

            if comm is None:
                comm = MPI.COMM_WORLD

            return comm.Get_size()
        except ImportError:
            return 1

    @staticmethod
    def is_root(root: int = 0, comm: Optional["MPI.Intracomm"] = None):
        root = int(root)
        if root < 0:
            root = max(P.size(comm=comm) + root, 0)

        try:
            from mpi4py import MPI

            if comm is None:
                comm = MPI.COMM_WORLD

            return comm.Get_rank() == root
        except ImportError:
            return True

    @staticmethod
    def gather(
        *data, comm: Optional["MPI.Intracomm"] = None, all: bool = False, root: int = 0
    ):
        try:
            from mpi4py import MPI
        except ImportError:
            if len(data) == 1:
                return [data[0]]
            return tuple([[d] for d in data])

        if comm is None:
            comm = MPI.COMM_WORLD

        # we explicitly convert into plain python/numpy to prevent
        #  lowering into MPI_Gather which requires every rank
        #  to use the same element count; by handling things
        #  manually we stay in pickle-land to support
        #  arbitrary buffer sizes

        import numpy as _np

        def _materialize(value):
            if isinstance(value, collections.defaultdict):
                value = dict(value)
            if isinstance(value, collections.abc.Mapping):
                return {k: _materialize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                converted = [_materialize(v) for v in value]
                return tuple(converted) if isinstance(value, tuple) else converted
            if hasattr(value, "__array__"):
                try:
                    return _np.asarray(value)
                except Exception:
                    return value
            return value

        prepared = [_materialize(d) for d in data]

        rank = comm.Get_rank()
        size = comm.Get_size()

        if size == 1:
            single = tuple([[d] for d in prepared])
            return single[0] if len(data) == 1 else single

        def _collect_to_root():
            tag = 0xCAFE

            if rank == root:
                buffers = [[None] * size for _ in prepared]
                for idx, item in enumerate(prepared):
                    buffers[idx][root] = item
                for src in range(size):
                    if src == root:
                        continue
                    recv_payload = comm.recv(source=src, tag=tag)
                    if not isinstance(recv_payload, (list, tuple)):
                        raise RuntimeError("P.gather received invalid payload")
                    if len(recv_payload) != len(prepared):
                        raise RuntimeError("P.gather payload length mismatch")
                    for idx, item in enumerate(recv_payload):
                        buffers[idx][src] = item
                return tuple(buffers)
            else:
                comm.send(list(prepared), dest=root, tag=tag)
                return tuple([None] * len(prepared))

        if all:
            collected = _collect_to_root()
            results = []
            for idx in range(len(prepared)):
                value = collected[idx] if rank == root else None
                results.append(comm.bcast(value, root=root))
            gathered = tuple(results)
        else:
            gathered = _collect_to_root()
            if rank != root:
                gathered = tuple([None] * len(prepared))

        if len(data) == 1:
            return gathered[0]

        return gathered

    @staticmethod
    def broadcast(*data, comm: Optional["MPI.Intracomm"] = None):
        try:
            from mpi4py import MPI
        except ImportError:
            if len(data) == 1:
                return data[0]
            return data

        if comm is None:
            comm = MPI.COMM_WORLD

        broadcasted = tuple(
            [
                comm.bcast(dict(d) if isinstance(d, collections.abc.Mapping) else d)
                for d in data
            ]
        )

        if len(data) == 1:
            return broadcasted[0]

        return broadcasted

    @staticmethod
    def merge(*data):
        return merge(*data)

    @staticmethod
    def reduce_sum(
        *data,
        comm: Optional["MPI.Intracomm"] = None,
        root: int = 0,
        all: bool = False,
    ):
        return reduce_sum(*data, comm=comm, root=root, all=all)


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


class Jsonable:
    def as_json(self, stringify=True, default=serialize, **dumps_kwargs):
        serialized = self.serialize()
        if stringify:
            serialized = json.dumps(serialized, default=default, **dumps_kwargs)
        return serialized

    @classmethod
    def from_json(
        cls, serialized, comm: Union["MPI.Intracomm", bool, None] = None, **loads_kwargs
    ):
        if isinstance(serialized, str):
            if serialized.endswith(".json"):
                load_error = None
                data = None
                if comm is False or P.is_root(comm=comm):
                    try:
                        with open(serialized, "r") as f:
                            data = json.load(f)
                    except Exception as e:
                        load_error = e

                if comm is not False:
                    load_error = P.broadcast(load_error, comm=comm)
                    if load_error is not None:
                        raise load_error
                    serialized = P.broadcast(data, comm=comm)
                else:
                    if load_error is not None:
                        raise load_error
                    serialized = data
            else:
                serialized = json.loads(serialized, **loads_kwargs)
        return cls.unserialize(serialized)

    def clone(self):
        return self.__class__.from_json(self.as_json())

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def unserialize(cls, serialized: dict) -> Any:
        return cls(**serialized)


class DotDict:
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    @staticmethod
    def create(d):
        result = DotDict.dotdict()
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = DotDict.create(v)
            else:
                result[k] = v
        return result
