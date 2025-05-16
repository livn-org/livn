import collections
import json
from typing import TYPE_CHECKING, Any, Optional, Union

from livn.types import Array

if TYPE_CHECKING:
    import numpy as np
    from mpi4py import MPI


def lnp() -> "np":
    from livn.backend import backend

    if backend() in ["diffrax"]:
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
    def is_root(comm: Optional["MPI.Intracomm"] = None):
        try:
            from mpi4py import MPI

            if comm is None:
                comm = MPI.COMM_WORLD

            return comm.Get_rank() == 0
        except ImportError:
            return True

    @staticmethod
    def gather(*data, comm: Optional["MPI.Intracomm"] = None, all: bool = False):
        try:
            from mpi4py import MPI
        except ImportError:
            if len(data) == 1:
                return [data[0]]
            return tuple([[d] for d in data])

        if comm is None:
            comm = MPI.COMM_WORLD

        op = comm.allgather if all is True else comm.gather

        gathered = tuple(
            [op(dict(d) if isinstance(d, collections.abc.Mapping) else d) for d in data]
        )

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
                if comm is False or P.is_root(comm=comm):
                    with open(serialized, "r") as f:
                        serialized = json.load(f)
                if comm is not False:
                    serialized = P.broadcast(serialized, comm=comm)
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
