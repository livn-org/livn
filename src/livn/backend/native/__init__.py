__all__ = ["Env", "available_threads", "num_threads", "set_num_threads"]


def __getattr__(name):
    if name == "Env":
        from livn.backend.native.env import Env

        return Env
    if name in ("available_threads", "num_threads", "set_num_threads"):
        from livn.backend.native import _lib

        return getattr(_lib, name)
    raise AttributeError(name)
