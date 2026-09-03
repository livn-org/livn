__all__ = ["Env"]


def __getattr__(name):
    if name == "Env":
        from livn.backend.native.env import Env

        return Env
    raise AttributeError(name)
