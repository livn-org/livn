import os

_BACKEND = "brian2"


if "LIVN_BACKEND" in os.environ:
    _backend = os.environ["LIVN_BACKEND"]
    if _backend:
        _BACKEND = _backend


def backend():
    return _BACKEND
