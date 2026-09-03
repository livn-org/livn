import os

_BACKEND = ""


def _autodetect() -> str:
    try:
        from livn.backend.native import _lib

        return "native" if _lib.available() else ""
    except Exception:
        return ""


# set but empty is the explicit spelling of "no backend", only absent autodetects
_BACKEND = os.environ["LIVN_BACKEND"] if "LIVN_BACKEND" in os.environ else _autodetect()


def backend():
    return _BACKEND
