from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping

import numpy as _np

from livn.backend import backend
from livn.utils import import_instance, import_object_by_path

_USES_JAX = "ax" in backend()

if _USES_JAX:
    import equinox as eqx

    _Base = eqx.Module

    def module(cls):
        return cls

    def static_field(default):
        return eqx.field(default=default, static=True)

else:
    _Base = object

    def module(cls):
        return dataclasses.dataclass(cls)

    def static_field(default):
        return dataclasses.field(default=default)


def with_fields(obj, **fields):
    new = copy.copy(obj)
    for name, value in fields.items():
        object.__setattr__(new, name, value)
    return new


def cls_path(obj) -> str:
    cls = type(obj)
    module_name = "livn.io" if cls.__module__.startswith("livn.io.") else cls.__module__
    return f"{module_name}.{cls.__qualname__}"


def plain(value):
    """A JSON-ready copy of a field value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, _np.floating)):
        return float(value)
    return _np.asarray(value, dtype=float).tolist()


def coerce(old, new):
    if old is None or (isinstance(old, (float, int)) and not isinstance(old, bool)):
        return None if new is None else float(new)
    if isinstance(old, bool):
        return bool(new)
    from livn.utils import lnp

    return lnp().asarray(new, dtype=getattr(old, "dtype", None))


def resolve_component(spec, expected=None):
    """An instance from an instance, a `{"cls": ..., **kwargs}` document, a
    `{"cls": ..., "kwargs": {...}}` document or an ObjSpec."""
    if isinstance(spec, Mapping):
        document = dict(spec)
        if "cls" not in document:
            raise ValueError(
                f"a component document names its class under 'cls': {spec!r}"
            )
        cls = import_object_by_path(document.pop("cls"))
        kwargs = dict(document.pop("kwargs", None) or {})
        kwargs.update(document)
        return cls(**kwargs)
    if isinstance(spec, (str, tuple, list)):
        return import_instance(spec)
    return spec


class Component(_Base):
    def serialize(self) -> dict:
        document = {"cls": cls_path(self)}
        for f in dataclasses.fields(self):
            document[f.name] = plain(getattr(self, f.name))
        return document

    def set_params(self, params: dict):
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(k for k in params if k not in known)
        if unknown:
            raise KeyError(
                f"{unknown} are not parameters of {type(self).__name__}; "
                f"its parameters are {sorted(known)}"
            )
        return with_fields(
            self, **{k: coerce(getattr(self, k), v) for k, v in params.items()}
        )
