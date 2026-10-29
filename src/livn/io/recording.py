from __future__ import annotations

import functools
import hashlib
import inspect
import re
from dataclasses import dataclass

import numpy as _np

from livn.backend import backend
from livn.conditions import Conditions
from livn.io.base import Component, _plain, static_field
from livn.io.transfer import stage_key
from livn.utils import class_path, import_instance, is_traced, with_fields

_USES_JAX = "ax" in backend()

if _USES_JAX:
    import jax
    import jax.numpy as np
else:
    import numpy as np

__all__ = [
    "Chain",
    "ExtracellularPotential",
    "LightArtifact",
    "Sum",
    "apply",
    "as_composition",
    "electrode_light",
    "fill",
    "gain_digest",
    "is_stack",
    "is_transfer",
    "stack",
]

_KEY = "key"
_SIGNAL = "signal"
_IO = "io"
_RESERVED = (_KEY, _SIGNAL, _IO)
_RESERVED_NAMES = ("cls", "kwargs")


@dataclass(frozen=True)
class _Needs:
    transfer: bool
    keywords: tuple[tuple[str, bool], ...]  # (name, required)
    var_keyword: bool


def _callable_of(block):
    if inspect.ismethod(block):
        return block.__func__, True
    if inspect.isroutine(block) or isinstance(block, functools.partial):
        return block, False
    return type(block).__call__, True


def _needs(block) -> _Needs:
    fn, bound = _callable_of(block)
    try:
        return _needs_of(fn, bound)
    except TypeError:
        # an unhashable callable: read it every time
        return _needs_of.__wrapped__(fn, bound)


@functools.cache
def _needs_of(fn, bound: bool) -> _Needs:
    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as e:
        raise TypeError(f"{fn!r} has no signature to read its inputs from") from e
    if bound and parameters:
        parameters = parameters[1:]
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    transfer = bool(parameters) and (
        parameters[0].name == _SIGNAL and parameters[0].kind in positional
    )
    if transfer:
        parameters = parameters[1:]
    keywords, var_keyword = [], False
    for p in parameters:
        if p.kind is p.VAR_KEYWORD:
            var_keyword = True
        elif p.kind is p.VAR_POSITIONAL or p.kind is p.POSITIONAL_ONLY:
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)} takes `{p.name}` positionally; a "
                "block names what it reads as keywords (only a transfer's first "
                "parameter, `signal`, is positional)"
            )
        elif p.name == _SIGNAL:
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)} takes `signal` but not first: a "
                "transfer's first positional parameter is `signal`"
            )
        else:
            keywords.append((p.name, p.default is p.empty))
    return _Needs(transfer, tuple(keywords), var_keyword)


def is_transfer(block) -> bool:
    """Whether `block` acts on the signal so far (its first parameter is `signal`)."""
    if isinstance(block, _Composition):
        return False
    return _needs(block).transfer


def _snake(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return name.lower()


def _default_name(block) -> str:
    if isinstance(block, functools.partial):
        block = block.func
    if inspect.isroutine(block):
        return block.__name__
    # a class may give its own, shorter than its name
    return getattr(type(block), "default_name", None) or _snake(type(block).__name__)


def _label(name: str, block) -> str:
    return f"{name!r} ({type(block).__name__})"


def _same(a, b) -> bool:
    if a is b:
        return True
    if is_traced(a, b):
        return True  # nothing to compare under a transform
    a, b = _np.asarray(a), _np.asarray(b)
    return a.shape == b.shape and bool(_np.array_equal(a, b))


def fill(conditions, owner: str, **facts):
    filled = {}
    for name, value in facts.items():
        if value is None:
            continue
        given = getattr(conditions, name)
        if given is None:
            filled[name] = value
        elif not _same(given, value):
            raise ValueError(
                f"conditions.{name} disagrees with the {owner}'s own; leave it "
                f"unset and the {owner} fills it in"
            )
    return conditions.replace(**filled) if filled else conditions


def _held_samples(rows: int, dt, inputs_dt) -> int:
    if inputs_dt is None:
        return int(rows)
    if dt is None:
        raise ValueError("holding inputs on the signal's grid needs its dt")
    return round(int(rows) * float(inputs_dt) / float(dt))


def _hold(inputs, dt, inputs_dt):
    if inputs_dt is None:
        return inputs
    n = _held_samples(inputs.shape[0], dt, inputs_dt)
    rows = _np.floor(_np.arange(n) * float(dt) / float(inputs_dt) + 1e-9).astype(
        _np.int64
    )
    return inputs[_np.minimum(rows, inputs.shape[0] - 1)]


def electrode_light(inputs, reach, dt=None, inputs_dt=None):
    inputs = np.asarray(inputs)
    if inputs.ndim != 2:
        raise ValueError(
            f"inputs is [timestep, n_input_channels], not shape {tuple(inputs.shape)}"
        )
    reach = np.asarray(reach)
    if reach.ndim != 2 or reach.shape[0] != inputs.shape[1]:
        raise ValueError(
            f"inputs drive {inputs.shape[1]} channels and the reach is "
            f"{tuple(reach.shape)}, not [n_input_channels, n_channels]"
        )
    return np.matmul(_hold(inputs, dt, inputs_dt), reach).T


def _call(name: str, block, conditions, key, signal=None, io=None):
    needs = _needs(block)
    kwargs = {}
    for field, required in needs.keywords:
        if field == _KEY:
            if key is None and required:
                raise ValueError(f"{_label(name, block)} needs a key")
            if key is not None:
                kwargs[_KEY] = key
            continue
        if field == _IO:
            if io is None and required:
                raise ValueError(
                    f"{_label(name, block)} needs the io it records through: "
                    "evaluate the composition with env.potential_recording or "
                    "io.observe, or pass it, recording(conditions, io=io)"
                )
            if io is not None:
                kwargs[_IO] = io
            continue
        value = getattr(conditions, field)
        if value is None:
            if required:
                raise ValueError(
                    f"{_label(name, block)} needs `{field}`, which the conditions "
                    "do not set"
                )
            continue
        kwargs[field] = value
    if needs.var_keyword:
        for field, value in conditions.as_dict().items():
            kwargs.setdefault(field, value)
    if needs.transfer:
        return block(signal, **kwargs)
    return block(**kwargs)


def apply(block, conditions, *, key=None, signal=None, io=None, name=None):
    if isinstance(block, _Composition):
        return block(conditions, key=key, io=io)
    name = _default_name(block) if name is None else name
    return _call(name, block, conditions, key, signal, io)


def _evaluate(name: str, block, conditions, key, signal=None, io=None):
    if isinstance(block, _Composition):
        return block._run(conditions, key, io)
    return _call(name, block, conditions, key, signal, io)


class _Composition:
    def __init__(self, *blocks, **named):
        if not blocks and not named:
            raise TypeError(f"a {type(self).__name__} needs at least one block")
        names, parts = [], []
        for block in blocks:
            block = import_instance(block)
            names.append(_default_name(block))
            parts.append(block)
        for name, block in named.items():
            names.append(name)
            parts.append(import_instance(block))
        self.names = tuple(names)
        self.blocks = tuple(parts)
        self._validate()

    @classmethod
    def _of(cls, blocks, names):
        new = cls.__new__(cls)
        new.blocks = tuple(blocks)
        new.names = tuple(names)
        return new

    def _validate(self) -> None:
        for name, block in zip(self.names, self.blocks, strict=True):
            if not name.isidentifier() or name in _RESERVED_NAMES:
                raise ValueError(
                    f"{name!r} cannot name a block: a name is an identifier "
                    f"(parameter paths split on '-'), and not one of {_RESERVED_NAMES}"
                )
            if not callable(block):
                raise TypeError(f"block {name!r} is {block!r}, which is not callable")
        seen: dict[str, object] = {}
        for name, block in self._flat():
            if name in seen:
                raise ValueError(
                    f"two blocks are named {name!r} in this composition; name them "
                    f"apart, e.g. {type(self).__name__}({name}_a=..., {name}_b=...), "
                    "so that parameter paths never depend on position"
                )
            seen[name] = block

    def _flat(self):
        """`(name, block)` of every block, nested ones included, depth first."""
        for name, block in zip(self.names, self.blocks, strict=True):
            yield name, block
            if isinstance(block, _Composition):
                yield from block._flat()

    def __getitem__(self, name: str):
        for own, block in self._flat():
            if own == name:
                return block
        raise KeyError(
            f"{name!r} names no block of this {type(self).__name__}; its blocks are "
            f"{[n for n, _ in self._flat()]}"
        )

    def __contains__(self, name) -> bool:
        return any(own == name for own, _ in self._flat())

    def __len__(self) -> int:
        return len(self.blocks)

    def items(self):
        """`(name, block)` of the direct blocks, in order."""
        return tuple(zip(self.names, self.blocks, strict=True))

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}={b!r}" for n, b in self.items())
        return f"{type(self).__name__}({inner})"

    def requires(self) -> frozenset[str]:
        """The condition fields the composition reads (`key` and `io` excluded)."""
        names: set[str] = set()
        for _, block in self._leaves():
            names.update(f for f, _ in _needs(block).keywords if f not in _RESERVED)
        return frozenset(names)

    def reads_io(self) -> bool:
        """Whether a block names `io`, the IO the composition records through."""
        return any(
            any(f == _IO for f, _ in _needs(block).keywords)
            for _, block in self._leaves()
        )

    def reads_everything(self) -> bool:
        """Whether a block takes `**kwargs`, and so every field that is set."""
        return any(_needs(block).var_keyword for _, block in self._leaves())

    def _leaves(self):
        return [(n, b) for n, b in self._flat() if not isinstance(b, _Composition)]

    def check(self, conditions=Conditions) -> None:
        """Every name a block asks for is a field of `conditions` (a class or an instance)."""
        cls = conditions if isinstance(conditions, type) else type(conditions)
        fields = set(cls.field_names())
        for name, block in self._leaves():
            for field, _required in _needs(block).keywords:
                if field not in _RESERVED and field not in fields:
                    raise TypeError(
                        f"{_label(name, block)} asks for `{field}`, which is not a "
                        f"field of {cls.__name__}; its fields are {sorted(fields)} "
                        "(subclass Conditions to add one)"
                    )

    @property
    def needs_dt(self) -> bool:
        return any(getattr(b, "needs_dt", False) for _, b in self._leaves())

    def __call__(self, conditions=None, key=None, io=None):
        conditions = Conditions() if conditions is None else conditions
        self.check(conditions)
        self._check_window(conditions)
        return self._run(conditions, key, io)

    def _check_window(self, conditions) -> None:
        if conditions.inputs is None or conditions.membrane_currents_uA is None:
            return
        if "inputs" not in self.requires() and not self.reads_everything():
            return
        n = int(conditions.membrane_currents_uA.shape[-1])
        rows = int(conditions.inputs.shape[0])
        held = _held_samples(rows, conditions.dt, conditions.inputs_dt)
        if held != n:
            spans = (
                f"{rows} rows"
                if conditions.inputs_dt is None
                else f"{rows} rows of {conditions.inputs_dt} ms ({held} samples)"
            )
            raise ValueError(
                f"conditions.inputs spans {spans} and membrane_currents_uA {n} "
                "samples; give the inputs over the same window (pad with zeros), "
                "and inputs_dt when they are on a coarser grid"
            )

    def _run(self, conditions, key, io=None):
        raise NotImplementedError

    def replace(self, name: str, block):
        if name not in self:
            raise KeyError(
                f"{name!r} names no block of this {type(self).__name__}; its blocks "
                f"are {[n for n, _ in self._flat()]}"
            )
        parts = []
        for own, part in self.items():
            if own == name:
                parts.append(block)
            elif isinstance(part, _Composition) and name in part:
                parts.append(part.replace(name, block))
            else:
                parts.append(part)
        new = type(self)._of(parts, self.names)
        new._validate()
        return new

    def set_params(self, params: dict):
        grouped: dict[str, dict] = {}
        for path, value in params.items():
            name, _, rest = path.partition("-")
            if not rest or name not in self:
                raise KeyError(
                    f"{path!r} is not a parameter of this {type(self).__name__}: a "
                    "path is '<block>-<parameter>' and its blocks are "
                    f"{[n for n, _ in self._flat()]}"
                )
            grouped.setdefault(name, {})[rest] = value
        new = self
        for name, values in grouped.items():
            block = new[name]
            if not hasattr(block, "set_params"):
                raise KeyError(f"block {name!r} has no parameters to set")
            new = new.replace(name, block.set_params(values))
        return new

    def serialize(self) -> dict:
        def document(name, block):
            if not hasattr(block, "serialize"):
                raise TypeError(
                    f"{_label(name, block)} cannot be written down: it has no "
                    "serialize()"
                )
            doc = block.serialize()
            if "cls" not in doc:
                doc = {"cls": class_path(block), **doc}
            return doc

        return {
            "cls": class_path(self),
            "kwargs": {n: document(n, b) for n, b in self.items()},
        }


class Sum(_Composition):
    """Sum sources."""

    def _validate(self) -> None:
        super()._validate()
        for name, block in self.items():
            if is_transfer(block):
                raise TypeError(
                    f"{_label(name, block)} is a transfer (its first parameter is "
                    "`signal`); a Sum adds sources, and transfers follow it in a "
                    "Chain: Chain(Sum(...), transfer)"
                )

    def _run(self, conditions, key, io=None):
        total = None
        shape = None
        for k, (name, block) in enumerate(self.items()):
            part = _evaluate(name, block, conditions, stage_key(key, k), io=io)
            if np.ndim(part) != 2:
                raise ValueError(
                    f"{_label(name, block)} made a signal of shape "
                    f"{tuple(np.shape(part))}, not [C, T]"
                )
            if shape is None:
                shape = tuple(np.shape(part))
            elif tuple(np.shape(part)) != shape:
                raise ValueError(
                    f"the parts of a Sum must have one shape [C, T]: "
                    f"{_label(self.names[0], self.blocks[0])} is {shape} and "
                    f"{_label(name, block)} {tuple(np.shape(part))}"
                )
            total = part if total is None else total + part
        return total


class Chain(_Composition):
    """A source and the transfers its signal passes through, in order."""

    def _validate(self) -> None:
        super()._validate()
        name, head = self.items()[0]
        if is_transfer(head):
            raise TypeError(
                f"the head of a Chain makes the signal, and {_label(name, head)} is "
                "a transfer (its first parameter is `signal`)"
            )
        for name, block in self.items()[1:]:
            if not is_transfer(block):
                raise TypeError(
                    f"{_label(name, block)} follows the head of a Chain but is a "
                    "source; add it to the head, Chain(Sum(head, it), ...), or "
                    "give it `signal` as its first parameter"
                )

    @property
    def head(self):
        return self.blocks[0]

    @property
    def transfers(self) -> tuple:
        return self.blocks[1:]

    def head_only(self) -> Chain:
        return Chain._of(self.blocks[:1], self.names[:1])

    def _run(self, conditions, key, io=None):
        n = len(self.blocks) - 1
        # the transfers take the keys a stack's transfers always had
        signal = _evaluate(
            self.names[0], self.head, conditions, stage_key(key, n), io=io
        )
        for k, (name, block) in enumerate(self.items()[1:]):
            signal = _evaluate(
                name, block, conditions, stage_key(key, k), signal, io=io
            )
        return signal


def _stack_names(transfers) -> tuple[str, ...]:
    names, count = ["extracellular"], {}
    for t in transfers:
        name = _default_name(t)
        count[name] = count.get(name, 0) + 1
        names.append(name if count[name] == 1 else f"{name}_{count[name]}")
    return tuple(names)


def stack(*transfers) -> Chain:
    chain = Chain._of((ExtracellularPotential(), *transfers), _stack_names(transfers))
    chain._validate()
    return chain


def is_stack(recording) -> bool:
    return (
        isinstance(recording, Chain)
        and type(recording.head) is ExtracellularPotential
        and recording.names == _stack_names(recording.transfers)
    )


def as_composition(recording) -> _Composition:
    recording = import_instance(recording)
    if isinstance(recording, _Composition):
        return recording
    if is_transfer(recording):
        raise TypeError(
            f"{type(recording).__name__} is a transfer; a recording starts with a "
            "source, Chain(source, transfer)"
        )
    return Chain(recording)


def gain_digest(gain) -> str:
    values = _np.ascontiguousarray(_np.asarray(gain, dtype=_np.float64))
    digest = hashlib.sha256(repr(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()[:32]


class ExtracellularPotential(Component):
    """The extracellular potential induced by transmembrane currents."""

    default_name = "extracellular"

    def __call__(
        self,
        *,
        io,
        neuron_coordinates,
        membrane_currents_uA=None,
        dt=None,
        extracellular_uV=None,
        extracellular_dt=None,
        extracellular_gain=None,
    ):
        device = io.recording_device()
        distances = io.distances(neuron_coordinates)
        if extracellular_uV is not None:
            why = _projection_mismatch(
                device, distances, dt, extracellular_dt, extracellular_gain
            )
            if why is None:
                return device.finish_projection(distances, extracellular_uV, dt=dt)
            if membrane_currents_uA is None:
                raise ValueError(
                    f"the projected extracellular recording does not apply: {why}. "
                    "Record membrane_current to project the currents under these "
                    "conditions"
                )
        if membrane_currents_uA is None:
            raise ValueError(
                f"{type(self).__name__} needs `membrane_currents_uA` "
                "(env.record('membrane_current')) or `extracellular_uV` "
                "(env.record('extracellular')), which the conditions do not set"
            )
        return device.potential_recording(distances, membrane_currents_uA, dt=dt)

    def serialize(self) -> dict:
        return {"cls": class_path(self)}

    def set_params(self, params: dict) -> ExtracellularPotential:
        if params:
            raise KeyError(
                f"{sorted(params)}: ExtracellularPotential has no parameters of its own"
            )
        return self


def _projection_mismatch(device, distances, dt, extracellular_dt, extracellular_gain):
    if not getattr(device, "projects", False):
        return f"{type(device).__name__}'s medium cannot be formed as a channel sum"
    if (
        dt is not None
        and extracellular_dt is not None
        and abs(dt - extracellular_dt) > 1e-9
    ):
        return (
            f"it was sampled every {extracellular_dt} ms and the signal is asked "
            f"for every {dt} ms"
        )
    if extracellular_gain is None:
        return "it carries no record of the gain it was summed through"
    gain = device.source_gain(distances)
    if is_traced(gain):
        return "the gain is traced, and the projection was summed through fixed numbers"
    if gain_digest(gain) != extracellular_gain:
        return (
            "it was summed through a different gain (the MEA, the medium or the "
            "sources changed since it was recorded)"
        )
    return None


class LightArtifact(Component):
    """The photoelectric artifact."""

    weights: object
    log_saturation: object = None
    taus: tuple = static_field((0.1, 0.5, 2.0, 10.0, 50.0))

    needs_dt = True

    def __init__(
        self,
        weights=None,
        taus=(0.1, 0.5, 2.0, 10.0, 50.0),
        saturation=None,
        n_channels: int | None = None,
    ):
        taus = tuple(float(tau) for tau in taus)
        if not taus or min(taus) <= 0.0:
            raise ValueError(f"the basis time constants must be positive, not {taus}")
        if weights is None:
            shape = (len(taus),) if n_channels is None else (n_channels, len(taus))
            weights = np.zeros(shape, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim not in (1, 2) or weights.shape[-1] != len(taus):
            raise ValueError(
                f"weights are [K] or [channels, K] over the {len(taus)} time "
                f"constants, got shape {tuple(weights.shape)}"
            )
        self.weights = weights
        self.log_saturation = (
            None
            if saturation is None
            else np.log(np.asarray(saturation, dtype=np.float32))
        )
        self.taus = taus

    def serialize(self) -> dict:
        return {
            "cls": class_path(self),
            "weights": _plain(self.weights),
            "taus": list(self.taus),
            "saturation": None
            if self.log_saturation is None
            else _plain(np.exp(self.log_saturation)),
        }

    def set_params(self, params: dict) -> LightArtifact:
        known = {"weights", "saturation", "log_saturation"}
        unknown = sorted(set(params) - known)
        if unknown:
            raise KeyError(
                f"{unknown} are not parameters of LightArtifact; they are "
                f"{sorted(known)} (the time constants `taus` fix its basis)"
            )
        fields = {}
        if "weights" in params:
            fields["weights"] = np.asarray(params["weights"], dtype=np.float32)
        if "saturation" in params:
            saturation = params["saturation"]
            fields["log_saturation"] = (
                None
                if saturation is None
                else np.log(np.asarray(saturation, dtype=np.float32))
            )
        if "log_saturation" in params:
            fields["log_saturation"] = np.asarray(params["log_saturation"], np.float32)
        return with_fields(self, **fields)

    def impulse_response(self, dt: float, n_samples: int):
        """The kernel `[C, n_samples]` (or `[1, n]` when shared) of a unit sample of light."""
        a = np.exp(-float(dt) / np.asarray(self.taus))
        n = np.arange(n_samples)[:, None]
        basis = (1.0 - a) * a**n  # [n, K]
        return np.atleast_2d(self.weights) @ basis.T

    def __call__(self, *, io, inputs, dt, inputs_dt=None):
        from livn.io.base import InputDevices
        from livn.io.optical import _Light

        coordinates = io.recording_device().electrode_coordinates
        stimulating = io.stimulating_device()
        if not isinstance(stimulating, InputDevices):
            light = electrode_light(
                inputs, stimulating.reach(coordinates), dt=dt, inputs_dt=inputs_dt
            )
            return self.response(light, dt=dt)
        # the light of every light source, each through its own reach
        light = None
        for device, commands in zip(
            stimulating.devices, stimulating.split(inputs), strict=True
        ):
            if not isinstance(device, _Light):
                continue
            part = electrode_light(
                commands, device.reach(coordinates), dt=dt, inputs_dt=inputs_dt
            )
            light = part if light is None else light + part
        if light is None:
            raise ValueError(
                "LightArtifact follows light on the electrodes, and none of the "
                f"stimulating devices {[type(d).__name__ for d in stimulating.devices]} "
                "is a light source"
            )
        return self.response(light, dt=dt)

    def response(self, field, *, dt):
        """The artifact of the irradiance `field` `[C, T]` on each electrode."""
        if dt is None:
            raise ValueError(
                "LightArtifact depends on the sampling interval: pass dt (ms)"
            )
        dt = float(dt)
        u = np.asarray(field)
        u = u.astype(np.promote_types(u.dtype, np.float32))
        if u.ndim != 2:
            raise ValueError(
                f"the light is one row per channel, [C, T], not shape {tuple(u.shape)}"
            )
        if self.weights.ndim == 2 and self.weights.shape[0] != u.shape[0]:
            raise ValueError(
                f"{self.weights.shape[0]} channels of weights for light on "
                f"{u.shape[0]} channels (shape {tuple(u.shape)})"
            )
        if self.log_saturation is not None:
            u = u / (1.0 + u / np.exp(self.log_saturation))
        a = np.exp(-dt / np.asarray(self.taus, dtype=u.dtype))
        weights = np.broadcast_to(self.weights, (u.shape[0], len(self.taus)))

        if _USES_JAX:

            def step(s, u_t):
                s = a * s + (1.0 - a) * u_t[:, None]
                return s, np.sum(weights * s, axis=-1)

            zero = np.zeros((u.shape[0], len(self.taus)), u.dtype)
            _, y = jax.lax.scan(step, zero, u.T)
            return y.T

        from scipy.signal import lfilter

        y = _np.zeros(u.shape)
        for k, a_k in enumerate(_np.asarray(a, dtype=float)):
            s = lfilter([1.0 - a_k], [1.0, -a_k], _np.asarray(u, dtype=float), axis=-1)
            y += _np.asarray(weights[:, k], dtype=float)[:, None] * s
        return y


if _USES_JAX:

    def _flatten(composition):
        return (composition.blocks,), composition.names

    def _unflatten_as(cls):
        def unflatten(names, children):
            (blocks,) = children
            return cls._of(blocks, names)

        return unflatten

    for _cls in (Sum, Chain):
        jax.tree_util.register_pytree_node(_cls, _flatten, _unflatten_as(_cls))
