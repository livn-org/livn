from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping

import numpy

from livn.utils import P, lnp, merge_array

if TYPE_CHECKING:
    from livn.types import Array

_TOL = 1e-9

IS_PYTREE = False


def _join(a, b):
    if a is None and b is None:
        return None
    return merge_array([a, b])


def _is_traced(value) -> bool:
    """Whether ``value`` is a JAX tracer, without importing jax if it is not already loaded"""
    jax = sys.modules.get("jax")
    if jax is None or value is None:
        return False
    return isinstance(value, jax.core.Tracer)


@dataclass(frozen=True, eq=False)
class PaddedEvents:
    """Rectangular event storage with one row per cell, non-finite where a row ran out of events.

    This is what an event-driven solve produces on device with a fixed shape it can allocate before
    it knows how many events there will be. It is the lossless form in that the times are the exact
    root-found event times, so it stays differentiable in them, and it needs no host sync, so it
    survives ``jit`` / ``grad`` / ``vmap``.

    Args:
        ids: ``(cells,)`` cell id of each row
        times: ``(..., cells, k)`` event times relative to the channel's ``t0``, padded with a
            non-finite value. A leading axis appears when the run had multiple samples.
    """

    ids: "Array"
    times: "Array"

    @property
    def fired(self) -> "Array":
        """``True`` where an entry is a real event rather than padding"""
        return lnp().isfinite(self.times)

    def __repr__(self) -> str:
        return f"PaddedEvents(shape={tuple(self.times.shape)})"


class Events:
    """Point events.

    Args:
        ids: Cell id per event
        times: Event time per event, relative to ``t0``
        t0: Absolute simulation time of this channel's time origin
        duration: Length of the recorded window in ms
        padded: Rectangular storage to compact from, instead of ``ids``/``times``
    """

    __slots__ = ("_ids", "_times", "_padded", "_compacted", "t0", "duration")

    kind = "events"

    def __init__(
        self,
        ids: "Array | None" = None,
        times: "Array | None" = None,
        t0: float = 0.0,
        duration: float | None = None,
        padded: "PaddedEvents | None" = None,
    ):
        put = object.__setattr__
        put(self, "_ids", ids)
        put(self, "_times", times)
        put(self, "_padded", padded)
        put(self, "_compacted", None)
        put(self, "t0", t0)
        put(self, "duration", duration)

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getstate__(self) -> dict:
        # pickle and copy would otherwise go through __setattr__, which refuses.
        # The compaction cache is derived, so it is not worth carrying.
        return {
            "_ids": self._ids,
            "_times": self._times,
            "_padded": self._padded,
            "t0": self.t0,
            "duration": self.duration,
        }

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_compacted", None)

    # -- storage ---------------------------------------------------------

    @property
    def padded(self) -> "PaddedEvents | None":
        """The rectangular device form, or ``None`` for a channel that never had one"""
        return self._padded

    def _compact(self) -> tuple:
        """``(ids, times)`` with the padding dropped and the events ordered by time"""
        if self._padded is None:
            return self._ids, self._times
        if self._compacted is not None:
            return self._compacted

        times = self._padded.times
        if _is_traced(times):
            raise RuntimeError(
                "these events are still a rectangular device array; compacting them "
                "drops padding and sorts by time, which needs concrete values and so "
                "cannot happen inside a JAX trace. Use `.padded.times` and "
                "`.padded.ids`, which keep the gradient, or move the access outside "
                "jit/grad/vmap."
            )
        if times.ndim > 2:
            raise ValueError(
                f"a batched run has a different number of events per sample, so its "
                f"{tuple(times.shape)} events have no ragged form; use `.padded` or "
                f"`.raster(dt)`"
            )

        times = numpy.asarray(times)
        rows = numpy.broadcast_to(numpy.asarray(self._padded.ids)[:, None], times.shape)

        keep = numpy.isfinite(times).reshape(-1)
        ids = rows.reshape(-1)[keep]
        times = times.reshape(-1)[keep]
        order = numpy.argsort(times, kind="stable")

        compacted = (ids[order], times[order])
        object.__setattr__(self, "_compacted", compacted)
        return compacted

    def compact(self) -> "Events":
        """This channel in the compact form, keeping the padded storage alongside"""
        ids, times = self._compact()
        return Events(
            ids=ids,
            times=times,
            t0=self.t0,
            duration=self.duration,
            padded=self._padded,
        )

    @property
    def ids(self) -> "Array | None":
        """Cell id per event, compact"""
        return self._compact()[0]

    @property
    def times(self) -> "Array | None":
        """Event time per event, compact and ordered by time"""
        return self._compact()[1]

    @property
    def values(self) -> "Array | None":
        """The event times"""
        return self.times

    def raster(self, dt: float, n_cells: int | None = None, steps: int | None = None):
        np = lnp()
        if steps is None:
            if self.duration is None:
                raise ValueError(
                    "raster needs `steps` when the channel has no duration"
                )
            steps = int(round(self.duration / dt)) + 1

        if self._padded is not None:
            times = self._padded.times  # (..., cells, k)
            cells = times.shape[-2] if n_cells is None else int(n_cells)
            rows = np.broadcast_to(
                np.arange(cells).reshape((cells, 1)), times.shape[-2:]
            )
            rows = np.broadcast_to(rows, times.shape)
            lead = times.shape[:-2]
        else:
            times, rows = self._times, self._ids
            if times is None or rows is None:
                raise ValueError("this channel holds no events")
            times, rows = np.asarray(times), np.asarray(rows)
            cells = (int(rows.max()) + 1) if n_cells is None else int(n_cells)
            lead = ()

        sample = int(numpy.prod(lead)) if lead else 1
        width = cells * steps
        bin_index = np.clip(np.round(times / dt), 0, steps - 1).astype(np.int32)
        flat = rows.reshape(sample, -1) * steps + bin_index.reshape(sample, -1)
        flat = np.where(np.isfinite(times).reshape(sample, -1), flat, width)

        offset = (np.arange(sample) * width).reshape(sample, 1)
        target = np.where(flat < width, flat + offset, sample * width).reshape(-1)

        grid = np.zeros(sample * width, bool)
        if hasattr(grid, "at"):
            grid = grid.at[target].set(True, mode="drop")
        else:
            inside = numpy.asarray(target) < sample * width
            grid = numpy.asarray(grid)
            grid[numpy.asarray(target)[inside]] = True

        return grid.reshape(lead + (cells, steps))

    # -- composition -----------------------------------------------------

    def concat(self, other: "Events", shift: float) -> "Events":
        """Append ``other``, whose time origin moves by ``shift``"""
        offset = other.t0 + shift - self.t0
        duration = None
        if self.duration is not None and other.duration is not None:
            duration = offset + other.duration

        return Events(
            ids=_join(self.ids, other.ids),
            times=_join(
                self.times, None if other.times is None else other.times + offset
            ),
            t0=self.t0,
            duration=duration,
        )

    def merge(self, other: "Events") -> "Events":
        """Union with the events of other cells over the same window"""
        return Events(
            ids=_join(self.ids, other.ids),
            times=_join(self.times, other.times),
            t0=self.t0,
            duration=self.duration,
        )

    def window(self, start: float, stop: float, name: str = "events") -> "Events":
        """Keep the events in ``[start, stop)``, given in absolute time"""
        lo, hi = start - self.t0, stop - self.t0

        ids, times = self.ids, self.times
        if ids is not None and times is not None:
            mask = (times >= lo) & (times < hi)
            ids, times = ids[mask], times[mask] - lo

        return Events(ids=ids, times=times, t0=start, duration=stop - start)

    def select(self, keep) -> "Events":
        """Keep only the events of the cells in ``keep``"""
        ids, times = self._compact()
        if ids is None:
            return self

        np = lnp()
        ids = np.asarray(ids)
        mask = np.isin(ids, keep)

        return Events(
            ids=ids[mask],
            times=None if times is None else np.asarray(times)[mask],
            t0=self.t0,
            duration=self.duration,
        )

    def __repr__(self) -> str:
        if self._padded is not None and self._compacted is None:
            return f"Events({self._padded!r}, t0={self.t0}, duration={self.duration})"
        times = self._compact()[1]
        n = 0 if times is None else len(times)
        return f"Events(n={n}, t0={self.t0}, duration={self.duration})"


@dataclass(frozen=True, eq=False)
class Series:
    """A regularly sampled trace per cell id.

    Args:
        ids: Cell id per row
        values: ``(n_ids, T)`` matrix of samples, or ``(n_ids, T, ...)`` when a
            sample is itself a vector (an after-spike current pair, say)
        dt: Sampling interval in ms
        t0: Absolute simulation time of sample ``0``
    """

    ids: "Array | None" = None
    values: "Array | None" = None
    dt: float = 0.1
    t0: float = 0.0

    kind = "series"

    @property
    def times(self) -> "Array | None":
        """Sample times relative to ``t0``"""
        if self.values is None:
            return None
        return lnp().arange(self.values.shape[1]) * self.dt

    @property
    def duration(self) -> float | None:
        if self.values is None:
            return None
        return float(self.values.shape[1] * self.dt)

    def concat(self, other: "Series", shift: float) -> "Series":
        """Append ``other``'s samples along the time axis"""
        if abs(self.dt - other.dt) > _TOL:
            raise ValueError(
                f"Cannot concat series with dt={self.dt} and dt={other.dt}"
            )
        if self.values is None:
            return replace(other, t0=other.t0 + shift)
        if other.values is None:
            return self
        if self.values.shape[0] != other.values.shape[0]:
            raise ValueError(
                "Cannot concat series recorded for a different number of cells "
                f"({self.values.shape[0]} vs {other.values.shape[0]})"
            )

        return replace(
            self, values=lnp().concatenate([self.values, other.values], axis=1)
        )

    def merge(self, other: "Series") -> "Series":
        """Union with the traces of other cells over the same window"""
        if abs(self.dt - other.dt) > _TOL:
            raise ValueError(f"Cannot merge series with dt={self.dt} and dt={other.dt}")
        if self.values is None:
            return other
        if other.values is None:
            return self
        if self.values.shape[1] != other.values.shape[1]:
            raise ValueError(
                "Cannot merge series of different length "
                f"({self.values.shape[1]} vs {other.values.shape[1]})"
            )

        return replace(
            self,
            ids=_join(self.ids, other.ids),
            values=lnp().concatenate([self.values, other.values], axis=0),
        )

    def window(self, start: float, stop: float, name: str = "series") -> "Series":
        """Keep the samples in ``[start, stop)``, given in absolute time"""
        if self.values is None:
            return replace(self, t0=start)

        def index(time: float, bound: str) -> int:
            offset = time - self.t0
            i = offset / self.dt
            if abs(i - round(i)) > _TOL:
                raise ValueError(
                    f"{bound}={offset} ms does not align with {name} recording "
                    f"dt={self.dt} ms yielding a fractional index {i}"
                )
            return int(round(i))

        return replace(
            self,
            values=self.values[:, index(start, "start") : index(stop, "stop")],
            t0=start,
        )

    def select(self, keep) -> "Series":
        """Keep only the rows of the cells in ``keep``"""
        if self.ids is None:
            return self

        np = lnp()
        ids = np.asarray(self.ids)
        mask = np.isin(ids, keep)

        return replace(
            self,
            ids=ids[mask],
            values=None if self.values is None else np.asarray(self.values)[mask, :],
        )

    def __repr__(self) -> str:
        shape = None if self.values is None else tuple(self.values.shape)
        return f"Series(shape={shape}, dt={self.dt}, t0={self.t0})"


@dataclass(frozen=True, eq=False)
class Run:
    """Named immutable recordings of one simulated time window.

    Args:
        channels: Mapping of channel name to :class:`Events` or :class:`Series`
        t0: Absolute simulation time at which the window starts
        duration: Length of the window in ms
    """

    channels: Mapping[str, Any] = field(default_factory=dict)
    t0: float = 0.0
    duration: float | None = None

    # -- construction ----------------------------------------------------

    def add(
        self,
        name: str,
        ids: "Array | None",
        values: "Array | None" = None,
        *,
        dt: float | None = None,
        kind: Literal["events", "series"] | None = None,
        t0: float | None = None,
        duration: float | None = None,
        padded: bool = False,
    ) -> "Run":
        if kind is None:
            if padded:
                kind = "events"
            elif dt is not None or getattr(values, "ndim", 1) >= 2:
                kind = "series"
            else:
                kind = "events"

        t0 = self.t0 if t0 is None else t0

        if kind == "series":
            if padded:
                raise ValueError("padded storage only applies to event channels")
            channel = Series(
                ids=ids, values=values, dt=0.1 if dt is None else float(dt), t0=t0
            )
        elif kind == "events":
            if padded:
                if getattr(values, "ndim", 1) < 2:
                    raise ValueError(
                        f"padded event times must be (..., cells, k), got "
                        f"{getattr(values, 'shape', type(values).__name__)}"
                    )
                if getattr(ids, "ndim", 1) != 1 or len(ids) != values.shape[-2]:
                    raise ValueError(
                        f"padded events need one row id per row: {len(ids)} ids for "
                        f"{values.shape[-2]} rows"
                    )
            channel = Events(
                ids=None if padded else ids,
                times=None if padded else values,
                t0=t0,
                duration=self.duration if duration is None else duration,
                padded=PaddedEvents(ids=ids, times=values) if padded else None,
            )
        else:
            raise ValueError(f"Unknown channel kind: {kind!r}")

        channels = dict(self.channels)
        channels[name] = channel

        return replace(self, channels=channels)

    def add_spikes(self, ids, times, *, padded: bool = False) -> "Run":
        """Add the ``spikes`` channel, or nothing when it was not recorded"""
        if ids is None and times is None:
            return self
        return self.add("spikes", ids, times, kind="events", padded=padded)

    def add_voltage(self, ids, values, dt: float = 0.1) -> "Run":
        """Add the ``voltage`` channel, or nothing when it was not recorded"""
        if ids is None and values is None:
            return self
        return self.add("voltage", ids, values, dt=dt, kind="series")

    def add_current(self, ids, values, dt: float = 0.1) -> "Run":
        """Add the ``current`` channel, or nothing when it was not recorded"""
        if ids is None and values is None:
            return self
        return self.add("current", ids, values, dt=dt, kind="series")

    def drop(self, name: str) -> "Run":
        """Return a copy of this run without ``name``"""
        channels = dict(self.channels)
        channels.pop(name, None)
        return replace(self, channels=channels)

    def drop_spikes(self) -> "Run":
        """Return a copy of this run without the ``spikes`` channel"""
        return self.drop("spikes")

    def drop_voltage(self) -> "Run":
        """Return a copy of this run without the ``voltage`` channel"""
        return self.drop("voltage")

    def drop_current(self) -> "Run":
        """Return a copy of this run without the ``current`` channel"""
        return self.drop("current")

    def channel(self, name: str):
        return self.channels.get(name)

    def ids(self, name: str):
        channel = self.channels.get(name)
        return None if channel is None else channel.ids

    def values(self, name: str):
        channel = self.channels.get(name)
        return None if channel is None else channel.values

    @property
    def spikes(self) -> Events | None:
        return self.channels.get("spikes")

    @property
    def spike_ids(self):
        return self.ids("spikes")

    @property
    def spike_times(self):
        return self.values("spikes")

    @property
    def voltage_ids(self):
        return self.ids("voltage")

    @property
    def voltage(self):
        return self.values("voltage")

    @property
    def voltage_dt(self) -> float | None:
        channel = self.channels.get("voltage")
        return None if channel is None else channel.dt

    @property
    def current_ids(self):
        return self.ids("current")

    @property
    def current(self):
        return self.values("current")

    @property
    def current_dt(self) -> float | None:
        channel = self.channels.get("current")
        return None if channel is None else channel.dt

    @property
    def t1(self) -> float | None:
        """Absolute simulation time at which the window ends"""
        if self.duration is None:
            return None
        return self.t0 + self.duration

    def _tuple(self) -> tuple:
        return (
            self.spike_ids,
            self.spike_times,
            self.voltage_ids,
            self.voltage,
            self.current_ids,
            self.current,
        )

    def __iter__(self) -> Iterator:
        return iter(self._tuple())

    def __len__(self) -> int:
        return 6  # an (ids, values) pair per channel

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.channels[key]
        return self._tuple()[key]

    def __contains__(self, key) -> bool:
        return key in self.channels

    def __repr__(self) -> str:
        channels = ", ".join(f"{k}={v!r}" for k, v in self.channels.items())
        return f"Run(t0={self.t0}, duration={self.duration}, {channels})"

    def concat(self, other: "Run") -> "Run":
        """Append a subsequent run, re-applying its time offset."""
        if not isinstance(other, Run):
            raise TypeError(f"Can only concat a Run, not {type(other).__name__}")
        if self.duration is None:
            raise ValueError("concat requires a known duration on the first run")

        gap = other.t0 - self.t0
        offset = gap if gap >= self.duration else self.duration
        # correction applied to other's channel origins
        shift = (self.t0 + offset) - other.t0

        duration = None if other.duration is None else offset + other.duration

        channels: dict[str, Any] = {}
        for name in self._names(other):
            a, b = self.channels.get(name), other.channels.get(name)
            if a is None:
                channels[name] = replace(b, t0=b.t0 + shift)
            elif b is None:
                channels[name] = a
            elif type(a) is not type(b):
                raise ValueError(f"Cannot concat {a.kind} with {b.kind} for {name!r}")
            else:
                channels[name] = a.concat(b, shift)

        return Run(channels=channels, t0=self.t0, duration=duration)

    def merge(self, other: "Run") -> "Run":
        """Union with a run covering the same window over disjoint cells."""
        if not isinstance(other, Run):
            raise TypeError(f"Can only merge a Run, not {type(other).__name__}")

        channels: dict[str, Any] = {}
        for name in self._names(other):
            a, b = self.channels.get(name), other.channels.get(name)
            if a is None or b is None:
                channels[name] = a if b is None else b
            elif type(a) is not type(b):
                raise ValueError(f"Cannot merge {a.kind} with {b.kind} for {name!r}")
            else:
                channels[name] = a.merge(b)

        duration = self.duration if self.duration is not None else other.duration

        return Run(channels=channels, t0=self.t0, duration=duration)

    def gather(self, comm=None, root: int = 0) -> "Run | None":
        """Collect the per-rank runs onto ``root`` and ``merge`` them into one.

        Returns ``None`` off the root rank.
        """
        runs = P.gather(self, comm=comm, root=root)
        if not P.is_root(comm=comm, root=root):
            return None

        merged = runs[0]
        for run in runs[1:]:
            merged = merged.merge(run)

        return merged

    def slice(self, start: float = 0.0, stop: float | None = None) -> "Run":
        """Window into ``[start, stop)``, both relative to this run's start."""
        if stop is None:
            if self.duration is None:
                raise ValueError("slice requires a stop or a known duration")
            stop = self.duration
        if stop < start:
            raise ValueError(f"stop={stop} must not precede start={start}")

        channels = {
            name: channel.window(self.t0 + start, self.t0 + stop, name)
            for name, channel in self.channels.items()
        }

        return Run(channels=channels, t0=self.t0 + start, duration=stop - start)

    def select(
        self,
        gids=None,
        population: str | list | tuple | None = None,
        population_ranges: Mapping[str, tuple] | None = None,
    ) -> "Run":
        """Restrict every channel to a subset of cells.

        Args:
            gids: Explicit cell ids to keep
            population: Population name(s), resolved against ``population_ranges``
            population_ranges: ``{name: (start, count)}``, as carried by
                ``system.population_ranges``
        """
        np = lnp()

        if population is None:
            if gids is None:
                return self
            keep = np.asarray(gids)
        else:
            if population_ranges is None:
                raise ValueError("select(population=...) requires population_ranges")
            if isinstance(population, str):
                population = [population]

            for name in population:
                if name not in population_ranges:
                    raise ValueError(f"Unknown population: {name!r}")
            keep = np.concatenate(
                [
                    np.arange(start, start + count)
                    for start, count in (population_ranges[n][:2] for n in population)
                ]
            )
            if gids is not None:
                keep = np.intersect1d(keep, np.asarray(gids))

        channels = {
            name: channel.select(keep) for name, channel in self.channels.items()
        }

        return replace(self, channels=channels)

    def _names(self, other: "Run") -> list[str]:
        """Channel names of both runs, this one's order first"""
        return list(self.channels) + [
            name for name in other.channels if name not in self.channels
        ]


# -- pytree registration -------------------------------------------------


def _events_flatten(events: Events):
    padded = events.padded
    children = (
        events._ids,
        events._times,
        None if padded is None else padded.ids,
        None if padded is None else padded.times,
    )
    return children, (events.t0, events.duration, padded is not None)


def _events_unflatten(aux, children) -> Events:
    ids, times, padded_ids, padded_times = children
    t0, duration, was_padded = aux
    return Events(
        ids=ids,
        times=times,
        t0=t0,
        duration=duration,
        padded=PaddedEvents(ids=padded_ids, times=padded_times) if was_padded else None,
    )


def _series_flatten(series: Series):
    return (series.ids, series.values), (series.dt, series.t0)


def _series_unflatten(aux, children) -> Series:
    ids, values = children
    dt, t0 = aux
    return Series(ids=ids, values=values, dt=dt, t0=t0)


def _run_flatten(run: Run):
    names = tuple(run.channels)
    return tuple(run.channels[name] for name in names), (names, run.t0, run.duration)


def _run_unflatten(aux, children) -> Run:
    names, t0, duration = aux
    return Run(channels=dict(zip(names, children)), t0=t0, duration=duration)


def _register() -> bool:
    from livn.backend.config import backend

    if "ax" not in backend():
        return False

    import jax

    jax.tree_util.register_pytree_node(Events, _events_flatten, _events_unflatten)
    jax.tree_util.register_pytree_node(Series, _series_flatten, _series_unflatten)
    jax.tree_util.register_pytree_node(Run, _run_flatten, _run_unflatten)

    return True


IS_PYTREE = _register()
