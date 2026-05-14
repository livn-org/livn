from __future__ import annotations

import logging
import time

from livn.backend.config import backend
from livn.types import Env
from livn.utils import P

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class _NeuronTimingLogger:
    def __init__(
        self,
        env: Env,
        log_interval_sim: float = 10_000.0,
        log_interval_wall: float = 10.0,
        tick_dt: float | None = None,
        progress: bool = True,
    ) -> None:
        from neuron import h

        self._h = h
        self.env = env
        self.rank = P.rank()
        self.log_interval_sim = float(log_interval_sim)
        self.log_interval_wall = float(log_interval_wall)
        self.tick_dt = float(tick_dt) if tick_dt is not None else self.log_interval_sim
        self.use_tqdm = progress and _tqdm is not None and self.rank == 0

        now = time.time()
        self._last_log_wall = now
        self._last_log_sim = 0.0

        self._pbar = None
        self._bar_duration: float | None = None
        self._bar_start_t: float = 0.0
        self._last_progress_t: float = 0.0

        # Keep a reference so the handler is not garbage-collected
        self.fih_status = h.FInitializeHandler(1, self.status)

    def _open_bar(self, sim_t: float, duration: float) -> None:
        self._bar_start_t = float(sim_t)
        self._bar_duration = float(round(duration))
        self._last_progress_t = 0.0
        self._pbar = _tqdm(
            total=self._bar_duration,
            unit="ms",
            desc=self._bar_desc(self._bar_start_t),
            leave=False,
            dynamic_ncols=True,
        )

    def _bar_desc(self, sim_t: float) -> str:
        return f"sim t={sim_t:.1f}ms"

    def _close_bar(self) -> None:
        if self._pbar is None:
            return
        if self._bar_duration is not None:
            remaining = self._bar_duration - self._last_progress_t
            if remaining > 0:
                self._pbar.update(remaining)
        self._pbar.close()
        self._pbar = None
        self._bar_duration = None

    def _update_bar(self, sim_t: float) -> None:
        if self._pbar is None or self._bar_duration is None:
            return
        elapsed = max(0.0, sim_t - self._bar_start_t)
        elapsed = min(elapsed, self._bar_duration)
        elapsed_int = float(round(elapsed))
        delta = elapsed_int - self._last_progress_t
        if delta > 0:
            self._pbar.update(delta)
            self._last_progress_t = elapsed_int
            self._pbar.set_description_str(self._bar_desc(sim_t), refresh=False)

    def _sync_bar(self, sim_t: float) -> None:
        if not self.use_tqdm:
            return

        duration = getattr(self.env, "duration", None)

        if duration is None:
            self._close_bar()
            return

        if self._pbar is None or float(duration) != self._bar_duration:
            self._close_bar()
            self._open_bar(sim_t, duration)

        self._update_bar(sim_t)

    def update(
        self,
        log_interval_sim: float | None = None,
        log_interval_wall: float | None = None,
        tick_dt: float | None = None,
        progress: bool | None = None,
    ) -> None:
        if log_interval_sim is not None:
            self.log_interval_sim = float(log_interval_sim)
        if log_interval_wall is not None:
            self.log_interval_wall = float(log_interval_wall)
        if tick_dt is not None:
            self.tick_dt = float(tick_dt)
        elif log_interval_sim is not None:
            # Re-derive default if caller bumped the sim interval but
            # didn't override tick_dt explicitly.
            self.tick_dt = self.log_interval_sim
        if progress is not None:
            new_use_tqdm = bool(progress) and _tqdm is not None and self.rank == 0
            if not new_use_tqdm:
                self._close_bar()
            self.use_tqdm = new_use_tqdm

    def status(self) -> None:
        h = self._h
        wt = time.time()
        sim_t = float(h.t)

        self._sync_bar(sim_t)

        if self.rank == 0 and self._pbar is None and sim_t > 0.0:
            sim_due = (sim_t - self._last_log_sim) >= self.log_interval_sim
            wall_due = (wt - self._last_log_wall) >= self.log_interval_wall
            if sim_due or wall_due:
                logger.info(
                    f"[timing] sim t={sim_t:.2f} ms "
                    f"(+{wt - self._last_log_wall:.2f} s wall)"
                )
                self._last_log_sim = sim_t
                self._last_log_wall = wt

        if sim_t == 0.0:
            self._last_log_sim = 0.0
            self._last_log_wall = wt

        h.cvode.event(h.t + self.tick_dt, self.status)


def with_progress_logging(
    env: Env,
    log_interval_sim: float = 10_000.0,
    log_interval_wall: float = 10.0,
    tick_dt: float | None = None,
    progress: bool = True,
) -> Env:
    """Enable live simulation-time logging on ``env``.

    For the NEURON backend this installs a periodic status handler that
    logs the simulated time and wall time elapsed since the last log
    line. Logs are emitted at most once per ``log_interval_sim`` ms of
    simulated time, and at least once per ``log_interval_wall`` seconds
    of wall-clock time (whichever threshold is crossed first).

    ``tick_dt`` controls how often the underlying ``cvode.event``
    callback fires (in ms of simulated time). When ``None`` it defaults
    to ``log_interval_sim``.

    If `tqdm` is installed and ``progress`` is true, a tqdm progress bar
    is shown on rank 0 for each ``env.run()``.

    For all other backends this is a no-op and the env is returned unchanged.
    """
    if backend() != "neuron":
        return env

    existing = getattr(env, "_timing_logger", None)
    if isinstance(existing, _NeuronTimingLogger):
        existing.update(
            log_interval_sim=log_interval_sim,
            log_interval_wall=log_interval_wall,
            tick_dt=tick_dt,
            progress=progress,
        )
        return env

    env._timing_logger = _NeuronTimingLogger(
        env,
        log_interval_sim=log_interval_sim,
        log_interval_wall=log_interval_wall,
        tick_dt=tick_dt,
        progress=progress,
    )
    return env
