from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
from machinable.config import Field as ConfigField

from systems.targets.EI.bursting import BurstingCulture
from systems.targets.EI.features import feature
from systems.targets.EI.measure import ASYNC_FEATURES
from systems.targets.EI.units import read_unit_targets


class MixedCulture(BurstingCulture):
    class Config(BurstingCulture.Config):
        frozen: str | dict | None = None
        free: list[str] = ConfigField(default_factory=list)

    UNIT_FEATURES: ClassVar[tuple] = ()
    SEARCH_OBJECTIVES: ClassVar[tuple] = (
        "unit_rate_median",
        "unit_rate_cv",
        "unit_corr_median",
        "coordination_excess",
    )
    GATED_FEATURES: ClassVar[tuple] = (
        "isi_cv",
        "unit_rate_cv",
        "unit_top10_share",
        "unit_corr_median",
        "coordination_excess",
    )
    ALWAYS_FREE: ClassVar[tuple] = ("io-volume_conductor-stimulation_gain",)

    def frozen(self) -> dict[str, float]:
        spec = self.config.frozen
        if spec is None:
            return {}
        if isinstance(spec, str):
            with open(spec) as f:
                spec = json.load(f)
        free = {*self.ALWAYS_FREE, *self.config.free}
        return {str(k): float(v) for k, v in dict(spec).items() if str(k) not in free}

    def search_space(self, model=None) -> dict[str, list[float]]:
        frozen = self.frozen()
        return {k: v for k, v in super().search_space(model).items() if k not in frozen}

    def decode_params(self, params: dict, model=None, strict: bool = False) -> dict:
        frozen = self.frozen()
        merged = {**frozen, **{k: v for k, v in params.items() if k not in frozen}}
        return super().decode_params(merged, model=model, strict=strict)

    def _score_units(self, observation) -> None:
        super()._score_units(observation)
        sidecar = read_unit_targets(observation)
        block = sidecar.at(self.config.spec.scale)
        for name, stat in sidecar.require(
            block, "async_features", ASYNC_FEATURES
        ).items():
            self._set_target(name, stat)

    def _async(self, env) -> dict:
        from livn.decoding import Slice, UnitCoordination, merged_spikes
        from livn.utils import P

        d = int(self.recording_duration)
        data = Slice(start=self.warmup_duration, stop=self.warmup_duration + d)(
            self.response_data
        )
        comm = getattr(env, "comm", None)
        units = self._units(env)
        it, tt = merged_spikes(data, env)
        result = None
        if P.is_root(comm=comm):
            it = np.asarray(it if it is not None else [], dtype=np.int64)
            tt = np.asarray(tt if tt is not None else [], dtype=np.float64)
            channel_of = {int(g): int(c) for c, g in units.items()}
            keep = np.isin(it, list(channel_of))
            channels = np.array([channel_of[int(g)] for g in it[keep]], dtype=np.int64)
            result = UnitCoordination(duration=d).decode(
                channels, tt[keep], sorted(units)
            )
        return P.broadcast(result, comm=comm)

    def compute_objectives(self, env) -> dict:
        result = dict(super().compute_objectives(env))
        values = self._async(env) or {}
        self.metrics.update(values)
        targets = self.targets()
        for name in self.SEARCH_OBJECTIVES:
            value = float(values.get(name, float("nan")))
            scoring = feature(name)
            if scoring.scale is not None:
                measured = 0.0 if np.isnan(value) else value
                objective = float(
                    ((measured - float(targets[name])) / scoring.scale) ** 2
                )
            else:
                objective = self._log_ratio_objective(
                    value, float(targets[name]), scoring.floor
                )
            result[name] = (objective, value)
        self.objectives = result
        return result
