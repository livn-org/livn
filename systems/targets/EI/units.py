from __future__ import annotations

import json
import os

from pydantic import BaseModel, ConfigDict, Field

QUANTILES = (0.05, 0.95)


class Stat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    median: float
    q_lo: float
    q_hi: float
    n: int


class UnitBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_windows: int
    n_units: int
    features: dict[str, Stat] = Field(default_factory=dict)
    channel_features: dict[str, Stat] = Field(default_factory=dict)
    async_features: dict[str, Stat] = Field(default_factory=dict)
    channels: list[int] | None = None


class UnitTargets(UnitBlock):
    observation: str
    recording: str
    sort: str
    window_ms: float
    n_channels: int
    noise_uv: float | None = None
    scales: dict[str, UnitBlock] = Field(default_factory=dict)
    source: str = ""

    def at(self, scale: float) -> UnitBlock:
        if float(scale) == 1.0:
            return self
        block = self.scales.get(scale_key(scale))
        if block is None:
            raise ValueError(
                f"{self.source}'s unit targets have no block for scale {scale_key(scale)}"
            )
        return block

    def require(self, block: UnitBlock, group: str, names) -> dict[str, Stat]:
        stated = getattr(block, group)
        missing = [name for name in names if name not in stated]
        if missing:
            where = "the full array" if block is self else f"a scale's {group}"
            raise ValueError(
                f"{self.source}'s unit targets lack {missing} in {where}; rewrite "
                "them from the sort"
            )
        return {name: stated[name] for name in names}


def sidecar_path(observation: str) -> str:
    return os.path.join(
        os.path.dirname(observation), "units", os.path.basename(observation)
    )


def scale_key(scale: float) -> str:
    return f"{float(scale):.4f}"


def read_unit_targets(observation: str) -> UnitTargets:
    path = sidecar_path(observation)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    with open(path) as f:
        return UnitTargets.model_validate({**json.load(f), "source": observation})
