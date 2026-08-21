from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA = 2

DEFAULT = "spontaneous"

CONDITION_NAMES = ("spontaneous", "evoked", "interstimulus", "periodic")

FREE_RUNNING = ("spontaneous", "interstimulus")
STIMULATED = tuple(c for c in CONDITION_NAMES if c not in FREE_RUNNING)


class Stat(BaseModel):
    model_config = ConfigDict(extra="allow")

    n: int
    median: float | None = None
    mean: float | None = None
    std: float | None = None
    q_lo: float | None = None
    q_hi: float | None = None
    min: float | None = None
    max: float | None = None


class Threshold(BaseModel):
    model_config = ConfigDict(extra="allow")

    censored: str | None = None
    below_mv: float | None = None
    above_mv: float | None = None
    highest_tested_mv: float
    amplitudes_mv: list[float]

    @field_validator("amplitudes_mv")
    @classmethod
    def _one_rung_brackets_nothing(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError(f"a sweep of {len(v)} amplitude(s) is degenerate")
        return v

    @model_validator(mode="after")
    def _a_crossing_names_its_bracket(self):
        if self.censored is None and self.above_mv is None:
            raise ValueError("Amplitude missing")
        return self


class AmplitudeRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    amplitude_mv: float
    n_windows: int = 0


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")

    condition: str | None = None
    window_ms: float
    onsets_ms: list[float] | None = None
    n_windows: int = 0
    features: dict[str, Stat | None] = Field(default_factory=dict)
    by_amplitude: dict[str, AmplitudeRow] | None = None


class Pooled(BaseModel):
    model_config = ConfigDict(extra="allow")

    summary: Summary | None = None
    ei_targets: dict = Field(default_factory=dict)
    response: dict = Field(default_factory=dict)
    threshold: Threshold | None = None
    error: str | None = None

    @field_validator("ei_targets", "response", mode="before")
    @classmethod
    def _absent_is_empty(cls, v):
        return {} if v is None else v


class Block(BaseModel):
    model_config = ConfigDict(extra="allow")

    pooled: Pooled


class Document(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(alias="schema")
    conditions: dict[str, dict] = Field(default_factory=dict)


class Measured(BaseModel):
    model_config = ConfigDict(extra="allow")

    condition: str
    summary: Summary
    ei_targets: dict
    response: dict
    threshold: Threshold | None
    by_amplitude: dict[str, AmplitudeRow] | None
    window_ms: float
    onsets_ms: list[float] | None

    @property
    def amplitudes_mv(self) -> tuple[float, ...]:
        rows = (self.by_amplitude or {}).values()
        return tuple(sorted(float(row.amplitude_mv) for row in rows))


def read_target(file: str, condition: str = DEFAULT) -> Measured:
    with open(file) as f:
        document = json.load(f)

    version = document.get("schema")
    if version != SCHEMA:
        raise ValueError(
            f"{file!r} declares schema {version!r}, and this reader expects {SCHEMA}"
        )

    if condition not in CONDITION_NAMES:
        raise ValueError(
            f"unknown condition {condition!r}; a target document is measured "
            f"as {', '.join(CONDITION_NAMES)}"
        )

    parsed = Document.model_validate(document)
    if condition not in parsed.conditions:
        raise KeyError(
            f"{file!r} holds no {condition!r} block; it was written with "
            f"{', '.join(sorted(parsed.conditions)) or 'nothing'}"
        )

    pooled = Block.model_validate(parsed.conditions[condition]).pooled
    if pooled.summary is None:
        raise ValueError(
            f"the {condition!r} block of {file!r} has no usable windows: "
            f"{pooled.error or 'no reason recorded'}"
        )

    summary = pooled.summary
    if condition in STIMULATED and not summary.onsets_ms:
        raise ValueError(f"the {condition!r} block of {file!r} records no `onsets_ms`")

    return Measured(
        condition=condition,
        summary=summary,
        ei_targets=pooled.ei_targets,
        response=pooled.response,
        threshold=pooled.threshold,
        by_amplitude=summary.by_amplitude,
        window_ms=summary.window_ms,
        onsets_ms=summary.onsets_ms,
    )
