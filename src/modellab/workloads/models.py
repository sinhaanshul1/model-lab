"""Validated, immutable workload models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkloadMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class GenerationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=128, ge=1, le=4_096)
    seed: int | None = None


class ExpectedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    scorer: Literal[
        "exact_match",
        "normalized_exact_match",
        "multiple_choice",
        "contains_all",
        "numeric_tolerance",
        "valid_json",
        "json_schema",
    ]
    value: Any | None = None
    tolerance: float | None = Field(default=None, ge=0)
    schema_definition: dict[str, Any] | None = Field(default=None, alias="schema")


class WorkloadCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    messages: tuple[WorkloadMessage, ...] = Field(min_length=1)
    expected: ExpectedAnswer | None = None


class WorkloadDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = Field(min_length=1, max_length=500)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    shared_messages: tuple[WorkloadMessage, ...] = ()
    cases: tuple[WorkloadCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> "WorkloadDefinition":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("workload case IDs must be unique")
        return self


class WorkloadSummary(BaseModel):
    name: str
    version: str
    description: str
    case_count: int
    content_hash: str


class RegisteredWorkload(WorkloadDefinition):
    content_hash: str
