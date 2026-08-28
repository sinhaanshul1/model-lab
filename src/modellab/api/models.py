"""Pydantic request/response models and status enums for the ModelLab API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ServingEngine(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"
    MOCK = "mock"


class ModelProfileStatus(StrEnum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    ARCHIVED = "archived"


class EvaluationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class MockChatCompletionRequest(BaseModel):
    """A small OpenAI-compatible request contract for the local mock server."""

    model: str = Field(default="modellab-mock", min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=4_096)


class MockChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[MockChatCompletionChoice]
    usage: MockUsage


class ModelProfileCreate(BaseModel):
    """An immutable candidate configuration for a model-serving deployment."""

    name: str = Field(min_length=3, max_length=100)
    model: str = Field(min_length=1, examples=["ibm-granite/granite-3.3-8b-instruct"])
    engine: ServingEngine = ServingEngine.MOCK
    quantization: str | None = Field(default=None, examples=["awq-4bit"])
    prefix_caching: bool = False
    max_concurrent_sequences: int = Field(default=8, ge=1, le=1_000)
    max_context_tokens: int = Field(default=4_096, ge=128, le=1_000_000)


class ModelProfile(ModelProfileCreate):
    id: UUID
    status: ModelProfileStatus = ModelProfileStatus.CANDIDATE
    created_at: datetime


class EvaluationRunCreate(BaseModel):
    model_profile_id: UUID
    workload_name: str = Field(default="smoke-test", min_length=1, max_length=100)
    request_count: int = Field(default=10, ge=1, le=100_000)
    concurrency: int = Field(default=1, ge=1, le=10_000)


class EvaluationMetrics(BaseModel):
    request_count: int
    successful_requests: int = 0
    p95_ttft_ms: float | None = None
    output_tokens_per_second: float | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)


class EvaluationRun(BaseModel):
    id: UUID
    model_profile_id: UUID
    workload_name: str
    request_count: int
    concurrency: int
    status: EvaluationRunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: EvaluationMetrics | None = None
