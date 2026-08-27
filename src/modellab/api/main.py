"""Initial FastAPI control plane for ModelLab.

This module intentionally uses in-memory storage during the first vertical
slice. A later migration will replace the dictionaries with PostgreSQL-backed
repositories without changing the HTTP contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
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



"""
Test ModelProfileCreate:
{
  "name": "test model",
  "model": "ibm-granite/granite-3.3-8b-instruct",
  "max_concurrent_sequences": 10
}

"""
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


app = FastAPI(
    title="ModelLab Control Plane",
    version="0.1.0",
    description="Configuration and evaluation-run control plane for LLM serving.",
)

# Temporary storage for the first local slice. Do not use this in production.
model_profiles: dict[UUID, ModelProfile] = {}
evaluation_runs: dict[UUID, EvaluationRun] = {}


def now() -> datetime:
    return datetime.now(UTC)


def deterministic_mock_completion(payload: MockChatCompletionRequest) -> MockChatCompletion:
    """Return a repeatable response for the same request payload.

    The mock endpoint lets ModelLab exercise request forwarding and benchmark
    collection before a real vLLM or SGLang server is available.
    """

    serialized_request = json.dumps(
        payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    request_digest = sha256(serialized_request.encode("utf-8")).hexdigest()
    last_user_message = next(
        (message.content for message in reversed(payload.messages) if message.role == "user"),
        payload.messages[-1].content,
    )
    response_content = (
        f"[ModelLab mock {request_digest[:12]}] Received: {last_user_message[:240]}"
    )
    prompt_tokens = sum(len(message.content.split()) for message in payload.messages)
    completion_tokens = len(response_content.split())

    return MockChatCompletion(
        id=f"chatcmpl-mock-{request_digest[:16]}",
        # A deterministic Unix timestamp derived from the request hash.
        created=1_700_000_000 + int(request_digest[16:24], 16) % 10_000_000,
        model=payload.model,
        choices=[
            MockChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response_content)
            )
        ],
        usage=MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.get("/healthz", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/mock-model/chat/completions",
    response_model=MockChatCompletion,
    tags=["mock model"],
)
def mock_chat_completions(payload: MockChatCompletionRequest) -> MockChatCompletion:
    """Generate a deterministic mock response without loading a real model."""

    return deterministic_mock_completion(payload)


@app.post(
    "/v1/model-profiles",
    response_model=ModelProfile,
    status_code=status.HTTP_201_CREATED,
    tags=["model profiles"],
)
def create_model_profile(payload: ModelProfileCreate) -> ModelProfile:
    profile = ModelProfile(id=uuid4(), created_at=now(), **payload.model_dump())
    model_profiles[profile.id] = profile
    return profile


@app.get("/v1/model-profiles", response_model=list[ModelProfile], tags=["model profiles"])
def list_model_profiles() -> list[ModelProfile]:
    return list(model_profiles.values())


@app.get("/v1/model-profiles/{profile_id}", response_model=ModelProfile, tags=["model profiles"])
def get_model_profile(profile_id: UUID) -> ModelProfile:
    profile = model_profiles.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model profile not found")
    return profile


@app.post(
    "/v1/evaluation-runs",
    response_model=EvaluationRun,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["evaluation runs"],
)
def create_evaluation_run(payload: EvaluationRunCreate) -> EvaluationRun:
    if payload.model_profile_id not in model_profiles:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model profile not found")

    run = EvaluationRun(
        id=uuid4(),
        created_at=now(),
        status=EvaluationRunStatus.QUEUED,
        **payload.model_dump(),
    )
    evaluation_runs[run.id] = run
    return run


@app.get("/v1/evaluation-runs", response_model=list[EvaluationRun], tags=["evaluation runs"])
def list_evaluation_runs() -> list[EvaluationRun]:
    return list(evaluation_runs.values())


@app.get("/v1/evaluation-runs/{run_id}", response_model=EvaluationRun, tags=["evaluation runs"])
def get_evaluation_run(run_id: UUID) -> EvaluationRun:
    run = evaluation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found")
    return run
