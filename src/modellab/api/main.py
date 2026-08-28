"""FastAPI control plane for ModelLab's PostgreSQL-backed state."""

from __future__ import annotations

from hashlib import sha256
import json
from uuid import UUID

from fastapi import Body, Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session
from modellab.api.models import (
    ChatMessage,
    EvaluationRun,
    EvaluationRunCreate,
    MockChatCompletion,
    MockChatCompletionChoice,
    MockChatCompletionRequest,
    MockUsage,
    ModelProfile,
    ModelProfileCreate,
)
from modellab.storage import repositories
from modellab.storage.database import get_session


app = FastAPI(
    title="ModelLab Control Plane",
    version="0.1.0",
    description="Configuration and evaluation-run control plane for LLM serving.",
)

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


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/mock-model/chat/completions",
    response_model=MockChatCompletion,
    tags=["mock model"],
)
def mock_chat_completions(
    payload: MockChatCompletionRequest = Body(
        openapi_examples={
            "basic_chat": {
                "summary": "Basic deterministic mock request",
                "value": {
                    "model": "modellab-mock",
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Explain prefix caching."},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 128,
                },
            }
        }
    ),
) -> MockChatCompletion:
    """Generate a deterministic mock response without loading a real model."""

    return deterministic_mock_completion(payload)


@app.post(
    "/v1/model-profiles",
    response_model=ModelProfile,
    status_code=status.HTTP_201_CREATED,
    tags=["model profiles"],
)
def create_model_profile(
    payload: ModelProfileCreate = Body(
        openapi_examples={
            "mock_candidate": {
                "summary": "Candidate mock-model profile",
                "value": {
                    "name": "mock-baseline",
                    "model": "modellab/mock-1",
                    "engine": "mock",
                    "quantization": "none",
                    "prefix_caching": True,
                    "max_concurrent_sequences": 8,
                    "max_context_tokens": 4096,
                },
            }
        }
    ),
    session: Session = Depends(get_session),
) -> ModelProfile:
    return repositories.create_model_profile(session, payload)


@app.get("/v1/model-profiles", response_model=list[ModelProfile], tags=["model profiles"])
def list_model_profiles(session: Session = Depends(get_session)) -> list[ModelProfile]:
    return repositories.list_model_profiles(session)


@app.get("/v1/model-profiles/{profile_id}", response_model=ModelProfile, tags=["model profiles"])
def get_model_profile(
    profile_id: UUID, session: Session = Depends(get_session)
) -> ModelProfile:
    profile = repositories.get_model_profile(session, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model profile not found")
    return profile


@app.post(
    "/v1/evaluation-runs",
    response_model=EvaluationRun,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["evaluation runs"],
)
def create_evaluation_run(
    payload: EvaluationRunCreate = Body(
        openapi_examples={
            "smoke_test": {
                "summary": "Small evaluation run",
                "value": {
                    "model_profile_id": "00000000-0000-0000-0000-000000000001",
                    "workload_name": "smoke-test",
                    "request_count": 10,
                    "concurrency": 2,
                },
            }
        }
    ),
    session: Session = Depends(get_session),
) -> EvaluationRun:
    if repositories.get_model_profile(session, payload.model_profile_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model profile not found")
    return repositories.create_evaluation_run(session, payload)


@app.get("/v1/evaluation-runs", response_model=list[EvaluationRun], tags=["evaluation runs"])
def list_evaluation_runs(session: Session = Depends(get_session)) -> list[EvaluationRun]:
    return repositories.list_evaluation_runs(session)


@app.get("/v1/evaluation-runs/{run_id}", response_model=EvaluationRun, tags=["evaluation runs"])
def get_evaluation_run(
    run_id: UUID, session: Session = Depends(get_session)
) -> EvaluationRun:
    run = repositories.get_evaluation_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found")
    return run
