"""FastAPI control plane for ModelLab's PostgreSQL-backed state."""

from __future__ import annotations

from hashlib import sha256
import json
from uuid import UUID

from fastapi import Body, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from modellab.api.models import (
    ChatMessage,
    EvaluationRun,
    EvaluationRunCreate,
    ModelDeployment,
    ModelDeploymentCreate,
    ModelDeploymentRequest,
    ModelDeploymentStatus,
    MockChatCompletion,
    MockChatCompletionChoice,
    MockChatCompletionRequest,
    MockUsage,
    ModelProfile,
    ModelProfileCreate,
)
from modellab.storage import repositories
from modellab.storage.database import get_session, get_session_factory
from modellab.deployments import DeploymentManager, build_deployment_providers
from modellab.deployments.manager import DeploymentLifecycleError


app = FastAPI(
    title="ModelLab Control Plane",
    version="0.1.0",
    description="Configuration and evaluation-run control plane for LLM serving.",
)

def get_deployment_manager() -> DeploymentManager:
    return DeploymentManager(get_session_factory(), build_deployment_providers())

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
) -> MockChatCompletion | StreamingResponse:
    """Generate a deterministic mock response without loading a real model."""

    completion = deterministic_mock_completion(payload)
    if not payload.stream:
        return completion

    content = completion.choices[0].message.content
    midpoint = max(1, len(content) // 2)
    chunks = (content[:midpoint], content[midpoint:])
    events: list[dict[str, object]] = [
        {
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            "usage": None,
        }
    ]
    events.extend(
        {
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [{"index": 0, "delta": {"content": chunk}}],
            "usage": None,
        }
        for chunk in chunks
        if chunk
    )
    events.append(
        {
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [],
            "usage": completion.usage.model_dump(),
        }
    )
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return StreamingResponse(iter((body,)), media_type="text/event-stream")


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
    "/v1/model-profiles/{profile_id}/deployments",
    response_model=ModelDeployment,
    status_code=status.HTTP_201_CREATED,
    tags=["model deployments"],
)
async def create_model_deployment(
    profile_id: UUID,
    payload: ModelDeploymentRequest,
    session: Session = Depends(get_session),
    manager: DeploymentManager = Depends(get_deployment_manager),
) -> ModelDeployment:
    if repositories.get_model_profile(session, profile_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model profile not found")
    if not manager.supports(payload.provider):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Deployment provider is not enabled: {payload.provider}",
        )
    deployment = repositories.create_model_deployment(
        session,
        ModelDeploymentCreate(
            model_profile_id=profile_id,
            provider=payload.provider,
            lifecycle_policy=payload.lifecycle_policy,
        ),
    )
    return await manager.start(deployment.id)


@app.get(
    "/v1/model-deployments/{deployment_id}",
    response_model=ModelDeployment,
    tags=["model deployments"],
)
def get_model_deployment(
    deployment_id: UUID, session: Session = Depends(get_session)
) -> ModelDeployment:
    deployment = repositories.get_model_deployment(session, deployment_id)
    if deployment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model deployment not found")
    return deployment


@app.post(
    "/v1/model-deployments/{deployment_id}/stop",
    response_model=ModelDeployment,
    tags=["model deployments"],
)
async def stop_model_deployment(
    deployment_id: UUID,
    session: Session = Depends(get_session),
    manager: DeploymentManager = Depends(get_deployment_manager),
) -> ModelDeployment:
    if repositories.get_model_deployment(session, deployment_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model deployment not found")
    try:
        return await manager.stop(deployment_id)
    except DeploymentLifecycleError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


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
                    "model_deployment_id": "00000000-0000-0000-0000-000000000001",
                    "workload_name": "smoke-test",
                    "request_count": 10,
                    "concurrency": 2,
                },
            }
        }
    ),
    session: Session = Depends(get_session),
) -> EvaluationRun:
    deployment = repositories.get_model_deployment(session, payload.model_deployment_id)
    if deployment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model deployment not found")
    if deployment.status is not ModelDeploymentStatus.READY:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model deployment is not ready")
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
