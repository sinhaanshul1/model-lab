"""End-to-end API, deployment, benchmark, persistence, and cleanup coverage."""

from __future__ import annotations

import asyncio
import json
import os
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from modellab.api.main import app, get_deployment_manager
from modellab.api.models import EvaluationRunStatus, ModelDeploymentStatus
from modellab.deployments import DeploymentManager, MockDeploymentProvider
from modellab.storage import repositories
from modellab.storage.database import create_schema, get_session, get_session_factory
from modellab.worker.benchmark_worker import BenchmarkWorker


TEST_DATABASE_URL = os.getenv(
    "MODELLAB_TEST_DATABASE_URL", "postgresql+psycopg://localhost/modellab_test"
)
MOCK_ENDPOINT_URL = "http://mock-model/v1/chat/completions"


def _require_dedicated_test_database() -> str:
    database_name = make_url(TEST_DATABASE_URL).database or ""
    if "test" not in database_name.lower():
        raise RuntimeError("MODELLAB_TEST_DATABASE_URL must point to a dedicated test database.")
    return TEST_DATABASE_URL


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    create_schema(_require_dedicated_test_database())


def _success_response(request: httpx.Request) -> httpx.Response:
    events = [
        {"choices": [{"index": 0, "delta": {"content": "hello"}}], "usage": None},
        {"choices": [], "usage": {"completion_tokens": 5}},
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        text=content,
        headers={"content-type": "text/event-stream"},
        request=request,
    )


def test_ephemeral_evaluation_lifecycle_from_api_to_cleanup() -> None:
    session_factory = get_session_factory(_require_dedicated_test_database())
    manager = DeploymentManager(
        session_factory, {"mock": MockDeploymentProvider(MOCK_ENDPOINT_URL)}
    )

    def test_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = test_session
    app.dependency_overrides[get_deployment_manager] = lambda: manager
    profile_id = None
    deployment_id = None
    run_id = None

    try:
        with TestClient(app) as client:
            profile_response = client.post(
                "/v1/model-profiles",
                json={
                    "name": "full-lifecycle-test-profile",
                    "model": "modellab/mock-1",
                    "engine": "mock",
                },
            )
            assert profile_response.status_code == 201
            profile_id = UUID(profile_response.json()["id"])

            deployment_response = client.post(
                f"/v1/model-profiles/{profile_id}/deployments",
                json={"provider": "mock", "lifecycle_policy": "ephemeral"},
            )
            assert deployment_response.status_code == 201
            assert deployment_response.json()["status"] == "ready"
            deployment_id = UUID(deployment_response.json()["id"])

            run_response = client.post(
                "/v1/evaluation-runs",
                json={
                    "model_deployment_id": str(deployment_id),
                    "workload_name": "smoke-test",
                    "workload_version": "1.0.0",
                    "request_count": 3,
                    "concurrency": 2,
                },
            )
            assert run_response.status_code == 202
            run_id = UUID(run_response.json()["id"])

            worker = BenchmarkWorker(
                session_factory,
                manager,
                transport=httpx.MockTransport(_success_response),
            )
            assert asyncio.run(worker.process_next_run())

            completed_response = client.get(f"/v1/evaluation-runs/{run_id}")
            assert completed_response.status_code == 200
            assert completed_response.json()["status"] == EvaluationRunStatus.SUCCEEDED
            assert completed_response.json()["metrics"]["successful_requests"] == 3

            stopped_response = client.get(f"/v1/model-deployments/{deployment_id}")
            assert stopped_response.status_code == 200
            assert stopped_response.json()["status"] == ModelDeploymentStatus.STOPPED
            assert stopped_response.json()["endpoint_url"] is None
    finally:
        app.dependency_overrides.clear()
        with session_factory() as session:
            if run_id is not None:
                repositories.delete_evaluation_run(session, run_id)
            if deployment_id is not None:
                repositories.delete_model_deployment(session, deployment_id)
            if profile_id is not None:
                repositories.delete_model_profile(session, profile_id)
