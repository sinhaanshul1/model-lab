"""Integration tests for the PostgreSQL-backed mock benchmark worker."""

from __future__ import annotations

import asyncio
import json
import os
from uuid import UUID

import httpx
import pytest
from sqlalchemy.engine import make_url

from modellab.api.models import (
    EvaluationRunCreate,
    EvaluationRunStatus,
    ModelDeploymentCreate,
    ModelDeploymentStatus,
    ModelProfileCreate,
    ServingEngine,
)
from modellab.storage import repositories
from modellab.deployments import DeploymentManager, MockDeploymentProvider
from modellab.storage.database import create_schema, get_session_factory
from modellab.worker.benchmark_worker import BenchmarkWorker


TEST_DATABASE_URL = os.getenv(
    "MODELLAB_TEST_DATABASE_URL", "postgresql+psycopg://localhost/modellab_test"
)


def _require_dedicated_test_database() -> str:
    database_name = make_url(TEST_DATABASE_URL).database or ""
    if "test" not in database_name.lower():
        raise RuntimeError(
            "MODELLAB_TEST_DATABASE_URL must point to a dedicated test database."
        )
    return TEST_DATABASE_URL


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    create_schema(_require_dedicated_test_database())


def _deployment_manager() -> DeploymentManager:
    return DeploymentManager(
        get_session_factory(_require_dedicated_test_database()),
        {"mock": MockDeploymentProvider("http://mock-model/v1/chat/completions")},
    )


def _create_profile_and_run(*, request_count: int = 4) -> tuple[UUID, UUID, UUID]:
    session_factory = get_session_factory(_require_dedicated_test_database())
    with session_factory() as session:
        profile = repositories.create_model_profile(
            session,
            ModelProfileCreate(
                name="benchmark-worker-test-profile",
                model="modellab/mock-1",
                engine=ServingEngine.MOCK,
            ),
        )
        deployment = repositories.create_model_deployment(
            session, ModelDeploymentCreate(model_profile_id=profile.id)
        )
        run = repositories.create_evaluation_run(
            session,
            EvaluationRunCreate(
                model_deployment_id=deployment.id,
                workload_name="smoke-test",
                request_count=request_count,
                concurrency=2,
            ),
        )
    asyncio.run(_deployment_manager().start(deployment.id))
    return profile.id, deployment.id, run.id


def _cleanup(profile_id: UUID, deployment_id: UUID, run_ids: list[UUID]) -> None:
    session_factory = get_session_factory(_require_dedicated_test_database())
    with session_factory() as session:
        for run_id in run_ids:
            repositories.delete_evaluation_run(session, run_id)
        repositories.delete_model_deployment(session, deployment_id)
        repositories.delete_model_profile(session, profile_id)


def _success_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    events = [
        {
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hello"}}
            ],
            "usage": None,
        },
        {"choices": [], "usage": {"completion_tokens": 7}},
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    return httpx.Response(
        200,
        text=content,
        headers={"content-type": "text/event-stream"},
        request=request,
    )


def _failure_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"detail": "mock outage"}, request=request)


class DelayedSSEStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}],"usage":null}\n\n'
        await asyncio.sleep(0.01)
        yield b'data: {"choices":[{"delta":{"content":"first token"}}],"usage":null}\n\n'
        await asyncio.sleep(0.01)
        yield b'data: {"choices":[],"usage":{"completion_tokens":4}}\n\n'
        yield b"data: [DONE]\n\n"


def _delayed_stream_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        stream=DelayedSSEStream(),
        headers={"content-type": "text/event-stream"},
        request=request,
    )


def test_only_one_session_can_claim_each_queued_run() -> None:
    profile_id, deployment_id, first_run_id = _create_profile_and_run()
    session_factory = get_session_factory(_require_dedicated_test_database())
    second_run_id = None

    try:
        with session_factory() as session:
            second_run = repositories.create_evaluation_run(
                session,
                EvaluationRunCreate(
                    model_deployment_id=deployment_id,
                    workload_name="smoke-test",
                    request_count=2,
                    concurrency=1,
                ),
            )
            second_run_id = second_run.id

        with session_factory() as first_session:
            first_claim = repositories.claim_next_evaluation_run(first_session)
        with session_factory() as second_session:
            second_claim = repositories.claim_next_evaluation_run(second_session)

        assert first_claim is not None
        assert second_claim is not None
        assert first_claim.id != second_claim.id
        assert {first_claim.id, second_claim.id} == {first_run_id, second_run_id}
    finally:
        _cleanup(
            profile_id, deployment_id, [first_run_id, *([second_run_id] if second_run_id else [])]
        )


def test_worker_completes_a_queued_mock_evaluation() -> None:
    profile_id, deployment_id, run_id = _create_profile_and_run(request_count=4)
    session_factory = get_session_factory(_require_dedicated_test_database())
    worker = BenchmarkWorker(
        session_factory,
        _deployment_manager(),
        transport=httpx.MockTransport(_success_response),
    )

    try:
        assert asyncio.run(worker.process_next_run())

        with session_factory() as session:
            completed_run = repositories.get_evaluation_run(session, run_id)
        assert completed_run is not None
        assert completed_run.status is EvaluationRunStatus.SUCCEEDED
        assert completed_run.metrics is not None
        assert completed_run.metrics.request_count == 4
        assert completed_run.metrics.successful_requests == 4
        assert completed_run.metrics.p50_ttft_ms is not None
        assert completed_run.metrics.p95_ttft_ms is not None
        assert completed_run.metrics.p99_ttft_ms is not None
        assert completed_run.metrics.p50_end_to_end_latency_ms is not None
        assert completed_run.metrics.p95_end_to_end_latency_ms is not None
        assert completed_run.metrics.p99_end_to_end_latency_ms is not None
        assert completed_run.metrics.output_tokens_per_second is not None
        assert len(completed_run.metrics.request_metrics) == 4
        assert {metric.request_index for metric in completed_run.metrics.request_metrics} == {
            0,
            1,
            2,
            3,
        }
        assert all(
            metric.ttft_ms <= metric.end_to_end_latency_ms
            for metric in completed_run.metrics.request_metrics
        )
        assert all(
            metric.completion_tokens == 7
            for metric in completed_run.metrics.request_metrics
        )
        with session_factory() as session:
            deployment = repositories.get_model_deployment(session, deployment_id)
        assert deployment is not None
        assert deployment.status is ModelDeploymentStatus.STOPPED
    finally:
        _cleanup(profile_id, deployment_id, [run_id])


def test_worker_marks_a_run_failed_when_mock_requests_fail() -> None:
    profile_id, deployment_id, run_id = _create_profile_and_run(request_count=1)
    session_factory = get_session_factory(_require_dedicated_test_database())
    worker = BenchmarkWorker(
        session_factory,
        _deployment_manager(),
        transport=httpx.MockTransport(_failure_response),
    )

    try:
        assert asyncio.run(worker.process_next_run())

        with session_factory() as session:
            failed_run = repositories.get_evaluation_run(session, run_id)
        assert failed_run is not None
        assert failed_run.status is EvaluationRunStatus.FAILED
        assert failed_run.finished_at is not None
        with session_factory() as session:
            deployment = repositories.get_model_deployment(session, deployment_id)
        assert deployment is not None
        assert deployment.status is ModelDeploymentStatus.STOPPED
    finally:
        _cleanup(profile_id, deployment_id, [run_id])


def test_worker_measures_first_token_before_stream_completion() -> None:
    profile_id, deployment_id, run_id = _create_profile_and_run(request_count=1)
    session_factory = get_session_factory(_require_dedicated_test_database())
    worker = BenchmarkWorker(
        session_factory,
        _deployment_manager(),
        transport=httpx.MockTransport(_delayed_stream_response),
    )

    try:
        assert asyncio.run(worker.process_next_run())

        with session_factory() as session:
            completed_run = repositories.get_evaluation_run(session, run_id)
        assert completed_run is not None
        assert completed_run.metrics is not None
        request_metric = completed_run.metrics.request_metrics[0]
        assert request_metric.ttft_ms >= 5
        assert request_metric.end_to_end_latency_ms > request_metric.ttft_ms
        assert request_metric.completion_tokens == 4
    finally:
        _cleanup(profile_id, deployment_id, [run_id])
