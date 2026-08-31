"""Integration tests for the PostgreSQL-backed mock benchmark worker."""

from __future__ import annotations

import asyncio
import os
from uuid import UUID

import httpx
import pytest
from sqlalchemy.engine import make_url

from modellab.api.models import (
    EvaluationRunCreate,
    EvaluationRunStatus,
    ModelProfileCreate,
    ServingEngine,
)
from modellab.storage import repositories
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


def _create_profile_and_run(*, request_count: int = 4) -> tuple[UUID, UUID]:
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
        run = repositories.create_evaluation_run(
            session,
            EvaluationRunCreate(
                model_profile_id=profile.id,
                workload_name="smoke-test",
                request_count=request_count,
                concurrency=2,
            ),
        )
    return profile.id, run.id


def _cleanup(profile_id: UUID, run_ids: list[UUID]) -> None:
    session_factory = get_session_factory(_require_dedicated_test_database())
    with session_factory() as session:
        for run_id in run_ids:
            repositories.delete_evaluation_run(session, run_id)
        repositories.delete_model_profile(session, profile_id)


def _success_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"usage": {"completion_tokens": 7}},
        request=request,
    )


def _failure_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"detail": "mock outage"}, request=request)


def test_only_one_session_can_claim_each_queued_run() -> None:
    profile_id, first_run_id = _create_profile_and_run()
    session_factory = get_session_factory(_require_dedicated_test_database())
    second_run_id = None

    try:
        with session_factory() as session:
            second_run = repositories.create_evaluation_run(
                session,
                EvaluationRunCreate(
                    model_profile_id=profile_id,
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
        _cleanup(profile_id, [first_run_id, *([second_run_id] if second_run_id else [])])


def test_worker_completes_a_queued_mock_evaluation() -> None:
    profile_id, run_id = _create_profile_and_run(request_count=4)
    session_factory = get_session_factory(_require_dedicated_test_database())
    worker = BenchmarkWorker(
        session_factory,
        "http://mock-model/v1/chat/completions",
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
        assert completed_run.metrics.p95_ttft_ms is not None
        assert completed_run.metrics.output_tokens_per_second is not None
    finally:
        _cleanup(profile_id, [run_id])


def test_worker_marks_a_run_failed_when_mock_requests_fail() -> None:
    profile_id, run_id = _create_profile_and_run(request_count=1)
    session_factory = get_session_factory(_require_dedicated_test_database())
    worker = BenchmarkWorker(
        session_factory,
        "http://mock-model/v1/chat/completions",
        transport=httpx.MockTransport(_failure_response),
    )

    try:
        assert asyncio.run(worker.process_next_run())

        with session_factory() as session:
            failed_run = repositories.get_evaluation_run(session, run_id)
        assert failed_run is not None
        assert failed_run.status is EvaluationRunStatus.FAILED
        assert failed_run.finished_at is not None
    finally:
        _cleanup(profile_id, [run_id])
