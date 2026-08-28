"""Integration tests for ModelLab's PostgreSQL persistence layer.

These tests only run against the dedicated ``modellab_test`` database and
delete every record they create, keeping repeated local test runs clean.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy.engine import make_url

from modellab.api.models import EvaluationRunCreate, ModelProfileCreate, ServingEngine
from modellab.storage import repositories
from modellab.storage.database import create_schema, get_session_factory


TEST_DATABASE_URL = os.getenv(
    "MODELLAB_TEST_DATABASE_URL", "postgresql+psycopg://localhost/modellab_test"
)


def _require_dedicated_test_database() -> str:
    """Refuse to run cleanup tests against a database not explicitly named test."""

    database_name = make_url(TEST_DATABASE_URL).database or ""
    if "test" not in database_name.lower():
        raise RuntimeError(
            "MODELLAB_TEST_DATABASE_URL must point to a dedicated test database."
        )
    return TEST_DATABASE_URL


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    create_schema(_require_dedicated_test_database())


def test_model_profile_and_evaluation_run_persist_then_are_cleaned_up() -> None:
    """Records survive a new session, can be queried, and are explicitly removed."""

    database_url = _require_dedicated_test_database()
    session_factory = get_session_factory(database_url)
    profile_id = None
    run_id = None

    try:
        with session_factory() as session:
            profile = repositories.create_model_profile(
                session,
                ModelProfileCreate(
                    name="postgres-storage-test-profile",
                    model="modellab/mock-1",
                    engine=ServingEngine.MOCK,
                    quantization="none",
                    prefix_caching=True,
                    max_concurrent_sequences=4,
                    max_context_tokens=2048,
                ),
            )
            profile_id = profile.id

        with session_factory() as session:
            saved_profile = repositories.get_model_profile(session, profile_id)
            assert saved_profile is not None
            assert saved_profile.name == "postgres-storage-test-profile"
            assert saved_profile.engine is ServingEngine.MOCK

            run = repositories.create_evaluation_run(
                session,
                EvaluationRunCreate(
                    model_profile_id=profile_id,
                    workload_name="postgres-storage-test-workload",
                    request_count=12,
                    concurrency=3,
                ),
            )
            run_id = run.id

        with session_factory() as session:
            saved_run = repositories.get_evaluation_run(session, run_id)
            assert saved_run is not None
            assert saved_run.model_profile_id == profile_id
            assert saved_run.request_count == 12
            assert saved_run.metrics is None
    finally:
        with session_factory() as session:
            if run_id is not None:
                assert repositories.delete_evaluation_run(session, run_id)
            if profile_id is not None:
                assert repositories.delete_model_profile(session, profile_id)

        with session_factory() as session:
            if run_id is not None:
                assert repositories.get_evaluation_run(session, run_id) is None
            if profile_id is not None:
                assert repositories.get_model_profile(session, profile_id) is None
