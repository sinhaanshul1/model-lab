"""Integration tests for deployment lifecycle state and mock runtime handling."""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy.engine import make_url

from modellab.api.models import (
    ModelDeploymentCreate,
    ModelDeploymentStatus,
    ModelProfileCreate,
    ServingEngine,
)
from modellab.deployments import DeploymentManager, MockDeploymentProvider
from modellab.storage import repositories
from modellab.storage.database import create_schema, get_session_factory


TEST_DATABASE_URL = os.getenv(
    "MODELLAB_TEST_DATABASE_URL", "postgresql+psycopg://localhost/modellab_test"
)


def _require_dedicated_test_database() -> str:
    database_name = make_url(TEST_DATABASE_URL).database or ""
    if "test" not in database_name.lower():
        raise RuntimeError("MODELLAB_TEST_DATABASE_URL must point to a dedicated test database.")
    return TEST_DATABASE_URL


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    create_schema(_require_dedicated_test_database())


def test_mock_deployment_transitions_from_pending_to_ready_to_stopped() -> None:
    session_factory = get_session_factory(_require_dedicated_test_database())
    manager = DeploymentManager(
        session_factory,
        {"mock": MockDeploymentProvider("http://mock/v1/chat/completions")},
    )
    profile_id = None
    deployment_id = None

    try:
        with session_factory() as session:
            profile = repositories.create_model_profile(
                session,
                ModelProfileCreate(
                    name="deployment-manager-test-profile",
                    model="modellab/mock-1",
                    engine=ServingEngine.MOCK,
                ),
            )
            profile_id = profile.id
            deployment = repositories.create_model_deployment(
                session, ModelDeploymentCreate(model_profile_id=profile.id)
            )
            deployment_id = deployment.id

        ready = asyncio.run(manager.start(deployment_id))
        assert ready.status is ModelDeploymentStatus.READY
        assert ready.runtime_id == f"mock:{deployment_id}"
        assert ready.endpoint_url == "http://mock/v1/chat/completions"
        assert ready.ready_at is not None

        stopped = asyncio.run(manager.stop(deployment_id))
        assert stopped.status is ModelDeploymentStatus.STOPPED
        assert stopped.runtime_id is None
        assert stopped.endpoint_url is None
        assert stopped.stopped_at is not None
    finally:
        with session_factory() as session:
            if deployment_id is not None:
                repositories.delete_model_deployment(session, deployment_id)
            if profile_id is not None:
                repositories.delete_model_profile(session, profile_id)
