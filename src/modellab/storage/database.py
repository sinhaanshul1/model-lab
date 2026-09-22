"""Database connection and schema helpers for local PostgreSQL development."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Generator

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from modellab.storage.orm_models import Base


DEFAULT_DATABASE_URL = "postgresql+psycopg://localhost/modellab_dev"
DATABASE_URL_ENV = "MODELLAB_DATABASE_URL"


def get_database_url() -> str:
    return os.getenv(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_database_url(), pool_pre_ping=True)


@lru_cache
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields one transaction-capable session."""

    with get_session_factory()() as session:
        yield session


def create_schema(database_url: str | None = None) -> None:
    """Create the local development/test tables when they do not already exist."""

    engine = get_engine(database_url)
    Base.metadata.create_all(engine)

    # ``create_all`` does not add columns to tables created by earlier
    # prototype versions. Keep the one additive compatibility upgrade here
    # until ModelLab adopts a versioned migration tool.
    evaluation_columns = {
        column["name"] for column in inspect(engine).get_columns("evaluation_runs")
    }
    if "model_deployment_id" not in evaluation_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE evaluation_runs "
                    "ADD COLUMN model_deployment_id UUID NULL "
                    "REFERENCES model_deployments(id) ON DELETE SET NULL"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_evaluation_runs_model_deployment_id "
                    "ON evaluation_runs (model_deployment_id)"
                )
            )

    metric_columns = {
        "workload_version": "VARCHAR(32) NULL",
        "workload_hash": "VARCHAR(64) NULL",
        "generation_settings": "JSON NULL",
        "warmup_request_count": "INTEGER NOT NULL DEFAULT 0",
        "p50_ttft_ms": "DOUBLE PRECISION NULL",
        "p99_ttft_ms": "DOUBLE PRECISION NULL",
        "p50_end_to_end_latency_ms": "DOUBLE PRECISION NULL",
        "p95_end_to_end_latency_ms": "DOUBLE PRECISION NULL",
        "p99_end_to_end_latency_ms": "DOUBLE PRECISION NULL",
        "request_metrics": "JSON NULL",
    }
    missing_metric_columns = {
        name: definition
        for name, definition in metric_columns.items()
        if name not in evaluation_columns
    }
    if missing_metric_columns:
        with engine.begin() as connection:
            for name, definition in missing_metric_columns.items():
                connection.execute(
                    text(f"ALTER TABLE evaluation_runs ADD COLUMN {name} {definition}")
                )


def reset_database_configuration() -> None:
    """Clear cached connections after changing database URLs in tests."""

    get_session_factory.cache_clear()
    get_engine.cache_clear()
