"""Database connection and schema helpers for local PostgreSQL development."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Generator

from sqlalchemy import Engine, create_engine
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

    Base.metadata.create_all(get_engine(database_url))


def reset_database_configuration() -> None:
    """Clear cached connections after changing database URLs in tests."""

    get_session_factory.cache_clear()
    get_engine.cache_clear()
