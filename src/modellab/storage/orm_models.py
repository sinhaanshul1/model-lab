"""SQLAlchemy table definitions for ModelLab's durable control-plane state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelProfileRecord(Base):
    __tablename__ = "model_profiles"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    quantization: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prefix_caching: Mapped[bool] = mapped_column(nullable=False, default=False)
    max_concurrent_sequences: Mapped[int] = mapped_column(Integer, nullable=False)
    max_context_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    evaluation_runs: Mapped[list["EvaluationRunRecord"]] = relationship(
        back_populates="model_profile", cascade="all, delete-orphan"
    )


class EvaluationRunRecord(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("model_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workload_name: Mapped[str] = mapped_column(String(100), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False)
    concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    successful_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    p95_ttft_ms: Mapped[float | None] = mapped_column(nullable=True)
    output_tokens_per_second: Mapped[float | None] = mapped_column(nullable=True)
    quality_score: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_profile: Mapped[ModelProfileRecord] = relationship(back_populates="evaluation_runs")
