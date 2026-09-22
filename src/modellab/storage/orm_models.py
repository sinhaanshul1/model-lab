"""SQLAlchemy table definitions for ModelLab's durable control-plane state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Uuid, func
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
    deployments: Mapped[list["ModelDeploymentRecord"]] = relationship(
        back_populates="model_profile", cascade="all, delete-orphan"
    )


class ModelDeploymentRecord(Base):
    __tablename__ = "model_deployments"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("model_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    lifecycle_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    runtime_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    endpoint_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_profile: Mapped[ModelProfileRecord] = relationship(back_populates="deployments")
    evaluation_runs: Mapped[list["EvaluationRunRecord"]] = relationship(
        back_populates="model_deployment"
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
    # Nullable only so evaluation rows created before deployments were introduced
    # remain readable. All new evaluation runs are linked to a deployment.
    model_deployment_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("model_deployments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workload_name: Mapped[str] = mapped_column(String(100), nullable=False)
    workload_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    workload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generation_settings: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    warmup_request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False)
    concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    successful_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_rate: Mapped[float | None] = mapped_column(nullable=True)
    benchmark_duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    request_throughput_per_second: Mapped[float | None] = mapped_column(nullable=True)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scored_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    passed_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    p50_ttft_ms: Mapped[float | None] = mapped_column(nullable=True)
    p95_ttft_ms: Mapped[float | None] = mapped_column(nullable=True)
    p99_ttft_ms: Mapped[float | None] = mapped_column(nullable=True)
    p50_end_to_end_latency_ms: Mapped[float | None] = mapped_column(nullable=True)
    p95_end_to_end_latency_ms: Mapped[float | None] = mapped_column(nullable=True)
    p99_end_to_end_latency_ms: Mapped[float | None] = mapped_column(nullable=True)
    output_tokens_per_second: Mapped[float | None] = mapped_column(nullable=True)
    quality_score: Mapped[float | None] = mapped_column(nullable=True)
    request_metrics: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_profile: Mapped[ModelProfileRecord] = relationship(back_populates="evaluation_runs")
    model_deployment: Mapped[ModelDeploymentRecord | None] = relationship(
        back_populates="evaluation_runs"
    )
