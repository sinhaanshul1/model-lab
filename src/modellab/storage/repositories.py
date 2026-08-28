"""Repository functions for ModelLab control-plane records."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from modellab.api.models import (
    EvaluationMetrics,
    EvaluationRun,
    EvaluationRunCreate,
    EvaluationRunStatus,
    ModelProfile,
    ModelProfileCreate,
    ModelProfileStatus,
)
from modellab.storage.orm_models import EvaluationRunRecord, ModelProfileRecord


def to_model_profile(record: ModelProfileRecord) -> ModelProfile:
    return ModelProfile(
        id=record.id,
        name=record.name,
        model=record.model,
        engine=record.engine,
        quantization=record.quantization,
        prefix_caching=record.prefix_caching,
        max_concurrent_sequences=record.max_concurrent_sequences,
        max_context_tokens=record.max_context_tokens,
        status=ModelProfileStatus(record.status),
        created_at=record.created_at,
    )


def to_evaluation_run(record: EvaluationRunRecord) -> EvaluationRun:
    has_metrics = any(
        value is not None
        for value in (
            record.p95_ttft_ms,
            record.output_tokens_per_second,
            record.quality_score,
        )
    ) or record.successful_requests > 0
    metrics = (
        EvaluationMetrics(
            request_count=record.request_count,
            successful_requests=record.successful_requests,
            p95_ttft_ms=record.p95_ttft_ms,
            output_tokens_per_second=record.output_tokens_per_second,
            quality_score=record.quality_score,
        )
        if has_metrics
        else None
    )
    return EvaluationRun(
        id=record.id,
        model_profile_id=record.model_profile_id,
        workload_name=record.workload_name,
        request_count=record.request_count,
        concurrency=record.concurrency,
        status=EvaluationRunStatus(record.status),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        metrics=metrics,
    )


def create_model_profile(session: Session, payload: ModelProfileCreate) -> ModelProfile:
    record = ModelProfileRecord(
        name=payload.name,
        model=payload.model,
        engine=payload.engine.value,
        quantization=payload.quantization,
        prefix_caching=payload.prefix_caching,
        max_concurrent_sequences=payload.max_concurrent_sequences,
        max_context_tokens=payload.max_context_tokens,
        status=ModelProfileStatus.CANDIDATE.value,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return to_model_profile(record)


def get_model_profile(session: Session, profile_id: UUID) -> ModelProfile | None:
    record = session.get(ModelProfileRecord, profile_id)
    return to_model_profile(record) if record is not None else None


def list_model_profiles(session: Session) -> list[ModelProfile]:
    records = session.scalars(
        select(ModelProfileRecord).order_by(ModelProfileRecord.created_at, ModelProfileRecord.id)
    ).all()
    return [to_model_profile(record) for record in records]


def create_evaluation_run(session: Session, payload: EvaluationRunCreate) -> EvaluationRun:
    record = EvaluationRunRecord(
        model_profile_id=payload.model_profile_id,
        workload_name=payload.workload_name,
        request_count=payload.request_count,
        concurrency=payload.concurrency,
        status=EvaluationRunStatus.QUEUED.value,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return to_evaluation_run(record)


def get_evaluation_run(session: Session, run_id: UUID) -> EvaluationRun | None:
    record = session.get(EvaluationRunRecord, run_id)
    return to_evaluation_run(record) if record is not None else None


def list_evaluation_runs(session: Session) -> list[EvaluationRun]:
    records = session.scalars(
        select(EvaluationRunRecord).order_by(EvaluationRunRecord.created_at, EvaluationRunRecord.id)
    ).all()
    return [to_evaluation_run(record) for record in records]


def delete_evaluation_run(session: Session, run_id: UUID) -> bool:
    record = session.get(EvaluationRunRecord, run_id)
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True


def delete_model_profile(session: Session, profile_id: UUID) -> bool:
    record = session.get(ModelProfileRecord, profile_id)
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True
