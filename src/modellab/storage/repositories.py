"""Repository functions for ModelLab control-plane records."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from modellab.api.models import (
    EvaluationMetrics,
    EvaluationRun,
    EvaluationRunCreate,
    EvaluationRunStatus,
    ModelDeployment,
    ModelDeploymentCreate,
    ModelDeploymentStatus,
    ModelProfile,
    ModelProfileCreate,
    ModelProfileStatus,
)
from modellab.storage.orm_models import (
    EvaluationRunRecord,
    ModelDeploymentRecord,
    ModelProfileRecord,
)


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
        model_deployment_id=record.model_deployment_id,
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


def to_model_deployment(record: ModelDeploymentRecord) -> ModelDeployment:
    return ModelDeployment(
        id=record.id,
        model_profile_id=record.model_profile_id,
        provider=record.provider,
        lifecycle_policy=record.lifecycle_policy,
        status=record.status,
        runtime_id=record.runtime_id,
        endpoint_url=record.endpoint_url,
        failure_reason=record.failure_reason,
        created_at=record.created_at,
        ready_at=record.ready_at,
        stopped_at=record.stopped_at,
    )


def create_model_deployment(
    session: Session, payload: ModelDeploymentCreate
) -> ModelDeployment:
    record = ModelDeploymentRecord(
        model_profile_id=payload.model_profile_id,
        provider=payload.provider,
        lifecycle_policy=payload.lifecycle_policy.value,
        status=ModelDeploymentStatus.PENDING.value,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def get_model_deployment(session: Session, deployment_id: UUID) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    return to_model_deployment(record) if record is not None else None


def mark_model_deployment_starting(
    session: Session, deployment_id: UUID
) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None or record.status != ModelDeploymentStatus.PENDING.value:
        return None
    record.status = ModelDeploymentStatus.STARTING.value
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def mark_model_deployment_ready(
    session: Session, deployment_id: UUID, *, runtime_id: str, endpoint_url: str
) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None or record.status != ModelDeploymentStatus.STARTING.value:
        return None
    record.status = ModelDeploymentStatus.READY.value
    record.runtime_id = runtime_id
    record.endpoint_url = endpoint_url
    record.ready_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def mark_model_deployment_stopping(
    session: Session, deployment_id: UUID
) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None or record.status != ModelDeploymentStatus.READY.value:
        return None
    record.status = ModelDeploymentStatus.STOPPING.value
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def mark_model_deployment_stopped(
    session: Session, deployment_id: UUID
) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None or record.status != ModelDeploymentStatus.STOPPING.value:
        return None
    record.status = ModelDeploymentStatus.STOPPED.value
    record.runtime_id = None
    record.endpoint_url = None
    record.stopped_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def mark_model_deployment_failed(
    session: Session, deployment_id: UUID, failure_reason: str
) -> ModelDeployment | None:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None or record.status in {
        ModelDeploymentStatus.STOPPED.value,
        ModelDeploymentStatus.FAILED.value,
    }:
        return None
    record.status = ModelDeploymentStatus.FAILED.value
    record.failure_reason = failure_reason[:2000]
    record.runtime_id = None
    record.endpoint_url = None
    record.stopped_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return to_model_deployment(record)


def list_model_deployments(session: Session) -> list[ModelDeployment]:
    records = session.scalars(
        select(ModelDeploymentRecord).order_by(
            ModelDeploymentRecord.created_at, ModelDeploymentRecord.id
        )
    ).all()
    return [to_model_deployment(record) for record in records]


def delete_model_deployment(session: Session, deployment_id: UUID) -> bool:
    record = session.get(ModelDeploymentRecord, deployment_id)
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True


def create_evaluation_run(session: Session, payload: EvaluationRunCreate) -> EvaluationRun:
    deployment = session.get(ModelDeploymentRecord, payload.model_deployment_id)
    if deployment is None:
        raise ValueError("Model deployment not found")
    record = EvaluationRunRecord(
        model_profile_id=deployment.model_profile_id,
        model_deployment_id=deployment.id,
        workload_name=payload.workload_name,
        request_count=payload.request_count,
        concurrency=payload.concurrency,
        status=EvaluationRunStatus.QUEUED.value,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return to_evaluation_run(record)


def claim_next_evaluation_run(session: Session) -> EvaluationRun | None:
    """Atomically claim the oldest queued run for one benchmark worker.

    The row lock is held only while its status changes to ``running``. A
    worker never holds a database lock while performing a benchmark.
    """

    record = session.scalars(
        select(EvaluationRunRecord)
        .where(EvaluationRunRecord.status == EvaluationRunStatus.QUEUED.value)
        .order_by(EvaluationRunRecord.created_at, EvaluationRunRecord.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    ).first()
    if record is None:
        return None

    record.status = EvaluationRunStatus.RUNNING.value
    record.started_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return to_evaluation_run(record)


def complete_evaluation_run(
    session: Session, run_id: UUID, metrics: EvaluationMetrics
) -> EvaluationRun | None:
    """Persist benchmark metrics and mark a claimed run as succeeded."""

    record = session.get(EvaluationRunRecord, run_id)
    if record is None or record.status != EvaluationRunStatus.RUNNING.value:
        return None

    record.status = EvaluationRunStatus.SUCCEEDED.value
    record.successful_requests = metrics.successful_requests
    record.p95_ttft_ms = metrics.p95_ttft_ms
    record.output_tokens_per_second = metrics.output_tokens_per_second
    record.quality_score = metrics.quality_score
    record.finished_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return to_evaluation_run(record)


def fail_evaluation_run(session: Session, run_id: UUID) -> EvaluationRun | None:
    """Mark a claimed run as failed after the worker logs its diagnostic."""

    record = session.get(EvaluationRunRecord, run_id)
    if record is None or record.status != EvaluationRunStatus.RUNNING.value:
        return None

    record.status = EvaluationRunStatus.FAILED.value
    record.finished_at = datetime.now(timezone.utc)
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
