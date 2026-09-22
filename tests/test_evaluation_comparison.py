from datetime import datetime, timezone
from uuid import uuid4

import pytest

from modellab.analysis import compare_evaluation_runs
from modellab.api.models import EvaluationMetrics, EvaluationRun, EvaluationRunStatus


def _run(*, p95_ttft_ms: float, throughput: float, workload_hash: str = "a" * 64):
    return EvaluationRun(
        id=uuid4(), model_profile_id=uuid4(), model_deployment_id=uuid4(),
        workload_name="smoke-test", workload_version="1.0.0",
        workload_hash=workload_hash,
        generation_settings={"temperature": 0.0, "max_tokens": 128, "seed": 42},
        warmup_request_count=2, request_count=100, concurrency=4,
        status=EvaluationRunStatus.SUCCEEDED, created_at=datetime.now(timezone.utc),
        metrics=EvaluationMetrics(
            request_count=100, successful_requests=100,
            p95_ttft_ms=p95_ttft_ms, output_tokens_per_second=throughput,
        ),
    )


def test_comparison_reports_absolute_and_percentage_changes() -> None:
    comparison = compare_evaluation_runs(
        _run(p95_ttft_ms=100, throughput=50),
        _run(p95_ttft_ms=80, throughput=60),
    )
    assert comparison.metrics["p95_ttft_ms"].absolute_change == -20
    assert comparison.metrics["p95_ttft_ms"].percent_change == -20
    assert comparison.metrics["output_tokens_per_second"].percent_change == 20


def test_comparison_rejects_different_workload_content() -> None:
    with pytest.raises(ValueError, match="content hash"):
        compare_evaluation_runs(
            _run(p95_ttft_ms=100, throughput=50),
            _run(p95_ttft_ms=80, throughput=60, workload_hash="b" * 64),
        )
