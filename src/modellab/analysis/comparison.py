"""Create apples-to-apples comparisons between completed evaluations."""

from __future__ import annotations

from modellab.api.models import EvaluationComparison, EvaluationRun, EvaluationRunStatus, MetricComparison


COMPARISON_METRICS = (
    "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "p50_end_to_end_latency_ms", "p95_end_to_end_latency_ms",
    "p99_end_to_end_latency_ms", "output_tokens_per_second",
    "request_throughput_per_second", "error_rate", "quality_score",
)


def compare_evaluation_runs(baseline: EvaluationRun, candidate: EvaluationRun) -> EvaluationComparison:
    for run, label in ((baseline, "baseline"), (candidate, "candidate")):
        if run.status is not EvaluationRunStatus.SUCCEEDED or run.metrics is None:
            raise ValueError(f"The {label} evaluation has not succeeded.")
    if not baseline.workload_hash or baseline.workload_hash != candidate.workload_hash:
        raise ValueError("Evaluations must use the same exact workload content hash.")
    if baseline.request_count != candidate.request_count:
        raise ValueError("Evaluations must use the same measured request count.")
    if baseline.concurrency != candidate.concurrency:
        raise ValueError("Evaluations must use the same concurrency.")
    if baseline.warmup_request_count != candidate.warmup_request_count:
        raise ValueError("Evaluations must use the same warm-up request count.")

    comparisons: dict[str, MetricComparison] = {}
    for metric_name in COMPARISON_METRICS:
        baseline_value = getattr(baseline.metrics, metric_name)
        candidate_value = getattr(candidate.metrics, metric_name)
        if baseline_value is None or candidate_value is None:
            continue
        comparisons[metric_name] = MetricComparison(
            baseline=baseline_value,
            candidate=candidate_value,
            absolute_change=candidate_value - baseline_value,
            percent_change=(
                ((candidate_value - baseline_value) / baseline_value) * 100
                if baseline_value != 0 else None
            ),
        )

    return EvaluationComparison(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        workload_name=baseline.workload_name,
        workload_version=baseline.workload_version,
        workload_hash=baseline.workload_hash,
        request_count=baseline.request_count,
        concurrency=baseline.concurrency,
        metrics=comparisons,
    )
