"""PostgreSQL-backed worker for ModelLab's first mock benchmark workload."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Sequence

import httpx
from sqlalchemy.orm import Session, sessionmaker

from modellab.api.models import (
    DeploymentLifecyclePolicy,
    EvaluationMetrics,
    EvaluationRequestMetrics,
    EvaluationRun,
    ModelDeploymentStatus,
    ModelProfile,
    ServingEngine,
)
from modellab.deployments import DeploymentManager, build_deployment_providers
from modellab.storage import repositories
from modellab.storage.database import get_session_factory
from modellab.workloads import get_workload_registry
from modellab.workloads.models import RegisteredWorkload
from modellab.workloads.scoring import score_answer


LOGGER = logging.getLogger(__name__)
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30.0
class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark cannot produce a usable result."""


class BenchmarkWorker:
    """Claims queued evaluations and benchmarks the configured mock profile."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        deployment_manager: DeploymentManager,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._deployment_manager = deployment_manager
        self._transport = transport

    async def process_next_run(self) -> bool:
        """Claim and process one run; return ``False`` when no work is queued."""

        with self._session_factory() as session:
            run = repositories.claim_next_evaluation_run(session)
        if run is None:
            return False

        deployment = None
        try:
            with self._session_factory() as session:
                profile = repositories.get_model_profile(session, run.model_profile_id)
                if run.model_deployment_id is not None:
                    deployment = repositories.get_model_deployment(
                        session, run.model_deployment_id
                    )
            if profile is None:
                raise BenchmarkExecutionError("The evaluation run's model profile no longer exists.")
            if deployment is None:
                raise BenchmarkExecutionError("The evaluation run's model deployment no longer exists.")
            if deployment.status is not ModelDeploymentStatus.READY:
                raise BenchmarkExecutionError("The evaluation run's model deployment is not ready.")
            if deployment.endpoint_url is None:
                raise BenchmarkExecutionError("The model deployment has no inference endpoint.")
            if profile.engine not in {ServingEngine.MOCK, ServingEngine.VLLM}:
                raise BenchmarkExecutionError(
                    "The benchmark worker requires an OpenAI-compatible mock or vLLM engine."
                )

            workload = get_workload_registry().get(
                run.workload_name, run.workload_version or "1.0.0"
            )
            if workload is None:
                raise BenchmarkExecutionError("The evaluation run's workload no longer exists.")
            if run.workload_hash and workload.content_hash != run.workload_hash:
                raise BenchmarkExecutionError(
                    "The workload definition changed after this evaluation was queued."
                )
            metrics = await self._benchmark_openai_compatible_profile(
                run, profile, deployment.endpoint_url, workload
            )
            with self._session_factory() as session:
                completed_run = repositories.complete_evaluation_run(session, run.id, metrics)
            if completed_run is None:
                raise BenchmarkExecutionError("The claimed evaluation run could not be completed.")
            LOGGER.info("Evaluation run %s succeeded.", run.id)
        except Exception:
            LOGGER.exception("Evaluation run %s failed.", run.id)
            with self._session_factory() as session:
                repositories.fail_evaluation_run(session, run.id)
        finally:
            if (
                deployment is not None
                and deployment.status is ModelDeploymentStatus.READY
                and deployment.lifecycle_policy is DeploymentLifecyclePolicy.EPHEMERAL
            ):
                try:
                    await self._deployment_manager.stop(deployment.id)
                except Exception:
                    LOGGER.exception("Could not clean up deployment %s.", deployment.id)
        return True

    async def run_forever(self, poll_interval_seconds: float) -> None:
        """Continuously process queued work, sleeping only when the queue is empty."""

        while True:
            processed_run = await self.process_next_run()
            if not processed_run:
                await asyncio.sleep(poll_interval_seconds)

    async def _benchmark_openai_compatible_profile(
        self,
        run: EvaluationRun,
        profile: ModelProfile,
        endpoint_url: str,
        workload: RegisteredWorkload,
    ) -> EvaluationMetrics:
        semaphore = asyncio.Semaphore(min(run.concurrency, run.request_count))

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS, transport=self._transport
        ) as client:
            warmup_responses = await asyncio.gather(
                *(
                    self._send_request(
                        client,
                        semaphore,
                        profile,
                        endpoint_url,
                        workload,
                        warmup_index,
                    )
                    for warmup_index in range(run.warmup_request_count)
                ),
                return_exceptions=True,
            )
            warmup_failures = [
                response for response in warmup_responses if isinstance(response, Exception)
            ]
            if warmup_failures:
                raise BenchmarkExecutionError(
                    f"{len(warmup_failures)} of {run.warmup_request_count} warm-up requests failed."
                ) from warmup_failures[0]

            started_at = time.perf_counter()
            responses = await asyncio.gather(
                *(
                    self._send_request(
                        client,
                        semaphore,
                        profile,
                        endpoint_url,
                        workload,
                        request_index,
                    )
                    for request_index in range(run.request_count)
                ),
                return_exceptions=True,
            )

        elapsed_seconds = time.perf_counter() - started_at
        request_results: list[EvaluationRequestMetrics] = []
        for request_index, response in enumerate(responses):
            case = workload.cases[request_index % len(workload.cases)]
            if isinstance(response, Exception):
                request_results.append(
                    EvaluationRequestMetrics(
                        request_index=request_index,
                        case_id=case.id,
                        success=False,
                        error=str(response)[:2_000],
                    )
                )
            else:
                request_results.append(response)

        successful_results = [result for result in request_results if result.success]
        if not successful_results:
            raise BenchmarkExecutionError(
                f"All {run.request_count} measured benchmark requests failed."
            )

        ttft_samples = [result.ttft_ms for result in successful_results if result.ttft_ms is not None]
        end_to_end_samples = [
            result.end_to_end_latency_ms
            for result in successful_results
            if result.end_to_end_latency_ms is not None
        ]
        input_tokens = sum(result.input_tokens or 0 for result in successful_results)
        output_tokens = sum(result.completion_tokens or 0 for result in successful_results)
        quality_scores = [
            result.quality_score
            for result in successful_results
            if result.quality_score is not None
        ]
        failed_requests = run.request_count - len(successful_results)
        return EvaluationMetrics(
            request_count=run.request_count,
            successful_requests=len(successful_results),
            failed_requests=failed_requests,
            error_rate=failed_requests / run.request_count,
            benchmark_duration_seconds=elapsed_seconds,
            request_throughput_per_second=(
                len(successful_results) / elapsed_seconds if elapsed_seconds else 0.0
            ),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            scored_requests=len(quality_scores),
            passed_requests=sum(score == 1.0 for score in quality_scores),
            p50_ttft_ms=_percentile(ttft_samples, 50),
            p95_ttft_ms=_percentile(ttft_samples, 95),
            p99_ttft_ms=_percentile(ttft_samples, 99),
            p50_end_to_end_latency_ms=_percentile(end_to_end_samples, 50),
            p95_end_to_end_latency_ms=_percentile(end_to_end_samples, 95),
            p99_end_to_end_latency_ms=_percentile(end_to_end_samples, 99),
            output_tokens_per_second=output_tokens / elapsed_seconds if elapsed_seconds else 0.0,
            quality_score=(
                sum(quality_scores) / len(quality_scores) if quality_scores else None
            ),
            request_metrics=request_results,
        )

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        profile: ModelProfile,
        endpoint_url: str,
        workload: RegisteredWorkload,
        request_index: int,
    ) -> EvaluationRequestMetrics:
        case = workload.cases[request_index % len(workload.cases)]
        payload = {
            "model": profile.model,
            "messages": [
                message.model_dump()
                for message in (*workload.shared_messages, *case.messages)
            ],
            **workload.generation.model_dump(exclude_none=True),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with semaphore:
            request_started_at = time.perf_counter()
            first_token_at: float | None = None
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            generated_parts: list[str] = []
            saw_done = False
            async with client.stream("POST", endpoint_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    event = json.loads(data)
                    usage = event.get("usage")
                    if usage and usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage and usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    output = _event_output(event)
                    if output:
                        generated_parts.append(output)
                    if first_token_at is None and output:
                        first_token_at = time.perf_counter()
            request_finished_at = time.perf_counter()

        if first_token_at is None:
            raise BenchmarkExecutionError(
                f"Streaming request {request_index} returned no generated content."
            )
        if not saw_done:
            raise BenchmarkExecutionError(
                f"Streaming request {request_index} ended without a [DONE] event."
            )
        if completion_tokens is None or completion_tokens < 1:
            raise BenchmarkExecutionError(
                f"Streaming request {request_index} returned no completion-token usage."
            )

        ttft_ms = (first_token_at - request_started_at) * 1_000
        end_to_end_latency_ms = (request_finished_at - request_started_at) * 1_000
        generated_text = "".join(generated_parts)
        return EvaluationRequestMetrics(
            request_index=request_index,
            case_id=case.id,
            success=True,
            ttft_ms=ttft_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            input_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_tokens_per_second=(
                completion_tokens / (end_to_end_latency_ms / 1_000)
                if end_to_end_latency_ms
                else 0.0
            ),
            generated_text=generated_text,
            quality_score=score_answer(generated_text, case.expected),
        )


def _event_output(event: object) -> str:
    if not isinstance(event, dict):
        return ""
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    output_parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for field in ("content", "reasoning_content"):
            value = delta.get(field)
            if isinstance(value, str) and value:
                output_parts.append(value)
    return "".join(output_parts)


def _percentile(samples: Sequence[float], percentile: int) -> float:
    """Return a nearest-rank percentile for a non-empty collection."""

    if not samples:
        raise ValueError("A percentile requires at least one sample.")
    if not 1 <= percentile <= 100:
        raise ValueError("Percentile must be between 1 and 100.")
    ordered_samples = sorted(samples)
    return ordered_samples[math.ceil(len(ordered_samples) * percentile / 100) - 1]


def main() -> None:
    logging.basicConfig(level=os.getenv("MODELLAB_LOG_LEVEL", "INFO"))
    session_factory = get_session_factory()
    deployment_manager = DeploymentManager(session_factory, build_deployment_providers())
    worker = BenchmarkWorker(session_factory, deployment_manager)
    poll_interval_seconds = float(
        os.getenv("MODELLAB_WORKER_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)
    )
    asyncio.run(worker.run_forever(poll_interval_seconds))


if __name__ == "__main__":
    main()
