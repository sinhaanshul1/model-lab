"""PostgreSQL-backed worker for ModelLab's first mock benchmark workload."""

from __future__ import annotations

import asyncio
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
    EvaluationRun,
    ModelDeploymentStatus,
    ModelProfile,
    ServingEngine,
)
from modellab.deployments import DeploymentManager, build_deployment_providers
from modellab.storage import repositories
from modellab.storage.database import get_session_factory


LOGGER = logging.getLogger(__name__)
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30.0
SMOKE_TEST_PROMPTS = (
    "Explain prefix caching in one sentence.",
    "Why should a database session be closed after a request?",
    "Write a short Python function that adds two integers.",
)


class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark cannot complete every scheduled request."""


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

            metrics = await self._benchmark_openai_compatible_profile(
                run, profile, deployment.endpoint_url
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
        self, run: EvaluationRun, profile: ModelProfile, endpoint_url: str
    ) -> EvaluationMetrics:
        semaphore = asyncio.Semaphore(min(run.concurrency, run.request_count))
        started_at = time.perf_counter()

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS, transport=self._transport
        ) as client:
            responses = await asyncio.gather(
                *(
                    self._send_request(
                        client, semaphore, profile, endpoint_url, request_index
                    )
                    for request_index in range(run.request_count)
                ),
                return_exceptions=True,
            )

        elapsed_seconds = time.perf_counter() - started_at
        failures = [response for response in responses if isinstance(response, Exception)]
        if failures:
            raise BenchmarkExecutionError(
                f"{len(failures)} of {run.request_count} benchmark requests failed."
            ) from failures[0]

        request_results = [response for response in responses if not isinstance(response, Exception)]
        latencies_ms = [result[0] for result in request_results]
        output_tokens = sum(result[1] for result in request_results)
        return EvaluationMetrics(
            request_count=run.request_count,
            successful_requests=len(request_results),
            # The non-streaming mock returns all tokens at once, so this is a
            # documented end-to-end response-latency proxy, not true TTFT.
            p95_ttft_ms=_p95(latencies_ms),
            output_tokens_per_second=output_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        )

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        profile: ModelProfile,
        endpoint_url: str,
        request_index: int,
    ) -> tuple[float, int]:
        prompt = SMOKE_TEST_PROMPTS[request_index % len(SMOKE_TEST_PROMPTS)]
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        async with semaphore:
            request_started_at = time.perf_counter()
            response = await client.post(endpoint_url, json=payload)
            elapsed_ms = (time.perf_counter() - request_started_at) * 1_000
        response.raise_for_status()
        completion_tokens = int(response.json()["usage"]["completion_tokens"])
        return elapsed_ms, completion_tokens


def _p95(samples: Sequence[float]) -> float:
    """Return nearest-rank P95 for a non-empty collection of latency samples."""

    if not samples:
        raise ValueError("P95 requires at least one latency sample.")
    ordered_samples = sorted(samples)
    return ordered_samples[math.ceil(len(ordered_samples) * 0.95) - 1]


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
