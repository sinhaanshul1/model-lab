"""Unit coverage for translating profiles into managed vLLM containers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest

from modellab.api.models import (
    DeploymentLifecyclePolicy,
    ModelDeployment,
    ModelDeploymentStatus,
    ModelProfile,
    ServingEngine,
)
from modellab.deployments import DockerDeploymentProvider


class FakeContainer:
    id = "container-123"

    def __init__(self) -> None:
        self.stop_calls: list[int] = []
        self.remove_calls = 0

    def stop(self, *, timeout: int) -> None:
        self.stop_calls.append(timeout)

    def remove(self) -> None:
        self.remove_calls += 1


class FakeContainers:
    def __init__(self) -> None:
        self.container = FakeContainer()
        self.run_args: tuple[object, ...] | None = None
        self.run_kwargs: dict[str, object] | None = None
        self.requested_container_id: str | None = None

    def run(self, *args: object, **kwargs: object) -> FakeContainer:
        self.run_args = args
        self.run_kwargs = kwargs
        return self.container

    def get(self, container_id: str) -> FakeContainer:
        self.requested_container_id = container_id
        return self.container


class FakeDockerClient:
    def __init__(self) -> None:
        self.containers = FakeContainers()


def _profile(*, prefix_caching: bool = True) -> ModelProfile:
    return ModelProfile(
        id=uuid4(),
        name="docker-provider-profile",
        model="org/model-awq",
        engine=ServingEngine.VLLM,
        quantization="awq",
        prefix_caching=prefix_caching,
        max_concurrent_sequences=16,
        max_context_tokens=8192,
        created_at=datetime.now(timezone.utc),
    )


def _deployment(profile: ModelProfile) -> ModelDeployment:
    return ModelDeployment(
        id=uuid4(),
        model_profile_id=profile.id,
        provider="docker",
        lifecycle_policy=DeploymentLifecyclePolicy.EPHEMERAL,
        status=ModelDeploymentStatus.STARTING,
        created_at=datetime.now(timezone.utc),
    )


def _healthy_response(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/health"
    return httpx.Response(200, request=request)


def test_docker_provider_starts_checks_and_removes_vllm_container() -> None:
    client = FakeDockerClient()
    profile = _profile()
    deployment = _deployment(profile)
    gpu_request = object()
    provider = DockerDeploymentProvider(
        client,
        image="vllm/vllm-openai:v0.21.0",
        network="modellab-runtime",
        gpu_device_request=gpu_request,
        model_cache_volume="modellab-model-cache",
        health_timeout_seconds=0,
        transport=httpx.MockTransport(_healthy_response),
    )

    handle = asyncio.run(provider.start(profile, deployment))

    assert handle.runtime_id == "container-123"
    assert handle.endpoint_url == (
        f"http://modellab-{deployment.id.hex}:8000/v1/chat/completions"
    )
    assert client.containers.run_args == ("vllm/vllm-openai:v0.21.0",)
    assert client.containers.run_kwargs is not None
    assert client.containers.run_kwargs["network"] == "modellab-runtime"
    assert client.containers.run_kwargs["device_requests"] == [gpu_request]
    assert client.containers.run_kwargs["volumes"] == {
        "modellab-model-cache": {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
        }
    }
    assert client.containers.run_kwargs["command"] == [
        "org/model-awq",
        "--served-model-name",
        "org/model-awq",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "16",
        "--quantization",
        "awq",
        "--enable-prefix-caching",
    ]
    assert asyncio.run(provider.health(handle))

    asyncio.run(provider.stop(handle))
    assert client.containers.requested_container_id == "container-123"
    assert client.containers.container.stop_calls == [30]
    assert client.containers.container.remove_calls == 1


def test_docker_provider_explicitly_disables_prefix_caching() -> None:
    client = FakeDockerClient()
    profile = _profile(prefix_caching=False)
    provider = DockerDeploymentProvider(
        client,
        image="vllm/vllm-openai:v0.21.0",
        network="modellab-runtime",
        gpu_device_request=object(),
        model_cache_volume="modellab-model-cache",
        health_timeout_seconds=0,
        transport=httpx.MockTransport(_healthy_response),
    )

    asyncio.run(provider.start(profile, _deployment(profile)))

    assert client.containers.run_kwargs is not None
    command = client.containers.run_kwargs["command"]
    assert "--no-enable-prefix-caching" in command
    assert "--enable-prefix-caching" not in command


def test_docker_provider_rejects_unpinned_images() -> None:
    with pytest.raises(ValueError, match="pinned"):
        DockerDeploymentProvider(
            FakeDockerClient(),
            image="vllm/vllm-openai:latest",
            network="modellab-runtime",
            gpu_device_request=object(),
            model_cache_volume="modellab-model-cache",
        )
