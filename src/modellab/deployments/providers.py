"""Runtime-provider contracts for model-server deployments."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from modellab.api.models import ModelDeployment, ModelProfile


@dataclass(frozen=True)
class DeploymentHandle:
    runtime_id: str
    endpoint_url: str


class DeploymentProvider(Protocol):
    async def start(
        self, profile: ModelProfile, deployment: ModelDeployment
    ) -> DeploymentHandle: ...

    async def health(self, handle: DeploymentHandle) -> bool: ...

    async def stop(self, handle: DeploymentHandle) -> None: ...


class MockDeploymentProvider:
    """Expose ModelLab's already-running deterministic model as a deployment."""

    def __init__(self, endpoint_url: str) -> None:
        self._endpoint_url = endpoint_url

    async def start(
        self, profile: ModelProfile, deployment: ModelDeployment
    ) -> DeploymentHandle:
        return DeploymentHandle(
            runtime_id=f"mock:{deployment.id}", endpoint_url=self._endpoint_url
        )

    async def health(self, handle: DeploymentHandle) -> bool:
        return True

    async def stop(self, handle: DeploymentHandle) -> None:
        return None


class DockerDeploymentProvider:
    """Start isolated vLLM servers through a trusted local Docker daemon."""

    def __init__(
        self,
        docker_client: Any,
        *,
        image: str,
        network: str,
        gpu_device_request: Any,
        health_timeout_seconds: float = 300.0,
        health_interval_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not image or image.endswith(":latest"):
            raise ValueError("The Docker provider requires a pinned vLLM image tag.")
        self._client = docker_client
        self._image = image
        self._network = network
        self._gpu_device_request = gpu_device_request
        self._health_timeout_seconds = health_timeout_seconds
        self._health_interval_seconds = health_interval_seconds
        self._transport = transport

    async def start(
        self, profile: ModelProfile, deployment: ModelDeployment
    ) -> DeploymentHandle:
        if profile.engine.value != "vllm":
            raise ValueError("The Docker provider currently supports only the vLLM engine.")

        container_name = f"modellab-{deployment.id.hex}"
        container = await asyncio.to_thread(
            self._client.containers.run,
            self._image,
            command=self._build_command(profile),
            name=container_name,
            detach=True,
            network=self._network,
            ipc_mode="host",
            device_requests=[self._gpu_device_request],
            labels={
                "modellab.managed": "true",
                "modellab.deployment_id": str(deployment.id),
                "modellab.profile_id": str(profile.id),
            },
        )
        return DeploymentHandle(
            runtime_id=container.id,
            endpoint_url=f"http://{container_name}:8000/v1/chat/completions",
        )

    async def health(self, handle: DeploymentHandle) -> bool:
        health_url = handle.endpoint_url.removesuffix("/v1/chat/completions") + "/health"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._health_timeout_seconds
        async with httpx.AsyncClient(timeout=5.0, transport=self._transport) as client:
            while True:
                try:
                    response = await client.get(health_url)
                    if response.is_success:
                        return True
                except httpx.HTTPError:
                    pass
                if loop.time() >= deadline:
                    return False
                await asyncio.sleep(self._health_interval_seconds)

    async def stop(self, handle: DeploymentHandle) -> None:
        container = await asyncio.to_thread(
            self._client.containers.get, handle.runtime_id
        )
        await asyncio.to_thread(container.stop, timeout=30)
        await asyncio.to_thread(container.remove)

    @staticmethod
    def _build_command(profile: ModelProfile) -> list[str]:
        command = [
            "--model",
            profile.model,
            "--served-model-name",
            profile.model,
            "--max-model-len",
            str(profile.max_context_tokens),
            "--max-num-seqs",
            str(profile.max_concurrent_sequences),
        ]
        if profile.quantization and profile.quantization.lower() != "none":
            command.extend(["--quantization", profile.quantization])
        if profile.prefix_caching:
            command.append("--enable-prefix-caching")
        return command
