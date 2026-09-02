"""Runtime-provider contracts for model-server deployments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
