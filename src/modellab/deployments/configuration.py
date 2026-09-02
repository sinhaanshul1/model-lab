"""Build deployment providers from trusted process configuration."""

from __future__ import annotations

import os

from modellab.deployments.providers import (
    DeploymentProvider,
    DockerDeploymentProvider,
    MockDeploymentProvider,
)


DEFAULT_MOCK_MODEL_URL = "http://127.0.0.1:8000/v1/mock-model/chat/completions"


def build_deployment_providers() -> dict[str, DeploymentProvider]:
    providers: dict[str, DeploymentProvider] = {
        "mock": MockDeploymentProvider(
            os.getenv("MODELLAB_MOCK_MODEL_URL", DEFAULT_MOCK_MODEL_URL)
        )
    }
    if os.getenv("MODELLAB_DOCKER_PROVIDER_ENABLED", "false").lower() != "true":
        return providers

    image = os.getenv("MODELLAB_VLLM_IMAGE")
    if not image:
        raise RuntimeError(
            "MODELLAB_VLLM_IMAGE must be a pinned image tag when the Docker provider is enabled."
        )

    import docker

    providers["docker"] = DockerDeploymentProvider(
        docker.from_env(),
        image=image,
        network=os.getenv("MODELLAB_DOCKER_NETWORK", "modellab-runtime"),
        gpu_device_request=docker.types.DeviceRequest(
            count=-1, capabilities=[["gpu"]]
        ),
        health_timeout_seconds=float(
            os.getenv("MODELLAB_DEPLOYMENT_HEALTH_TIMEOUT_SECONDS", "300")
        ),
    )
    return providers
