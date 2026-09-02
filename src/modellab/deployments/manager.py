"""Coordinate durable deployment state with runtime providers."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from modellab.api.models import ModelDeployment
from modellab.deployments.providers import DeploymentHandle, DeploymentProvider
from modellab.storage import repositories


class DeploymentLifecycleError(RuntimeError):
    pass


class DeploymentManager:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        providers: dict[str, DeploymentProvider],
    ) -> None:
        self._session_factory = session_factory
        self._providers = providers

    async def start(self, deployment_id: UUID) -> ModelDeployment:
        with self._session_factory() as session:
            deployment = repositories.mark_model_deployment_starting(session, deployment_id)
            if deployment is None:
                raise DeploymentLifecycleError("Deployment is not pending.")
            profile = repositories.get_model_profile(session, deployment.model_profile_id)
        if profile is None:
            return self._fail(deployment_id, "Model profile not found.")

        provider = self._providers.get(deployment.provider)
        if provider is None:
            return self._fail(deployment_id, f"Unsupported provider: {deployment.provider}")

        try:
            handle = await provider.start(profile, deployment)
            if not await provider.health(handle):
                raise DeploymentLifecycleError("Deployment failed its health check.")
            with self._session_factory() as session:
                ready = repositories.mark_model_deployment_ready(
                    session,
                    deployment_id,
                    runtime_id=handle.runtime_id,
                    endpoint_url=handle.endpoint_url,
                )
            if ready is None:
                raise DeploymentLifecycleError("Deployment could not be marked ready.")
            return ready
        except Exception as exc:
            return self._fail(deployment_id, str(exc))

    async def stop(self, deployment_id: UUID) -> ModelDeployment:
        with self._session_factory() as session:
            deployment = repositories.mark_model_deployment_stopping(session, deployment_id)
        if deployment is None:
            raise DeploymentLifecycleError("Deployment is not ready.")

        provider = self._providers.get(deployment.provider)
        if provider is None:
            return self._fail(deployment_id, f"Unsupported provider: {deployment.provider}")
        if deployment.runtime_id is None or deployment.endpoint_url is None:
            return self._fail(deployment_id, "Deployment has no runtime handle.")

        try:
            await provider.stop(
                DeploymentHandle(
                    runtime_id=deployment.runtime_id,
                    endpoint_url=deployment.endpoint_url,
                )
            )
            with self._session_factory() as session:
                stopped = repositories.mark_model_deployment_stopped(session, deployment_id)
            if stopped is None:
                raise DeploymentLifecycleError("Deployment could not be marked stopped.")
            return stopped
        except Exception as exc:
            return self._fail(deployment_id, str(exc))

    def _fail(self, deployment_id: UUID, reason: str) -> ModelDeployment:
        with self._session_factory() as session:
            failed = repositories.mark_model_deployment_failed(session, deployment_id, reason)
        if failed is None:
            raise DeploymentLifecycleError(reason)
        return failed
