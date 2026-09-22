"""Load versioned workloads from repository-owned JSON definitions."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from modellab.workloads.models import RegisteredWorkload, WorkloadDefinition, WorkloadSummary


class WorkloadRegistry:
    def __init__(self, data_directory: Path | None = None) -> None:
        self._data_directory = data_directory or Path(__file__).with_name("data")
        self._workloads = self._load()

    def _load(self) -> dict[tuple[str, str], RegisteredWorkload]:
        workloads: dict[tuple[str, str], RegisteredWorkload] = {}
        for path in sorted(self._data_directory.glob("*.json")):
            definition = WorkloadDefinition.model_validate_json(path.read_text())
            key = (definition.name, definition.version)
            if key in workloads:
                raise ValueError(f"duplicate workload definition: {definition.name}@{definition.version}")
            canonical = json.dumps(
                definition.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            )
            workloads[key] = RegisteredWorkload(
                **definition.model_dump(),
                content_hash=sha256(canonical.encode("utf-8")).hexdigest(),
            )
        if not workloads:
            raise ValueError(f"no workload definitions found in {self._data_directory}")
        return workloads

    def list(self) -> list[WorkloadSummary]:
        return [
            WorkloadSummary(
                name=workload.name,
                version=workload.version,
                description=workload.description,
                case_count=len(workload.cases),
                content_hash=workload.content_hash,
            )
            for workload in self._workloads.values()
        ]

    def get(self, name: str, version: str) -> RegisteredWorkload | None:
        return self._workloads.get((name, version))


@lru_cache(maxsize=1)
def get_workload_registry() -> WorkloadRegistry:
    return WorkloadRegistry()
