from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modellab.api.main import app
from modellab.workloads.registry import WorkloadRegistry


def test_built_in_workloads_are_versioned_and_hashed() -> None:
    workloads = WorkloadRegistry().list()

    assert {workload.name for workload in workloads} == {
        "smoke-test",
        "short-chat",
        "shared-prefix",
        "long-context",
        "quality",
    }
    assert all(workload.version == "1.0.0" for workload in workloads)
    assert all(len(workload.content_hash) == 64 for workload in workloads)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "invalid.json").write_text(
        '{"name":"bad","version":"1.0.0","description":"invalid duplicate IDs",'
        '"cases":[{"id":"same","messages":[{"role":"user","content":"one"}]},'
        '{"id":"same","messages":[{"role":"user","content":"two"}]}]}'
    )

    with pytest.raises(ValueError, match="case IDs must be unique"):
        WorkloadRegistry(tmp_path)


def test_workload_endpoints() -> None:
    client = TestClient(app)

    listing = client.get("/v1/workloads")
    assert listing.status_code == 200
    assert any(item["name"] == "quality" for item in listing.json())

    detail = client.get("/v1/workloads/quality/1.0.0")
    assert detail.status_code == 200
    assert detail.json()["cases"][0]["id"] == "arithmetic"

    assert client.get("/v1/workloads/missing/1.0.0").status_code == 404
