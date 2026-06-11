"""Tests for on-disk index persistence in the HTTP API."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ragforge.api.app import create_app
from ragforge.config import Settings


def test_ingest_autosaves_and_new_app_loads_it(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    cfg = Settings(index_path=str(index))

    # First app instance: ingest, which should auto-persist to disk.
    app1 = create_app(cfg)
    client1 = TestClient(app1)
    resp = client1.post(
        "/ingest",
        json={"documents": [{"id": "d1", "text": "Rust guarantees memory safety."}]},
    )
    assert resp.status_code == 200
    assert index.exists()

    # A fresh app instance must load the persisted index on startup.
    app2 = create_app(cfg)
    client2 = TestClient(app2)
    health = client2.get("/health").json()
    assert health["indexed_chunks"] >= 1
    assert health["index_path"] == str(index)

    # And it can answer queries against the loaded index.
    results = client2.post("/query", json={"query": "memory safety"}).json()["results"]
    assert results and results[0]["doc_id"] == "d1"


def test_save_endpoint_without_path_is_400() -> None:
    client = TestClient(create_app(Settings(index_path=None)))
    assert client.post("/save").status_code == 400


def test_save_endpoint_writes_file(tmp_path: Path) -> None:
    index = tmp_path / "saved.json"
    client = TestClient(create_app(Settings(index_path=str(index))))
    client.post("/ingest", json={"documents": [{"id": "d", "text": "hello world"}]})
    index.unlink(missing_ok=True)  # remove the auto-saved copy
    resp = client.post("/save")
    assert resp.status_code == 200
    assert resp.json()["saved"] is True
    assert index.exists()
