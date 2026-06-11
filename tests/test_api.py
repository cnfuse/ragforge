"""Tests for the FastAPI service using Starlette's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragforge.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _seed(client: TestClient) -> None:
    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "py", "text": "Python is a readable high-level programming language."},
                {"id": "rust", "text": "Rust is a systems language with memory safety guarantees."},
            ]
        },
    )


def test_health_starts_empty(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 0
    assert "model" in body


def test_ingest_then_health_reflects_count(client: TestClient) -> None:
    resp = client.post(
        "/ingest",
        json={"documents": [{"id": "d1", "text": "hello world from ragforge"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["indexed_chunks"] >= 1
    assert client.get("/health").json()["indexed_chunks"] >= 1


def test_ingest_rejects_empty_document_list(client: TestClient) -> None:
    resp = client.post("/ingest", json={"documents": []})
    assert resp.status_code == 422  # min_length=1 validation


def test_query_returns_ranked_hits(client: TestClient) -> None:
    _seed(client)
    resp = client.post("/query", json={"query": "memory safe systems language", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results
    assert results[0]["doc_id"] == "rust"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_validates_blank_query(client: TestClient) -> None:
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_ask_returns_answer_with_citations(client: TestClient) -> None:
    _seed(client)
    resp = client.post("/ask", json={"question": "which language is memory safe?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["citations"]
    assert body["citations"][0]["doc_id"] == "rust"


def test_openapi_schema_is_served(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "ragforge"
