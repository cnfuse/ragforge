"""Tests for metadata filtering on retrieval (pipeline + API)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ragforge.api.app import create_app
from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.types import Document

CORPUS = [
    Document(
        id="a", text="Rust guarantees memory safety.", metadata={"lang": "rust", "kind": "doc"}
    ),
    Document(
        id="b", text="Python is readable and high level.", metadata={"lang": "py", "kind": "doc"}
    ),
    Document(
        id="c", text="Rust uses the cargo build tool.", metadata={"lang": "rust", "kind": "note"}
    ),
]


def _pipeline() -> Pipeline:
    p = Pipeline(Settings())
    p.ingest(CORPUS)
    return p


def test_filter_restricts_to_matching_metadata() -> None:
    results = _pipeline().retrieve("language", top_k=5, where={"lang": "rust"})
    assert results
    assert {r.chunk.doc_id for r in results} == {"a", "c"}


def test_filter_requires_all_keys_to_match() -> None:
    results = _pipeline().retrieve("rust", top_k=5, where={"lang": "rust", "kind": "note"})
    assert {r.chunk.doc_id for r in results} == {"c"}


def test_filter_with_no_matches_returns_empty() -> None:
    assert _pipeline().retrieve("anything", top_k=5, where={"lang": "go"}) == []


def test_no_filter_returns_across_sources() -> None:
    results = _pipeline().retrieve("language", top_k=5)
    assert {r.chunk.doc_id for r in results} == {"a", "b", "c"}


def test_filter_composes_with_hybrid_and_mmr() -> None:
    p = Pipeline(Settings(hybrid_enabled=True, mmr_enabled=True))
    p.ingest(CORPUS)
    results = p.retrieve("cargo build tool", top_k=5, where={"lang": "rust"})
    assert {r.chunk.doc_id for r in results} <= {"a", "c"}


def test_api_query_accepts_where_filter() -> None:
    client = TestClient(create_app(Settings()))
    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": d.id, "text": d.text, "metadata": d.metadata} for d in CORPUS
            ]
        },
    )
    resp = client.post("/query", json={"query": "language", "where": {"lang": "rust"}})
    assert resp.status_code == 200
    doc_ids = {r["doc_id"] for r in resp.json()["results"]}
    assert doc_ids == {"a", "c"}
