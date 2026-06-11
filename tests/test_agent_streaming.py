"""Tests for the agent's event stream and the SSE /ask/stream endpoint."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ragforge.agent.rag_agent import RagAgent
from ragforge.api.app import create_app
from ragforge.config import Settings
from ragforge.llm.mock import MockLLM
from ragforge.pipeline import Pipeline
from ragforge.types import Document

CORPUS = [
    Document(id="py", text="Python is a readable high-level programming language."),
    Document(id="rust", text="Rust is a systems language with memory safety guarantees."),
]


def _agent() -> RagAgent:
    pipeline = Pipeline()
    pipeline.ingest(CORPUS)
    return RagAgent(pipeline, MockLLM())


def test_iter_events_emits_search_results_then_answer() -> None:
    events = list(_agent().iter_events("which language is memory safe?"))
    types = [e.type for e in events]
    assert "search" in types
    assert "results" in types
    assert types[-1] == "answer"
    # Terminal event carries the full answer with citations.
    final = events[-1]
    assert final.answer is not None
    assert final.answer.citations
    assert final.answer.citations[0].chunk.doc_id == "rust"


def test_iter_events_results_carry_hit_count() -> None:
    events = list(_agent().iter_events("memory safety"))
    results = [e for e in events if e.type == "results"]
    assert results
    assert results[0].data["count"] >= 1


def test_answer_matches_terminal_event() -> None:
    agent = _agent()
    streamed = list(agent.iter_events("readable language"))[-1].answer
    direct = agent.answer("readable language")
    assert streamed is not None
    assert direct.text == streamed.text


def test_sse_endpoint_streams_events() -> None:
    client = TestClient(create_app(Settings()))
    client.post(
        "/ingest",
        json={"documents": [{"id": d.id, "text": d.text} for d in CORPUS]},
    )
    resp = client.post("/ask/stream", json={"question": "which language is memory safe?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Parse the SSE frames into AgentEvent payloads.
    frames = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames
    assert frames[-1]["type"] in {"answer", "budget_exhausted"}
    assert frames[-1]["answer"]["text"]
