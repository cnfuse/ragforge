"""FastAPI application factory for ragforge.

Endpoints:
    GET  /health   liveness + index size + model info
    POST /ingest   add documents to the in-memory index
    POST /query    retrieve relevant chunks
    POST /ask       answer a question with the agent (Claude or offline mock)

The app owns a single :class:`Pipeline` and a lazily-built agent in application
state, so the index persists across requests within a process. Build it with
:func:`create_app` (used by tests) or import :data:`app` for ``uvicorn``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from ragforge.agent import RagAgent
from ragforge.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    Hit,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SaveResponse,
)
from ragforge.config import Settings
from ragforge.config import settings as default_settings
from ragforge.llm import build_llm
from ragforge.logging import get_logger
from ragforge.pipeline import Pipeline
from ragforge.types import Document

log = get_logger("api")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct a FastAPI app with its own pipeline and agent."""
    cfg = settings or default_settings
    app = FastAPI(
        title="ragforge",
        version="0.1.0",
        description="Agentic RAG platform with a built-in evaluation harness, built on Claude.",
    )
    # Load a persisted index on startup when configured and present.
    if cfg.index_path and Path(cfg.index_path).exists():
        try:
            pipeline = Pipeline.from_index(cfg.index_path, cfg)
            log.info("loaded index from %s (%d chunks)", cfg.index_path, len(pipeline.store))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("could not load index %s: %s; starting empty", cfg.index_path, exc)
            pipeline = Pipeline(cfg)
    else:
        pipeline = Pipeline(cfg)
    agent = RagAgent(pipeline, build_llm(cfg))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            indexed_chunks=len(pipeline.store),
            model=cfg.model,
            live_llm=cfg.has_live_llm,
            index_path=cfg.index_path,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        docs = [Document(id=d.id, text=d.text, metadata=d.metadata) for d in req.documents]
        added = pipeline.ingest(docs)
        log.info("ingested %d documents -> %d chunks via API", len(docs), added)
        if cfg.index_path:  # auto-persist so the service survives restarts
            pipeline.save_index(cfg.index_path)
        return IngestResponse(indexed_chunks=added, total_chunks=len(pipeline.store))

    @app.post("/save", response_model=SaveResponse)
    def save() -> SaveResponse:
        if not cfg.index_path:
            raise HTTPException(status_code=400, detail="No RAGFORGE_INDEX_PATH configured.")
        pipeline.save_index(cfg.index_path)
        return SaveResponse(saved=True, path=cfg.index_path, total_chunks=len(pipeline.store))

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        hits = pipeline.retrieve(req.query, top_k=req.top_k)
        return QueryResponse(
            query=req.query,
            results=[
                Hit(
                    chunk_id=h.chunk.id,
                    doc_id=h.chunk.doc_id,
                    score=h.score,
                    text=h.chunk.text,
                )
                for h in hits
            ],
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        answer = agent.answer(req.question)
        cites = answer.citations[: req.top_k] if req.top_k else answer.citations
        return AskResponse(
            question=answer.question,
            answer=answer.text,
            model=answer.model,
            citations=[
                Citation(chunk_id=c.chunk.id, doc_id=c.chunk.doc_id, score=c.score)
                for c in cites
            ],
        )

    return app


# Module-level app for `uvicorn ragforge.api.app:app`.
app = create_app()
