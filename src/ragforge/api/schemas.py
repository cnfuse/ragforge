"""Request and response models for the HTTP API.

Keeping the wire contract in dedicated pydantic models (rather than reusing the
internal domain types directly) decouples the public API shape from internal
refactors and gives FastAPI a clean OpenAPI schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """A document submitted for ingestion."""

    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class IngestResponse(BaseModel):
    indexed_chunks: int
    total_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    where: dict[str, str] | None = Field(
        default=None, description="Optional metadata equality filter (all keys must match)."
    )


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    results: list[Hit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str | None = None
    citations: list[Citation]


class SaveResponse(BaseModel):
    saved: bool
    path: str | None = None
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    model: str
    live_llm: bool
    index_path: str | None = None
