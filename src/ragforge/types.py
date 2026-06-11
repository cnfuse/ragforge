"""Core domain models shared across the pipeline.

These small, immutable-ish pydantic models are the contracts that flow between
ingestion, embedding, storage, retrieval, and the agent. Keeping them in one
place keeps the seams between components explicit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document prior to chunking."""

    id: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    id: str
    doc_id: str
    text: str
    ordinal: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk paired with its similarity score from retrieval."""

    chunk: Chunk
    score: float


class Answer(BaseModel):
    """The agent's response together with the evidence it relied on."""

    question: str
    text: str
    citations: list[ScoredChunk] = Field(default_factory=list)
    model: str | None = None
