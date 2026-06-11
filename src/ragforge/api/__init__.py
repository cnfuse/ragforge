"""HTTP service exposing the RAG pipeline over a small FastAPI app."""

from __future__ import annotations

from ragforge.api.app import create_app

__all__ = ["create_app"]
