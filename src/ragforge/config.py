"""Central configuration for ragforge.

Settings are resolved from environment variables (prefixed ``RAGFORGE_``) and an
optional ``.env`` file, then validated by pydantic. Importing :data:`settings`
gives every component one consistent, typed view of configuration.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the whole platform."""

    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # allow init by field name as well as the env alias
    )

    # --- Language model ---------------------------------------------------
    # Anthropic resolves the key from ANTHROPIC_API_KEY itself; we surface it
    # here only so callers can detect whether live calls are possible.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    model: str = "claude-opus-4-8"
    max_tokens: int = 2048
    effort: str = "high"  # low | medium | high | xhigh | max

    # --- Embeddings -------------------------------------------------------
    embedding_provider: str = "hashing"  # hashing | voyage
    embedding_dim: int = 512
    voyage_model: str = "voyage-3"

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = 800  # target characters per chunk
    chunk_overlap: int = 120

    # --- Retrieval --------------------------------------------------------
    top_k: int = 5
    # Hybrid retrieval: fuse dense (embedding) and sparse (BM25) rankings via RRF.
    hybrid_enabled: bool = False
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    # MMR diversification of the final selection (reduce near-duplicate passages).
    mmr_enabled: bool = False
    mmr_lambda: float = 0.5  # 1.0 = pure relevance, lower = more diverse

    # --- Reranking (optional second stage) --------------------------------
    rerank_enabled: bool = False
    rerank_provider: str = "lexical"  # lexical | llm
    rerank_alpha: float = 0.5  # weight on embedding score vs BM25 in the blend
    rerank_fetch_multiplier: int = 4  # first-pass candidates = top_k * this

    # --- Service ----------------------------------------------------------
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    # Optional on-disk index. If set, the API loads it on startup (when present)
    # and persists to it after each ingest, so the service survives restarts.
    index_path: str | None = None

    @property
    def has_live_llm(self) -> bool:
        """True when a live Anthropic call is possible (API key present)."""
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
