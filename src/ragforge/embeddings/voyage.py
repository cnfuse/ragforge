"""Optional Voyage AI embedding provider for production-quality semantics.

Selected by setting ``RAGFORGE_EMBEDDING_PROVIDER=voyage`` (requires the
``voyage`` extra and ``VOYAGE_API_KEY``). Vectors are L2-normalised so the rest
of the pipeline can treat dot products as cosine similarity, exactly as with the
local hashing embedder.
"""

from __future__ import annotations

import numpy as np


class VoyageEmbedder:
    """Adapter from the Voyage embeddings API to the :class:`Embedder` protocol."""

    def __init__(self, model: str = "voyage-3") -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The 'voyageai' package is required for the Voyage embedder; "
                "install it with `pip install 'ragforge[voyage]'`."
            ) from exc
        self._client = voyageai.Client()
        self.model = model
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe once to learn the model's output dimensionality.
            self._dim = int(self.embed_one("dimension probe").shape[0])
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim if self._dim else 0), dtype=np.float32)
        result = self._client.embed(texts, model=self.model, input_type="document")
        arr = np.asarray(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = arr / norms
        self._dim = normalised.shape[1]
        return normalised

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
