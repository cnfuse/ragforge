"""A local, deterministic hashing embedder (feature hashing over token n-grams).

This is the "batteries included" default: no API key, no network, identical
output on every machine. It uses the hashing trick to project token unigrams and
bigrams into a fixed-dimensional space, with sublinear term weighting. It will
never match a learned semantic model, but it is more than enough to exercise the
full retrieval pipeline and to keep tests hermetic and fast.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _features(text: str) -> list[str]:
    toks = _tokens(text)
    grams = list(toks)
    grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
    return grams


class HashingEmbedder:
    """Feature-hashing embedder producing L2-normalised vectors."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _hash(self, feature: str) -> tuple[int, float]:
        """Return a (bucket, sign) pair from a stable hash of ``feature``."""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % self._dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for feat in _features(text):
                counts[feat] = counts.get(feat, 0) + 1
            for feat, count in counts.items():
                bucket, sign = self._hash(feat)
                out[row, bucket] += sign * (1.0 + math.log(count))
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out

    def embed_one(self, text: str) -> np.ndarray:
        vector: np.ndarray = self.embed([text])[0]
        return vector
