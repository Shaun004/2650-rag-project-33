"""Configurable embedding providers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic bag-of-hashes embedder for offline unit tests (no model download)."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._one(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32)


def get_embedder(model_name: str, *, use_hash: bool = False) -> Embedder:
    if use_hash or model_name.lower() in {"hash", "test", "fake"}:
        return HashEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except ImportError:
        # Demo/CI machines may not have sentence-transformers installed.
        return HashEmbedder()
