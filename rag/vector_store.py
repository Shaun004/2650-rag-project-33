"""FAISS vector store with JSON metadata sidecar (WALert-inspired, metadata-aware)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd


class FaissMetadataStore:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def build(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata length must match vectors")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine if vectors are normalised
        index.add(np.asarray(vectors, dtype=np.float32))
        self.index = index
        self.metadata = metadata
        self.save()

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found under {self.index_dir}. Run scripts/ingest_manuals.py first."
            )
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        model_number: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            self.load()
        assert self.index is not None

        candidate_k = min(len(self.metadata), max(top_k * 10, top_k))
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, candidate_k)

        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self.metadata[int(idx)])
            if model_number and str(meta.get("model_number", "")).upper() not in {
                model_number.upper(),
                model_number.upper().replace("UA", "").replace("WXXY", ""),
            }:
                # Allow family short codes like DU8000 / QN90D against full model strings
                model_field = str(meta.get("model_number", "")).upper()
                family = str(meta.get("model_family", "")).upper()
                needle = model_number.upper()
                if needle not in model_field and needle not in family:
                    continue
            meta["score"] = float(score)
            hits.append(meta)
            if len(hits) >= top_k:
                break
        return hits


def collection_to_metadata(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    # Replace NaN with None for JSON
    cleaned: list[dict[str, Any]] = []
    for row in records:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                item[key] = None
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned
