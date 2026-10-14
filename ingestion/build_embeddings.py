"""Build FAISS index from collection.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import get_settings
from rag.embeddings import get_embedder
from rag.vector_store import FaissMetadataStore, collection_to_metadata


def build_faiss_index(
    collection_path: Path | None = None,
    index_dir: Path | None = None,
    embedding_model: str | None = None,
    use_hash: bool = False,
) -> Path:
    settings = get_settings()
    collection_path = collection_path or settings.resolve(settings.collection_path)
    index_dir = index_dir or settings.resolve(settings.vector_database_path)
    embedding_model = embedding_model or settings.embedding_model

    df = pd.read_csv(collection_path)
    if df.empty:
        raise ValueError(f"Collection is empty: {collection_path}")

    texts = df["passage"].fillna("").astype(str).tolist()
    embedder = get_embedder(embedding_model, use_hash=use_hash)
    vectors = embedder.embed_documents(texts)
    metadata = collection_to_metadata(df)

    store = FaissMetadataStore(index_dir)
    store.build(vectors, metadata)
    return index_dir


if __name__ == "__main__":
    path = build_faiss_index()
    print(f"Wrote FAISS index to {path}")
