"""Central configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    vector_database: str = "faiss"
    vector_database_path: str = "data/indexes/faiss"
    vector_database_url: str = "data/indexes/faiss"

    top_k: int = 3
    retrieval_threshold: float = 0.25
    retriever_backend: str = "bm25"  # bm25 | dense — bm25 works without sentence-transformers

    samsung_manual_source: str = (
        "https://www.samsung.com/au/support/model/UA75DU8000WXXY/"
    )
    default_tv_model: str = "UA75DU8000WXXY"
    manual_manifest_dir: str = "data/manifests"
    manual_text_dir: str = "data/manuals"
    collection_path: str = "data/collection.csv"
    metadata_path: str = "data/indexes/faiss/metadata.json"

    api_port: int = 8000
    api_host: str = "0.0.0.0"

    # Chunking
    chunk_strategy: str = "section"
    max_chunk_chars: int = 1200
    chunk_overlap: int = 150

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else ROOT / path


def get_settings() -> Settings:
    return Settings()
