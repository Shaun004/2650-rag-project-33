"""FastAPI chat endpoint."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from rag.pipeline import RagPipeline

app = FastAPI(
    title="Samsung TV Manual RAG Chatbot",
    description="WALert-adapted RAG API for Samsung TV user manuals",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model: str | None = Field(
        default=None,
        description="TV model number, e.g. UA75DU8000WXXY or DU8000",
    )


class Source(BaseModel):
    manual: str
    model: str | None = None
    page: int | None = None
    section: str | None = None


class ChatResponseModel(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str | None = None
    retrieval_scores: list[float] = []
    refused: bool = False


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    settings = get_settings()
    # Prefer BM25 if dense index missing or sentence-transformers unavailable
    index_dir = settings.resolve(settings.vector_database_path)
    backend = settings.retriever_backend
    use_hash = False
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        use_hash = True
        backend = "bm25"
    if backend == "dense" and not (index_dir / "index.faiss").exists():
        backend = "bm25"
    return RagPipeline(
        settings=settings,
        backend=backend,  # type: ignore[arg-type]
        use_hash_embeddings=use_hash,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponseModel)
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        pipeline = get_pipeline()
        result = pipeline.ask(request.question, model=request.model)
        return result.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.chat:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
