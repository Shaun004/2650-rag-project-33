"""End-to-end RAG orchestration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from config import Settings, get_settings
from rag.embeddings import Embedder, get_embedder
from rag.generator import EchoLLM, LLMClient, OllamaClient
from rag.prompts import build_prompt, insufficient_context_message
from rag.retriever import BM25Retriever, DenseRetriever, RetrievedChunk
from rag.vector_store import FaissMetadataStore

_OUT_OF_SCOPE_CUES = (
    "ceo",
    "weather",
    "stock price",
    "stock price of",
    "cook pasta",
    "joke",
    "who is the president",
    "football",
    "bitcoin",
    "overclock",
    "unknown apk",
    "apk sites",
)


def _looks_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(cue in q for cue in _OUT_OF_SCOPE_CUES)


@dataclass
class ChatResponse:
    answer: str
    sources: list[dict[str, Any]]
    model_used: str | None
    retrieval_scores: list[float]
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        backend: Literal["dense", "bm25"] | None = None,
        use_hash_embeddings: bool = False,
    ):
        self.settings = settings or get_settings()
        self.backend = backend or (
            "bm25" if self.settings.retriever_backend == "bm25" else "dense"
        )
        self._embedder = embedder
        self._use_hash_embeddings = use_hash_embeddings
        self.llm = llm or OllamaClient(
            self.settings.ollama_base_url,
            self.settings.llm_model,
        )
        self._dense: DenseRetriever | None = None
        self._bm25: BM25Retriever | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            # Only construct when needed (dense path); BM25 never requires this.
            self._embedder = get_embedder(
                self.settings.embedding_model,
                use_hash=self._use_hash_embeddings,
            )
        return self._embedder

    def _get_dense(self) -> DenseRetriever:
        if self._dense is None:
            store = FaissMetadataStore(self.settings.resolve(self.settings.vector_database_path))
            store.load()
            self._dense = DenseRetriever(store, self.embedder)
        return self._dense

    def _get_bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.settings.resolve(self.settings.collection_path))
        return self._bm25

    def retrieve(
        self,
        question: str,
        model: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.settings.top_k
        thr = self.settings.retrieval_threshold if threshold is None else threshold
        if self.backend == "bm25":
            return self._get_bm25().retrieve(question, top_k=k, model=model, threshold=thr)
        return self._get_dense().retrieve(question, top_k=k, model=model, threshold=thr)

    def ask(self, question: str, model: str | None = None) -> ChatResponse:
        question = (question or "").strip()
        if not question:
            return ChatResponse(
                answer="Please ask a question about your Samsung TV.",
                sources=[],
                model_used=model,
                retrieval_scores=[],
                refused=True,
            )

        chunks = self.retrieve(question, model=model)
        # Extra Out-of-KB guard: refuse clearly non-manual topics even if BM25 finds weak brand hits
        if chunks and _looks_out_of_scope(question):
            chunks = []
        if not chunks:
            return ChatResponse(
                answer=insufficient_context_message(model),
                sources=[],
                model_used=model,
                retrieval_scores=[],
                refused=True,
            )

        prompt = build_prompt(question, chunks, model=model)
        try:
            answer = self.llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - surface friendly API/UI error
            answer = (
                "The language model is unavailable "
                f"({exc}). Retrieved sources are still shown below."
            )

        sources = []
        seen: set[tuple[Any, ...]] = set()
        for chunk in chunks:
            key = (chunk.manual, chunk.page, chunk.section)
            if key in seen:
                continue
            seen.add(key)
            sources.append(chunk.to_source())

        return ChatResponse(
            answer=answer,
            sources=sources,
            model_used=model,
            retrieval_scores=[c.score for c in chunks],
            refused=False,
        )


def build_test_pipeline(collection_csv: Path, index_dir: Path) -> RagPipeline:
    """Helper for tests: BM25 + Echo LLM, no network."""
    settings = Settings(
        collection_path=str(collection_csv),
        vector_database_path=str(index_dir),
        retriever_backend="bm25",
        retrieval_threshold=0.0,
        top_k=3,
    )
    return RagPipeline(
        settings=settings,
        llm=EchoLLM(),
        backend="bm25",
        use_hash_embeddings=True,
    )
