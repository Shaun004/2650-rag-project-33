from fastapi.testclient import TestClient

from api.chat import app, get_pipeline
from rag.generator import EchoLLM
from rag.pipeline import RagPipeline
from config import Settings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_health():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"


def test_chat_endpoint_with_override(tmp_path, monkeypatch):
    from ingestion.build_collection import ingest_manifest

    collection = tmp_path / "collection.csv"
    ingest_manifest(ROOT / "data" / "manifests" / "UA75DU8000WXXY.json", collection)

    settings = Settings(
        collection_path=str(collection),
        retrieval_threshold=0.0,
        top_k=3,
        retriever_backend="bm25",
    )
    pipeline = RagPipeline(settings=settings, llm=EchoLLM(), backend="bm25", use_hash_embeddings=True)

    app.dependency_overrides = {}
    get_pipeline.cache_clear()
    monkeypatch.setattr("api.chat.get_pipeline", lambda: pipeline)

    client = TestClient(app)
    res = client.post(
        "/api/chat",
        json={"question": "How do I connect to Wi-Fi?", "model": "UA75DU8000WXXY"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "answer" in body
    assert "sources" in body
