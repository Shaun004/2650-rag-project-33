# Architecture — WALert → Samsung TV Manual RAG

## WALert (source)

WALert answers RMIT SCT FAQ questions using:

1. **Intent track** (Alexa) — omitted here (not needed for a manual chatbot).
2. **RAG track** — passages in `collection.csv` → BM25 or FAISS dense retrieval → Falcon-7B prompt (“answer from documents else NA”) → offline eval with topics/qrels/NDCG + ROUGE/BERTScore.

Key reusable ideas: passage collection, top-k retrieval, grounded “else NA” prompting, Known/Inferred/Out-of-KB evaluation, BM25 vs dense comparison.

## Samsung TV RAG (this repo)

```text
Samsung Support PDF/text + manifest
        ↓
parse / clean / section-aware chunk
        ↓
embeddings (configurable) + FAISS (+ BM25)
        ↓
model-aware retriever (top-k, threshold, metadata filter)
        ↓
Ollama LLM (grounded prompt + citations)
        ↓
FastAPI `/api/chat` + Streamlit UI
```

## Why these choices

| Choice | Rationale |
|--------|-----------|
| FAISS + metadata JSON | Matches WALert dense path; simple local deploy; metadata filter without new infra |
| Ollama instead of Falcon | Sprint requirement; free/local for student machines |
| Streamlit + FastAPI | Sprint demo + testable API; UI stays thin |
| Manifest per manual | Multi-model ready; Sprint 1 uses UA75DU8000WXXY only |
| WALert-like `collection.csv` / `questions.csv` | Keeps evaluation comparable to WALert |

## Module map

| Package | Responsibility |
|---------|----------------|
| `ingestion/` | Load manuals, clean, chunk, build collection + indexes |
| `rag/` | Embeddings, BM25/dense retrieve, prompts, generate |
| `api/` | FastAPI chat endpoint |
| `frontend/` | Streamlit chatbot |
| `eval/` | Score questions.csv (WALert-adapted metrics) |
| `tests/` | Unit + integration |

## Configuration

All sensitive/tunable values come from environment variables (see `.env.example`). No API keys are committed.
