# 2650-rag-project-33 — Samsung TV Manual RAG Chatbot

LLM chatbot that answers questions about **Samsung TV user manuals** using a Retrieval-Augmented Generation (RAG) pipeline adapted from **RMIT WALert**.

Primary product (Sprint decision): **Samsung Crystal UHD DU8000** — `UA75DU8000WXXY`  
Official manuals: [Samsung AU support](https://www.samsung.com/au/support/model/UA75DU8000WXXY/)

---

## 1. Project overview

Users ask questions such as “How do I connect Wi-Fi?” or “Why isn’t my remote working?”.  
The system retrieves relevant manual chunks (with model/page/section metadata), then generates a grounded answer with citations via **Ollama** (Sprint replacement for WALert’s Falcon-7B).

## 2. Architecture (WALert → this project)

```text
WALert FAQ passages + FAISS/BM25 + Falcon
                 ↓ adapted to
Samsung manuals + section chunks + FAISS/BM25 + Ollama
                 ↓
FastAPI /api/chat + Streamlit UI
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/USER_STORIES.md`](docs/USER_STORIES.md).

## 3. RAG pipeline

```text
Samsung Manual (manifest)
 → load/clean text
 → section-aware chunking (+ optional split)
 → embeddings (configurable)
 → FAISS (+ BM25)
 → model-aware retrieval (top-k, threshold)
 → grounded LLM prompt
 → answer + sources (manual / page / section)
```

## 4. Quick start — run the entire thing (Windows)

Do this once on a fresh machine, then use the daily commands below.

### One-time setup

```powershell
cd 2650-rag-project-33

# 1) Python env
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# 2) Ollama (local LLM) — install if missing
winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
# Then open a new terminal so `ollama` is on PATH

# 3) Pull the model used by this project
ollama pull llama3

# 4) Build the manual index (required before chat)
python scripts\ingest_manuals.py --hash-embeddings
```

Confirm Ollama is up:

```powershell
ollama list
# should show llama3
curl http://localhost:11434/api/tags
```

### Every time you want to run the chatbot

**Option A — Streamlit only (simplest demo)**

```powershell
cd 2650-rag-project-33
.\.venv\Scripts\activate
ollama list   # if this fails, start the Ollama app from the Start menu
streamlit run frontend\app.py
```

Open the browser URL Streamlit prints (usually http://localhost:8501).  
Keep **“Call local Python pipeline”** checked in the sidebar. Set TV model to `UA75DU8000WXXY` if empty.

**Option B — API + Streamlit**

Terminal 1:

```powershell
.\.venv\Scripts\activate
python -m api.chat
```

Terminal 2:

```powershell
.\.venv\Scripts\activate
streamlit run frontend\app.py
```

In the Streamlit sidebar: uncheck “Call local Python pipeline” if you want requests to go through `http://127.0.0.1:8000`.

### If something fails

| Symptom | Fix |
|---------|-----|
| Yellow “Ollama not reachable” | Start **Ollama** from the Start menu, wait a few seconds, retry. Run `ollama pull llama3` if the model is missing. |
| `No module named 'sentence_transformers'` | Not required for the default BM25 path. Ensure `.env` has `RETRIEVER_BACKEND=bm25` (this is the default in `.env.example`). |
| Missing collection / empty answers | Run `python scripts\ingest_manuals.py --hash-embeddings` again. |
| Wrong Python packages | Always activate `.venv` before running Streamlit/API. |

### Optional: tests & evaluation

```powershell
pytest -q
python eval\score.py --backend bm25
```

## 5. Environment variables

See [`.env.example`](.env.example):

| Variable | Purpose |
|----------|---------|
| `LLM_MODEL` | Ollama model name |
| `EMBEDDING_MODEL` | Sentence-Transformers model id |
| `VECTOR_DATABASE_PATH` | FAISS directory |
| `TOP_K` | Retrieved chunks |
| `RETRIEVAL_THRESHOLD` | Min similarity (dense) |
| `SAMSUNG_MANUAL_SOURCE` | Official support URL |
| `API_PORT` | FastAPI port |

Do not commit secrets (none required for Ollama/local).

## 6. Obtain / add manuals

1. Open the Samsung support page for the model.
2. Download the **User Manual** PDF from Samsung’s download centre.
3. Keep/clean text under `data/manuals/` (DU8000 text is already present).
4. Register the manual in `data/manifests/<MODEL>.json`.

## 7. Run ingestion

```bash
# Hash embeddings = fast offline index (good for CI/tests)
python scripts/ingest_manuals.py --hash-embeddings

# Or real sentence-transformer embeddings (downloads model once)
python scripts/ingest_manuals.py
```

## 8. Start the backend

```bash
python -m api.chat
# → http://127.0.0.1:8000/api/chat
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How do I connect Bluetooth headphones?\",\"model\":\"UA75DU8000WXXY\"}"
```

## 9. Start the frontend

```bash
streamlit run frontend/app.py
```

Features: chat UI, TV model field, sources/pages, loading state, local pipeline or API mode.

## 10. Run tests

```bash
pytest -q
```

## 11. Evaluate retrieval / answers

```bash
python scripts/ingest_manuals.py --hash-embeddings
python eval/score.py --backend bm25
```

Outputs under `eval/results/` (page-hit rate, refusal rate, answer_ok by question type).

## 12. Example questions

- How do I connect my Samsung TV to Wi-Fi?
- How do I turn on subtitles?
- How do I enable Game Mode on the DU8000?
- Why isn’t my remote working?
- Who is the CEO of Samsung? *(should refuse)*

Full set: `data/questions.csv`.

## 13. Known limitations

- Sprint focus is **one** primary model (DU8000); multi-model is supported via manifests/metadata but extra manuals must be added.
- PDF download is manual from Samsung (reproducible via manifest URLs); committed artifact is cleaned text.
- Generation quality depends on the local Ollama model; without Ollama the UI falls back to a grounded Echo stub.
- Dense threshold tuning is dataset-specific; BM25 is the reliable default before embeddings are built.
- Safety: never rely on the bot for electrical / wall-mount hazard decisions — see ethics notes in the sprint report.

## Sprint process

Work follows the WIL Project 33 Trello board (`sprint_timeline - wil-project-33.json`): role-based Sprint 1–3 cards with **Done when** acceptance criteria. Traceability lives in `docs/USER_STORIES.md`.
