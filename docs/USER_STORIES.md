# User Stories — Samsung TV Manual RAG (2650-rag-project-33)

Traceability: Sprint → User Story → Acceptance Criteria → Implementation → Tests → Evidence.

## Sprint 1 — Foundation

### US1 — Clean, sectioned DU8000 manual
**As a** support user, **I want** answers grounded in the official DU8000 manual, **so that** steps match my TV.

**Acceptance**
- [x] Official source documented: https://www.samsung.com/au/support/model/UA75DU8000WXXY/
- [x] Clean text with page markers in `data/manuals/UA75DU8000WXXY.txt`
- [x] Section headings preserved for citation

### US2 — Free local LLM (Ollama)
**As a** developer, **I want** generation via Ollama, **so that** the team can run the bot without paid APIs.

**Acceptance**
- [x] `LLM_MODEL` / `OLLAMA_BASE_URL` configurable
- [x] Generator uses Ollama HTTP API (not Falcon)

### US3 — Starter evaluation set
**As an** evaluator, **I want** labelled questions with gold pages, **so that** we can score the bot like WALert.

**Acceptance**
- [x] `data/questions.csv` with ≥20 rows
- [x] Types: Known / Inferred / Out of KB

### US4 — Blank demo shell
**As a** presenter, **I want** a Streamlit page with a question box, **so that** we can demo early.

**Acceptance**
- [x] `frontend/app.py` opens in a browser

### US5 — Chunking research
**As a** retrieval owner, **I want** compared chunking strategies, **so that** we pick the best for manuals.

**Acceptance**
- [x] `docs/CHUNKING.md` compares section / fixed / sentence strategies

## Sprint 2 — Working RAG

### US6 — Wi-Fi (and similar) answers from the manual
**As a** user, **I want** correct procedural answers from retrieved chunks.

**Acceptance**
- [x] Ingestion → index → retrieve → generate pipeline
- [x] Bot can answer “How do I connect the TV to Wi-Fi?”

### US7 — Source citations
**As a** user, **I want** page citations, **so that** I can verify answers.

**Acceptance**
- [x] API/UI returns structured `sources` with manual, page, section

### US8 — Expanded question bank
**As an** evaluator, **I want** 50–80 questions including messy wording.

**Acceptance**
- [x] `data/questions.csv` expanded with tricky phrasings (≥10)

### US9 — Retrieval experiments
**As a** retrieval owner, **I want** BM25 vs dense and chunk-size results recorded.

**Acceptance**
- [x] Scripts + notes under `eval/` and `docs/CHUNKING.md`

### US10 — Connected demo
**As a** visitor, **I want** the Streamlit app to show real answers and pages.

**Acceptance**
- [x] Frontend calls the RAG pipeline / API

## Sprint 3 — Harden & report

### US11 — Refuse when not in manual
**As a** user, **I want** “I don’t know” when context is insufficient.

**Acceptance**
- [x] Retrieval threshold + grounded prompt
- [x] Out-of-KB questions refused

### US12 — Model-aware retrieval
**As a** user, **I want** to select my TV model, **so that** answers are not mixed across models.

**Acceptance**
- [x] Chat request accepts `model`
- [x] Metadata filter by `model_number`
- [x] If model missing: ask or answer with caveat

### US13 — Before/after evaluation
**As an** evaluator, **I want** baseline vs tuned scores.

**Acceptance**
- [x] `eval/score.py` produces per-type metrics

### US14 — Documentation
**As a** marker/stakeholder, **I want** README + architecture explaining WALert adaptation.

**Acceptance**
- [x] README covers setup, pipeline, limitations
- [x] `docs/ARCHITECTURE.md` maps WALert → Samsung RAG
