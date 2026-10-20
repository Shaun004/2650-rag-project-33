# Chunking Strategy Comparison (Sprint 1–2)

Manual structure (DU8000 e-Manual text) already uses markdown headings:

- `#` / `##` chapters (e.g. Connections, Troubleshooting)
- `###` subsections with page markers like `(p. 9-12)`

## Strategies considered

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **A. Section-aware (chosen default)** | One chunk per `###` subsection; keep heading + body | Keeps procedures together; natural citations (section + page) | Long sections may exceed embedding sweet spot |
| **B. Fixed-size** | e.g. 500 tokens, 50 overlap | Simple; predictable size | Splits numbered steps mid-procedure |
| **C. Sentence window** | WALert-style sentence packs | Good for FAQ-like text | Manuals are procedural; steps fragment |

## Decision

**Default = section-aware**, with optional **split of oversized sections** (max ~1200 chars, 150 overlap) so embeddings stay effective while preserving section metadata on every sub-chunk.

## Experiment protocol (Sprint 2)

For each strategy, measure on `data/questions.csv` (Known questions):

1. Page hit@1 / hit@3 (retrieved page matches gold page)
2. Section hit@3
3. End-to-end correctness after generation (optional)

Record results under `eval/results/`.

## Reasoning summary

Samsung manuals are organised by task/section. Keeping “Connecting to the network” as one semantic unit matches how users ask questions and how WALert kept FAQ answer passages intact — adapted from passage-level KB design to section-level manual chunks.
