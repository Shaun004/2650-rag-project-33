# Baseline evaluation notes (Sprint 2)

Command:

```bash
python scripts/ingest_manuals.py --hash-embeddings
python eval/score.py --backend bm25
```

## Results (EchoLLM stub — retrieval-focused baseline)

| Type | n | page_hit_rate | refusal_rate | answer_ok_rate |
|------|---|---------------|--------------|----------------|
| Known | 38 | 0.763 | 0.00 | 1.00 |
| Inferred | 10 | 0.600 | 0.00 | 1.00 |
| Out of KB | 7 | 0.00 | 1.00 | 1.00 |

Interpretation:

- Section-aware chunks + BM25 with title boosting + light query expansion recover the correct page for most Known questions (~76% page hit@top-k with ±2 page tolerance).
- Out-of-KB refusals use empty/weak retrieval plus explicit out-of-scope cues (WALert-style unanswered %).
- Re-run after `pip install sentence-transformers` + `python scripts/ingest_manuals.py` and with Ollama running for dense retrieval + real generation quality.

See `summary_bm25.json` and `detail_bm25.csv` in this folder.
