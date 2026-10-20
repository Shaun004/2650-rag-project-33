# Data layout (WALert-adapted)

| File | Role |
|------|------|
| `manifests/*.json` | Per-manual metadata + paths (add a new TV by adding a manifest) |
| `manuals/*.txt` | Cleaned manual text with page markers |
| `manuals/*.pdf` | Optional official PDF (download from Samsung support; not always committed) |
| `collection.csv` | Passage KB (WALert `collection.csv` + Samsung metadata columns) |
| `questions.csv` | Evaluation set (Known / Inferred / Out of KB) |
| `indexes/faiss/` | FAISS index + `metadata.json` sidecar |

## Add another Samsung TV manual

1. Download the User Manual from the official Samsung support page for that model.
2. Place PDF/text under `data/manuals/`.
3. Copy `manifests/UA75DU8000WXXY.json` and edit model fields + paths.
4. Run `python scripts/ingest_manuals.py`.
