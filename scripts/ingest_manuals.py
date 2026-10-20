#!/usr/bin/env python3
"""Reproducible ingestion: manifests → collection.csv → FAISS index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from ingestion.build_collection import ingest_manifest
from ingestion.build_embeddings import build_faiss_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Samsung TV manuals into the RAG store")
    parser.add_argument(
        "--manifest",
        action="append",
        help="Manifest JSON path (default: all files in MANUAL_MANIFEST_DIR)",
    )
    parser.add_argument("--strategy", default=None, help="section | fixed | sentence")
    parser.add_argument("--hash-embeddings", action="store_true", help="Use hash embedder (tests/offline)")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    manifest_dir = settings.resolve(settings.manual_manifest_dir)
    collection_path = settings.resolve(settings.collection_path)
    strategy = args.strategy or settings.chunk_strategy

    manifests = [Path(p) for p in args.manifest] if args.manifest else sorted(manifest_dir.glob("*.json"))
    if not manifests:
        print(f"No manifests found in {manifest_dir}")
        return 1

    for i, manifest in enumerate(manifests):
        print(f"Ingesting {manifest} ...")
        df = ingest_manifest(
            manifest,
            collection_path,
            strategy=strategy,
            max_chars=settings.max_chunk_chars,
            overlap=settings.chunk_overlap,
            append=i > 0,
        )
        print(f"  rows for this manual: {len(df[df['model_number'] == df['model_number'].iloc[0]])}")

    import pandas as pd

    n_rows = len(pd.read_csv(collection_path))
    print(f"Collection written to {collection_path} ({n_rows} rows)")

    if not args.skip_index:
        index_dir = build_faiss_index(use_hash=args.hash_embeddings)
        print(f"Index written to {index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
