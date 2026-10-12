"""Build WALert-style collection.csv rows with rich Samsung metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .chunk_documents import Chunk, chunk_manual_text
from .parse_documents import load_manual_text, parse_manual_text


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:40] or "chunk"


def chunk_id_for(model_number: str, chunk: Chunk) -> str:
    page = chunk.page_start if chunk.page_start is not None else 0
    return f"{_slug(model_number)}_p{page}_c{chunk.chunk_index:03d}"


def build_collection_rows(
    chunks: list[Chunk],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model = manifest["model_number"]
    for chunk in chunks:
        cid = chunk_id_for(model, chunk)
        rows.append(
            {
                "passage_id": cid,
                "passage": chunk.text,
                "manufacturer": manifest.get("manufacturer", "Samsung"),
                "brand": manifest.get("brand", "Samsung"),
                "product_type": manifest.get("product_type", "TV"),
                "model_number": model,
                "model_family": manifest.get("model_family", ""),
                "manual_title": manifest.get("manual_title", ""),
                "manual_version": manifest.get("manual_version", ""),
                "language": manifest.get("language", "English"),
                "document_url": manifest.get("document_url", ""),
                "page_number": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "chapter": chunk.chapter,
                "chunk_id": cid,
            }
        )
    return rows


def write_collection(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_manifest(
    manifest_path: Path,
    collection_path: Path,
    strategy: str = "section",
    max_chars: int = 1200,
    overlap: int = 150,
    append: bool = False,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    from config import ROOT

    manifest = load_manifest(manifest_path)
    text_path = Path(manifest["local_text"])
    if not text_path.is_absolute():
        text_path = (repo_root or ROOT) / text_path
    text = load_manual_text(text_path)
    sections = parse_manual_text(text)
    chunks = chunk_manual_text(
        sections,
        strategy=strategy,  # type: ignore[arg-type]
        max_chars=max_chars,
        overlap=overlap,
    )
    rows = build_collection_rows(chunks, manifest)
    if append and collection_path.exists():
        existing = pd.read_csv(collection_path)
        # Drop prior rows for same model, then append
        existing = existing[existing["model_number"] != manifest["model_number"]]
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    write_collection(df.to_dict(orient="records"), collection_path)
    return df
