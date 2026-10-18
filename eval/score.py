#!/usr/bin/env python3
"""WALert-adapted scoring for Samsung TV questions.csv."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from rag.generator import EchoLLM
from rag.pipeline import RagPipeline


def page_hit(predicted_pages: list[int | None], gold_page: float | int | None) -> bool:
    if gold_page is None or (isinstance(gold_page, float) and pd.isna(gold_page)):
        return False
    gold = int(gold_page)
    for page in predicted_pages:
        if page is None:
            continue
        if abs(int(page) - gold) <= 2:  # allow nearby pages in multi-page sections
            return True
    return False


def is_refusal(answer: str) -> bool:
    text = answer.lower()
    cues = [
        "does not provide enough information",
        "could not find enough information",
        "i don't know",
        "not in the manual",
        "manual does not",
    ]
    return any(c in text for c in cues)


def score_run(
    questions_path: Path,
    output_dir: Path,
    backend: str = "bm25",
    limit: int | None = None,
) -> dict:
    settings = get_settings()
    settings_override = settings.model_copy(
        update={
            "retriever_backend": backend,
            "retrieval_threshold": 0.05 if backend == "dense" else 0.0,
        }
    )
    pipeline = RagPipeline(
        settings=settings_override,
        llm=EchoLLM(),
        backend=backend,  # type: ignore[arg-type]
        use_hash_embeddings=True,
    )

    df = pd.read_csv(questions_path)
    if limit:
        df = df.head(limit)

    rows = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        qtype = str(row["question_type"])
        t0 = time.perf_counter()
        result = pipeline.ask(str(row["question"]), model=settings.default_tv_model)
        latency = time.perf_counter() - t0
        pages = [s.get("page") for s in result.sources]
        hit = page_hit(pages, row.get("gold_page"))
        refused = result.refused or is_refusal(result.answer)

        correct_cite = hit
        # Out of KB: success if refused
        if qtype == "Out of KB":
            answer_ok = refused
            correct_cite = refused
        else:
            answer_ok = (not refused) and (hit or bool(result.sources))

        by_type[qtype]["total"] += 1
        by_type[qtype]["page_hits"] += int(hit)
        by_type[qtype]["refused"] += int(refused)
        by_type[qtype]["answer_ok"] += int(answer_ok)

        rows.append(
            {
                "question_id": row["question_id"],
                "question_type": qtype,
                "question": row["question"],
                "gold_page": row.get("gold_page"),
                "pred_pages": pages,
                "page_hit": hit,
                "refused": refused,
                "answer_ok": answer_ok,
                "latency_sec": round(latency, 3),
                "answer": result.answer[:500],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"detail_{backend}.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)

    summary = {}
    for qtype, stats in by_type.items():
        total = max(stats["total"], 1)
        summary[qtype] = {
            "n": stats["total"],
            "page_hit_rate": round(stats["page_hits"] / total, 3),
            "refusal_rate": round(stats["refused"] / total, 3),
            "answer_ok_rate": round(stats["answer_ok"] / total, 3),
        }
    summary_path = output_dir / f"summary_{backend}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--backend", default="bm25", choices=["bm25", "dense"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    score_run(Path(args.questions), Path(args.out), backend=args.backend, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
