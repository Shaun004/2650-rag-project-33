"""Semantic + BM25 retrieval with optional model filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .vector_store import FaissMetadataStore


@dataclass
class RetrievedChunk:
    passage_id: str
    passage: str
    score: float
    manual: str
    model: str
    page: int | None
    section: str
    chapter: str
    document_url: str

    def to_source(self) -> dict[str, Any]:
        return {
            "manual": self.manual,
            "model": self.model,
            "page": self.page,
            "section": self.section,
        }


def _normalise_model(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def model_matches(requested: str | None, row_model: str, row_family: str = "") -> bool:
    if not requested:
        return True
    req = _normalise_model(requested)
    model = _normalise_model(row_model)
    family = _normalise_model(row_family)
    if not req:
        return True
    return req in model or model in req or req in family or family in req


def _tokenize(text: str) -> list[str]:
    # Normalise common compounds before splitting (Wi-Fi → wifi)
    normalised = text.lower().replace("wi-fi", "wifi").replace("e-arc", "earc")
    normalised = normalised.replace("smart hub", "smarthub")
    return re.findall(r"[a-z0-9]+", normalised)


class DenseRetriever:
    def __init__(self, store: FaissMetadataStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        model: str | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(question)
        # Fetch extra then filter — store already filters loosely
        hits = self.store.search(vector, top_k=max(top_k * 8, top_k), model_number=None)
        filtered = []
        for hit in hits:
            passage = str(hit.get("passage") or "")
            section = str(hit.get("section") or "")
            if _has_term_overlap(question, passage, section):
                filtered.append(hit)
        return _to_chunks(filtered, top_k=top_k, model=model, threshold=threshold)


_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "and",
    "or",
    "on",
    "in",
    "for",
    "my",
    "me",
    "i",
    "is",
    "do",
    "how",
    "what",
    "why",
    "when",
    "where",
    "can",
    "with",
    "from",
    "it",
    "this",
    "that",
    "tv",
    "samsung",
    "manual",
    "please",
    "help",
}


def _content_terms(question: str) -> set[str]:
    return {t for t in _tokenize(question) if t not in _STOPWORDS and len(t) > 2}


def _has_term_overlap(question: str, passage: str, section: str = "") -> bool:
    terms = _content_terms(question)
    if not terms:
        return True
    haystack = _tokenize(f"{section} {passage}")
    haystack_set = set(haystack)
    return any(term in haystack_set for term in terms)


class BM25Retriever:
    def __init__(self, collection_path: Path, absolute_min_score: float = 1.5):
        self.df = pd.read_csv(collection_path)
        # Boost section/chapter titles so procedural headings rank above incidental mentions
        self.corpus = []
        for _, row in self.df.iterrows():
            section = str(row.get("section") or "")
            chapter = str(row.get("chapter") or "")
            passage = str(row.get("passage") or "")
            enriched = f"{section} {section} {section} {chapter} {passage}"
            self.corpus.append(enriched)
        self.tokenized = [_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.absolute_min_score = absolute_min_score

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        model: str | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        expanded = _expand_query(question)
        scores = self.bm25.get_scores(_tokenize(expanded))
        order = np.argsort(scores)[::-1]
        hits: list[dict[str, Any]] = []
        for idx in order:
            raw = float(scores[int(idx)])
            if raw < self.absolute_min_score:
                continue
            row = self.df.iloc[int(idx)]
            if not model_matches(model, str(row.get("model_number", "")), str(row.get("model_family", ""))):
                continue
            passage = str(row.get("passage") or "")
            section = str(row.get("section") or "")
            if not _has_term_overlap(expanded, passage, section):
                continue
            meta = row.to_dict()
            # Keep raw BM25 in metadata; also expose a 0-1-ish score for thresholds
            meta["raw_score"] = raw
            meta["score"] = raw
            hits.append(meta)
            if len(hits) >= max(top_k * 3, top_k):
                break
        if threshold is not None and hits:
            max_score = max(h["score"] for h in hits) or 1.0
            for h in hits:
                h["score"] = h["score"] / max_score
        return _to_chunks(hits, top_k=top_k, model=None, threshold=threshold)


def _expand_query(question: str) -> str:
    q = question
    lower = question.lower()
    expansions = []
    if "wi-fi" in lower or "wifi" in lower:
        expansions.append("wireless network open network settings")
    if "subtitle" in lower or "caption" in lower:
        expansions.append("subtitle settings accessibility")
    if "bluetooth" in lower and ("headphone" in lower or "speaker" in lower):
        expansions.append("audio device bluetooth speaker connection guide")
    if "game mode" in lower:
        expansions.append("game mode settings connection")
    if "factory reset" in lower or "reset the tv" in lower:
        expansions.append("reset smart hub self diagnosis")
    if "remote" in lower and ("work" in lower or "dead" in lower or "not" in lower or "isnt" in lower or "isn't" in lower):
        expansions.append("remote control does not work pair battery")
    if expansions:
        q = question + " " + " ".join(expansions)
    return q


def _to_chunks(
    hits: list[dict[str, Any]],
    top_k: int,
    model: str | None,
    threshold: float | None,
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for hit in hits:
        if not model_matches(
            model,
            str(hit.get("model_number") or ""),
            str(hit.get("model_family") or ""),
        ):
            continue
        score = float(hit.get("score", 0.0))
        if threshold is not None and score < threshold:
            continue
        page = hit.get("page_number")
        page_int = int(page) if page is not None and str(page) != "nan" and page == page else None
        chunks.append(
            RetrievedChunk(
                passage_id=str(hit.get("passage_id") or hit.get("chunk_id") or ""),
                passage=str(hit.get("passage") or ""),
                score=score,
                manual=str(hit.get("manual_title") or "Samsung TV Manual"),
                model=str(hit.get("model_number") or ""),
                page=page_int,
                section=str(hit.get("section") or ""),
                chapter=str(hit.get("chapter") or ""),
                document_url=str(hit.get("document_url") or ""),
            )
        )
        if len(chunks) >= top_k:
            break
    return chunks
