"""Chunking strategies for Samsung TV manuals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .parse_documents import ManualSection

ChunkStrategy = Literal["section", "fixed", "sentence"]


@dataclass
class Chunk:
    text: str
    chapter: str
    section: str
    page_start: int | None
    page_end: int | None
    chunk_index: int


def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # Prefer breaking on paragraph/newline
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if break_at > max_chars * 0.4:
                end = start + break_at + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]


def chunk_section_aware(
    sections: list[ManualSection],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        header = f"Section: {section.section}\nChapter: {section.chapter}\n\n"
        body = section.content.strip()
        pieces = _split_oversized(header + body, max_chars=max_chars, overlap=overlap)
        for piece in pieces:
            chunks.append(
                Chunk(
                    text=piece,
                    chapter=section.chapter,
                    section=section.section,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_fixed(
    sections: list[ManualSection],
    max_chars: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    full = "\n\n".join(
        f"### {s.section}\n{s.content}" for s in sections
    )
    pieces = _split_oversized(full, max_chars=max_chars, overlap=overlap)
    # Best-effort page/section: use first section metadata
    default = sections[0] if sections else None
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                text=piece,
                chapter=default.chapter if default else "General",
                section="fixed_window",
                page_start=default.page_start if default else None,
                page_end=default.page_end if default else None,
                chunk_index=i,
            )
        )
    return chunks


def chunk_sentence(
    sections: list[ManualSection],
    sentences_per_chunk: int = 5,
) -> list[Chunk]:
    sentence_split = re.compile(r"(?<=[.!?])\s+")
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        sentences = [s.strip() for s in sentence_split.split(section.content) if s.strip()]
        for i in range(0, len(sentences), sentences_per_chunk):
            group = sentences[i : i + sentences_per_chunk]
            text = f"Section: {section.section}\n\n" + " ".join(group)
            chunks.append(
                Chunk(
                    text=text,
                    chapter=section.chapter,
                    section=section.section,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_manual_text(
    sections: list[ManualSection],
    strategy: ChunkStrategy = "section",
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    if strategy == "section":
        return chunk_section_aware(sections, max_chars=max_chars, overlap=overlap)
    if strategy == "fixed":
        return chunk_fixed(sections, max_chars=max_chars, overlap=overlap)
    if strategy == "sentence":
        return chunk_sentence(sections)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
