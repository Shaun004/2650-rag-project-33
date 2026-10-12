"""Parse and lightly clean Samsung manual text/PDF content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAGE_SPAN_RE = re.compile(
    r"\(p\.\s*(\d+)(?:\s*[-–]\s*(\d+))?\)",
    re.IGNORECASE,
)
NOISE_LINE_RE = re.compile(
    r"^(page\s+\d+|confidential|©\s*samsung|all rights reserved)\b",
    re.IGNORECASE,
)


@dataclass
class ManualSection:
    chapter: str
    section: str
    page_start: int | None
    page_end: int | None
    content: str


def parse_page_span(heading: str) -> tuple[str, int | None, int | None]:
    """Extract title and page span from a heading like 'Connecting to the network  (p. 9-12)'."""
    match = PAGE_SPAN_RE.search(heading)
    if not match:
        return heading.strip(), None, None
    title = heading[: match.start()].strip()
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return title, start, end


def clean_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if NOISE_LINE_RE.match(line.strip()):
            continue
        # Drop lone repeated running headers that are exactly "User Manual"
        if line.strip().lower() in {"user manual", "e-manual"}:
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def parse_manual_text(text: str) -> list[ManualSection]:
    """Split cleaned markdown-like manual text into subsection blocks."""
    text = clean_text(text)
    sections: list[ManualSection] = []
    chapter = "General"
    current_heading = "Introduction"
    page_start: int | None = None
    page_end: int | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content:
            sections.append(
                ManualSection(
                    chapter=chapter,
                    section=current_heading,
                    page_start=page_start,
                    page_end=page_end,
                    content=content,
                )
            )
        body = []

    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("##"):
            flush()
            chapter = line[2:].strip() or chapter
            continue
        if line.startswith("## ") and not line.startswith("###"):
            flush()
            chapter = line[3:].strip() or chapter
            current_heading = chapter
            page_start = page_end = None
            continue
        if line.startswith("### "):
            flush()
            title, page_start, page_end = parse_page_span(line[4:].strip())
            current_heading = title or "Untitled"
            continue
        body.append(line)

    flush()
    return sections


def load_manual_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_text(path: Path) -> str:
    """Optional PDF path for reproducible ingestion when a PDF is present."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(f"### PDF Page {i}  (p. {i})\n{extracted}")
    return "\n\n".join(pages)


def iter_manifest_paths(manifest_dir: Path) -> Iterable[Path]:
    yield from sorted(manifest_dir.glob("*.json"))
