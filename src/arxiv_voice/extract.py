from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


SECTION_PATTERN = re.compile(
    r"^(?:abstract|introduction|conclusion|references|appendix|"
    r"\d+(?:\.\d+)*[\s\-:]+[A-Z][A-Za-z0-9 ,:/()\-]{2,120})$",
    re.IGNORECASE,
)


@dataclass
class PdfExtraction:
    title: str
    authors: List[str]
    page_count: int
    pages: List[str]
    full_text: str
    sections: List[dict]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(full_text: str) -> List[dict]:
    sections: List[dict] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(full_text.splitlines(), start=1):
        line = raw_line.strip()
        if len(line) < 4 or len(line) > 120:
            continue
        lowered = line.lower()
        if lowered in seen:
            continue
        if SECTION_PATTERN.match(line):
            seen.add(lowered)
            sections.append({"line": line_number, "heading": line})
    return sections


def extract_pdf(pdf_path: Path, page_limit: int | None = None) -> PdfExtraction:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page_index, page in enumerate(reader.pages):
        if page_limit is not None and page_index >= page_limit:
            break
        page_text = normalize_text(page.extract_text() or "")
        pages.append(page_text)

    metadata = reader.metadata or {}
    title = (metadata.title or pdf_path.stem).strip()
    authors = []
    author_field = getattr(metadata, "author", None)
    if author_field:
        authors = [part.strip() for part in re.split(r",| and ", author_field) if part.strip()]

    page_blocks = []
    for index, page_text in enumerate(pages, start=1):
        page_blocks.append(f"[[Page {index}]]\n{page_text}")

    full_text = "\n\n".join(page_blocks).strip()
    sections = extract_sections(full_text)

    return PdfExtraction(
        title=title,
        authors=authors,
        page_count=len(pages),
        pages=pages,
        full_text=full_text,
        sections=sections,
    )
