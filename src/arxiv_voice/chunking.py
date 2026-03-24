from __future__ import annotations

from typing import List


def chunk_text(text: str, max_chars: int, overlap_chars: int = 0) -> List[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph) + 2
        if current and current_len + paragraph_len > max_chars:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if overlap_chars > 0 and chunk:
                overlap = chunk[-overlap_chars:].strip()
                current = [overlap, paragraph]
                current_len = len(overlap) + len(paragraph) + 4
            else:
                current = [paragraph]
                current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += paragraph_len

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks
