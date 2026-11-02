from __future__ import annotations


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("chunk overlap must be >= 0 and smaller than max_chars")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            candidates = [
                text.rfind("\n\n", start, hard_end),
                text.rfind(". ", start, hard_end),
                text.rfind("! ", start, hard_end),
                text.rfind("? ", start, hard_end),
                text.rfind("; ", start, hard_end),
                text.rfind(" ", start, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start + max_chars // 2:
                end = boundary
                if text[boundary : boundary + 2] in {". ", "! ", "? ", "; "}:
                    end += 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
