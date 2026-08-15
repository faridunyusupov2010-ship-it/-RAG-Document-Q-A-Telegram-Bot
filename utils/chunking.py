"""
chunking.py

A small recursive text splitter, built from scratch (no LangChain) so the
logic is fully visible and explainable in an interview: split on paragraph
breaks first, then sentences, then hard-wrap only as a last resort, and keep
a small overlap between chunks so a fact split across a chunk boundary isn't
lost.
"""

import re

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """If a single paragraph is already short enough, keep it whole.
    Otherwise split it into sentences and greedily pack them."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = SENTENCE_BREAK.split(paragraph)
    pieces, current = [], ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            # Edge case: a single sentence longer than max_chars on its own
            # (rare, but real documents have run-on sentences / tables).
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    pieces.append(sentence[i:i + max_chars])
                current = ""
            else:
                current = sentence

    if current:
        pieces.append(current)

    return pieces


def chunk_text(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    """
    max_chars: target chunk size in characters (not tokens — good enough
        for a demo, and language-agnostic, which matters since we support
        non-English documents too).
    overlap_chars: how much of the end of one chunk is repeated at the
        start of the next, so retrieval doesn't miss a fact that happens
        to sit right on a chunk boundary.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    
    raw_pieces = []
    for paragraph in paragraphs:
        raw_pieces.extend(_split_paragraph(paragraph, max_chars))

    # Merge small pieces together up to max_chars, and add overlap between
    # consecutive chunks.
    chunks = []
    current = ""

    for piece in raw_pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(f"{prev_tail}\n\n{chunks[i]}")
        chunks = overlapped

    return chunks
