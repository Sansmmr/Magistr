"""
Helpers for limiting and splitting source text sent to LLM agents.
"""

from typing import List, Optional

# ~3-4k tokens — safe for gpt-4o-mini TPM limits when combined with prompts
MAX_SOURCE_CHARS = 8000

CONTEXT_COLUMN_CANDIDATES = (
    "support",
    "context",
    "passage",
    "article",
    "text",
    "document",
    "story",
    "content",
    "body",
)


def truncate_source_text(text: str, max_chars: int = MAX_SOURCE_CHARS) -> str:
    """Trim source text to a safe size for LLM prompts."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[... text truncated for API limits ...]"


def pick_source_for_question(
    source_text: str,
    question_index: int,
    source_chunks: Optional[List[str]] = None,
    max_chars: int = MAX_SOURCE_CHARS,
) -> str:
    """
    Pick one manageable passage for a given question index.
    Uses per-row chunks when available (CSV/Parquet datasets).
    """
    if source_chunks:
        chunk = source_chunks[question_index % len(source_chunks)]
        return truncate_source_text(chunk, max_chars)

    if not source_text:
        return ""

    # Fallback: split large plain text into roughly equal parts
    if len(source_text) > max_chars:
        part_size = max(max_chars, len(source_text) // max(question_index + 1, 1))
        start = (question_index * part_size) % max(len(source_text) - max_chars, 1)
        snippet = source_text[start : start + max_chars]
        return truncate_source_text(snippet, max_chars)

    return truncate_source_text(source_text, max_chars)
