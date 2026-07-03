from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")
SECTION_RE = re.compile(
    r"^\s*(abstract|introduction|background|related work|methodology|methods|experiments|results|discussion|conclusion|references)\s*$",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def split_words(text: str, max_words: int = 220, overlap: int = 45) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def vectorize(text: str) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = sum(counts.values())
    if not total:
        return {}
    return {term: count / total for term, count in counts.items() if len(term) > 2}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt(text: str, limit: int = 360) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def detect_section(page_text: str) -> str:
    for line in page_text.splitlines()[:18]:
        if SECTION_RE.match(line):
            return line.strip().title()
    return "General"
