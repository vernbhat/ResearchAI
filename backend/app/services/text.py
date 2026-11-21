from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']+")

STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours yourself yourselves
    paper papers use used using via may might must shall one two also e.g i.e et al fig figure table
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


try:  # Snowball stemmer shipped with fastembed; falls back to light suffix stripping.
    from py_rust_stemmers import SnowballStemmer

    _stemmer = SnowballStemmer("english")

    def stem(word: str) -> str:
        return _stemmer.stem_word(word)

except ImportError:  # pragma: no cover

    def stem(word: str) -> str:
        for suffix in ("ings", "ing", "ies", "es", "ed", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


def content_tokens(text: str) -> list[str]:
    """Tokens used for keyword search: lowercase, stemmed, no stopwords, no 1-2 letter words."""
    return [stem(t) for t in tokenize(text) if len(t) > 2 and t not in STOPWORDS]


def split_words(text: str, max_words: int = 160, overlap: int = 40) -> list[str]:
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
    """Term-frequency vector (kept for the keyword index stored in SQLite)."""
    counts = Counter(content_tokens(text))
    total = sum(counts.values())
    if not total:
        return {}
    return {term: count / total for term, count in counts.items()}


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


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of every document for the query."""
    query_terms = set(content_tokens(query))
    if not query_terms or not documents:
        return [0.0] * len(documents)
    tokenized = [content_tokens(doc) for doc in documents]
    n = len(tokenized)
    avg_len = sum(len(doc) for doc in tokenized) / n or 1.0
    doc_freq = Counter(term for doc in tokenized for term in set(doc) if term in query_terms)
    scores = []
    for doc in tokenized:
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))
        scores.append(score)
    return scores


def clean_text(text: str) -> str:
    text = (
        text.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("–", "-")
        .replace("—", "-")
    )
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def excerpt(text: str, limit: int = 360) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."
