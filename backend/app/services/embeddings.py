"""Local semantic embeddings via fastembed (ONNX, no PyTorch needed).

If fastembed is unavailable or the model cannot be downloaded, the app keeps
working with BM25 keyword retrieval only.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from app.config import EMBEDDING_MODEL, EMBEDDINGS_ENABLED

log = logging.getLogger("researchai.embeddings")
_failed = False


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(EMBEDDING_MODEL)


def available() -> bool:
    global _failed
    if not EMBEDDINGS_ENABLED or _failed:
        return False
    try:
        _model()
        return True
    except Exception as exc:  # import error, download error, etc.
        log.warning("Embeddings disabled: %s", exc)
        _failed = True
        return False


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / norm, 6) for v in vector]


def embed_documents(texts: list[str]) -> list[list[float]] | None:
    if not texts or not available():
        return None
    return [_normalize(list(map(float, v))) for v in _model().embed(texts, batch_size=32)]


def embed_query(text: str) -> list[float] | None:
    if not available():
        return None
    # bge models recommend this instruction prefix for retrieval queries
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return _normalize(list(map(float, next(iter(_model().embed([prefixed]))))))


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    size = len(vectors[0])
    total = [0.0] * size
    for vector in vectors:
        for i, value in enumerate(vector):
            total[i] += value
    return _normalize(total)
