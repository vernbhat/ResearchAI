"""Hybrid retrieval: semantic embeddings + BM25 keyword scores."""

from __future__ import annotations

from app.database import connect, loads, row_to_dict
from app.models import Citation
from app.services import embeddings
from app.services.text import bm25_scores, excerpt

SEMANTIC_WEIGHT = 0.55
KEYWORD_WEIGHT = 0.45


def load_chunks(paper_ids: list[int] | None = None) -> list[dict]:
    params: list[object] = []
    where = ""
    if paper_ids:
        where = f"WHERE p.id IN ({','.join('?' for _ in paper_ids)})"
        params.extend(paper_ids)
    with connect() as db:
        rows = db.execute(
            f"""
            SELECT c.id AS chunk_id, c.paper_id, c.page_number, c.chunk_index, c.text,
                   c.embedding_json, p.title AS paper_title
            FROM chunks c
            JOIN papers p ON p.id = c.paper_id
            {where}
            ORDER BY c.paper_id, c.chunk_index
            """,
            params,
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def search_chunks(
    query: str,
    top_k: int = 5,
    paper_ids: list[int] | None = None,
    exclude_paper_ids: set[int] | None = None,
) -> list[Citation]:
    chunks = load_chunks(paper_ids)
    if exclude_paper_ids:
        chunks = [c for c in chunks if c["paper_id"] not in exclude_paper_ids]
    if not chunks:
        return []

    keyword = bm25_scores(query, [c["text"] for c in chunks])
    max_kw = max(keyword) or 1.0
    keyword = [score / max_kw for score in keyword]

    semantic = [0.0] * len(chunks)
    query_vector = embeddings.embed_query(query)
    if query_vector is not None:
        for i, chunk in enumerate(chunks):
            if chunk.get("embedding_json"):
                semantic[i] = max(0.0, embeddings.dot(query_vector, loads(chunk["embedding_json"])))

    use_semantic = query_vector is not None and any(semantic)
    scored: list[tuple[float, dict]] = []
    for i, chunk in enumerate(chunks):
        if use_semantic:
            score = SEMANTIC_WEIGHT * semantic[i] + KEYWORD_WEIGHT * keyword[i]
        else:
            score = keyword[i]
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        Citation(
            paper_id=chunk["paper_id"],
            paper_title=chunk["paper_title"],
            page_number=chunk["page_number"],
            chunk_id=chunk["chunk_id"],
            score=round(score, 4),
            excerpt=excerpt(chunk["text"]),
            text=chunk["text"],
        )
        for score, chunk in scored[:top_k]
    ]


def paper_vectors() -> dict[int, list[float]]:
    """Mean embedding per paper, used for related-paper search."""
    grouped: dict[int, list[list[float]]] = {}
    for chunk in load_chunks():
        if chunk.get("embedding_json"):
            grouped.setdefault(chunk["paper_id"], []).append(loads(chunk["embedding_json"]))
    return {pid: vec for pid, vecs in grouped.items() if (vec := embeddings.mean_vector(vecs))}
