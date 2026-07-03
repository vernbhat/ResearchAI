from __future__ import annotations

from app.database import connect, loads, row_to_dict
from app.models import Citation
from app.services.text import cosine, excerpt, vectorize


def search_chunks(query: str, top_k: int = 5, paper_ids: list[int] | None = None) -> list[Citation]:
    query_vector = vectorize(query)
    params: list[object] = []
    where = ""
    if paper_ids:
        placeholders = ",".join("?" for _ in paper_ids)
        where = f"WHERE p.id IN ({placeholders})"
        params.extend(paper_ids)

    with connect() as db:
        rows = db.execute(
            f"""
            SELECT c.id AS chunk_id, c.paper_id, c.page_number, c.text, c.vector_json,
                   p.title AS paper_title
            FROM chunks c
            JOIN papers p ON p.id = c.paper_id
            {where}
            """,
            params,
        ).fetchall()

    scored: list[Citation] = []
    for row in rows:
        data = row_to_dict(row)
        score = cosine(query_vector, loads(data["vector_json"]))
        if score <= 0:
            continue
        scored.append(
            Citation(
                paper_id=data["paper_id"],
                paper_title=data["paper_title"],
                page_number=data["page_number"],
                chunk_id=data["chunk_id"],
                score=round(score, 4),
                excerpt=excerpt(data["text"]),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]
