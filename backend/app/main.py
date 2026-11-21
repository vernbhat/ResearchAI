from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.database import UPLOAD_DIR, connect, dumps, ensure_storage, loads, row_to_dict
from app.models import AskRequest, AskResponse, Citation, CompareRequest, RelatedRequest
from app.services import embeddings
from app.services.llm import llm_status
from app.services.pdf import chunk_pages, extract_pdf
from app.services.qa import (
    answer_question,
    compare_papers,
    expand_query,
    is_broad_overview_question,
    summarize_paper,
)
from app.services.retrieval import paper_vectors, search_chunks
from app.services.text import excerpt, vectorize

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_storage()
    backfill_embeddings()
    status = llm_status()
    log.info("LLM: %s (%s) enabled=%s", status["provider"], status["model"], status["enabled"])
    log.info("Embeddings available: %s", embeddings.available())
    yield


app = FastAPI(title="ResearchAI", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def backfill_embeddings() -> None:
    """Embed chunks that were indexed before embeddings existed (or while they were unavailable)."""
    if not embeddings.available():
        return
    with connect() as db:
        rows = db.execute("SELECT id, text FROM chunks WHERE embedding_json IS NULL").fetchall()
        if not rows:
            return
        log.info("Embedding %d existing chunks...", len(rows))
        vectors = embeddings.embed_documents([row["text"] for row in rows]) or []
        db.executemany(
            "UPDATE chunks SET embedding_json = ? WHERE id = ?",
            [(dumps(vec), row["id"]) for vec, row in zip(vectors, rows)],
        )


def _paper_chunks(paper_id: int) -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, page_number, chunk_index, text FROM chunks WHERE paper_id = ? ORDER BY chunk_index",
            (paper_id,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def _get_paper(paper_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return row_to_dict(row)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "llm": llm_status(), "embeddings": embeddings.available()}


@app.post("/papers")
async def upload_paper(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    ensure_storage()
    safe_name = Path(file.filename).name
    with connect() as db:
        exists = db.execute("SELECT 1 FROM papers WHERE lower(filename) = lower(?)", (safe_name,)).fetchone()
    if exists:
        raise HTTPException(status_code=409, detail=f"{safe_name} is already uploaded.")

    destination = UPLOAD_DIR / safe_name
    suffix = 1
    while destination.exists():
        destination = UPLOAD_DIR / f"{Path(safe_name).stem}-{suffix}{Path(safe_name).suffix}"
        suffix += 1
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        parsed = extract_pdf(destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    pages = parsed["pages"]
    chunks = chunk_pages(pages)  # type: ignore[arg-type]
    if not chunks:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted. The PDF may be scanned images (OCR is not supported).",
        )

    fallback_title = Path(safe_name).stem.replace("_", " ").replace("-", " ").strip().title() or safe_name
    title = str(parsed["title"] or fallback_title)
    vectors = embeddings.embed_documents([str(chunk["text"]) for chunk in chunks])

    with connect() as db:
        cursor = db.execute(
            "INSERT INTO papers (title, filename, path, page_count) VALUES (?, ?, ?, ?)",
            (title, safe_name, str(destination), int(parsed["page_count"])),
        )
        paper_id = int(cursor.lastrowid)
        db.executemany(
            """
            INSERT INTO chunks (paper_id, page_number, chunk_index, text, vector_json, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    paper_id,
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["text"],
                    dumps(vectorize(str(chunk["text"]))),
                    dumps(vectors[i]) if vectors else None,
                )
                for i, chunk in enumerate(chunks)
            ],
        )

    return {"id": paper_id, "title": title, "page_count": parsed["page_count"], "chunks": len(chunks)}


@app.get("/papers")
def list_papers() -> list[dict[str, object]]:
    ensure_storage()
    with connect() as db:
        rows = db.execute(
            """
            SELECT p.id, p.title, p.filename, p.page_count, p.uploaded_at, COUNT(c.id) AS chunk_count
            FROM papers p
            LEFT JOIN chunks c ON c.paper_id = p.id
            GROUP BY p.id
            ORDER BY p.uploaded_at DESC, p.id DESC
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: int) -> dict[str, object]:
    paper = _get_paper(paper_id)
    with connect() as db:
        db.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
        db.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    path = Path(str(paper["path"]))
    if path.exists():
        path.unlink()
    return {"deleted": True, "id": paper_id, "title": paper["title"]}


@app.get("/papers/{paper_id}/summary")
def get_summary(paper_id: int, refresh: bool = False) -> dict[str, object]:
    paper = _get_paper(paper_id)
    if paper.get("summary") and not refresh:
        return {"summary": paper["summary"], "llm": True, "cached": True}
    summary, used_llm = summarize_paper(str(paper["title"]), _paper_chunks(paper_id))
    if used_llm:  # cache LLM summaries to save API calls
        with connect() as db:
            db.execute("UPDATE papers SET summary = ? WHERE id = ?", (summary, paper_id))
    return {"summary": summary, "llm": used_llm, "cached": False}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    ensure_storage()
    with connect() as db:
        if request.paper_ids:
            scope = db.execute(
                f"SELECT id, title FROM papers WHERE id IN ({','.join('?' for _ in request.paper_ids)})",
                request.paper_ids,
            ).fetchall()
        else:
            scope = db.execute("SELECT id, title FROM papers").fetchall()
        history: list[dict[str, str]] = []
        if request.chat_id is not None:
            if not db.execute("SELECT 1 FROM chats WHERE id = ?", (request.chat_id,)).fetchone():
                raise HTTPException(status_code=404, detail="Chat not found.")
            history = [
                {"role": row["role"], "content": row["content"]}
                for row in db.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 4",
                    (request.chat_id,),
                ).fetchall()
            ][::-1]
    if not scope:
        raise HTTPException(status_code=400, detail="Upload at least one paper first.")
    scope_ids = [int(row["id"]) for row in scope]

    if is_broad_overview_question(request.question):
        # Overview questions need each paper's opening (abstract/intro), not keyword matches.
        citations = []
        for row in scope[:6]:
            for chunk in _paper_chunks(int(row["id"]))[:2]:
                citations.append(
                    Citation(
                        paper_id=int(row["id"]),
                        paper_title=str(row["title"]),
                        page_number=int(chunk["page_number"]),
                        chunk_id=int(chunk["id"]),
                        score=1.0,
                        excerpt=excerpt(str(chunk["text"])),
                        text=str(chunk["text"]),
                    )
                )
    else:
        last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        if llm_status()["enabled"]:
            queries = expand_query(request.question, history)
        else:
            # Fold the previous question in so short follow-ups ("and its limitations?") still retrieve well.
            queries = [f"{request.question} {last_user}" if len(request.question.split()) < 8 and last_user else request.question]
        citations = _multi_query_search(queries, request.top_k, scope_ids)

    answer, used_llm = answer_question(request.question, citations, history)

    with connect() as db:
        chat_id = request.chat_id
        if chat_id is None:
            cursor = db.execute("INSERT INTO chats (title) VALUES (?)", (request.question[:80],))
            chat_id = int(cursor.lastrowid)
        db.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)", (chat_id, request.question))
        db.execute(
            "INSERT INTO messages (chat_id, role, content, citations_json) VALUES (?, 'assistant', ?, ?)",
            (chat_id, answer, dumps([item.model_dump() for item in citations])),
        )

    return AskResponse(chat_id=chat_id, answer=answer, citations=citations, llm=used_llm)


def _multi_query_search(queries: list[str], top_k: int, paper_ids: list[int]) -> list[Citation]:
    """Run several queries and merge results, keeping each chunk's best score."""
    best: dict[int, Citation] = {}
    hits: dict[int, int] = {}
    for query in queries:
        for citation in search_chunks(query, top_k * 2, paper_ids):
            hits[citation.chunk_id] = hits.get(citation.chunk_id, 0) + 1
            if citation.chunk_id not in best or citation.score > best[citation.chunk_id].score:
                best[citation.chunk_id] = citation
    # Chunks found by several query variants get a small boost.
    ranked = sorted(best.values(), key=lambda c: c.score + 0.05 * (hits[c.chunk_id] - 1), reverse=True)
    return ranked[:top_k]


@app.get("/chats")
def list_chats() -> list[dict[str, object]]:
    ensure_storage()
    with connect() as db:
        rows = db.execute("SELECT * FROM chats ORDER BY created_at DESC, id DESC").fetchall()
    return [row_to_dict(row) for row in rows]


@app.get("/chats/{chat_id}")
def get_chat(chat_id: int) -> dict[str, object]:
    with connect() as db:
        chat = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found.")
        messages = db.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)).fetchall()
    payload = row_to_dict(chat)
    payload["messages"] = [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
            "citations": loads(row["citations_json"]),
        }
        for row in messages
    ]
    return payload


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int) -> dict[str, object]:
    with connect() as db:
        db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return {"deleted": True, "id": chat_id}


@app.post("/compare")
def compare(request: CompareRequest) -> dict[str, object]:
    ids = list(dict.fromkeys(request.paper_ids))
    with connect() as db:
        rows = db.execute(
            f"SELECT id, title FROM papers WHERE id IN ({','.join('?' for _ in ids)})", ids
        ).fetchall()
    papers = [row_to_dict(row) for row in rows]
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="Choose at least two different papers.")
    if len(papers) > 4:
        raise HTTPException(status_code=400, detail="Compare up to 4 papers at a time.")
    chunks_by_paper = {int(p["id"]): _paper_chunks(int(p["id"])) for p in papers}
    return {"comparison": compare_papers(papers, chunks_by_paper)}


@app.post("/related")
def related_work(request: RelatedRequest) -> dict[str, object]:
    paper = _get_paper(request.paper_id)
    seed_chunks = _paper_chunks(request.paper_id)[:3]
    seed_text = " ".join(str(c["text"]) for c in seed_chunks)

    with connect() as db:
        duplicates = {
            int(row["id"])
            for row in db.execute(
                "SELECT id FROM papers WHERE id = ? OR lower(filename) = lower(?) OR lower(title) = lower(?)",
                (request.paper_id, paper["filename"], paper["title"]),
            ).fetchall()
        }

    # Rank other papers by similarity of their mean embedding to this paper's.
    vectors = paper_vectors()
    seed_vector = vectors.get(request.paper_id)
    ranked_ids: list[tuple[int, float]] = []
    if seed_vector:
        ranked_ids = sorted(
            ((pid, embeddings.dot(seed_vector, vec)) for pid, vec in vectors.items() if pid not in duplicates),
            key=lambda item: item[1],
            reverse=True,
        )

    related: list[Citation] = []
    if ranked_ids:
        for pid, paper_score in ranked_ids[: request.top_k]:
            best = search_chunks(seed_text, 1, [pid])
            if best:
                related.append(best[0].model_copy(update={"score": round(paper_score, 4)}))
    else:  # keyword fallback: best chunk per other paper
        seen: set[int] = set()
        for citation in search_chunks(seed_text, 200, exclude_paper_ids=duplicates):
            if citation.paper_id in seen:
                continue
            seen.add(citation.paper_id)
            related.append(citation)
            if len(related) >= request.top_k:
                break

    return {"paper": paper, "related": related}
