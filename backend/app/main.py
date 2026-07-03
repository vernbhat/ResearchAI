from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.database import UPLOAD_DIR, connect, dumps, ensure_storage, loads, row_to_dict
from app.models import AskRequest, AskResponse, CompareRequest, RelatedRequest
from app.services.pdf import chunk_pages, extract_pdf_pages
from app.services.qa import is_broad_overview_question, summarize_paper_chunks, synthesize_answer
from app.services.retrieval import search_chunks
from app.services.text import vectorize

app = FastAPI(title="ResearchHub AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    ensure_storage()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/papers")
async def upload_paper(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    ensure_storage()
    safe_name = Path(file.filename).name
    destination = UPLOAD_DIR / safe_name
    suffix = 1
    while destination.exists():
        destination = UPLOAD_DIR / f"{destination.stem}-{suffix}{destination.suffix}"
        suffix += 1

    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        pages = extract_pdf_pages(destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not parse PDF: {exc}") from exc

    chunks = chunk_pages(pages)
    title = Path(safe_name).stem.replace("_", " ").replace("-", " ").strip().title() or safe_name

    with connect() as db:
        cursor = db.execute(
            "INSERT INTO papers (title, filename, path, page_count) VALUES (?, ?, ?, ?)",
            (title, safe_name, str(destination), len(pages)),
        )
        paper_id = int(cursor.lastrowid)
        db.executemany(
            """
            INSERT INTO chunks (paper_id, page_number, chunk_index, text, vector_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    paper_id,
                    chunk["page_number"],
                    chunk["chunk_index"],
                    chunk["text"],
                    dumps(vectorize(str(chunk["text"]))),
                )
                for chunk in chunks
            ],
        )

    return {"id": paper_id, "title": title, "page_count": len(pages), "chunks": len(chunks)}


@app.get("/papers")
def list_papers() -> list[dict[str, object]]:
    ensure_storage()
    with connect() as db:
        rows = db.execute(
            """
            SELECT p.*, COUNT(c.id) AS chunk_count
            FROM papers p
            LEFT JOIN chunks c ON c.paper_id = p.id
            GROUP BY p.id
            ORDER BY p.uploaded_at DESC
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


@app.get("/papers/{paper_id}/summary")
def summarize_paper(paper_id: int) -> dict[str, str]:
    with connect() as db:
        paper = db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        chunks = db.execute(
            """
            SELECT page_number, chunk_index, text
            FROM chunks
            WHERE paper_id = ?
            ORDER BY chunk_index
            """,
            (paper_id,),
        ).fetchall()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return {"summary": summarize_paper_chunks(paper["title"], [row_to_dict(row) for row in chunks])}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    ensure_storage()
    citations = search_chunks(request.question, request.top_k)
    answer = synthesize_answer(request.question, citations)

    if is_broad_overview_question(request.question):
        with connect() as db:
            paper_rows = db.execute("SELECT * FROM papers ORDER BY uploaded_at DESC").fetchall()
            if len(paper_rows) == 1:
                paper = paper_rows[0]
                chunk_rows = db.execute(
                    """
                    SELECT page_number, chunk_index, text
                    FROM chunks
                    WHERE paper_id = ?
                    ORDER BY chunk_index
                    """,
                    (paper["id"],),
                ).fetchall()
                answer = summarize_paper_chunks(
                    paper["title"],
                    [row_to_dict(row) for row in chunk_rows],
                )

    with connect() as db:
        chat_id = request.chat_id
        if chat_id is None:
            cursor = db.execute("INSERT INTO chats (title) VALUES (?)", (request.question[:80],))
            chat_id = int(cursor.lastrowid)
        db.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)",
            (chat_id, request.question),
        )
        db.execute(
            "INSERT INTO messages (chat_id, role, content, citations_json) VALUES (?, 'assistant', ?, ?)",
            (chat_id, answer, dumps([item.model_dump() for item in citations])),
        )

    return AskResponse(chat_id=chat_id, answer=answer, citations=citations)


@app.get("/chats")
def list_chats() -> list[dict[str, object]]:
    ensure_storage()
    with connect() as db:
        rows = db.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
    return [row_to_dict(row) for row in rows]


@app.get("/chats/{chat_id}")
def get_chat(chat_id: int) -> dict[str, object]:
    with connect() as db:
        chat = db.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found.")
        messages = db.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC, id ASC",
            (chat_id,),
        ).fetchall()
    payload = row_to_dict(chat)
    payload["messages"] = [
        {**row_to_dict(row), "citations": loads(row["citations_json"])} for row in messages
    ]
    return payload


@app.post("/compare")
def compare_papers(request: CompareRequest) -> dict[str, object]:
    with connect() as db:
        rows = db.execute(
            f"SELECT id, title FROM papers WHERE id IN ({','.join('?' for _ in request.paper_ids)})",
            request.paper_ids,
        ).fetchall()
    papers = [row_to_dict(row) for row in rows]
    if len(papers) < 2:
        raise HTTPException(status_code=400, detail="Choose at least two existing papers.")

    comparison = []
    for paper in papers:
        citations = search_chunks("method approach dataset results contribution limitation", 5, [paper["id"]])
        comparison.append({"paper": paper, "highlights": citations})
    return {"comparison": comparison}


@app.post("/related")
def related_work(request: RelatedRequest) -> dict[str, object]:
    with connect() as db:
        paper = db.execute("SELECT * FROM papers WHERE id = ?", (request.paper_id,)).fetchone()
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found.")
        seed_rows = db.execute(
            "SELECT text FROM chunks WHERE paper_id = ? ORDER BY chunk_index LIMIT 4",
            (request.paper_id,),
        ).fetchall()
    seed = " ".join(row["text"] for row in seed_rows)
    duplicate_ids = _duplicate_paper_ids(request.paper_id, paper["filename"], paper["title"])
    citations = [
        item
        for item in search_chunks(seed, request.top_k + len(duplicate_ids) + 5)
        if item.paper_id not in duplicate_ids
    ]
    return {"paper": row_to_dict(paper), "related": citations[: request.top_k]}


def _duplicate_paper_ids(paper_id: int, filename: str, title: str) -> set[int]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT id
            FROM papers
            WHERE id = ?
               OR lower(filename) = lower(?)
               OR lower(title) = lower(?)
            """,
            (paper_id, filename, title),
        ).fetchall()
    return {int(row["id"]) for row in rows}
