# Research AI

An intelligent literature review assistant MVP. Upload research PDFs, ask grounded questions, get citations, generate summaries, compare papers, find related work, and keep chat history locally.

## Stack

- Backend: FastAPI, SQLite, PyMuPDF
- Retrieval: local TF-IDF style vector search
- Frontend: React + Vite

## Quick Start

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL from Vite, usually `http://localhost:5173`.

## Notes

This MVP is intentionally local-first. It does not require an LLM API key. Answers are synthesized from retrieved paper chunks with citations. You can later swap `backend/app/services/retrieval.py` for FAISS + sentence-transformers and `backend/app/services/qa.py` for an LLM-backed answer generator.
