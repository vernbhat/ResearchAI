# ResearchAI

An AI literature-review assistant. Upload research-paper PDFs, then ask questions and get answers grounded in the papers with page citations, generate structured summaries, compare papers side by side, find related work in your library, and keep chat history.

## How it works

```text
React (Vite)  ──►  FastAPI
                     ├─ PyMuPDF: text + title extraction, reference lists removed
                     ├─ Chunking: 160-word chunks with 40-word overlap
                     ├─ Embeddings: fastembed (BAAI/bge-small-en-v1.5, local ONNX, no GPU)
                     ├─ Hybrid retrieval: cosine similarity on embeddings + BM25 keyword scores
                     ├─ Multi-query retrieval: the LLM rewrites each question into search queries
                     ├─ LLM (Groq free API or local Ollama): cited answers, summaries, comparisons
                     └─ SQLite: papers, chunks + vectors, chats, messages, cached summaries
```

- **Ask**: question → query expansion → hybrid search over chunks → top excerpts go to the LLM with instructions to answer only from them and cite `[n]`. The last few chat turns go along too, so follow-up questions work.
- **Summary**: abstract/introduction and conclusion chunks → structured summary (Overview, Problem, Method, Key results, Contributions, Limitations). Cached in SQLite after the first call.
- **Compare**: 2–4 selected papers → JSON comparison across research focus, methods/data, findings and limitations.
- **Related work**: ranks the other papers in your library by the cosine similarity of their mean chunk embeddings.
- **No LLM key?** Everything still runs: answers fall back to extracted passages and summaries to ranked sentences.

## Quick start

### 1. Get a free LLM key (recommended)

Create a free key at <https://console.groq.com/keys>, then:

```bash
cd backend
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk_...
```

To run fully offline with [Ollama](https://ollama.com) instead, use `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.1:8b` (after `ollama pull llama3.1:8b`).

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The first start downloads the ~130 MB embedding model once. Check <http://127.0.0.1:8000/health>: it shows whether the LLM and embeddings are active.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Backend, LLM and embedding status |
| POST | `/papers` | Upload and index a PDF |
| GET | `/papers` | List papers |
| DELETE | `/papers/{id}` | Delete a paper and its chunks |
| GET | `/papers/{id}/summary?refresh=false` | Structured summary (cached) |
| POST | `/ask` | `{question, chat_id?, paper_ids?, top_k?}` → cited answer |
| GET / DELETE | `/chats`, `/chats/{id}` | Chat history |
| POST | `/compare` | `{paper_ids: [..]}` → aspect-by-aspect comparison |
| POST | `/related` | `{paper_id}` → most similar papers in the library |

## Configuration (`backend/.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | – | Free at console.groq.com |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Falls back to `openai/gpt-oss-20b`, then `llama-3.3-70b-versatile` |
| `LLM_BASE_URL` | provider default | Any OpenAI-compatible endpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed model |
| `EMBEDDINGS_ENABLED` | `true` | `false` = BM25 keyword search only |

Databases from the earlier version are migrated automatically, and existing chunks are embedded on startup.

## Limitations

- Scanned (image-only) PDFs need OCR, which isn't supported.
- Related work searches only your uploaded papers, not the internet.
- Small local models (≤3B) follow citation instructions less reliably than the Groq models.
