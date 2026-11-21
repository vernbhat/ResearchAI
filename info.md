# Research AI Project Info

## Project Name

Research AI is a local literature review assistant. It lets a user upload research paper PDFs, ask questions about them, generate summaries, compare papers, find related work, and save chat history.

The project is an MVP version of a larger ResearchHub AI idea. It already has the complete basic architecture:

```text
React frontend
    |
    v
FastAPI backend
    |
    +-- PDF processing
    +-- Text chunking
    +-- Local vector search
    +-- Answer generation with citations
    +-- SQLite storage
```

## Main Technologies Used

### Backend

- Python
- FastAPI
- Uvicorn
- SQLite
- PyMuPDF, imported as `fitz`
- Pydantic
- Python Multipart

### Frontend

- React
- Vite
- JavaScript / JSX
- CSS
- Lucide React icons

### Storage

- Uploaded PDFs are stored in `backend/uploads/`
- App data is stored in `backend/data/researchhub.db`
- SQLite tables store papers, chunks, chats, and messages

### Retrieval Method (v0.2)

- PDF text is split into 160-word chunks (40-word overlap); reference lists are skipped
- Each chunk is embedded locally with fastembed (`BAAI/bge-small-en-v1.5`, 384-dim) and stored in SQLite
- A question is embedded and compared with every chunk using cosine similarity
- BM25 keyword scores (stemmed, stopwords removed) are combined with the embedding score (55% / 45%)
- If an LLM is configured, it first rewrites the question into extra search queries (multi-query retrieval)
- The top chunks go to the LLM (Groq `openai/gpt-oss-120b` or local Ollama), which writes an answer citing [1], [2]...
- Without an LLM key, the app falls back to showing the extracted passages

## File Structure

```text
researchai/
├── README.md
├── info.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── data/
│   │   └── researchhub.db
│   ├── uploads/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── database.py
│       ├── models.py
│       └── services/
│           ├── __init__.py
│           ├── pdf.py
│           ├── text.py
│           ├── retrieval.py
│           └── qa.py
└── frontend/
    ├── package.json
    ├── package-lock.json
    ├── index.html
    └── src/
        ├── main.jsx
        └── styles.css
```

Generated folders such as `frontend/node_modules/`, `frontend/dist/`, `backend/.venv/`, `backend/data/`, and `backend/uploads/` are ignored by Git.

## How The Project Works

### 1. User Uploads A PDF

The user uploads one or more PDF files from the frontend.

Frontend function:

```javascript
upload(files)
```

This sends each PDF to:

```text
POST /papers
```

Backend file:

```text
backend/app/main.py
```

Backend route:

```python
@app.post("/papers")
async def upload_paper(file: UploadFile = File(...)):
```

What happens:

1. The backend checks that the file is a PDF.
2. The PDF is saved inside `backend/uploads/`.
3. The backend extracts text from the PDF.
4. The text is split into chunks.
5. Each chunk is converted into a vector.
6. Paper metadata and chunks are stored in SQLite.

## Backend Explanation

### `backend/app/main.py`

This is the main FastAPI application. It defines all API routes and connects the frontend to the backend services.

Important routes:

```text
GET /health
```

Checks whether the backend is running.

```text
POST /papers
```

Uploads and indexes a PDF.

```text
GET /papers
```

Returns all uploaded papers and their chunk counts.

```text
GET /papers/{paper_id}/summary
```

Returns a structured summary for one paper.

```text
POST /ask
```

Answers a natural language question using the uploaded papers.

```text
GET /chats
```

Returns saved chat sessions.

```text
GET /chats/{chat_id}
```

Returns a chat with all its messages.

```text
POST /compare
```

Compares selected papers by retrieving important chunks from each paper.

```text
POST /related
```

Finds related papers based on similarity. Duplicate uploads with the same filename or title are excluded so the app does not show the same paper as related work.

### `backend/app/database.py`

This file handles local storage.

Important constants:

```python
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = DATA_DIR / "researchhub.db"
```

Important functions:

```python
ensure_storage()
```

Creates the data folders and SQLite tables if they do not exist.

```python
connect()
```

Opens a SQLite database connection.

```python
row_to_dict(row)
```

Converts SQLite rows into normal Python dictionaries.

```python
dumps(value)
loads(value)
```

Convert data to and from JSON strings. These are used for vectors and citations.

SQLite tables:

```text
papers
```

Stores uploaded paper metadata:

- id
- title
- filename
- path
- page_count
- uploaded_at

```text
chunks
```

Stores extracted paper chunks:

- id
- paper_id
- page_number
- chunk_index
- text
- vector_json

```text
chats
```

Stores chat sessions:

- id
- title
- created_at

```text
messages
```

Stores chat messages:

- id
- chat_id
- role
- content
- citations_json
- created_at

### `backend/app/models.py`

This file defines Pydantic models for request and response validation.

Important models:

```python
Citation
```

Represents a retrieved source chunk with:

- paper id
- paper title
- page number
- chunk id
- relevance score
- excerpt

```python
AskRequest
```

Input for asking a question:

- question
- optional chat_id
- top_k number of chunks to retrieve

```python
AskResponse
```

Output for a question:

- chat_id
- answer
- citations

```python
CompareRequest
```

Input for comparing papers.

```python
RelatedRequest
```

Input for finding related papers.

## Backend Services

### `backend/app/services/pdf.py`

This file extracts text from PDFs and splits pages into chunks.

Important function:

```python
extract_pdf_pages(path)
```

Uses PyMuPDF:

```python
doc = fitz.open(path)
page.get_text("text")
```

It returns a list like:

```python
[
    {"page_number": 1, "text": "..."},
    {"page_number": 2, "text": "..."}
]
```

Important function:

```python
chunk_pages(pages)
```

Splits each page into smaller text chunks and keeps the page number for citations.

### `backend/app/services/text.py`

This file contains text processing logic.

Important functions:

```python
clean_text(text)
```

Removes extra whitespace.

```python
split_words(text, max_words=220, overlap=45)
```

Splits long paper text into overlapping chunks. Overlap helps avoid losing context between chunks.

```python
tokenize(text)
```

Extracts normalized word tokens.

```python
vectorize(text)
```

Turns text into a word-frequency vector.

Example idea:

```text
"adversarial attacks affect models"
```

becomes something like:

```python
{
    "adversarial": 0.25,
    "attacks": 0.25,
    "affect": 0.25,
    "models": 0.25
}
```

```python
cosine(left, right)
```

Measures similarity between two vectors.

```python
excerpt(text)
```

Creates a shorter readable citation excerpt.

### `backend/app/services/retrieval.py`

This file searches for relevant chunks.

Main function:

```python
search_chunks(query, top_k=5, paper_ids=None)
```

How it works:

1. Convert the query into a vector.
2. Load chunk vectors from SQLite.
3. Compare query vector with each chunk vector.
4. Score chunks using cosine similarity.
5. Sort by score.
6. Return the top chunks as citations.

This is the main local retrieval engine.

### `backend/app/services/qa.py`

This file creates answers and summaries.

Important function:

```python
synthesize_answer(question, citations)
```

Groups citations by paper and builds a grounded answer from the retrieved chunks.

Important function:

```python
summarize_paper_chunks(title, chunks)
```

Creates a structured summary with:

- What the paper is about
- What it tests
- Main findings
- Why it matters
- Key contributions
- Evidence pages

Important function:

```python
is_broad_overview_question(question)
```

Detects questions like:

```text
what is this paper about
summarize this paper
main idea
overview
```

If the user asks a broad overview question and only one paper exists, the backend uses the structured paper summary instead of a noisy retrieval answer.

## Frontend Explanation

### `frontend/package.json`

Defines frontend dependencies and scripts.

Important scripts:

```bash
npm run dev
```

Runs the Vite development server.

```bash
npm run build
```

Builds the frontend for production.

Important dependencies:

- React
- React DOM
- Vite
- TypeScript package
- Lucide React

### `frontend/index.html`

This is the HTML entry point. It loads:

```text
/src/main.jsx
```

### `frontend/src/main.jsx`

This is the main React app.

Important state variables:

```javascript
papers
```

List of uploaded papers.

```javascript
chats
```

Saved chat sessions.

```javascript
question
```

The current user question.

```javascript
answer
```

The latest answer from the backend.

```javascript
summary
```

The selected paper summary.

```javascript
comparison
```

Comparison output for selected papers.

```javascript
related
```

Related-work results.

Important functions:

```javascript
api(path, options)
```

Helper function for calling the FastAPI backend.

```javascript
refresh()
```

Loads papers and chats from the backend.

```javascript
upload(files)
```

Uploads PDFs to `POST /papers`.

```javascript
ask()
```

Sends the user question to `POST /ask`.

```javascript
loadSummary(paperId)
```

Calls `GET /papers/{paperId}/summary`.

```javascript
compare()
```

Calls `POST /compare`.

```javascript
findRelated(paperId)
```

Calls `POST /related`.

The UI has these main areas:

- Left sidebar with upload, paper list, and chat list
- Top area with compare button
- Question input box
- Answer panel
- Summary panel
- Comparison panel
- Related Work panel

### `frontend/src/styles.css`

This file styles the app.

It controls:

- Sidebar layout
- Upload button
- Paper cards
- Chat list
- Main workspace grid
- Ask box
- Answer and citation panels
- Responsive mobile layout

## API Routes Summary

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

Used to check that the backend is running.

### `POST /papers`

Uploads a PDF.

Input:

```text
multipart/form-data with file
```

Response:

```json
{
  "id": 1,
  "title": "Paper Title",
  "page_count": 13,
  "chunks": 57
}
```

### `GET /papers`

Returns uploaded papers.

Response:

```json
[
  {
    "id": 1,
    "title": "Paper Title",
    "filename": "paper.pdf",
    "path": "backend/uploads/paper.pdf",
    "page_count": 13,
    "uploaded_at": "...",
    "chunk_count": 57
  }
]
```

### `GET /papers/{paper_id}/summary`

Returns a structured paper summary.

Response:

```json
{
  "summary": "Summary for Paper Title..."
}
```

### `POST /ask`

Asks a question.

Request:

```json
{
  "question": "What is this paper about?",
  "chat_id": null,
  "top_k": 5
}
```

Response:

```json
{
  "chat_id": 1,
  "answer": "...",
  "citations": []
}
```

### `GET /chats`

Returns saved chat sessions.

### `GET /chats/{chat_id}`

Returns one chat and its messages.

### `POST /compare`

Compares papers.

Request:

```json
{
  "paper_ids": [1, 2]
}
```

Response:

```json
{
  "comparison": []
}
```

### `POST /related`

Finds related papers.

Request:

```json
{
  "paper_id": 1,
  "top_k": 5
}
```

Response:

```json
{
  "paper": {},
  "related": []
}
```

If only duplicate copies of the same paper exist, `related` returns an empty list.

## Complete Data Flow

### PDF Upload Flow

```text
User selects PDF
    |
    v
React upload(files)
    |
    v
POST /papers
    |
    v
FastAPI saves file to backend/uploads/
    |
    v
PyMuPDF extracts page text
    |
    v
Text is cleaned
    |
    v
Text is split into chunks
    |
    v
Each chunk is vectorized
    |
    v
Paper + chunks are saved in SQLite
    |
    v
Frontend refreshes paper list
```

### Question Answering Flow

```text
User asks a question
    |
    v
React ask()
    |
    v
POST /ask
    |
    v
Question is vectorized
    |
    v
Backend compares question vector against chunk vectors
    |
    v
Top chunks are returned as citations
    |
    v
Backend builds answer from cited excerpts
    |
    v
Question and answer are saved to chat history
    |
    v
Frontend displays answer and citations
```

### Summary Flow

```text
User clicks Summary
    |
    v
GET /papers/{paper_id}/summary
    |
    v
Backend loads all chunks for that paper
    |
    v
Backend ranks useful sentences by summary category
    |
    v
Structured summary is returned
    |
    v
Frontend displays summary
```

### Related Work Flow

```text
User clicks Related
    |
    v
POST /related
    |
    v
Backend uses early chunks from selected paper as a search seed
    |
    v
Backend searches other papers
    |
    v
Duplicate copies of the same paper are excluded
    |
    v
Frontend displays related chunks or an empty-state message
```

## Current Limitations

- Scanned PDFs (images only) are not supported (no OCR).
- Related work only searches among uploaded PDFs, not the internet.
- Answer quality depends on the LLM; without a key, answers are extractive.

## Possible Future Improvements

1. Use a vector index such as FAISS or sqlite-vec for large libraries.
2. Add a cross-encoder reranker.
3. Detect sections (Abstract, Method, Results) for better summaries.
4. Add PDF page preview when clicking a citation.
5. Export summaries to Markdown/PDF.
6. Add user accounts.

## How To Run The Project

First copy `backend/.env.example` to `backend/.env` and add a free Groq API key from https://console.groq.com/keys.

### Backend

```bash
cd /Users/kaushikshankarbhat/Documents/researchai/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

### Frontend

```bash
cd /Users/kaushikshankarbhat/Documents/researchai/frontend
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173/
```

## How To Install From Scratch

### Backend setup

```bash
cd /Users/kaushikshankarbhat/Documents/researchai/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend setup

```bash
cd /Users/kaushikshankarbhat/Documents/researchai/frontend
npm install
```

## Important Project Files

```text
backend/app/main.py
```

Main FastAPI app and all API routes.

```text
backend/app/database.py
```

SQLite setup and helper functions.

```text
backend/app/models.py
```

Pydantic request and response models.

```text
backend/app/services/pdf.py
```

PDF extraction and page chunking.

```text
backend/app/services/text.py
```

Text cleaning, tokenization, vectorization, and cosine similarity.

```text
backend/app/services/retrieval.py
```

Searches relevant chunks for questions, summaries, comparison, and related work.

```text
backend/app/services/qa.py
```

Generates grounded answers and structured paper summaries.

```text
frontend/src/main.jsx
```

Main React app logic.

```text
frontend/src/styles.css
```

Frontend design and layout.

## Short Explanation For Presentation

Research AI is a web-based literature review assistant. The frontend is built with React, and the backend is built with FastAPI. Users upload PDF research papers. The backend extracts text from the PDFs using PyMuPDF, splits the text into smaller chunks, converts each chunk into a local vector, and stores it in SQLite. When the user asks a question, the system converts the question into a vector and searches for the most similar paper chunks using cosine similarity. The answer is then generated from those chunks and displayed with paper names and page citations. The project also supports summaries, paper comparison, related-work search, and local chat history.
