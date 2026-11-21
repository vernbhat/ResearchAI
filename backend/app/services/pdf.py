from __future__ import annotations

import re
from pathlib import Path

import fitz

from app.services.text import clean_text, split_words

REFERENCES_RE = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.IGNORECASE)
APPENDIX_RE = re.compile(r"^\s*(appendix|supplementary material)\b", re.IGNORECASE)


def extract_pdf(path: Path) -> dict[str, object]:
    """Return the paper title and per-page body text (reference lists removed)."""
    doc = fitz.open(path)
    try:
        title = _extract_title(doc)
        pages: list[dict[str, object]] = []
        in_references = False
        for index, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            kept: list[str] = []
            for line in raw.splitlines():
                if index > 1 and REFERENCES_RE.match(line):
                    in_references = True
                    continue
                if in_references and APPENDIX_RE.match(line):
                    in_references = False  # appendix after references is real content
                if not in_references:
                    kept.append(line)
            text = clean_text("\n".join(kept))
            if text:
                pages.append({"page_number": index, "text": text})
        return {"title": title, "pages": pages, "page_count": doc.page_count}
    finally:
        doc.close()


def extract_pdf_pages(path: Path) -> list[dict[str, object]]:
    return list(extract_pdf(path)["pages"])  # type: ignore[arg-type]


BOILERPLATE = re.compile(
    r"contents lists available|sciencedirect|journal homepage|www\.|https?://|elsevier|springer|"
    r"ieee|transactions on|proceedings of|journal of|vol\.|volume|issn|doi:|©|copyright|"
    r"received|accepted|available online|preprint|^arxiv",
    re.IGNORECASE,
)


def looks_like_title(text: str) -> bool:
    """A real title has several words and is not a journal header or boilerplate line."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    return 3 <= len(words) <= 40 and 8 <= len(text) <= 250 and not BOILERPLATE.search(text)


def _extract_title(doc: fitz.Document) -> str | None:
    meta_title = clean_text((doc.metadata or {}).get("title", "") or "")
    if looks_like_title(meta_title) and not re.search(r"\.(pdf|docx?|tex)$|^untitled|^microsoft word", meta_title, re.I):
        return meta_title
    if doc.page_count == 0:
        return None
    # Fallback: the largest-font lines near the top of page 1, skipping journal headers
    # (e.g. Elsevier puts "Neurocomputing" above the real title).
    lines = []
    for block in doc[0].get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            if abs(line.get("dir", (1, 0))[1]) > 0.1:  # skip rotated text such as the arXiv side stamp
                continue
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans or spans[0]["bbox"][1] > doc[0].rect.height * 0.5:
                continue
            text = clean_text(" ".join(s["text"] for s in spans))
            if len(text) > 2 and not re.match(r"^arxiv:", text, re.I):
                lines.append((round(max(s["size"] for s in spans), 1), spans[0]["bbox"][1], text))
    for size in sorted({size for size, _, _ in lines}, reverse=True):
        parts = [text for s, _, text in sorted(lines, key=lambda l: l[1]) if s == size and not BOILERPLATE.search(text)]
        title = clean_text(" ".join(parts))
        if looks_like_title(title):
            return title
    return None


def chunk_pages(pages: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    chunk_index = 0
    for page in pages:
        for chunk in split_words(str(page["text"])):
            records.append(
                {
                    "page_number": int(page["page_number"]),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
            chunk_index += 1
    return records
