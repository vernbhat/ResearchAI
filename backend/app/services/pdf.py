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


def _extract_title(doc: fitz.Document) -> str | None:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if 8 <= len(meta_title) <= 250 and not re.search(r"\.(pdf|docx?|tex)$|^untitled|^microsoft word|^arxiv", meta_title, re.I):
        return clean_text(meta_title)
    if doc.page_count == 0:
        return None
    # Fallback: the largest-font text near the top of page 1.
    spans = []
    for block in doc[0].get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            if abs(line.get("dir", (1, 0))[1]) > 0.1:  # skip rotated text such as the arXiv side stamp
                continue
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if re.match(r"^arxiv:", text, re.I):
                    continue
                if len(text) > 2 and span["bbox"][1] < doc[0].rect.height * 0.5:
                    spans.append((round(span["size"], 1), span["bbox"][1], text))
    if not spans:
        return None
    biggest = max(size for size, _, _ in spans)
    parts = [text for size, _, text in sorted(spans, key=lambda s: s[1]) if size == biggest]
    title = clean_text(" ".join(parts))
    return title if 8 <= len(title) <= 250 else None


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
