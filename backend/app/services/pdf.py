from __future__ import annotations

from pathlib import Path

import fitz

from app.services.text import clean_text, split_words


def extract_pdf_pages(path: Path) -> list[dict[str, object]]:
    doc = fitz.open(path)
    pages: list[dict[str, object]] = []
    try:
        for index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"page_number": index, "text": text})
    finally:
        doc.close()
    return pages


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
