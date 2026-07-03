from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from app.models import Citation

BROAD_OVERVIEW_RE = re.compile(
    r"\b(what('?s| is)\s+(this|the)\s+paper\s+about|summari[sz]e|overview|main idea|explain\s+this\s+paper)\b",
    re.IGNORECASE,
)


def is_broad_overview_question(question: str) -> bool:
    return bool(BROAD_OVERVIEW_RE.search(question))


def synthesize_answer(question: str, citations: list[Citation]) -> str:
    if not citations:
        return (
            "I could not find a grounded answer in the uploaded papers. "
            "Try uploading more PDFs or rephrasing the question with terms used in the papers."
        )

    grouped: dict[str, list[Citation]] = defaultdict(list)
    for citation in citations:
        grouped[citation.paper_title].append(citation)

    lines = [
        f"Based on the most relevant passages, here is a grounded answer to: \"{question}\"",
        "",
    ]
    for paper_title, paper_citations in grouped.items():
        pages = sorted({item.page_number for item in paper_citations})
        evidence = " ".join(item.excerpt for item in paper_citations[:2])
        lines.append(f"- {paper_title} (p. {', '.join(map(str, pages))}): {evidence}")

    lines.append("")
    lines.append("Use the cited excerpts below to verify the answer against the source papers.")
    return "\n".join(lines)


def summarize_chunks(title: str, citations: list[Citation]) -> str:
    if not citations:
        return f"No searchable text was found for {title}."
    points = [f"- p. {item.page_number}: {item.excerpt}" for item in citations[:6]]
    return f"Summary for {title}\n\n" + "\n".join(points)


def summarize_paper_chunks(title: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return f"No searchable text was found for {title}."

    sections = [
        (
            "What the paper is about",
            [
                "paper",
                "study",
                "evaluate",
                "propose",
                "adversarial",
                "attack",
                "image",
                "classification",
                "model",
            ],
        ),
        (
            "What it tests",
            [
                "fgsm",
                "pgd",
                "bim",
                "deepfool",
                "ensemble",
                "resnet",
                "densenet",
                "vgg",
                "mobilenet",
                "inception",
                "resnet18",
                "resnet50",
                "resnet152",
                "densenet201",
                "dataset",
            ],
        ),
        (
            "Main findings",
            [
                "results",
                "show",
                "less vulnerable",
                "robust",
                "performance",
                "accuracy",
                "resnet152",
                "densenet201",
                "ensemble",
            ],
        ),
        (
            "Why it matters",
            [
                "robustness",
                "vulnerability",
                "defensive",
                "misclassification",
                "real-world",
                "security",
                "reliability",
            ],
        ),
    ]

    lines = [f"Summary for {title}", ""]
    used: set[str] = set()
    cited_pages: set[int] = set()

    for heading, keywords in sections:
        sentence = _best_sentence(chunks, keywords, used)
        if sentence is None:
            continue
        used.add(sentence["text"])
        cited_pages.add(sentence["page_number"])
        lines.append(f"{heading}: {sentence['text']} (p. {sentence['page_number']})")

    contribution_sentences = _best_sentences(
        chunks,
        ["contribution", "propose", "analysis", "evaluation", "defensive", "dataset"],
        used,
        limit=3,
    )
    if contribution_sentences:
        lines.append("")
        lines.append("Key contributions:")
        for sentence in contribution_sentences:
            used.add(sentence["text"])
            cited_pages.add(sentence["page_number"])
            lines.append(f"- {sentence['text']} (p. {sentence['page_number']})")

    if cited_pages:
        lines.append("")
        lines.append("Evidence pages: " + ", ".join(f"p. {page}" for page in sorted(cited_pages)))

    return "\n".join(lines)


def _best_sentences(
    chunks: list[dict[str, Any]],
    keywords: list[str],
    used: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = _rank_sentences(chunks, keywords, used)
    return ranked[:limit]


def _best_sentence(
    chunks: list[dict[str, Any]],
    keywords: list[str],
    used: set[str],
) -> dict[str, Any] | None:
    ranked = _rank_sentences(chunks, keywords, used)
    return ranked[0] if ranked else None


def _rank_sentences(
    chunks: list[dict[str, Any]],
    keywords: list[str],
    used: set[str],
) -> list[dict[str, Any]]:
    keyword_set = {keyword.lower() for keyword in keywords}
    candidates: list[dict[str, Any]] = []

    for chunk in chunks:
        page_number = int(chunk["page_number"])
        for sentence in _sentences(str(chunk["text"])):
            normalized = sentence.lower()
            if sentence in used or not _useful_sentence(sentence):
                continue
            score = sum(1 for keyword in keyword_set if keyword in normalized)
            if score == 0:
                continue
            if page_number <= 2:
                score += 1
            if "we " in normalized or "this paper" in normalized:
                score += 1
            candidates.append({"text": sentence, "page_number": page_number, "score": score})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def _useful_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < 45 or len(sentence) > 360:
        return False
    if not sentence.endswith((".", "!", "?")):
        return False
    noisy = [
        "received ",
        "digital object identifier",
        "copyright",
        "corresponding author",
        " et al.",
    ]
    if re.match(r"^\[\d+\]", sentence):
        return False
    return not any(marker in lowered for marker in noisy)
