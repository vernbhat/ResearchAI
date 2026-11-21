"""Answer generation, summaries and comparisons.

Each feature uses the configured LLM (Groq / Ollama) grounded on retrieved
chunks. If no LLM is configured or the call fails, it falls back to an
extractive answer built directly from the paper text.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from app.models import Citation
from app.services.llm import LLMError, chat, chat_json, llm_status
from app.services.text import clean_text

log = logging.getLogger("researchai.qa")

BROAD_OVERVIEW_RE = re.compile(
    r"\b(what('?s| is| are)\s+(this|the|these)\s+papers?\s+about|summari[sz]e|overview|main idea|"
    r"explain\s+(this|the)\s+paper|tl;?dr)\b",
    re.IGNORECASE,
)

COMPARISON_ASPECTS = ["Research focus", "Methods and data", "Main findings", "Limitations or challenges"]

ANSWER_SYSTEM = (
    "You are ResearchAI, an assistant that answers questions about research papers the user uploaded. "
    "Use ONLY the numbered excerpts provided. Cite every claim inline with the excerpt number, e.g. [1] or [2][3]. "
    "If the excerpts do not contain the answer, say clearly that the uploaded papers do not cover it. "
    "Be concise and precise: a short direct answer first, then supporting details or bullet points. "
    "Do not invent numbers, results or references. Use plain text or simple Markdown."
)

SUMMARY_SYSTEM = (
    "You summarise research papers for a literature review. Use ONLY the provided excerpts, "
    "and cite pages like (p. 3). Write in clear plain language."
)


def is_broad_overview_question(question: str) -> bool:
    return bool(BROAD_OVERVIEW_RE.search(question))


def _sources_block(citations: list[Citation]) -> str:
    return "\n\n".join(
        f"[{i}] {c.paper_title} (page {c.page_number})\n{c.text or c.excerpt}" for i, c in enumerate(citations, 1)
    )


# --------------------------------------------------------------------------- answers


def answer_question(
    question: str, citations: list[Citation], history: list[dict[str, str]] | None = None
) -> tuple[str, bool]:
    """Return (answer, generated_by_llm)."""
    if not citations:
        return (
            "I could not find anything relevant in the uploaded papers. "
            "Try rephrasing the question or uploading more PDFs.",
            False,
        )
    if llm_status()["enabled"]:
        messages = [{"role": "system", "content": ANSWER_SYSTEM}]
        for turn in (history or [])[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"][:1500]})
        messages.append(
            {
                "role": "user",
                "content": f"Excerpts from the uploaded papers:\n\n{_sources_block(citations)}\n\nQuestion: {question}",
            }
        )
        try:
            return chat(messages), True
        except LLMError as exc:
            log.warning("Falling back to extractive answer: %s", exc)
    return synthesize_answer(question, citations), False


def synthesize_answer(question: str, citations: list[Citation]) -> str:
    """Extractive fallback: the best passages grouped by paper."""
    grouped: dict[str, list[tuple[int, Citation]]] = defaultdict(list)
    for index, citation in enumerate(citations, 1):
        grouped[citation.paper_title].append((index, citation))

    lines = [f'Most relevant passages for: "{question}"', ""]
    for paper_title, items in grouped.items():
        lines.append(f"{paper_title}:")
        for index, citation in items[:2]:
            lines.append(f"- [{index}] (p. {citation.page_number}) {citation.excerpt}")
        lines.append("")
    lines.append("(No LLM configured - showing extracted passages. Add a GROQ_API_KEY for written answers.)")
    return "\n".join(lines)


def expand_query(question: str, history: list[dict[str, str]] | None = None) -> list[str]:
    """Use the LLM to turn a question into a few keyword-rich search queries (multi-query retrieval)."""
    if not llm_status()["enabled"]:
        return [question]
    context = ""
    last_user = next((h["content"] for h in reversed(history or []) if h["role"] == "user"), "")
    if last_user:
        context = f"Previous question (for resolving pronouns like 'it'): {last_user}\n"
    prompt = (
        f"{context}Question: {question}\n\n"
        "Write 3 short search queries that would find the passages answering this question inside academic papers. "
        "Use the vocabulary papers typically use (e.g. 'corpus', 'trained on', 'evaluation', 'benchmark'), "
        "resolve pronouns using the previous question, and keep each query under 12 words. "
        'Return JSON only: {"queries": ["...", "...", "..."]}'
    )
    try:
        data = chat_json([{"role": "user", "content": prompt}], max_tokens=400)
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]  # type: ignore[union-attr]
    except (LLMError, AttributeError) as exc:
        log.info("Query expansion skipped: %s", exc)
        queries = []
    return [question] + queries[:3]


# --------------------------------------------------------------------------- summaries


def summary_context(chunks: list[dict[str, Any]], limit: int = 9) -> list[dict[str, Any]]:
    """Pick representative chunks: the opening (abstract/intro) plus the ending (conclusion)."""
    if len(chunks) <= limit:
        return chunks
    head = chunks[:6]
    tail = chunks[-3:]
    return head + [c for c in tail if c not in head]


def summarize_paper(title: str, chunks: list[dict[str, Any]]) -> tuple[str, bool]:
    if not chunks:
        return f"No searchable text was found for {title}.", False
    if llm_status()["enabled"]:
        context = "\n\n".join(f"(page {c['page_number']})\n{c['text']}" for c in summary_context(chunks))
        prompt = (
            f"Paper: {title}\n\nExcerpts:\n{context}\n\n"
            "Write a structured summary with these sections (Markdown headings):\n"
            "## Overview (2-3 sentences)\n## Problem\n## Method\n## Key results\n## Contributions\n## Limitations\n"
            "Use bullet points where helpful and cite pages like (p. 2). "
            "If a section is not covered by the excerpts, write 'Not stated in the extracted text.'"
        )
        try:
            return chat([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=1400), True
        except LLMError as exc:
            log.warning("Falling back to extractive summary: %s", exc)
    return summarize_paper_chunks(title, chunks), False


def summarize_paper_chunks(title: str, chunks: list[dict[str, Any]]) -> str:
    """Extractive fallback summary using generic, domain-independent cue words."""
    if not chunks:
        return f"No searchable text was found for {title}."
    sections = [
        ("What the paper is about", ["in this paper", "this paper", "we propose", "we present", "we introduce", "this work", "we study"]),
        ("Method", ["method", "approach", "architecture", "framework", "algorithm", "model", "we use", "trained"]),
        ("Main findings", ["results show", "we show", "we find", "outperform", "achieve", "improve", "accuracy", "state-of-the-art"]),
        ("Why it matters", ["important", "enable", "practical", "impact", "significant", "future work", "conclusion"]),
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

    contributions = _rank_sentences(chunks, ["contribution", "we propose", "novel", "first", "new"], used)[:3]
    if contributions:
        lines += ["", "Key contributions:"]
        for sentence in contributions:
            cited_pages.add(sentence["page_number"])
            lines.append(f"- {sentence['text']} (p. {sentence['page_number']})")
    if cited_pages:
        lines += ["", "Evidence pages: " + ", ".join(map(str, sorted(cited_pages)))]
    return "\n".join(lines)


# --------------------------------------------------------------------------- comparison


def compare_papers(papers: list[dict[str, Any]], chunks_by_paper: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    if llm_status()["enabled"]:
        try:
            return _compare_with_llm(papers, chunks_by_paper)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            log.warning("Falling back to extractive comparison: %s", exc)
    return _compare_extractive(papers, chunks_by_paper)


def _compare_with_llm(papers: list[dict[str, Any]], chunks_by_paper: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    per_paper = max(3, 9 // len(papers))
    blocks = []
    for paper in papers:
        context = summary_context(chunks_by_paper.get(paper["id"], []), limit=per_paper)[:per_paper]
        text = "\n".join(f"(page {c['page_number']}) {c['text']}" for c in context)
        blocks.append(f"=== paper_id={paper['id']} | {paper['title']} ===\n{text}")
    prompt = (
        "Compare these research papers using ONLY the excerpts.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON only, in this exact shape:\n"
        '{"overview": "2-3 sentence comparison of how the papers relate and differ",\n'
        ' "aspects": [{"label": "<aspect>", "points": [{"paper_id": <id>, "text": "<1-2 sentences>", "page_number": <int or null>}]}]}\n'
        f"Use exactly these aspect labels in order: {COMPARISON_ASPECTS}. "
        "Include one point per paper for every aspect. If something is not in the excerpts, say 'Not stated in the extracted text.'"
    )
    data = chat_json([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=1800)
    titles = {paper["id"]: paper["title"] for paper in papers}
    aspects = []
    for aspect in data["aspects"]:
        points = []
        for point in aspect["points"]:
            pid = int(point["paper_id"])
            if pid not in titles:
                continue
            page = point.get("page_number")
            points.append({"paper_id": pid, "paper_title": titles[pid], "text": str(point["text"]), "page_number": int(page) if isinstance(page, (int, float)) else None})
        aspects.append({"label": str(aspect["label"]), "points": points})
    if not aspects:
        raise ValueError("empty comparison")
    return {"papers": papers, "overview": str(data.get("overview", "")), "aspects": aspects}


def _compare_extractive(papers: list[dict[str, Any]], chunks_by_paper: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    profiles = {paper["id"]: comparison_profile(paper["title"], chunks_by_paper.get(paper["id"], [])) for paper in papers}
    aspects = [
        {
            "label": label,
            "points": [
                {"paper_id": paper["id"], "paper_title": paper["title"], **profiles[paper["id"]]["aspects"][label]}
                for paper in papers
            ],
        }
        for label in COMPARISON_ASPECTS
    ]
    return {"papers": papers, "overview": "", "aspects": aspects}


def comparison_profile(title: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    aspect_keywords = {
        "Research focus": ["in this paper", "this paper", "we propose", "we present", "this work", "we study", "objective", "focus"],
        "Methods and data": ["method", "model", "dataset", "data", "architecture", "approach", "trained", "experiment"],
        "Main findings": ["results show", "we show", "we find", "outperform", "achieve", "accuracy", "improve", "performance"],
        "Limitations or challenges": ["limitation", "challenge", "however", "future work", "fail", "drawback", "difficult", "not "],
    }
    profile: dict[str, Any] = {"title": title, "aspects": {}}
    used: set[str] = set()
    for aspect, keywords in aspect_keywords.items():
        sentence = _best_sentence(chunks, keywords, used)
        if sentence is None:
            profile["aspects"][aspect] = {"text": "No clear evidence found in the extracted text.", "page_number": None}
            continue
        used.add(sentence["text"])
        profile["aspects"][aspect] = {"text": sentence["text"], "page_number": sentence["page_number"]}
    return profile


# --------------------------------------------------------------------------- sentence ranking helpers


def _best_sentence(chunks: list[dict[str, Any]], keywords: list[str], used: set[str]) -> dict[str, Any] | None:
    ranked = _rank_sentences(chunks, keywords, used)
    return ranked[0] if ranked else None


def _rank_sentences(chunks: list[dict[str, Any]], keywords: list[str], used: set[str]) -> list[dict[str, Any]]:
    keyword_set = {keyword.lower() for keyword in keywords}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        page_number = int(chunk["page_number"])
        for sentence in _sentences(str(chunk["text"])):
            normalized = sentence.lower()
            if sentence in used or sentence in seen or not _useful_sentence(sentence):
                continue
            seen.add(sentence)
            score = sum(1 for keyword in keyword_set if keyword in normalized)
            if score == 0:
                continue
            if page_number <= 2:
                score += 1
            if "we " in normalized or "this paper" in normalized or "our " in normalized:
                score += 1
            candidates.append({"text": sentence, "page_number": page_number, "score": score})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


NOISY_MARKERS = [
    "received ", "digital object identifier", "copyright", "corresponding author", " et al.", "permission",
    "license", "licence", "arxiv", "preprint", "proceedings", "conference on", "http", "@", "equal contribution",
    "all rights reserved", "grants permission", "work performed while",
]


def _useful_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < 45 or len(sentence) > 360:
        return False
    if not sentence.endswith((".", "!", "?")):
        return False
    if re.match(r"^\[\d+\]", sentence):
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    if letters / max(len(sentence), 1) < 0.6:  # tables, equations, figure residue
        return False
    return not any(marker in lowered for marker in NOISY_MARKERS)
