"""Find related papers in the published literature (not just the local library).

Sources are tried in order until one returns results:
  1. Semantic Scholar recommendations (best quality, free; optional key for higher limits)
  2. OpenAlex "related works" (free; optional key / email)
  3. arXiv keyword search (free, no key) - always-available fallback

Every call is wrapped so a network problem never breaks the app; the caller gets
an empty list plus a human-readable notice instead.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

import httpx

from app.services.text import STOPWORDS, clean_text

log = logging.getLogger("researchai.scholar")

TIMEOUT = float(os.getenv("SCHOLAR_TIMEOUT", "15"))
USER_AGENT = "ResearchAI/0.3 (research paper assistant; https://github.com/vernbhat/ResearchAI)"

S2_FIELDS = "title,year,authors,venue,citationCount,externalIds,url,abstract,openAccessPdf"
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_FILE_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$", re.IGNORECASE)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+[^\s\"<>,;.)\]])")


class SourceError(RuntimeError):
    pass


@dataclass
class SeedPaper:
    title: str
    text: str = ""
    arxiv_id: str | None = None
    doi: str | None = None


@dataclass
class LookupResult:
    papers: list[dict] = field(default_factory=list)
    source: str | None = None
    errors: list[str] = field(default_factory=list)


def identify(title: str, filename: str, text: str) -> SeedPaper:
    """Pull identifiers (arXiv id, DOI) out of the PDF text/filename."""
    arxiv = ARXIV_RE.search(text) or ARXIV_FILE_RE.search(filename or "")
    doi = DOI_RE.search(text)
    return SeedPaper(
        title=title.strip(),
        text=text,
        arxiv_id=arxiv.group(1) if arxiv else None,
        doi=doi.group(1).lower() if doi else None,
    )


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def _get_json(client: httpx.Client, url: str, **kwargs) -> dict:
    response = client.get(url, **kwargs)
    if response.status_code == 429:
        raise SourceError("rate limited (too many requests from this network)")
    if response.status_code == 404:
        raise SourceError("paper not found")
    if response.status_code >= 400:
        raise SourceError(f"HTTP {response.status_code}")
    return response.json()


def _authors(names: list[str]) -> str:
    names = [n for n in names if n]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)


def _short(text: str | None, limit: int = 420) -> str:
    text = clean_text(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."


def _same_title(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    return bool(a and b) and norm(a) == norm(b)


# --------------------------------------------------------------------------- Semantic Scholar


def _s2_headers() -> dict[str, str]:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    return {"x-api-key": key} if key else {}


def _from_s2(item: dict) -> dict:
    ids = item.get("externalIds") or {}
    pdf = (item.get("openAccessPdf") or {}).get("url") or (
        f"https://arxiv.org/pdf/{ids['ArXiv']}" if ids.get("ArXiv") else None
    )
    return {
        "title": item.get("title") or "Untitled",
        "authors": _authors([a.get("name", "") for a in item.get("authors") or []]),
        "year": item.get("year"),
        "venue": item.get("venue") or None,
        "citations": item.get("citationCount"),
        "url": item.get("url") or (f"https://doi.org/{ids['DOI']}" if ids.get("DOI") else None),
        "pdf_url": pdf,
        "abstract": _short(item.get("abstract")),
    }


def semantic_scholar(seed: SeedPaper, limit: int) -> list[dict]:
    base = "https://api.semanticscholar.org"
    headers = _s2_headers()
    with _client() as client:
        if seed.arxiv_id:
            paper = _get_json(client, f"{base}/graph/v1/paper/arXiv:{seed.arxiv_id}",
                              params={"fields": "paperId,title"}, headers=headers)
        elif seed.doi:
            paper = _get_json(client, f"{base}/graph/v1/paper/DOI:{seed.doi}",
                              params={"fields": "paperId,title"}, headers=headers)
        else:
            match = _get_json(client, f"{base}/graph/v1/paper/search/match",
                              params={"query": seed.title, "fields": "paperId,title"}, headers=headers)
            paper = (match.get("data") or [{}])[0]
        paper_id = paper.get("paperId")
        if not paper_id:
            raise SourceError("paper not found")

        results: list[dict] = []
        for pool in ("all-cs", "recent"):
            data = _get_json(client, f"{base}/recommendations/v1/papers/forpaper/{paper_id}",
                             params={"fields": S2_FIELDS, "limit": limit, "from": pool}, headers=headers)
            results = data.get("recommendedPapers") or []
            if results:
                break
    return [_from_s2(item) for item in results]


# --------------------------------------------------------------------------- OpenAlex


def _openalex_params(extra: dict) -> dict:
    params = dict(extra)
    if key := os.getenv("OPENALEX_API_KEY", "").strip():
        params["api_key"] = key
    if email := os.getenv("OPENALEX_EMAIL", "").strip():
        params["mailto"] = email
    return params


def _abstract_from_index(index: dict | None) -> str:
    if not index:
        return ""
    positions = [(pos, word) for word, spots in index.items() for pos in spots]
    return " ".join(word for _, word in sorted(positions))


def _from_openalex(item: dict) -> dict:
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = item.get("open_access") or {}
    return {
        "title": item.get("display_name") or "Untitled",
        "authors": _authors([(a.get("author") or {}).get("display_name", "") for a in item.get("authorships") or []]),
        "year": item.get("publication_year"),
        "venue": source.get("display_name"),
        "citations": item.get("cited_by_count"),
        "url": item.get("doi") or location.get("landing_page_url") or item.get("id"),
        "pdf_url": location.get("pdf_url") or open_access.get("oa_url"),
        "abstract": _short(_abstract_from_index(item.get("abstract_inverted_index"))),
    }


def openalex(seed: SeedPaper, limit: int) -> list[dict]:
    base = "https://api.openalex.org"
    with _client() as client:
        if seed.doi:
            work = _get_json(client, f"{base}/works/doi:{seed.doi}", params=_openalex_params({}))
        else:
            found = _get_json(client, f"{base}/works", params=_openalex_params(
                {"search": seed.title, "per_page": 1, "select": "id,display_name,related_works"}))
            work = (found.get("results") or [{}])[0]
        related_ids = [w.rsplit("/", 1)[-1] for w in work.get("related_works") or []][:limit]
        if not related_ids:
            raise SourceError("no related works listed")
        data = _get_json(client, f"{base}/works", params=_openalex_params({
            "filter": "ids.openalex:" + "|".join(related_ids),
            "per_page": limit,
            "select": "id,display_name,publication_year,authorships,primary_location,"
                      "cited_by_count,doi,abstract_inverted_index,open_access",
        }))
    papers = [_from_openalex(item) for item in data.get("results") or []]
    return sorted(papers, key=lambda p: p.get("citations") or 0, reverse=True)


# --------------------------------------------------------------------------- arXiv


GENERIC = {
    "paper", "propose", "proposed", "using", "based", "results", "result", "show", "shows", "approach",
    "method", "methods", "new", "also", "work", "however", "different", "used", "use", "well", "number",
    "first", "two", "one", "can", "may", "university", "department", "google", "research", "com", "edu",
    "abstract", "introduction", "arxiv", "et", "al", "fig", "figure", "table", "section", "model", "models",
    "need", "via", "towards", "toward", "improving", "learning",
}


def abstract_text(text: str, words: int = 260) -> str:
    """Return the abstract if we can find it, otherwise the opening words of the paper."""
    match = re.search(r"\babstract\b[\s.:\-]*", text, flags=re.IGNORECASE)
    start = match.end() if match else 0
    end = re.search(r"\b(1\.?\s+)?introduction\b", text[start:], flags=re.IGNORECASE)
    body = text[start: start + end.start()] if end and end.start() > 300 else text[start:]
    return " ".join(body.split()[:words])


def keywords(seed: SeedPaper, count: int = 8) -> list[str]:
    """Distinctive terms: frequent phrases/words from the abstract plus title words."""
    def tokens(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower()) if w not in STOPWORDS and w not in GENERIC]

    abstract = abstract_text(seed.text)
    words = re.findall(r"[a-z][a-z\-]+", abstract.lower())
    bigrams = Counter(
        f"{a} {b}" for a, b in zip(words, words[1:])
        if a not in STOPWORDS and b not in STOPWORDS and a not in GENERIC and b not in GENERIC
        and len(a) > 2 and len(b) > 2
    )
    unigrams = Counter(tokens(abstract))
    terms = [p for p, n in bigrams.most_common(4) if n >= 2]
    terms += [w for w in dict.fromkeys(tokens(seed.title)) if w not in " ".join(terms)]
    terms += [w for w, _ in unigrams.most_common(20) if w not in " ".join(terms)]
    return terms[:count]


def arxiv(seed: SeedPaper, limit: int, query_terms: list[str] | None = None, rerank=None) -> list[dict]:
    terms = query_terms or keywords(seed)
    if not terms:
        raise SourceError("no keywords to search with")
    query = " OR ".join(f'abs:"{t}"' for t in terms)
    with _client() as client:
        response = client.get("https://export.arxiv.org/api/query", params={
            "search_query": query, "start": 0, "max_results": 40, "sortBy": "relevance",
        })
    if response.status_code >= 400:
        raise SourceError(f"HTTP {response.status_code}")

    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(response.text)
    papers: list[dict] = []
    for entry in root.findall("a:entry", ns):
        link = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        arxiv_id = re.sub(r"v\d+$", "", link.rsplit("/abs/", 1)[-1])
        title = clean_text(entry.findtext("a:title", default="", namespaces=ns))
        if (seed.arxiv_id and arxiv_id == seed.arxiv_id) or _same_title(title, seed.title):
            continue
        published = entry.findtext("a:published", default="", namespaces=ns)
        summary = clean_text(entry.findtext("a:summary", default="", namespaces=ns))
        papers.append({
            "title": title,
            "authors": _authors([a.findtext("a:name", default="", namespaces=ns)
                                 for a in entry.findall("a:author", ns)]),
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
            "citations": None,
            "url": link.replace("http://", "https://"),
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abstract": _short(summary),
            "_full": f"{title}. {summary}",
        })
    if rerank and papers:
        # Semantic re-ranking: order candidates by embedding similarity to the uploaded paper.
        scores = rerank([p["_full"] for p in papers])
        if scores:
            papers = [p for _, p in sorted(zip(scores, papers), key=lambda x: x[0], reverse=True)]
    for p in papers:
        p.pop("_full", None)
    return papers[:limit]


# --------------------------------------------------------------------------- orchestration

SOURCES = (
    ("Semantic Scholar", semantic_scholar),
    ("OpenAlex", openalex),
    ("arXiv", arxiv),
)


def find_related(seed: SeedPaper, limit: int = 8, arxiv_terms: list[str] | None = None, rerank=None) -> LookupResult:
    result = LookupResult()
    if os.getenv("RELATED_ONLINE", "true").strip().lower() in {"0", "false", "no"}:
        result.errors.append("Online search is turned off (RELATED_ONLINE=false).")
        return result
    for name, fn in SOURCES:
        try:
            papers = fn(seed, limit, arxiv_terms, rerank) if fn is arxiv else fn(seed, limit)
        except (SourceError, httpx.HTTPError, ET.ParseError, ValueError) as exc:
            log.info("%s lookup failed: %s", name, exc)
            result.errors.append(f"{name}: {exc}")
            continue
        papers = [p for p in papers if not _same_title(p["title"], seed.title)]
        if papers:
            result.papers, result.source = papers[:limit], name
            return result
        result.errors.append(f"{name}: no results")
    return result
