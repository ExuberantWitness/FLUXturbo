"""Citation resolver (I3) — tries multiple academic APIs in order.

Order: Semantic Scholar → arXiv → OpenAlex → CrossRef.
Each client is a thin `requests.get` wrapper with timeout, exponential backoff,
and per-API token bucket. The dispatcher returns the first hit; all-fail yields None.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests


# Per-API token bucket state (very simple: track last call timestamp).
_last_call: dict[str, float] = {}
_RATE_LIMIT_S: dict[str, float] = {
    # conservative public/anonymous rate limits
    "semantic_scholar": 1.0,    # ~100 req / 5 min ≈ 1 req / 3s — we use 1s as best-effort
    "arxiv": 1.0,
    "openalex": 0.2,             # 100k/day, generous
    "crossref": 1.0,
}


def resolve_citation(
    raw: str,
    *,
    api_keys: dict[str, str] | None = None,
    timeout: float = 2.0,
    max_retries: int = 3,
) -> dict | None:
    """Try each academic API to resolve a raw citation string.

    Returns: {doi, title, authors, year, venue, source_api, confidence} or None if all fail.
    """
    if not raw or not raw.strip():
        return None

    api_keys = api_keys or {}
    raw = raw.strip()

    candidates = [
        ("semantic_scholar", lambda: _try_semantic_scholar(raw, api_keys, timeout, max_retries)),
        ("arxiv",            lambda: _try_arxiv(raw, timeout, max_retries)),
        ("openalex",         lambda: _try_openalex(raw, api_keys, timeout, max_retries)),
        ("crossref",         lambda: _try_crossref(raw, api_keys, timeout, max_retries)),
    ]

    for source_api, fn in candidates:
        _wait_rate_limit(source_api)
        try:
            hit = fn()
        except Exception:
            hit = None
        if hit:
            hit["source_api"] = source_api
            return hit
    return None


# ---------------------------------------------------------------------------
# Per-API clients
# ---------------------------------------------------------------------------
def _try_semantic_scholar(raw: str, api_keys: dict, timeout: float, max_retries: int) -> dict | None:
    """Search Semantic Scholar Graph API by raw query string."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    api_key = api_keys.get("semantic_scholar")
    if api_key:
        headers["x-api-key"] = api_key
    params = {
        "query": raw[:256],
        "limit": 1,
        "fields": "title,authors,year,venue,externalIds",
    }
    data = _get_with_retry(url, params=params, headers=headers, timeout=timeout, max_retries=max_retries)
    if not data or not data.get("data"):
        return None
    paper = data["data"][0]
    ext = paper.get("externalIds") or {}
    return {
        "doi": ext.get("DOI"),
        "title": paper.get("title"),
        "authors": [a.get("name") for a in (paper.get("authors") or [])][:5],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "confidence": 0.9,
    }


def _try_arxiv(raw: str, timeout: float, max_retries: int) -> dict | None:
    """Search arXiv via the Atom export API.

    Best for arXiv-style IDs ("arxiv:1706.03762") or strong title matches.
    """
    arxiv_id = _extract_arxiv_id(raw)
    if arxiv_id:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    else:
        query = raw[:200].replace(" ", "+")
        url = f"http://export.arxiv.org/api/query?search_query=ti:{query}&max_results=1"
    data = _get_with_retry(url, timeout=timeout, max_retries=max_retries, parser="xml")
    if not data:
        return None
    # The XML parser returns a dict with title/year/authors already extracted.
    return {
        "doi": None,
        "title": data.get("title"),
        "authors": data.get("authors", [])[:5],
        "year": data.get("year"),
        "venue": "arXiv",
        "arxiv_id": data.get("arxiv_id") or arxiv_id,
        "confidence": 0.85 if arxiv_id else 0.6,
    }


def _try_openalex(raw: str, api_keys: dict, timeout: float, max_retries: int) -> dict | None:
    """Search OpenAlex by title/author/raw query."""
    url = "https://api.openalex.org/works"
    headers = {}
    # mailto param is polite-pool requirement for OpenAlex
    params = {
        "search": raw[:256],
        "per-page": 1,
        "mailto": "ccchain@example.com",
    }
    data = _get_with_retry(url, params=params, headers=headers, timeout=timeout, max_retries=max_retries)
    if not data or not data.get("results"):
        return None
    work = data["results"][0]
    return {
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": work.get("title"),
        "authors": [a.get("author", {}).get("display_name") for a in (work.get("authorships") or [])][:5],
        "year": work.get("publication_year"),
        "venue": (work.get("primary_location", {}) or {}).get("source", {}).get("display_name") if work.get("primary_location") else None,
        "confidence": 0.75,
    }


def _try_crossref(raw: str, api_keys: dict, timeout: float, max_retries: int) -> dict | None:
    """Search CrossRef by raw bibliographic query."""
    url = "https://api.crossref.org/works"
    headers = {"User-Agent": "ccchain/0.3 (mailto:ccchain@example.com)"}
    params = {
        "query.bibliographic": raw[:256],
        "rows": 1,
    }
    data = _get_with_retry(url, params=params, headers=headers, timeout=timeout, max_retries=max_retries)
    if not data or not data.get("message", {}).get("items"):
        return None
    item = data["message"]["items"][0]
    return {
        "doi": item.get("DOI"),
        "title": (item.get("title") or [None])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in (item.get("author") or [])][:5],
        "year": (item.get("issued", {}).get("date-parts") or [[None]])[0][0],
        "venue": (item.get("container-title") or [None])[0],
        "confidence": 0.7,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_ARXIV_PATTERN = re.compile(r"(?:arxiv[:\s])?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


def _extract_arxiv_id(raw: str) -> str | None:
    m = _ARXIV_PATTERN.search(raw)
    return m.group(1) if m else None


def _wait_rate_limit(api: str) -> None:
    """Simple per-API throttle: sleep until min interval elapsed."""
    interval = _RATE_LIMIT_S.get(api, 0.5)
    last = _last_call.get(api, 0.0)
    elapsed = time.time() - last
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _last_call[api] = time.time()


def _get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 2.0,
    max_retries: int = 3,
    parser: str = "json",
) -> Any:
    """GET with exponential backoff. Returns parsed dict or None on failure."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                if parser == "json":
                    return resp.json()
                elif parser == "xml":
                    return _parse_arxiv_atom(resp.text)
            elif resp.status_code == 404:
                return None
            # 4xx/5xx: retry with backoff
        except requests.RequestException as e:
            last_exc = e
        time.sleep(0.5 * (2 ** attempt))
    return None


def _parse_arxiv_atom(xml_text: str) -> dict | None:
    """Parse arXiv Atom feed response. Returns first entry or None.

    Very lightweight regex-based parser; avoids xml.etree namespace headaches.
    """
    import re
    # Find first <entry>...</entry> block
    entry_match = re.search(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
    if not entry_match:
        return None
    entry = entry_match.group(1)

    def _first(pattern: str, text: str, group: int = 1) -> str | None:
        m = re.search(pattern, text)
        return m.group(group) if m else None

    title = _first(r"<title[^>]*>(.*?)</title>", entry)
    if title:
        title = re.sub(r"\s+", " ", title).strip()
    year_raw = _first(r"<published>(\d{4})", entry)
    arxiv_id_raw = _first(r'<id>(http://arxiv\.org/abs/)?([^<]+)</id>', entry, group=2)

    authors = re.findall(r"<name>(.*?)</name>", entry)

    return {
        "title": title,
        "year": int(year_raw) if year_raw else None,
        "authors": authors,
        "arxiv_id": arxiv_id_raw,
    }
