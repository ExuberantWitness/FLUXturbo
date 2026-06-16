"""Test citation resolver (I3) with mocked HTTP."""

from unittest.mock import patch, MagicMock

import pytest

from ccchain.core import references as ref_mod
from ccchain.core.references import resolve_citation, _extract_arxiv_id


def _mock_response(json_data=None, status=200, text=""):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data or {}
    m.text = text
    return m


@patch("ccchain.core.references.requests.get")
def test_resolve_via_semantic_scholar(mock_get):
    """Semantic Scholar hit returns immediately."""
    mock_get.return_value = _mock_response({
        "data": [{
            "title": "Attention Is All You Need",
            "authors": [{"name": "A. Vaswani"}, {"name": "N. Shazeer"}],
            "year": 2017,
            "venue": "NeurIPS",
            "externalIds": {"DOI": "10.5555/3295222.3295349"},
        }]
    })
    result = resolve_citation("Vaswani et al. 2017 Attention is all you need",
                              api_keys={"semantic_scholar": "test_key"},
                              timeout=0.1, max_retries=1)
    assert result is not None
    assert result["title"] == "Attention Is All You Need"
    assert result["year"] == 2017
    assert result["doi"] == "10.5555/3295222.3295349"
    assert result["source_api"] == "semantic_scholar"
    assert result["confidence"] >= 0.8


@patch("ccchain.core.references.requests.get")
def test_resolve_via_arxiv_with_id(mock_get):
    """Direct arXiv ID lookup."""
    arxiv_xml = """<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/1706.03762v5</id>
        <title>Attention Is All You Need</title>
        <published>2017-06-12T17:57:34Z</published>
        <author><name>Ashish Vaswani</name></author>
    </entry>
    </feed>"""
    mock_get.return_value = _mock_response(text=arxiv_xml)
    # First API (Semantic Scholar) returns nothing, second (arXiv) hits.
    # But for simplicity, patch to return arxiv xml directly on first call:
    def side(*args, **kwargs):
        return _mock_response(text=arxiv_xml)
    mock_get.side_effect = side

    # Disable sem scholar by making it return empty
    def smart_side(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response({"data": []})
        return _mock_response(text=arxiv_xml)
    mock_get.side_effect = smart_side

    result = resolve_citation("arxiv:1706.03762",
                              timeout=0.1, max_retries=1)
    assert result is not None
    assert result["title"] == "Attention Is All You Need"
    assert result["venue"] == "arXiv"
    assert result["source_api"] == "arxiv"


@patch("ccchain.core.references.requests.get")
def test_resolve_returns_none_when_all_fail(mock_get):
    """Garbage citation → all APIs miss → None."""
    def side(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response({"data": []})
        if "arxiv" in url:
            return _mock_response(text="<feed></feed>")
        if "openalex" in url:
            return _mock_response({"results": []})
        if "crossref" in url:
            return _mock_response({"message": {"items": []}})
        return _mock_response(status=404)
    mock_get.side_effect = side

    result = resolve_citation("Smith et al., definitely not real, 2099",
                              timeout=0.1, max_retries=1)
    assert result is None


def test_resolve_empty_string_returns_none():
    assert resolve_citation("") is None
    assert resolve_citation("   ") is None


def test_extract_arxiv_id():
    assert _extract_arxiv_id("arxiv:1706.03762") == "1706.03762"
    assert _extract_arxiv_id("see arXiv:2301.12345v2 for details") == "2301.12345v2"
    assert _extract_arxiv_id("not an arxiv ref") is None


@patch("ccchain.core.references.requests.get")
def test_resolve_handles_network_error(mock_get):
    """Network errors during a call fall through to next API."""
    import requests as req_lib
    arxiv_xml = """<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2301.99999v1</id>
        <title>Found via arxiv fallback</title>
        <published>2023-01-15T00:00:00Z</published>
        <author><name>Test Author</name></author>
    </entry>
    </feed>"""

    def side(url, **kwargs):
        if "semanticscholar" in url:
            raise req_lib.RequestException("network down")
        if "arxiv" in url:
            return _mock_response(text=arxiv_xml)
        return _mock_response(status=404)
    mock_get.side_effect = side

    result = resolve_citation("some paper", timeout=0.1, max_retries=1)
    # Semantic Scholar fails → arxiv tries → returns hit
    assert result is not None
    assert result["source_api"] == "arxiv"
    assert result["title"] == "Found via arxiv fallback"


@patch("ccchain.core.references.requests.get")
def test_resolve_crossref_fallback(mock_get):
    """If first 3 APIs miss, crossref resolves."""
    def side(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response({"data": []})
        if "arxiv" in url:
            return _mock_response(text="<feed></feed>")
        if "openalex" in url:
            return _mock_response({"results": []})
        if "crossref" in url:
            return _mock_response({
                "message": {
                    "items": [{
                        "DOI": "10.1000/test",
                        "title": ["Found in CrossRef"],
                        "author": [{"given": "Test", "family": "Author"}],
                        "issued": {"date-parts": [[2021]]},
                        "container-title": ["Test Journal"],
                    }]
                }
            })
        return _mock_response(status=404)
    mock_get.side_effect = side

    result = resolve_citation("some obscure paper", timeout=0.1, max_retries=1)
    assert result is not None
    assert result["title"] == "Found in CrossRef"
    assert result["doi"] == "10.1000/test"
    assert result["source_api"] == "crossref"
