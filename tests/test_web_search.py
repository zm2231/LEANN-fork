"""
Tests for the Exa + SearXNG parallel WebSearcher.
"""
from unittest.mock import MagicMock, patch

import pytest

from leann.web_search import WebSearcher


def _exa_result(url, title="Exa result", text="exa snippet"):
    return {"title": title, "link": url, "snippet": text, "_source": "exa"}


def _searxng_result(url, title="SearXNG result", snippet="searxng snippet"):
    return {"title": title, "link": url, "snippet": snippet, "_source": "searxng"}


# ── Init / availability ───────────────────────────────────────────────


def test_available_with_exa_only():
    ws = WebSearcher(exa_api_key="test_key")
    assert ws.available is True


def test_available_with_searxng_only():
    ws = WebSearcher(searxng_url="http://127.0.0.1:8888")
    assert ws.available is True


def test_available_with_both():
    ws = WebSearcher(exa_api_key="k", searxng_url="http://127.0.0.1:8888")
    assert ws.available is True


def test_not_available_with_neither():
    ws = WebSearcher()
    assert ws.available is False


def test_legacy_api_key_compat():
    """api_key kwarg (Serper legacy) should not crash and available stays False."""
    ws = WebSearcher(api_key="ignored")
    assert ws.available is False


# ── Search: parallel merge + dedup ───────────────────────────────────


def test_both_providers_called_and_merged():
    ws = WebSearcher(exa_api_key="k", searxng_url="http://x")
    ws._exa_search = MagicMock(return_value=[_exa_result("https://a.com"), _exa_result("https://b.com")])
    ws._searxng_search = MagicMock(return_value=[_searxng_result("https://c.com"), _searxng_result("https://b.com")])

    results = ws.search("test query", top_k=10)

    ws._exa_search.assert_called_once()
    ws._searxng_search.assert_called_once()
    urls = [r["link"] for r in results]
    assert "https://a.com" in urls
    assert "https://b.com" in urls
    assert "https://c.com" in urls
    # b.com appears in both but should be deduplicated
    assert urls.count("https://b.com") == 1


def test_exa_results_come_before_searxng():
    """Exa results should appear first in merged output (semantic quality priority)."""
    ws = WebSearcher(exa_api_key="k", searxng_url="http://x")
    ws._exa_search = MagicMock(return_value=[_exa_result("https://exa-first.com")])
    ws._searxng_search = MagicMock(return_value=[_searxng_result("https://searxng-second.com")])

    results = ws.search("test", top_k=10)
    assert results[0]["link"] == "https://exa-first.com"
    assert results[1]["link"] == "https://searxng-second.com"


def test_top_k_caps_results():
    ws = WebSearcher(exa_api_key="k", searxng_url="http://x")
    ws._exa_search = MagicMock(return_value=[_exa_result(f"https://{i}.com") for i in range(5)])
    ws._searxng_search = MagicMock(return_value=[_searxng_result(f"https://s{i}.com") for i in range(5)])

    results = ws.search("test", top_k=3)
    assert len(results) == 3


def test_exa_only_search():
    ws = WebSearcher(exa_api_key="k")
    ws._exa_search = MagicMock(return_value=[_exa_result("https://a.com")])

    results = ws.search("test")
    ws._exa_search.assert_called_once()
    assert results[0]["link"] == "https://a.com"


def test_searxng_only_search():
    ws = WebSearcher(searxng_url="http://x")
    ws._searxng_search = MagicMock(return_value=[_searxng_result("https://b.com")])

    results = ws.search("test")
    ws._searxng_search.assert_called_once()
    assert results[0]["link"] == "https://b.com"


def test_unavailable_returns_error_result():
    ws = WebSearcher()
    results = ws.search("anything")
    assert len(results) == 1
    assert "not configured" in results[0]["snippet"].lower()


def test_exa_failure_falls_back_to_searxng():
    ws = WebSearcher(exa_api_key="k", searxng_url="http://x")
    ws._exa_search = MagicMock(return_value=[])  # Exa fails silently → returns []
    ws._searxng_search = MagicMock(return_value=[_searxng_result("https://fallback.com")])

    results = ws.search("test")
    assert any(r["link"] == "https://fallback.com" for r in results)


# ── Page content ──────────────────────────────────────────────────────


def test_get_page_content_exa_primary():
    ws = WebSearcher(exa_api_key="k")
    with patch("leann.web_search.requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "results": [{"text": "Full page text from Exa"}]
        }
        content = ws.get_page_content("https://example.com")
    assert content == "Full page text from Exa"


def test_get_page_content_falls_back_to_jina():
    ws = WebSearcher(exa_api_key="k")
    with patch("leann.web_search.requests.post") as mock_post, \
         patch("leann.web_search.requests.get") as mock_get:
        # Exa returns empty text
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"results": [{"text": ""}]}
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.text = "Jina markdown content"

        content = ws.get_page_content("https://example.com")
    assert content == "Jina markdown content"


def test_get_page_content_exa_exception_falls_back_to_jina():
    """If Exa /contents raises, Jina fallback is used."""
    ws = WebSearcher(exa_api_key="k")
    with patch("leann.web_search.requests.post", side_effect=Exception("exa down")), \
         patch("leann.web_search.requests.get") as mock_get:
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.text = "Jina fallback"
        content = ws.get_page_content("https://example.com")
    assert content == "Jina fallback"


def test_get_page_content_both_fail_returns_error_string():
    """If both Exa and Jina fail, an error string is returned (no raise)."""
    ws = WebSearcher(exa_api_key="k")
    with patch("leann.web_search.requests.post", side_effect=Exception("exa down")), \
         patch("leann.web_search.requests.get", side_effect=Exception("jina down")):
        content = ws.get_page_content("https://example.com")
    assert "Error" in content or "error" in content.lower()


def test_future_exception_isolated_other_provider_still_returns():
    """If one provider's future raises, the other's results are still returned."""
    ws = WebSearcher(exa_api_key="k", searxng_url="http://x")

    def bad_exa(*_):
        raise RuntimeError("network error")

    ws._exa_search = bad_exa
    ws._searxng_search = MagicMock(return_value=[_searxng_result("https://ok.com")])

    results = ws.search("test")
    assert any(r["link"] == "https://ok.com" for r in results)


def test_searxng_base_url_env_fallback(monkeypatch):
    """SEARXNG_BASE_URL env var should be accepted as the SearXNG URL."""
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://127.0.0.1:9999")
    ws = WebSearcher()
    assert ws.searxng_url == "http://127.0.0.1:9999"
    assert ws.available is True
