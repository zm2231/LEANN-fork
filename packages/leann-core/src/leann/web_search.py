"""
Web search backed by Exa (neural/semantic) + SearXNG (meta-search) in parallel.

Both providers run concurrently; results are deduplicated by URL and the merged
list is returned, capped at top_k.  Either provider may be absent:
  - Exa absent: falls back to SearXNG-only
  - SearXNG absent: falls back to Exa-only
  - Both absent: web search disabled (warns at init)

Env vars:
  EXA_API_KEY       — Exa neural search (https://exa.ai)
  SEARXNG_URL       — SearXNG base URL, e.g. http://127.0.0.1:8888
  JINA_API_KEY      — optional, improves Jina page-fetch rate limits
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

logger = logging.getLogger(__name__)

_EXA_SEARCH_URL = "https://api.exa.ai/search"
_EXA_CONTENTS_URL = "https://api.exa.ai/contents"
_JINA_READER_URL = "https://r.jina.ai/"


class WebSearcher:
    def __init__(
        self,
        exa_api_key: str | None = None,
        searxng_url: str | None = None,
        jina_api_key: str | None = None,
        # Legacy Serper arg ignored — kept for call-site compat
        api_key: str | None = None,
    ):
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        self.searxng_url = (
            searxng_url
            or os.getenv("SEARXNG_URL")
            or os.getenv("SEARXNG_BASE_URL")
        )
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")

        if not self.exa_api_key and not self.searxng_url:
            logger.warning(
                "Neither EXA_API_KEY nor SEARXNG_URL is set. Web search disabled."
            )

        # Legacy compat: react_agent.py checks bool(web_searcher.api_key)
        self.api_key = self.exa_api_key or self.searxng_url

    @property
    def available(self) -> bool:
        return bool(self.exa_api_key or self.searxng_url)

    # ── Exa ─────────────────────────────────────────────────────────────

    def _exa_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            resp = requests.post(
                _EXA_SEARCH_URL,
                headers={"x-api-key": self.exa_api_key, "Content-Type": "application/json"},
                json={"query": query, "numResults": top_k, "useAutoprompt": True, "type": "neural"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "snippet": r.get("text", r.get("highlights", [""])[0] if r.get("highlights") else ""),
                    "_source": "exa",
                }
                for r in data.get("results", [])
            ]
        except Exception as exc:
            logger.error("Exa search failed: %s", exc)
            return []

    # ── SearXNG ─────────────────────────────────────────────────────────

    def _searxng_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "_source": "searxng",
                }
                for r in data.get("results", [])[:top_k]
            ]
        except Exception as exc:
            logger.error("SearXNG search failed: %s", exc)
            return []

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.available:
            return [{"title": "Error", "link": "", "snippet": "Web search not configured."}]

        futures: dict = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if self.exa_api_key:
                futures["exa"] = pool.submit(self._exa_search, query, top_k)
            if self.searxng_url:
                futures["searxng"] = pool.submit(self._searxng_search, query, top_k)

        # Merge, dedup by URL, preserve order (Exa first for semantic quality)
        seen_urls: set[str] = set()
        merged: list[dict[str, Any]] = []
        for key in ("exa", "searxng"):
            if key in futures:
                try:
                    items = futures[key].result()
                except Exception as exc:
                    logger.error("%s search raised an exception: %s", key, exc)
                    items = []
                for item in items:
                    url = item.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        merged.append(item)

        return merged[:top_k]

    def get_page_content(self, url: str) -> str:
        """Fetch full page content. Tries Exa /contents first, falls back to Jina."""
        if self.exa_api_key:
            try:
                resp = requests.post(
                    _EXA_CONTENTS_URL,
                    headers={"x-api-key": self.exa_api_key, "Content-Type": "application/json"},
                    json={"ids": [url], "text": True},
                    timeout=15,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results and results[0].get("text"):
                    return results[0]["text"]
            except Exception as exc:
                logger.warning("Exa content fetch failed for %s: %s — falling back to Jina", url, exc)

        # Jina fallback
        jina_url = f"{_JINA_READER_URL}{url}"
        headers: dict[str, str] = {"X-Return-Format": "markdown"}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"
        try:
            resp = requests.get(jina_url, headers=headers, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.error("Jina fallback failed for %s: %s", url, exc)
            return f"Error fetching content: {exc!s}"
