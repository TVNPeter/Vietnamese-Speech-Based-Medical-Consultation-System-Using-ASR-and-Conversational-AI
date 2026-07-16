"""Trusted web fallback for medical RAG through Tavily."""

import logging
from urllib.parse import urlparse

import httpx

from src.config import settings
from src.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class TavilyMedicalSearch:
    """Fetch compact evidence only from an explicit medical-domain allowlist."""

    _SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self) -> None:
        self._cache: dict[str, list[RetrievedChunk]] = {}

    @property
    def is_enabled(self) -> bool:
        return settings.TAVILY_ENABLED and bool(settings.TAVILY_API_KEY.strip())

    async def search(self, query: str) -> list[RetrievedChunk]:
        """Return trusted Tavily excerpts, or an empty list on a safe failure."""
        normalized_query = " ".join(query.split())
        if not normalized_query or not self.is_enabled:
            return []
        if normalized_query in self._cache:
            return list(self._cache[normalized_query])

        domains = [
            domain.strip()
            for domain in settings.TAVILY_TRUSTED_DOMAINS.split(",")
            if domain.strip()
        ]
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": normalized_query,
            "search_depth": "advanced",
            "max_results": settings.TAVILY_MAX_RESULTS,
            "include_domains": domains,
            "include_answer": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=settings.TAVILY_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(self._SEARCH_URL, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as error:
            logger.warning("Tavily fallback failed: %s", error)
            return []

        chunks: list[RetrievedChunk] = []
        for position, result in enumerate(response.json().get("results", []), start=1):
            if not isinstance(result, dict):
                continue
            content = " ".join(str(result.get("content", "")).split())
            if len(content) < 80:
                continue
            url = str(result.get("url", "")).strip()
            domain = urlparse(url).netloc.casefold() or "trusted medical web"
            title = " ".join(str(result.get("title", "")).split())
            source = f"{domain} (Tavily)"
            if title:
                source = f"{source} — {title[:120]}"
            chunks.append(
                RetrievedChunk(
                    content=content,
                    source=source,
                    score=1.0 / position,
                    url=url or None,
                )
            )

        self._cache[normalized_query] = chunks
        logger.info("Tavily returned %d trusted evidence excerpts.", len(chunks))
        return list(chunks)
