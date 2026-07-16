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

    @staticmethod
    def _domains(value: str) -> list[str]:
        return [domain.strip() for domain in value.split(",") if domain.strip()]

    async def _request(self, query: str, domains: list[str]) -> list[dict]:
        """Ask Tavily for results limited to an explicit domain allowlist."""
        if not domains:
            return []
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
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
        return [
            result
            for result in response.json().get("results", [])
            if isinstance(result, dict)
        ]

    async def search(self, query: str) -> list[RetrievedChunk]:
        """Return trusted Tavily excerpts, or an empty list on a safe failure."""
        normalized_query = " ".join(query.split())
        if not normalized_query or not self.is_enabled:
            return []
        if normalized_query in self._cache:
            return list(self._cache[normalized_query])

        domains = self._domains(settings.TAVILY_TRUSTED_DOMAINS)
        preferred_domains = [
            domain
            for domain in self._domains(settings.TAVILY_PREFERRED_DOMAINS)
            if domain in domains
        ]
        # Prefer Vietnamese clinical sources. If none has a relevant page, use
        # the broader allowlist rather than returning an unsupported answer.
        results = await self._request(normalized_query, preferred_domains)
        if not results:
            results = await self._request(normalized_query, domains)

        chunks: list[RetrievedChunk] = []
        for position, result in enumerate(results, start=1):
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
