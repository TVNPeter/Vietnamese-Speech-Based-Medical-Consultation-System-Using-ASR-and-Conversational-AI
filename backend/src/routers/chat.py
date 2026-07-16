"""Chat router: SSE streaming chat with RAG."""

import json
import logging
import re
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.config import settings
from src.dependencies import get_llm, get_rag
from src.prompts import RAG_USER_TEMPLATE, SYSTEM_PROMPT
from src.schemas import ChatRequest, RAGReindexResponse
from src.services.llm import LLMService
from src.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE event."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _clean_answer(text: str) -> str:
    """Keep model output concise without relying on topic-specific rules."""
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    cleaned = re.sub(
        r"^(?:\*{2})?câu trả lời(?:\*{2})?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if len(cleaned) <= settings.LLM_MAX_RESPONSE_CHARS:
        return cleaned

    limit = settings.LLM_MAX_RESPONSE_CHARS
    sentence_end = max(
        cleaned.rfind(marker, 0, limit) for marker in (".", "!", "?")
    )
    if sentence_end >= limit // 2:
        return cleaned[: sentence_end + 1].rstrip()
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip() + "…"


def _needs_extractive_fallback(answer: str, context: str, question: str) -> bool:
    """Reject process narration and unsupported named entities generically."""
    normalized_answer = answer.casefold()
    process_markers = (
        "tôi sẽ",
        "phân tích",
        "đoạn [",
        "tài liệu được cung cấp",
        "reference material",
        "i will",
    )
    if any(marker in normalized_answer for marker in process_markers):
        return True

    normalized_context = context.casefold()
    named_terms = re.findall(r"\b[A-ZÀ-Ỹ][a-zà-ỹ]{6,}\b", answer)
    if any(term.casefold() not in normalized_context for term in named_terms):
        return True

    question_entities = {
        term.casefold()
        for term in re.findall(r"\b[\wÀ-ỹ]{6,}\b", question)
    }
    return len(question_entities) == 1 and any(
        term.casefold() not in question_entities for term in named_terms
    )


def _extractive_fallback(question: str, chunks: list) -> str:
    """Return the most focused local evidence when generation is ungrounded."""
    query_terms = {
        token.casefold()
        for token in re.findall(r"[\wÀ-ỹ]+", question)
        if len(token) >= 4
    }
    best: tuple[int, int, str] | None = None
    for source_index, chunk in enumerate(chunks, start=1):
        for fragment in re.split(r"[\n*]+", chunk.content):
            candidate = " ".join(fragment.split()).strip(" :;,-")
            if not 30 <= len(candidate) <= 300:
                continue
            candidate = re.sub(r"\s+(?:và|hoặc|nếu|khi)$", "", candidate)
            candidate_terms = {
                token.casefold()
                for token in re.findall(r"[\wÀ-ỹ]+", candidate)
            }
            overlap = len(query_terms & candidate_terms)
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, source_index, candidate)

    if best is None:
        return (
            "Không đủ bằng chứng đáng tin cậy trong các nguồn được truy xuất để "
            "trả lời an toàn."
        )
    _, source_index, candidate = best
    return (
        f"- {candidate}. [{source_index}]\n\n"
        "Thông tin chỉ mang tính tham khảo; hãy trao đổi với bác sĩ hoặc dược sĩ "
        "trước khi thay đổi điều trị."
    )


async def _stream_chat(
    request: ChatRequest,
    llm: LLMService,
    rag: RAGService,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for a chat request."""
    chunks = await rag.query_with_fallback(request.message)
    context_chunks = chunks[: settings.RAG_CONTEXT_MAX_CHUNKS]
    context = "\n\n".join(
        f"[{i + 1}] (Nguồn: {chunk.source})\n"
        f"{' '.join(chunk.content.split())[:settings.RAG_CONTEXT_CHARS_PER_CHUNK]}"
        for i, chunk in enumerate(context_chunks)
    )
    user_message = RAG_USER_TEMPLATE.format(
        context=context if context else "Không có tài liệu tham khảo.",
        question=request.message,
    )

    yield _sse_event(
        {
            "type": "metadata",
            "data": {
                "sources": [
                    {
                        "index": i + 1,
                        "title": chunk.source,
                        "content": chunk.content[:200],
                        "score": chunk.score,
                        "url": chunk.url,
                    }
                    for i, chunk in enumerate(context_chunks)
                ],
            },
        }
    )

    answer_parts = []
    async for token in llm.generate_stream(user_message, SYSTEM_PROMPT):
        answer_parts.append(token)
    answer = _clean_answer("".join(answer_parts))
    if _needs_extractive_fallback(answer, context, request.message):
        logger.warning("Rejected ungrounded LLM output; returning extractive RAG answer.")
        answer = _extractive_fallback(request.message, context_chunks)
    if answer:
        yield _sse_event({"type": "token", "content": answer})

    yield _sse_event({"type": "done", "content": ""})


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm: LLMService = Depends(get_llm),
    rag: RAGService = Depends(get_rag),
) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""
    return StreamingResponse(
        _stream_chat(request, llm, rag),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stats")
async def get_stats(rag: RAGService = Depends(get_rag)) -> dict:
    """Return index statistics."""
    return {
        "stats": {
            "document_count": rag.document_count,
            "index_loaded": rag.document_count > 0,
        }
    }


@router.post("/rag/reindex", response_model=RAGReindexResponse)
async def reindex_documents(
    rag: RAGService = Depends(get_rag),
) -> RAGReindexResponse:
    """Reload the active prebuilt index or rebuild the configured FAISS index."""
    document_count, chunk_count = await rag.rebuild()
    return RAGReindexResponse(
        document_count=document_count,
        chunk_count=chunk_count,
    )
