"""Chat router: SSE streaming chat with RAG."""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

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


async def _stream_chat(
    request: ChatRequest,
    llm: LLMService,
    rag: RAGService,
) -> AsyncGenerator[str, None]:
    """Generate SSE events for a chat request."""
    chunks = rag.query(request.message)
    context = "\n\n".join(
        f"[{i + 1}] (Nguồn: {chunk.source})\n{chunk.content}"
        for i, chunk in enumerate(chunks)
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
                    }
                    for i, chunk in enumerate(chunks)
                ],
            },
        }
    )

    async for token in llm.generate_stream(user_message, SYSTEM_PROMPT):
        yield _sse_event({"type": "token", "content": token})

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
