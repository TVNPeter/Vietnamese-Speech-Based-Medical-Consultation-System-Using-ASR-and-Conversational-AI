"""FastAPI dependency injection helpers."""

from fastapi import Request

from src.services.asr import ASRService
from src.services.llm import LLMService
from src.services.rag import RAGService
from src.services.tts import TTSService


def get_llm(request: Request) -> LLMService:
    """Retrieve the LLM service from application state."""
    return request.app.state.llm


def get_rag(request: Request) -> RAGService:
    """Retrieve the RAG service from application state."""
    return request.app.state.rag


def get_asr(request: Request) -> ASRService:
    """Retrieve the ASR service from application state."""
    return request.app.state.asr


def get_tts(request: Request) -> TTSService:
    """Retrieve the TTS service from application state."""
    return request.app.state.tts
