"""FastAPI application factory with lifespan management."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routers import asr, chat, tts
from src.services.asr import ASRService
from src.services.llm import LLMService
from src.services.rag import RAGService
from src.services.tts import TTSService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    logger.info("Starting services...")

    llm_service = LLMService()
    await llm_service.start()
    app.state.llm = llm_service

    rag_service = RAGService()
    await rag_service.initialize()
    app.state.rag = rag_service

    app.state.asr = ASRService()
    app.state.tts = TTSService()

    logger.info("All services started.")
    yield

    logger.info("Shutting down services...")
    await llm_service.stop()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Vietnamese Medical Consultation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(asr.router, prefix="/api")
app.include_router(tts.router, prefix="/api")
