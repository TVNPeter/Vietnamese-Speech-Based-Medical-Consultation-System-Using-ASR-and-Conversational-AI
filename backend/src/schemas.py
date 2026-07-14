"""Request and response schemas."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat message from the user."""

    message: str


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the RAG index."""

    content: str
    source: str
    score: float


class ASRResponse(BaseModel):
    """Transcription result from ASR."""

    text: str


class TTSRequest(BaseModel):
    """Text to synthesize into speech."""

    text: str
