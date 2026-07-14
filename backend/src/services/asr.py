"""ASR service: speech-to-text transcription."""

import logging

from src.config import settings

logger = logging.getLogger(__name__)


class ASRService:
    """Automatic Speech Recognition service."""

    def __init__(self) -> None:
        self._model_path = settings.resolve(settings.ASR_MODEL_PATH)
        self._model: object | None = None

    def ensure_model_loaded(self) -> None:
        """Verify the ASR model directory exists."""
        if not self._model_path.is_dir():
            raise FileNotFoundError(
                f"ASR model directory not found: {self._model_path}. "
                "Place your ASR model files in this directory."
            )

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        self.ensure_model_loaded()
        raise NotImplementedError(
            "ASR model not yet integrated. "
            f"Place model files in {self._model_path} and implement "
            "transcription."
        )
