"""TTS service: text-to-speech synthesis."""

import logging

from src.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech synthesis service."""

    def __init__(self) -> None:
        self._model_path = settings.resolve(settings.TTS_MODEL_PATH)
        self._model: object | None = None

    def ensure_model_loaded(self) -> None:
        """Verify the TTS model directory exists."""
        if not self._model_path.is_dir():
            raise FileNotFoundError(
                f"TTS model directory not found: {self._model_path}. "
                "Place your TTS model files in this directory."
            )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into audio bytes (WAV format)."""
        self.ensure_model_loaded()
        raise NotImplementedError(
            "TTS model not yet integrated. "
            f"Place model files in {self._model_path} and implement "
            "synthesis."
        )
