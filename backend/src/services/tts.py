"""VieNeu-TTS v3 Turbo synthesis service."""

import asyncio
import logging
import tempfile
import threading
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Lazy, process-local VieNeu engine used by the speaker button."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> object:
        if not settings.TTS_ENABLED:
            raise RuntimeError("TTS is disabled. Set TTS_ENABLED=true to enable it.")
        if self._model is None:
            from vieneu import Vieneu

            logger.info("Loading VieNeu-TTS v3 Turbo on the first request.")
            self._model = Vieneu()
        return self._model

    async def synthesize(self, text: str) -> bytes:
        """Synthesize a response into a WAV byte stream without blocking FastAPI."""
        return await asyncio.to_thread(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> bytes:
        with self._lock:
            model = self._load_model()
            audio = model.infer(text=text, voice=settings.TTS_VOICE)  # type: ignore[attr-defined]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                output_path = Path(handle.name)
            try:
                model.save(audio, str(output_path))  # type: ignore[attr-defined]
                return output_path.read_bytes()
            finally:
                output_path.unlink(missing_ok=True)
