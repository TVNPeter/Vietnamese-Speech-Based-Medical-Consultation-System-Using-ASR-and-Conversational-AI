"""Regression tests for Vietnamese VieNeu-TTS handling."""

import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from src.config import settings
from src.schemas import TTSRequest
from src.services.tts import TTSService


class _FakeVieNeu:
    """Small in-memory substitute that avoids loading model weights in tests."""

    def __init__(self) -> None:
        self.text: str | None = None
        self.voice: str | None = None

    def get_preset_voice(self, name: str) -> str:
        return name

    def infer(self, *, text: str, voice: str) -> bytes:
        self.text = text
        self.voice = voice
        return b"synthetic audio"

    def save(self, audio: bytes, output_path: str) -> None:
        Path(output_path).write_bytes(b"RIFF" + audio)


class VieNeuTTSServiceTests(unittest.TestCase):
    def test_synthesis_preserves_vietnamese_unicode(self) -> None:
        service = TTSService()
        model = _FakeVieNeu()
        service._model = model
        text = "L-cystine giúp tóc khỏe; giữ nguyên tiếng Việt: Trúc Ly."

        audio = asyncio.run(service.synthesize(text))

        self.assertEqual(model.text, text)
        self.assertEqual(model.voice, "Trúc Ly")
        self.assertEqual(audio, b"RIFFsynthetic audio")

    def test_endpoint_uses_standard_utf8_test_filename(self) -> None:
        service = TTSService()
        service._model = _FakeVieNeu()
        dependencies = ModuleType("src.dependencies")
        dependencies.get_tts = lambda: None  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"src.dependencies": dependencies}):
            sys.modules.pop("src.routers.tts", None)
            router = importlib.import_module("src.routers.tts")
            response = asyncio.run(
                router.synthesize_speech(TTSRequest(text="Kiểm tra UTF-8"), service)
            )
        sys.modules.pop("src.routers.tts", None)

        self.assertEqual(
            response.headers["content-disposition"],
            "inline; filename=vieneu-tts-l-cystine-utf8.wav",
        )
        self.assertEqual(
            settings.TTS_OUTPUT_FILENAME, "vieneu-tts-l-cystine-utf8.wav"
        )
