"""TTS router: text-to-speech endpoint."""

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from src.config import settings
from src.dependencies import get_tts
from src.schemas import TTSRequest
from src.services.tts import TTSService

router = APIRouter()


@router.post("/tts")
async def synthesize_speech(
    request: TTSRequest,
    tts: TTSService = Depends(get_tts),
) -> Response:
    """Synthesize text into speech audio."""
    audio_bytes = await tts.synthesize(request.text)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": (
                f"inline; filename={settings.TTS_OUTPUT_FILENAME}"
            )
        },
    )
