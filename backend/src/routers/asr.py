"""ASR router: speech-to-text endpoint."""

from fastapi import APIRouter, Depends, UploadFile

from src.dependencies import get_asr
from src.schemas import ASRResponse
from src.services.asr import ASRService

router = APIRouter()


@router.post("/asr", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile,
    asr: ASRService = Depends(get_asr),
) -> ASRResponse:
    """Transcribe uploaded audio to text."""
    audio_bytes = await file.read()
    text = await asr.transcribe(audio_bytes)
    return ASRResponse(text=text)
