"""ASR router: speech-to-text endpoint."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile

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
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Audio upload is empty.")
    try:
        text = await asr.transcribe(audio_bytes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ASRResponse(text=text)
