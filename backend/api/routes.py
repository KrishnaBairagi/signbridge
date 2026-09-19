"""
api/routes.py

REST endpoints per §17. These are convenience/testing endpoints -- the LIVE
conversation flow itself runs over the WebSocket (backend/websocket/
live_conversation.py) so recognition is properly temporal, not per-request.
Use these REST endpoints for: unit testing the recognizer/sentence builder
in isolation, or non-live batch use cases.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.sign_language.recognizer import recognize_sign
from backend.services.sentence_builder import build_sentence_rule_based
from backend.services.ai_conversation_service import get_ai_conversation_service
from backend.tts.tts_service import get_tts_provider, BrowserTTSProvider

router = APIRouter(prefix="/api")


class LandmarkFrame(BaseModel):
    left_hand: list[list[float]] | None = None
    right_hand: list[list[float]] | None = None
    pose: list[list[float]] | None = None
    timestamp_ms: int


class PredictRequest(BaseModel):
    sequence: list[LandmarkFrame] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    sign: str | None
    confidence: float
    model: str


@router.post("/sign/predict", response_model=PredictResponse)
def predict_sign(req: PredictRequest):
    sequence = [frame.model_dump() for frame in req.sequence]
    result = recognize_sign(sequence)
    return PredictResponse(**result)


class TranslateRequest(BaseModel):
    tokens: list[str] = Field(..., min_length=1)


class TranslateResponse(BaseModel):
    text: str
    method: str


@router.post("/translate", response_model=TranslateResponse)
def translate_tokens(req: TranslateRequest):
    result = build_sentence_rule_based(req.tokens)
    return TranslateResponse(text=result.text, method=result.method)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@router.post("/tts")
def synthesize_speech(req: TTSRequest):
    provider = get_tts_provider()
    if isinstance(provider, BrowserTTSProvider):
        # Tell the frontend to speak it locally instead of streaming audio back.
        return {"mode": "browser", "text": req.text}
    try:
        audio_bytes = provider.speak(req.text)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    from fastapi.responses import Response
    return Response(content=audio_bytes, media_type="audio/mpeg")
