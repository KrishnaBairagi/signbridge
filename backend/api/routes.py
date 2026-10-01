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


@router.get("/model/info")
def get_model_info():
    from backend.config import get_settings
    settings = get_settings()
    demo_mode = settings.demo_mode
    if demo_mode:
        return {
            "model_type": "demo",
            "name": "HeuristicSignClassifier",
            "class_count": 21,
            "sequence_len": settings.max_seq_len,
            "features_per_frame": 228,
        }
    else:
        return {
            "model_type": "production",
            "name": "BiLSTM 61-Class ONNX Model",
            "class_count": 61,
            "sequence_len": settings.max_seq_len,
            "features_per_frame": 228,
            "model_path": settings.onnx_model_path,
            "labels_path": settings.labels_path,
        }


@router.get("/sessions/{session_id}")
def get_session_details(session_id: str):
    from backend.db import database
    with database.get_connection() as conn:
        session = conn.execute(
            "SELECT id, started_at, consent_to_store, vocabulary_mode FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        session_dict = dict(session)
        if not session_dict["consent_to_store"]:
            return {
                "session": session_dict,
                "messages": [],
                "note": "Session transcript storage was not consented by user."
            }

        messages = conn.execute(
            "SELECT id, direction, content, confidence, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return {
            "session": session_dict,
            "messages": [dict(m) for m in messages],
        }
