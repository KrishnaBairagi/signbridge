"""
websocket/live_conversation.py

The core of the live, two-way conversation system (§12). One WebSocket
connection = one session with its own TemporalSignBuffer, so multiple
concurrent conversations don't interfere.

Message protocol (JSON over the WebSocket), client -> server:

  {"type": "start_session", "consent_to_store": false}
  {"type": "landmarks", "frame": {...}}                # from MediaPipe, ~30fps
  {"type": "speech_text", "text": "Where is the station?"}  # from Web Speech API
  {"type": "end_session"}

Server -> client:

  {"type": "recognition_update", "sign": "HELLO", "confidence": 0.82, "status": "confirmed"}
  {"type": "sentence", "text": "Hello, I need help.", "tokens": [...]}
  {"type": "speak", "text": "Hello, I need help."}         # tells client to call speechSynthesis
  {"type": "recipient_text", "text": "Sure, how can I help?"}
  {"type": "error", "code": "no_hands", "message": "..."}
"""

from __future__ import annotations

import uuid

from fastapi import WebSocket, WebSocketDisconnect

from backend.sign_language.temporal_buffer import TemporalSignBuffer
from backend.services.sentence_builder import build_sentence_rule_based
from backend.db import database

# How long to wait after a confirmed sign, with no new signs, before we
# treat the accumulated tokens as "end of utterance" and build a sentence.
UTTERANCE_GAP_SIGNS = 1  # simplistic: build+emit a sentence after every confirmed sign in this demo,
                         # since the demo vocab has no reliable "pause" detector beyond gesture end.
                         # A production model would use a longer inter-sign silence timer instead.


class LiveSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())
        self.buffer = TemporalSignBuffer()
        self.token_history: list[str] = []
        self.consent_to_store = False

    async def send(self, payload: dict):
        await self.websocket.send_json(payload)

    async def handle_landmarks(self, frame: dict):
        event = self.buffer.add_frame(frame)
        if event is None:
            return

        if event.status == "no_hands":
            await self.send({
                "type": "error", "code": "no_hands",
                "message": "No hands detected. Please position your hands inside the camera frame.",
            })
            return

        if event.status == "low_confidence":
            await self.send({
                "type": "recognition_update", "sign": None, "confidence": event.confidence,
                "status": "low_confidence",
                "message": "Not sure what that sign was. Please repeat it clearly.",
            })
            return

        await self.send({
            "type": "recognition_update", "sign": event.sign,
            "confidence": event.confidence, "status": event.status,
        })

        if event.status != "confirmed":
            return  # only build sentences from confirmed signs

        self.token_history.append(event.sign)
        database.log_message(self.session_id, "sign_to_speech", event.sign, event.confidence)

        result = build_sentence_rule_based(self.token_history)
        await self.send({"type": "sentence", "text": result.text, "tokens": list(self.token_history)})
        await self.send({"type": "speak", "text": result.text})

        # In this simple demo policy, clear history after each emitted sentence
        # so consecutive signs don't endlessly re-accumulate. A production
        # system would use a smarter end-of-utterance detector (e.g. a pause
        # threshold or an explicit "done signing" gesture).
        self.token_history = []

    async def handle_speech_text(self, text: str):
        database.log_message(self.session_id, "speech_to_text", text)
        await self.send({"type": "recipient_text", "text": text})


from backend.config import get_settings
from backend.logging_config import logger


async def live_conversation_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = LiveSession(websocket)

    try:
        while True:
            try:
                message = await websocket.receive_json()
            except (ValueError, TypeError, Exception) as parse_err:
                if isinstance(parse_err, WebSocketDisconnect):
                    raise parse_err
                await session.send({
                    "type": "error", "code": "bad_request",
                    "message": "Invalid JSON format or malformed WebSocket payload."
                })
                continue

            if not isinstance(message, dict):
                await session.send({
                    "type": "error", "code": "bad_request",
                    "message": "WebSocket JSON payload must be an object."
                })
                continue

            msg_type = message.get("type")

            if msg_type == "start_session":
                session.consent_to_store = bool(message.get("consent_to_store", False))
                database.create_session(session.session_id, session.consent_to_store)
                demo_mode = get_settings().demo_mode
                model_tag = "demo" if demo_mode else "production"
                logger.info(f"WebSocket session started: {session.session_id} | model: {model_tag}")
                await session.send({"type": "session_started", "session_id": session.session_id,
                                     "model": model_tag})

            elif msg_type == "landmarks":
                frame_data = message.get("frame")
                if not isinstance(frame_data, dict):
                    frame_data = {}
                await session.handle_landmarks(frame_data)

            elif msg_type == "speech_text":
                text_val = str(message.get("text", ""))
                await session.handle_speech_text(text_val)

            elif msg_type == "end_session":
                logger.info(f"WebSocket session ended gracefully: {session.session_id}")
                await session.send({"type": "session_ended"})
                break

            else:
                await session.send({"type": "error", "code": "unknown_message_type",
                                     "message": f"Unrecognized message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session.session_id}")

