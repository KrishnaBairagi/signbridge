"""
speech/stt_service.py

Speech-to-text abstraction (§11). Like TTS, the prototype's primary path is
the browser's native SpeechRecognition (Web Speech API) running client-side
for near-real-time, no-API-key transcription of the hearing person's voice.
This module is the seam for a server-side Whisper/Azure/Google fallback
(e.g. for browsers without SpeechRecognition support, like Firefox).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError


class BrowserSTTProvider(STTProvider):
    """Client-side Web Speech API handles this; server is a pass-through."""

    def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError("Transcription happens client-side via Web Speech API.")


class WhisperSTTProvider(STTProvider):
    """
    Server-side fallback using OpenAI Whisper (local or API). Requires
    `openai-whisper` or the OpenAI API depending on WHISPER_MODE.
    """

    def __init__(self):
        self.mode = os.getenv("WHISPER_MODE", "api")  # "api" | "local"
        self.api_key = os.getenv("OPENAI_API_KEY")

    def transcribe(self, audio_bytes: bytes) -> str:
        if self.mode == "api" and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set for Whisper API mode")
        # Real implementation: POST audio_bytes to the Whisper endpoint, or
        # run the local whisper model. Left as an integration point.
        raise NotImplementedError("Whisper integration point -- add SDK/API call here.")


from backend.config import get_settings


def get_stt_provider() -> STTProvider:
    provider = get_settings().stt_provider
    if provider == "browser":
        return BrowserSTTProvider()
    if provider == "whisper":
        return WhisperSTTProvider()
    raise ValueError(f"Unknown STT_PROVIDER: {provider}")

