"""
tts/tts_service.py

TTS provider abstraction (§10). The prototype's PRIMARY path is actually the
browser's own SpeechSynthesis API (zero latency, zero cost, no API key,
works offline) -- the frontend calls speak() locally. This backend module
exists for:
  1. Server-driven TTS when you later want a consistent voice/provider
     regardless of client, or need to synthesize into an audio file.
  2. A clean seam to swap in Azure/Google/OpenAI TTS without touching
     callers.

Usage:
    from backend.tts.tts_service import get_tts_provider
    provider = get_tts_provider()
    audio_bytes = provider.speak("I need help.")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str) -> bytes:
        """Return audio bytes (e.g. WAV/MP3) for the given text."""
        raise NotImplementedError


class BrowserTTSProvider(TTSProvider):
    """
    No-op server-side placeholder: signals to the frontend that it should
    use window.speechSynthesis locally instead of requesting server audio.
    This is the default for the prototype (§10: "Browser SpeechSynthesis
    for initial prototype").
    """

    def speak(self, text: str) -> bytes:
        raise NotImplementedError(
            "BrowserTTSProvider performs synthesis client-side. "
            "The /api/tts endpoint should return {'mode': 'browser', 'text': text} "
            "instead of calling this."
        )


class AzureTTSProvider(TTSProvider):
    def __init__(self):
        self.api_key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")
        if not self.api_key:
            raise RuntimeError("AZURE_SPEECH_KEY not set")

    def speak(self, text: str) -> bytes:
        # Real implementation would call the Azure Speech REST/SDK endpoint.
        # Left unimplemented here to avoid making network calls with fake
        # keys; wire this up once AZURE_SPEECH_KEY is configured.
        raise NotImplementedError("Azure TTS integration point -- add SDK call here.")


class GoogleTTSProvider(TTSProvider):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_TTS_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_TTS_API_KEY not set")

    def speak(self, text: str) -> bytes:
        raise NotImplementedError("Google Cloud TTS integration point -- add SDK call here.")


from backend.config import get_settings


def get_tts_provider() -> TTSProvider:
    provider = get_settings().tts_provider
    if provider == "browser":
        return BrowserTTSProvider()
    if provider == "azure":
        return AzureTTSProvider()
    if provider == "google":
        return GoogleTTSProvider()
    raise ValueError(f"Unknown TTS_PROVIDER: {provider}")

