"""
services/ai_conversation_service.py

§22: a provider-agnostic abstraction so Gemini Live / OpenAI Realtime /
Azure AI / Google Cloud AI can be swapped without touching call sites.
The sign recognition model NEVER depends on this -- it's used only for
optional sentence polishing (services/sentence_builder.py) and, later,
free-form conversational assistance if you choose to add it.

Disabled by default (LLM_PROVIDER=none) so the prototype runs with zero
API keys and zero hallucination risk out of the box.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class AIConversationService(ABC):
    @abstractmethod
    def process_text(self, text: str) -> str: ...

    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str: ...

    def stream_audio(self, text: str):
        raise NotImplementedError("Streaming audio not implemented for this provider.")


class NullAIConversationService(AIConversationService):
    """Default no-op provider. Sentence builder falls back to rule-based."""

    def process_text(self, text: str) -> str:
        return text

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("LLM disabled (LLM_PROVIDER=none)")


class OpenAIConversationService(AIConversationService):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def process_text(self, text: str) -> str:
        return text

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        # Integration point: call the OpenAI Chat Completions / Responses API.
        raise NotImplementedError("Add OpenAI SDK call here using self.api_key/self.model.")


class GeminiConversationService(AIConversationService):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def process_text(self, text: str) -> str:
        return text

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Add Gemini Live/API call here using self.api_key.")


def get_ai_conversation_service() -> AIConversationService:
    provider = os.getenv("LLM_PROVIDER", "none").lower()
    if provider == "none":
        return NullAIConversationService()
    if provider == "openai":
        return OpenAIConversationService()
    if provider == "gemini":
        return GeminiConversationService()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
