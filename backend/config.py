"""
backend/config.py — Centralized SignBridge Application Settings.

Provides a unified Pydantic v2 Settings model reading environment variables
and providing defaults for recognition mode, model paths, CORS origins,
database configuration, and provider selections.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, field_validator


def _get_default_model_path() -> str:
    env_path = os.getenv("SIGNBRIDGE_ONNX_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists("checkpoints/full_61/model.onnx"):
        return "checkpoints/full_61/model.onnx"
    if os.path.exists("checkpoints/full_61/model.pt"):
        return "checkpoints/full_61/model.pt"
    return "checkpoints/model.pt"


def _get_default_labels_path() -> str:
    env_path = os.getenv("SIGNBRIDGE_LABELS_PATH")
    if env_path:
        return env_path
    if os.path.exists("checkpoints/full_61/model_labels.json"):
        return "checkpoints/full_61/model_labels.json"
    return "checkpoints/model_labels.json"


def _parse_bool(val: str | bool | None, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() not in ("false", "0", "no", "off")


class Settings(BaseModel):
    demo_mode: bool = Field(default_factory=lambda: _parse_bool(os.getenv("DEMO_MODE"), False))
    onnx_model_path: str = Field(default_factory=_get_default_model_path)
    labels_path: str = Field(default_factory=_get_default_labels_path)
    max_seq_len: int = Field(default_factory=lambda: int(os.getenv("SIGNBRIDGE_MAX_SEQ_LEN", "48")))
    cors_allowed_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip() for origin in os.getenv(
                "CORS_ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000"
            ).split(",") if origin.strip()
        ]
    )
    db_path: str = Field(default_factory=lambda: os.getenv("SIGNBRIDGE_DB_PATH", "signbridge.db"))
    tts_provider: str = Field(default_factory=lambda: os.getenv("TTS_PROVIDER", "browser").lower())
    stt_provider: str = Field(default_factory=lambda: os.getenv("STT_PROVIDER", "browser").lower())
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none").lower())

    @field_validator("max_seq_len")
    @classmethod
    def validate_max_seq_len(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_seq_len must be positive")
        return v

    @field_validator("tts_provider")
    @classmethod
    def validate_tts_provider(cls, v: str) -> str:
        valid = {"browser", "azure", "google"}
        if v not in valid:
            raise ValueError(f"Unknown TTS_PROVIDER: {v}. Allowed: {valid}")
        return v

    @field_validator("stt_provider")
    @classmethod
    def validate_stt_provider(cls, v: str) -> str:
        valid = {"browser", "whisper"}
        if v not in valid:
            raise ValueError(f"Unknown STT_PROVIDER: {v}. Allowed: {valid}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        valid = {"none", "openai", "gemini"}
        if v not in valid:
            raise ValueError(f"Unknown LLM_PROVIDER: {v}. Allowed: {valid}")
        return v


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
