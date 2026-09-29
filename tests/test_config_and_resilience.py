"""
tests/test_config_and_resilience.py — Comprehensive tests for Production Configuration,
Resilience, and Service Abstractions.
"""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.config import Settings, get_settings
from backend.logging_config import logger, setup_logging
from backend.main import app
from backend.db.database import DB_PATH, get_connection
from backend.tts.tts_service import get_tts_provider, BrowserTTSProvider
from backend.speech.stt_service import get_stt_provider, BrowserSTTProvider
from backend.services.ai_conversation_service import get_ai_conversation_service, NullAIConversationService


class TestConfigAndResilience(unittest.TestCase):

    def test_default_config_loading(self):
        """1. Default configuration loading."""
        settings = Settings()
        self.assertIsInstance(settings.demo_mode, bool)
        self.assertFalse(settings.demo_mode)
        self.assertEqual(settings.max_seq_len, 48)
        self.assertIn("http://localhost:8000", settings.cors_allowed_origins)
        self.assertEqual(settings.tts_provider, "browser")
        self.assertEqual(settings.stt_provider, "browser")
        self.assertEqual(settings.llm_provider, "none")

    def test_env_var_override(self):
        """2. Environment variable overrides."""
        env = {
            "DEMO_MODE": "false",
            "SIGNBRIDGE_MAX_SEQ_LEN": "32",
            "SIGNBRIDGE_DB_PATH": "test_override.db",
            "CORS_ALLOWED_ORIGINS": "http://example.com, http://test.com",
            "TTS_PROVIDER": "browser",
            "STT_PROVIDER": "whisper",
            "LLM_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env):
            settings = get_settings()
            self.assertFalse(settings.demo_mode)
            self.assertEqual(settings.max_seq_len, 32)
            self.assertEqual(settings.db_path, "test_override.db")
            self.assertEqual(settings.cors_allowed_origins, ["http://example.com", "http://test.com"])
            self.assertEqual(settings.stt_provider, "whisper")
            self.assertEqual(settings.llm_provider, "openai")

    def test_config_validation_error(self):
        """3. Validation errors for invalid values."""
        with self.assertRaises(ValidationError):
            Settings(max_seq_len=0)

        with self.assertRaises(ValidationError):
            Settings(max_seq_len=-5)

        with self.assertRaises(ValidationError):
            Settings(tts_provider="unsupported_provider")

        with self.assertRaises(ValidationError):
            Settings(stt_provider="invalid_stt")

        with self.assertRaises(ValidationError):
            Settings(llm_provider="unknown_llm")

    def test_logging_configuration(self):
        """4. Structured logging configuration."""
        custom_logger = setup_logging()
        self.assertEqual(custom_logger.name, "signbridge")
        self.assertTrue(len(custom_logger.handlers) > 0)
        # Verify logging executes without raising exceptions
        logger.info("Test log message for resilience verification")

    def test_health_endpoint_demo_mode(self):
        """5. GET /health status and demo_mode reflection."""
        client = TestClient(app)
        with patch.dict(os.environ, {"DEMO_MODE": "true"}):
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["demo_mode"], "true")

        with patch.dict(os.environ, {"DEMO_MODE": "false"}):
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["demo_mode"], "false")

    def test_ws_session_started_model_badge(self):
        """6. WebSocket session_started payload model reflection."""
        client = TestClient(app)
        with patch.dict(os.environ, {"DEMO_MODE": "true"}):
            with client.websocket_connect("/ws/live-conversation") as websocket:
                websocket.send_json({"type": "start_session", "consent_to_store": False})
                data = websocket.receive_json()
                self.assertEqual(data["type"], "session_started")
                self.assertEqual(data["model"], "demo")

        with patch.dict(os.environ, {"DEMO_MODE": "false"}):
            with client.websocket_connect("/ws/live-conversation") as websocket:
                websocket.send_json({"type": "start_session", "consent_to_store": False})
                data = websocket.receive_json()
                self.assertEqual(data["type"], "session_started")
                self.assertEqual(data["model"], "production")

    def test_cors_configuration(self):
        """7. CORS middleware configuration."""
        client = TestClient(app)
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        self.assertIn(response.status_code, (200, 204))

    def test_db_and_provider_abstraction(self):
        """8. Database path and service provider factory resolution."""
        with patch.dict(os.environ, {"TTS_PROVIDER": "browser", "STT_PROVIDER": "browser", "LLM_PROVIDER": "none"}):
            tts = get_tts_provider()
            stt = get_stt_provider()
            llm = get_ai_conversation_service()
            self.assertIsInstance(tts, BrowserTTSProvider)
            self.assertIsInstance(stt, BrowserSTTProvider)
            self.assertIsInstance(llm, NullAIConversationService)


if __name__ == "__main__":
    unittest.main()
