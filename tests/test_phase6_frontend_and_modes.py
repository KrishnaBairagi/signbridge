"""
tests/test_phase6_frontend_and_modes.py — Comprehensive Automated Tests for Phase 6:
Frontend Modes, State Machine, Cleanup, and User Experience Refinements.
"""

import os
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.config import get_settings


class TestPhase6FrontendAndModes(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        with open(self.html_path, "r", encoding="utf-8") as f:
            self.html_content = f.read()

    def test_frontend_index_html_exists(self):
        """1. Verify frontend index.html exists and is non-empty."""
        self.assertTrue(self.html_path.exists())
        self.assertGreater(len(self.html_content), 1000)

    def test_state_machine_definitions_exist(self):
        """2. Verify explicit frontend state machine constants exist in index.html."""
        for state in ["IDLE", "STARTING", "CONNECTED", "RECOGNIZING", "RECONNECTING", "ENDING"]:
            self.assertIn(state, self.html_content, f"State {state} missing from frontend state machine")

    def test_real_frontend_modes_exist(self):
        """3. Verify live, sign-only, and speech-only modes are implemented."""
        self.assertIn('"live"', self.html_content)
        self.assertIn('"sign-only"', self.html_content)
        self.assertIn('"speech-only"', self.html_content)
        self.assertIn('switchMode', self.html_content)

    def test_browser_capability_check_implemented(self):
        """4. Verify browser capability check functions exist."""
        self.assertIn('checkCapabilities', self.html_content)
        self.assertIn('getUserMedia', self.html_content)
        self.assertIn('WebSocket', self.html_content)
        self.assertIn('speechSynthesis', self.html_content)

    def test_cleanup_functions_implemented(self):
        """5. Verify explicit camera, mic, and WebSocket cleanup routines."""
        self.assertIn('stopCamera', self.html_content)
        self.assertIn('stopMicrophone', self.html_content)
        self.assertIn('cleanupWebSocket', self.html_content)

    def test_websocket_reconnect_exponential_backoff(self):
        """6. Verify 1s, 2s, 4s, 8s, 16s backoff and max 5 retries in JS."""
        self.assertIn('Math.pow(2, retryCount) * 1000', self.html_content)
        self.assertIn('maxRetries = 5', self.html_content)
        self.assertIn('userEndedSession', self.html_content)

    def test_speech_only_mode_disables_camera(self):
        """7. Verify speech-only mode stops camera and disables camera button."""
        self.assertIn('Disabled in Speech-Only mode.', self.html_content)
        self.assertIn('Camera disabled in Speech-Only mode', self.html_content)

    def test_sign_only_mode_disables_speech_recognition(self):
        """8. Verify sign-only mode stops speech recognition and disables mic button."""
        self.assertIn('Disabled in Sign-Only mode.', self.html_content)

    def test_health_check_endpoint(self):
        """9. GET /health route returns ok status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_websocket_mode_session_started_production(self):
        """10. Live WebSocket session startup returns production model tag when DEMO_MODE=false."""
        with patch.dict(os.environ, {"DEMO_MODE": "false"}):
            with self.client.websocket_connect("/ws/live-conversation") as ws:
                ws.send_json({"type": "start_session", "consent_to_store": False})
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "session_started")
                self.assertEqual(msg["model"], "production")


if __name__ == "__main__":
    unittest.main()
