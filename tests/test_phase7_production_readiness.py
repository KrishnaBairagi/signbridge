"""
tests/test_phase7_production_readiness.py — Automated Test Suite for Phase 7:
Production Readiness, WebSocket Heartbeat, Extended Protocols, and Session Metadata API.
"""

import os
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

import backend.sign_language.recognizer as rec
from backend.main import app
from backend.db import database
from backend.config import get_settings


class TestPhase7ProductionReadiness(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        database.init_db()

    def test_model_info_endpoint_demo_mode(self):
        """1. Verify GET /api/model/info returns demo mode metadata when DEMO_MODE=true."""
        rec._DEMO_MODE = True
        os.environ["DEMO_MODE"] = "true"
        res = self.client.get("/api/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_type"], "demo")
        self.assertEqual(data["class_count"], 21)
        self.assertEqual(data["features_per_frame"], 228)

    def test_model_info_endpoint_production_mode(self):
        """2. Verify GET /api/model/info returns production 61-class metadata when DEMO_MODE=false."""
        rec._DEMO_MODE = False
        os.environ["DEMO_MODE"] = "false"
        res = self.client.get("/api/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_type"], "production")
        self.assertEqual(data["class_count"], 61)
        self.assertEqual(data["features_per_frame"], 228)
        self.assertTrue(data["model_path"].endswith("model.onnx"))

    def test_websocket_ping_pong_heartbeat(self):
        """3. Verify WebSocket ping message returns matching pong with timestamp_ms."""
        with self.client.websocket_connect("/ws/live-conversation") as ws:
            ws.send_json({"type": "start_session", "consent_to_store": False})
            started = ws.receive_json()
            self.assertEqual(started["type"], "session_started")

            ts = 1700000000000
            ws.send_json({"type": "ping", "timestamp_ms": ts})
            pong = ws.receive_json()
            self.assertEqual(pong["type"], "pong")
            self.assertEqual(pong["timestamp_ms"], ts)

    def test_websocket_clear_history_protocol(self):
        """4. Verify WebSocket clear_history resets token history and returns confirmation."""
        with self.client.websocket_connect("/ws/live-conversation") as ws:
            ws.send_json({"type": "start_session", "consent_to_store": False})
            ws.receive_json()

            ws.send_json({"type": "clear_history"})
            res = ws.receive_json()
            self.assertEqual(res["type"], "history_cleared")

    def test_websocket_get_session_info_protocol(self):
        """5. Verify WebSocket get_session_info returns session metadata."""
        with self.client.websocket_connect("/ws/live-conversation") as ws:
            ws.send_json({"type": "start_session", "consent_to_store": True})
            started = ws.receive_json()

            ws.send_json({"type": "get_session_info"})
            info = ws.receive_json()
            self.assertEqual(info["type"], "session_info")
            self.assertEqual(info["session_id"], started["session_id"])
            self.assertTrue(info["consent_to_store"])

    def test_session_details_endpoint_not_found(self):
        """6. Verify GET /api/sessions/{session_id} returns 404 for unknown session."""
        res = self.client.get("/api/sessions/nonexistent-session-id-12345")
        self.assertEqual(res.status_code, 404)

    def test_session_details_endpoint_consent_gated(self):
        """7. Verify GET /api/sessions/{session_id} respects consent_to_store privacy rules."""
        import uuid
        session_id = f"test-no-consent-session-{uuid.uuid4()}"
        database.create_session(session_id, consent_to_store=False)
        database.log_message(session_id, "sign_to_speech", "HELLO", 0.95)

        res = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["session"]["consent_to_store"])
        self.assertEqual(len(data["messages"]), 0)

        # Now test with consent_to_store = True
        consent_session_id = f"test-with-consent-session-{uuid.uuid4()}"
        database.create_session(consent_session_id, consent_to_store=True)
        database.log_message(consent_session_id, "sign_to_speech", "HELP", 0.92)

        res = self.client.get(f"/api/sessions/{consent_session_id}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["session"]["consent_to_store"])
        self.assertEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["content"], "HELP")

    def test_frontend_index_html_phase7_enhancements(self):
        """8. Verify index.html contains Phase 7 heartbeat and protocol handlers."""
        html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
        content = html_path.read_text(encoding="utf-8")
        self.assertIn("pingInterval", content)
        self.assertIn('"ping"', content)
        self.assertIn('"pong"', content)
        self.assertIn('"history_cleared"', content)
