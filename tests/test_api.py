"""
tests/test_api.py

Requires `pip install -r requirements.txt` (fastapi, httpx) since it uses
FastAPI's TestClient. Not runnable in an offline sandbox without those
packages -- run this yourself after installing dependencies:

    pip install -r requirements.txt
    pytest tests/test_api.py -v
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_translate_endpoint():
    response = client.post("/api/translate", json={"tokens": ["HELLO", "I", "NEED", "HELP"]})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hello I need help."


def test_translate_requires_tokens():
    response = client.post("/api/translate", json={"tokens": []})
    assert response.status_code == 422  # min_length=1 validation


def test_tts_browser_mode_returns_text_passthrough():
    response = client.post("/api/tts", json={"text": "Hello there."})
    assert response.status_code == 200
    assert response.json() == {"mode": "browser", "text": "Hello there."}


def test_predict_no_hands():
    frame = {"left_hand": None, "right_hand": None, "pose": None, "timestamp_ms": 0}
    response = client.post("/api/sign/predict", json={"sequence": [frame]})
    assert response.status_code == 200
    assert response.json()["sign"] is None


def test_websocket_start_and_end_session():
    with client.websocket_connect("/ws/live-conversation") as ws:
        ws.send_json({"type": "start_session", "consent_to_store": False})
        msg = ws.receive_json()
        assert msg["type"] == "session_started"
        assert msg["model"] == "demo"

        ws.send_json({"type": "end_session"})
        msg = ws.receive_json()
        assert msg["type"] == "session_ended"


def test_websocket_speech_text_echoes_to_recipient_display():
    with client.websocket_connect("/ws/live-conversation") as ws:
        ws.send_json({"type": "start_session", "consent_to_store": False})
        ws.receive_json()  # session_started
        ws.send_json({"type": "speech_text", "text": "Where is the station?"})
        msg = ws.receive_json()
        assert msg["type"] == "recipient_text"
        assert msg["text"] == "Where is the station?"
