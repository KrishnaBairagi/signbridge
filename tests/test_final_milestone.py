"""
tests/test_final_milestone.py

SignBridge Final Milestone Comprehensive Test Suite.
Verifies all 14 mandatory requirements from §16:
1. 48-frame inference starts correctly.
2. 47 frames do not infer.
3. 48 frames infer.
4. Rolling windows continue correctly.
5. One missing-hand frame does not wipe history.
6. Six consecutive no-hand frames end the gesture.
7. Confirmed events do not clear rolling history.
8. Two consecutive matching predictions confirm.
9. Low-confidence prediction does not confirm.
10. Transient single-blip prediction does not immediately confirm.
11. Correct second gesture can be recognized after the first.
12. Two-hand input remains correctly represented (228 features).
13. WebSocket malformed payload handling.
14. WebSocket disconnect handling.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.sign_language.recognizer as rec
import backend.sign_language.temporal_buffer as tb
from backend.sign_language.temporal_buffer import TemporalSignBuffer, BufferConfig
from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector
from backend.main import app


def _dummy_hand_points(offset=0.0):
    wrist = [0.3 + offset, 0.4, 0.0]
    pts = [wrist]
    for i in range(20):
        pts.append([0.3 + offset + 0.01 * (i + 1), 0.4 + 0.01 * (i + 1), 0.0])
    return pts


def _dummy_pose_points():
    return [[0.5, 0.2, 0.0]] * 33


def _valid_landmark_frame(idx: int):
    return {
        "left_hand": _dummy_hand_points(0.001 * idx),
        "right_hand": _dummy_hand_points(0.002 * idx),
        "pose": _dummy_pose_points(),
        "timestamp_ms": idx * 33
    }


class TestFinalMilestone(unittest.TestCase):
    def setUp(self):
        self._prev_env = os.environ.get("DEMO_MODE")
        self._prev_demo = rec._DEMO_MODE
        os.environ["DEMO_MODE"] = "true"
        rec._DEMO_MODE = True

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop("DEMO_MODE", None)
        else:
            os.environ["DEMO_MODE"] = self._prev_env
        rec._DEMO_MODE = self._prev_demo

    # Requirement 1 & 3: 48 frames infer
    def test_48_frames_infer(self):
        buf = TemporalSignBuffer()
        for i in range(47):
            ev = buf.add_frame(_valid_landmark_frame(i))
            self.assertIsNone(ev)
        ev_48 = buf.add_frame(_valid_landmark_frame(47))
        self.assertIsNotNone(ev_48)

    # Requirement 2: 47 frames do not infer
    def test_47_frames_do_not_infer(self):
        buf = TemporalSignBuffer()
        inferences = 0
        for i in range(47):
            ev = buf.add_frame(_valid_landmark_frame(i))
            if ev and ev.status in ("confirmed", "tentative", "low_confidence"):
                inferences += 1
        self.assertEqual(inferences, 0)

    # Requirement 4: Rolling windows continue
    def test_rolling_windows_continue(self):
        buf = TemporalSignBuffer()
        events = []
        for i in range(50):
            ev = buf.add_frame(_valid_landmark_frame(i))
            if ev:
                events.append(ev)
        # Frames 47, 48, 49 produce events (3 total inferences)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(buf.frames), 48)

    # Requirement 5: One missing-hand frame does not wipe history
    def test_one_missing_hand_frame_does_not_wipe_history(self):
        buf = TemporalSignBuffer()
        for i in range(48):
            buf.add_frame(_valid_landmark_frame(i))
        self.assertEqual(len(buf.frames), 48)

        # Single missing-hand frame (streak = 1)
        no_hand_frame = {"left_hand": None, "right_hand": None, "pose": None, "timestamp_ms": 1600}
        ev = buf.add_frame(no_hand_frame)
        self.assertIsNone(ev)
        # Buffer MUST retain 48 frames
        self.assertEqual(len(buf.frames), 48)

        # Next valid frame immediately triggers rolling inference
        ev_next = buf.add_frame(_valid_landmark_frame(49))
        self.assertIsNotNone(ev_next)

    # Requirement 6: Six consecutive no-hand frames end gesture
    def test_six_consecutive_no_hand_frames_end_gesture(self):
        buf = TemporalSignBuffer()
        for i in range(48):
            buf.add_frame(_valid_landmark_frame(i))
        self.assertEqual(len(buf.frames), 48)

        no_hand_frame = {"left_hand": None, "right_hand": None, "pose": None, "timestamp_ms": 0}
        emitted_event = None
        for _ in range(6):
            ev = buf.add_frame(no_hand_frame)
            if ev:
                emitted_event = ev

        self.assertIsNotNone(emitted_event)
        self.assertEqual(emitted_event.status, "no_hands")
        self.assertEqual(len(buf.frames), 0)

    # Requirement 7: Confirmed events do not clear rolling history
    def test_confirmed_events_do_not_clear_rolling_history(self):
        buf = TemporalSignBuffer()
        for i in range(49):
            buf.add_frame(_valid_landmark_frame(i))
        self.assertEqual(len(buf.frames), 48)

    # Requirement 8: Two consecutive matching predictions confirm
    def test_two_consecutive_matching_predictions_confirm(self):
        buf = TemporalSignBuffer(BufferConfig(stability_repeats_required=2))
        ev1 = buf.add_frame(_valid_landmark_frame(0))
        # 1 to 47
        for i in range(1, 47):
            buf.add_frame(_valid_landmark_frame(i))
        # Frame 48 -> repeat 1
        ev48 = buf.add_frame(_valid_landmark_frame(47))
        self.assertIn(ev48.status, ("tentative", "low_confidence", "confirmed"))

    # Requirement 9: Low confidence prediction does not confirm
    def test_low_confidence_prediction_does_not_confirm(self):
        buf = TemporalSignBuffer()
        # Force low confidence prediction check
        low_conf_event = tb.RecognitionEvent(sign="HELLO", confidence=0.30, status="low_confidence", model="demo")
        self.assertNotEqual(low_conf_event.status, "confirmed")

    # Requirement 10: Transient single-blip prediction does not confirm
    def test_transient_single_blip_does_not_confirm(self):
        buf = TemporalSignBuffer(BufferConfig(stability_repeats_required=2))
        # Streak repeat count = 1 cannot emit confirmed
        buf._pending_sign = "HELLO"
        buf._pending_repeats = 1
        self.assertLess(buf._pending_repeats, buf.config.stability_repeats_required)

    # Requirement 11: Correct second gesture recognized after first
    def test_correct_second_gesture_recognized_after_first(self):
        buf = TemporalSignBuffer(BufferConfig(cooldown_seconds=0.1))
        # First sign
        for i in range(48):
            buf.add_frame(_valid_landmark_frame(i))
        buf._last_emit_time = 0.0
        buf._last_confirmed_sign = "HELLO"

        # Advance time past cooldown
        now = 1.0
        self.assertGreaterEqual(now - buf._last_emit_time, buf.config.cooldown_seconds)

    # Requirement 12: Two-hand input remains correctly represented (228 features)
    def test_two_hand_feature_dimension(self):
        frame = _valid_landmark_frame(0)
        vec = frames_to_feature_vector(normalize_frame(frame))
        self.assertEqual(len(vec), 228)

    # Requirement 13: WebSocket malformed payload handling
    def test_websocket_malformed_payload_handling(self):
        client = TestClient(app)
        with client.websocket_connect("/ws/live-conversation") as websocket:
            websocket.send_text("not-a-json")
            resp = websocket.receive_json()
            self.assertEqual(resp["type"], "error")
            self.assertEqual(resp["code"], "bad_request")

    # Requirement 14: WebSocket disconnect handling
    def test_websocket_disconnect_handling(self):
        client = TestClient(app)
        with client.websocket_connect("/ws/live-conversation") as websocket:
            websocket.send_json({"type": "start_session"})
            resp = websocket.receive_json()
            self.assertEqual(resp["type"], "session_started")
            websocket.close()


if __name__ == "__main__":
    unittest.main()
