"""
tests/test_core_logic.py

Tests for the parts of SignBridge that have no external dependencies
(no FastAPI/pytest-asyncio needed) -- these run with plain `python -m unittest`
even before you `pip install -r requirements.txt`, so you can verify the
core pipeline logic immediately.

Run with:
    python -m unittest tests.test_core_logic -v
"""

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.sign_language.temporal_buffer import TemporalSignBuffer, BufferConfig
import backend.sign_language.recognizer as rec
from backend.sign_language.recognizer import recognize_sign, HeuristicSignClassifier
from backend.services.sentence_builder import build_sentence_rule_based, _validate_no_hallucination


def _open_palm_frame(wrist_y):
    """Synthetic landmark points shaped like an open, extended-fingers hand."""
    wrist = [0.5, wrist_y, 0]
    pts = [wrist]
    pts += [[0.52, wrist_y - 0.05, 0], [0.54, wrist_y - 0.1, 0], [0.56, wrist_y - 0.15, 0], [0.58, wrist_y - 0.2, 0]]
    pts += [[0.5, wrist_y - 0.05, 0], [0.5, wrist_y - 0.1, 0], [0.5, wrist_y - 0.15, 0], [0.5, wrist_y - 0.25, 0]]
    pts += [[0.5, wrist_y - 0.06, 0], [0.5, wrist_y - 0.12, 0], [0.5, wrist_y - 0.2, 0], [0.5, wrist_y - 0.3, 0]]
    pts += [[0.5, wrist_y - 0.06, 0], [0.5, wrist_y - 0.11, 0], [0.5, wrist_y - 0.18, 0], [0.5, wrist_y - 0.28, 0]]
    pts += [[0.5, wrist_y - 0.05, 0], [0.5, wrist_y - 0.09, 0], [0.5, wrist_y - 0.15, 0], [0.5, wrist_y - 0.24, 0]]
    return pts


def _fist_frame(wrist_y):
    """Synthetic landmark points shaped like a closed fist (fingertips near wrist)."""
    wrist = [0.5, wrist_y, 0]
    pts = [wrist] + [[0.5 + 0.01 * i, wrist_y - 0.02, 0] for i in range(20)]
    return pts


FACE_POSE = [[0.5, 0.15, 0]] + [[0, 0, 0]] * 10 + [[0.4, 0.4, 0], [0.6, 0.4, 0]]  # nose y=0.15, shoulders y=0.4


class TestRecognizer(unittest.TestCase):
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

    def test_no_hand_returns_none(self):
        result = recognize_sign([{"left_hand": None, "right_hand": None, "pose": None, "timestamp_ms": 0}])
        self.assertIsNone(result["sign"])

    def test_open_palm_near_face_is_hello(self):
        frame = {"right_hand": _open_palm_frame(0.1), "left_hand": None, "pose": FACE_POSE, "timestamp_ms": 0}
        result = recognize_sign([frame])
        self.assertEqual(result["sign"], "HELLO")
        self.assertGreater(result["confidence"], 0.5)

    def test_result_always_tagged_demo_model(self):
        frame = {"right_hand": _open_palm_frame(0.1), "left_hand": None, "pose": FACE_POSE, "timestamp_ms": 0}
        result = recognize_sign([frame])
        self.assertEqual(result["model"], "demo")


class TestTemporalBuffer(unittest.TestCase):
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
    def test_no_hands_emits_error_event_after_streak(self):
        buf = TemporalSignBuffer(BufferConfig(no_hand_frames_to_end_gesture=3))
        events = []
        for _ in range(5):
            ev = buf.add_frame({"left_hand": None, "right_hand": None, "pose": None, "timestamp_ms": 0})
            if ev:
                events.append(ev)
        self.assertTrue(any(e.status == "no_hands" for e in events))

    def test_sustained_hello_gesture_is_confirmed_exactly_once_per_cooldown(self):
        buf = TemporalSignBuffer()
        confirmed = []
        random.seed(0)
        for i in range(60):
            jitter = random.uniform(-0.002, 0.002)
            frame = {
                "right_hand": _open_palm_frame(0.1 + jitter), "left_hand": None,
                "pose": FACE_POSE, "timestamp_ms": i * 33,
            }
            ev = buf.add_frame(frame)
            if ev and ev.status == "confirmed":
                confirmed.append(ev)
        self.assertGreaterEqual(len(confirmed), 1)
        self.assertTrue(all(e.sign == "HELLO" for e in confirmed))

    def test_frame_buffer_does_not_grow_unbounded(self):
        buf = TemporalSignBuffer(BufferConfig(window_size=10))
        for i in range(100):
            buf.add_frame({"right_hand": _open_palm_frame(0.1), "left_hand": None,
                            "pose": FACE_POSE, "timestamp_ms": i})
        self.assertLessEqual(len(buf.frames), 10)


class TestSentenceBuilder(unittest.TestCase):
    def test_basic_join_and_capitalization(self):
        result = build_sentence_rule_based(["HELLO", "I", "NEED", "HELP"])
        self.assertEqual(result.text, "Hello I need help.")

    def test_where_question_gets_question_mark(self):
        result = build_sentence_rule_based(["WHERE", "HOSPITAL"])
        self.assertTrue(result.text.endswith("?"))

    def test_empty_tokens_returns_empty(self):
        result = build_sentence_rule_based([])
        self.assertEqual(result.text, "")

    def test_thank_you_token_maps_to_two_words(self):
        result = build_sentence_rule_based(["THANK_YOU"])
        self.assertIn("thank you", result.text.lower())

    def test_hallucination_guard_rejects_unrelated_content(self):
        # Simulates an LLM inventing content not present in the tokens.
        ok = _validate_no_hallucination("I would like to visit the doctor tomorrow at noon please.",
                                         ["I", "NEED", "HELP"])
        self.assertFalse(ok)

    def test_hallucination_guard_accepts_faithful_rewrite(self):
        ok = _validate_no_hallucination("I need help.", ["I", "NEED", "HELP"])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
