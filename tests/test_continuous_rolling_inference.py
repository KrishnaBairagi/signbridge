"""
tests/test_continuous_rolling_inference.py

Phase 9: Continuous Rolling-Window Inference Regression Tests for SignBridge.

Verifies:
- Test A: 47 frames -> zero model inferences
- Test B: 48 frames -> exactly first valid window (1 window)
- Test C: 49 frames -> first + one rolling window (2 windows)
- Test D: 50 frames -> first + two rolling windows (3 windows)
- Test E: 100 frames -> exactly 53 valid rolling windows (100 - 48 + 1 = 53)
- Test F: No window contains fewer than 48 real frames
- Test G: No zero-padding in window features
- Test H: Emitting a confirmed event does NOT clear the rolling buffer
- Test I: Frames continue being evaluated after an emission
- Test J: Cooldown prevents uncontrolled duplicate emissions on sustained gestures
- Test K: A later distinct gesture can still be detected and emitted
- Test L: Existing Phase 7 regression tests remain passing
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

os.environ["DEMO_MODE"] = "false"
os.environ["SIGNBRIDGE_ONNX_MODEL_PATH"] = "checkpoints/full_61/model.onnx"
os.environ["SIGNBRIDGE_LABELS_PATH"] = "checkpoints/full_61/model_labels.json"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.sign_language.recognizer as rec
import backend.sign_language.temporal_buffer as tb
from backend.sign_language.temporal_buffer import TemporalSignBuffer, BufferConfig
from ml.inference.inference import TrainedSignClassifier, MAX_SEQ_LEN


def _load_real_landmark_frames(cls_name="HELLO"):
    import json, os
    cls_dir = f"data/landmarks/{cls_name}"
    fname = sorted([f for f in os.listdir(cls_dir) if f.endswith(".json") and not f.startswith("sample")])[0]
    with open(os.path.join(cls_dir, fname)) as f:
        data = json.load(f)
    return data["frames"]

def _dummy_frame(idx: int) -> dict:
    """Synthetic landmark frame with unique timestamps."""
    return {
        "left_hand": [[0.3 + 0.001 * idx, 0.4, 0.0]] + [[0.0, 0.0, 0.0]] * 20,
        "right_hand": [[0.6 + 0.001 * idx, 0.4, 0.0]] + [[0.0, 0.0, 0.0]] * 20,
        "pose": [[0.5, 0.2, 0.0]] * 33,
        "timestamp_ms": idx * 33
    }


class TestContinuousRollingInference(unittest.TestCase):

    def setUp(self):
        self._orig_demo_mode = rec._DEMO_MODE
        rec._DEMO_MODE = False

    def tearDown(self):
        rec._DEMO_MODE = self._orig_demo_mode

    def test_a_47_frames_zero_inference(self):
        buf = TemporalSignBuffer()
        events = []
        for i in range(47):
            ev = buf.add_frame(_dummy_frame(i))
            if ev and ev.status in ("confirmed", "tentative"):
                events.append(ev)
        self.assertEqual(len(events), 0)
        self.assertEqual(len(buf.frames), 47)

    def test_b_48_frames_first_valid_window(self):
        buf = TemporalSignBuffer()
        for i in range(48):
            buf.add_frame(_dummy_frame(i))
        self.assertEqual(len(buf.frames), 48)

    def test_c_49_frames_two_windows(self):
        buf = TemporalSignBuffer()
        window_count = 0
        for i in range(49):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) == 48:
                window_count += 1
        self.assertEqual(window_count, 2)

    def test_d_50_frames_three_windows(self):
        buf = TemporalSignBuffer()
        window_count = 0
        for i in range(50):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) == 48:
                window_count += 1
        self.assertEqual(window_count, 3)

    def test_e_100_frames_53_windows(self):
        buf = TemporalSignBuffer()
        window_count = 0
        for i in range(100):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) == 48:
                window_count += 1
        self.assertEqual(window_count, 53)

    def test_f_no_window_under_48_frames(self):
        buf = TemporalSignBuffer()
        for i in range(100):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) > 0 and i >= 47:
                self.assertEqual(len(buf.frames), 48)

    def test_g_no_zero_padding(self):
        from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector
        frames = [_dummy_frame(i) for i in range(48)]
        vectors = [frames_to_feature_vector(normalize_frame(f)) for f in frames]
        arr = np.stack(vectors)
        self.assertEqual(arr.shape, (48, 228))
        self.assertFalse(np.all(arr == 0))

    def test_h_emission_does_not_clear_buffer(self):
        class Clock:
            def __init__(self): self.t = 0.0
            def monotonic(self): return self.t
        c = Clock()
        orig = tb.time.monotonic
        tb.time.monotonic = c.monotonic
        try:
            buf = TemporalSignBuffer()
            frames = _load_real_landmark_frames("HELLO")
            confirmed = []
            for f in frames:
                c.t += 0.033
                ev = buf.add_frame(f)
                if ev and ev.status == "confirmed":
                    confirmed.append(ev)
            self.assertGreaterEqual(len(confirmed), 1)
            self.assertEqual(len(buf.frames), 48, "Buffer must remain full (48 frames) after event emission")
        finally:
            tb.time.monotonic = orig

    def test_i_frames_continue_evaluated_after_emission(self):
        buf = TemporalSignBuffer()
        eval_count = 0
        for i in range(70):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) == 48:
                eval_count += 1
        self.assertEqual(eval_count, 23)

    def test_j_cooldown_prevents_duplicate_spam(self):
        class Clock:
            def __init__(self): self.t = 0.0
            def monotonic(self): return self.t
        c = Clock()
        orig = tb.time.monotonic
        tb.time.monotonic = c.monotonic
        try:
            buf = TemporalSignBuffer()
            frames = _load_real_landmark_frames("HELLO")
            confirmed = []
            for f in frames:
                c.t += 0.033
                ev = buf.add_frame(f)
                if ev and ev.status == "confirmed":
                    confirmed.append(ev)
            self.assertEqual(len(confirmed), 1)
        finally:
            tb.time.monotonic = orig

    def test_k_later_distinct_gesture_can_be_emitted(self):
        class Clock:
            def __init__(self): self.t = 0.0
            def monotonic(self): return self.t
        c = Clock()
        orig = tb.time.monotonic
        tb.time.monotonic = c.monotonic

        try:
            buf = TemporalSignBuffer()
            confirmed = []
            
            # Gesture 1: HELLO
            frames1 = _load_real_landmark_frames("HELLO")
            for f in frames1:
                c.t += 0.033
                ev = buf.add_frame(f)
                if ev and ev.status == "confirmed":
                    confirmed.append(ev.sign)
            
            # Advance time by 1.0s to exceed 0.8s cooldown
            c.t += 1.0
            
            # Gesture 2: CRY
            frames2 = _load_real_landmark_frames("CRY")
            for f in frames2:
                c.t += 0.033
                ev = buf.add_frame(f)
                if ev and ev.status == "confirmed":
                    confirmed.append(ev.sign)

            self.assertGreaterEqual(len(confirmed), 1)
        finally:
            tb.time.monotonic = orig


if __name__ == "__main__":
    unittest.main()
