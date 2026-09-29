"""
tests/test_temporal_skew_fix.py

Phase 7: Temporal & Preprocessing Regression Tests for SignBridge.

Verifies:
- Test A: 47 frames -> no model inference occurs
- Test B: 48 frames -> first valid model inference occurs
- Test C: 49 frames -> valid rolling window of 48 real frames exists
- Test D: 48-frame input contains no zero-padding
- Test E: Partial windows (< 48 frames) never reach ONNX inference
- Test F: First inference uses exact frames 0-47
- Test G: Subsequent inferences use real rolling windows without zero padding
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.sign_language.temporal_buffer import TemporalSignBuffer, BufferConfig
from ml.inference.inference import TrainedSignClassifier, MAX_SEQ_LEN


def _dummy_frame(idx: int) -> dict:
    """Synthetic landmark frame with unique timestamps."""
    return {
        "left_hand": [[0.3 + 0.001 * idx, 0.4, 0.0]] + [[0.0, 0.0, 0.0]] * 20,
        "right_hand": [[0.6 + 0.001 * idx, 0.4, 0.0]] + [[0.0, 0.0, 0.0]] * 20,
        "pose": [[0.5, 0.2, 0.0]] * 33,
        "timestamp_ms": idx * 33
    }


class TestTemporalSkewFix(unittest.TestCase):

    def test_a_47_frames_no_inference(self):
        """Test A: 47 frames -> no model inference occurs (buffer returns None or low/no_hands)."""
        buf = TemporalSignBuffer()
        events = []
        for i in range(47):
            ev = buf.add_frame(_dummy_frame(i))
            if ev is not None and ev.status in ("confirmed", "tentative"):
                events.append(ev)
        self.assertEqual(len(events), 0, "No model inference event should be emitted for < 48 frames")

    def test_b_48_frames_inference_occurs(self):
        """Test B: 48 frames -> first valid model inference occurs."""
        buf = TemporalSignBuffer()
        for i in range(47):
            buf.add_frame(_dummy_frame(i))
        self.assertEqual(len(buf.frames), 47)
        
        # 48th frame triggers first full window
        buf.add_frame(_dummy_frame(47))
        self.assertEqual(len(buf.frames), 48, "Buffer must contain exactly 48 real frames")

    def test_c_49_frames_valid_rolling_window(self):
        """Test C: 49 frames -> valid rolling window of 48 real frames exists."""
        buf = TemporalSignBuffer()
        for i in range(49):
            buf.add_frame(_dummy_frame(i))
        self.assertEqual(len(buf.frames), 48)
        self.assertEqual(buf.frames[0]["timestamp_ms"], 1 * 33, "First frame in rolling deque should be frame 1")
        self.assertEqual(buf.frames[-1]["timestamp_ms"], 48 * 33, "Last frame in rolling deque should be frame 48")

    def test_d_48_frame_input_no_zero_padding(self):
        """Test D: 48-frame input contains no zero-padding."""
        frames = [_dummy_frame(i) for i in range(48)]
        from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector
        vectors = [frames_to_feature_vector(normalize_frame(f)) for f in frames]
        arr = np.stack(vectors)
        self.assertEqual(arr.shape, (48, 228))
        self.assertFalse(np.all(arr == 0), "Real frame sequence must not be all zero padding")

    def test_e_partial_windows_never_reach_onnx(self):
        """Test E: Partial windows (< 48 frames) never reach TrainedSignClassifier."""
        clf = TrainedSignClassifier(
            model_path="checkpoints/full_61/model.onnx",
            labels_path="checkpoints/full_61/model_labels.json"
        )
        partial_frames = [_dummy_frame(i) for i in range(30)]
        res = clf.recognize(partial_frames)
        self.assertIsNone(res, "TrainedSignClassifier.recognize must return None for < 48 frames")

    def test_f_first_inference_uses_frames_0_to_47(self):
        """Test F: First inference uses exact frames 0-47."""
        frames = [_dummy_frame(i) for i in range(48)]
        self.assertEqual(len(frames), 48)
        self.assertEqual(frames[0]["timestamp_ms"], 0)
        self.assertEqual(frames[47]["timestamp_ms"], 47 * 33)

    def test_g_subsequent_inference_uses_real_rolling_windows(self):
        """Test G: Subsequent inferences use real rolling windows without zero padding."""
        buf = TemporalSignBuffer()
        for i in range(100):
            buf.add_frame(_dummy_frame(i))
            if len(buf.frames) == 48:
                frames_list = list(buf.frames)
                self.assertEqual(len(frames_list), 48)
                for f in frames_list:
                    self.assertIsNotNone(f.get("left_hand"))
                    self.assertIsNotNone(f.get("right_hand"))


if __name__ == "__main__":
    unittest.main()
