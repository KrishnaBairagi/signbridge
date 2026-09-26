"""
sign_language/recognizer.py

THIS IS THE MODULE YOU REPLACE WHEN YOU HAVE A TRAINED MODEL.

Contract (must not change, so the rest of the app never needs to know
whether it's talking to a demo heuristic or a trained LSTM/Transformer):

    recognize_sign(sequence: LandmarkSequence) -> SignPrediction
        SignPrediction = {"sign": str, "confidence": float}

`sequence` is a rolling window of normalized MediaPipe landmark frames
(see ml/feature_extraction/landmarks.py for the exact schema). The demo
classifier below uses simple, explainable geometric heuristics on hand
shape + position. It recognizes a small subset of the requested vocabulary
and is NOT a substitute for a trained model.

DEMO MODE vs PRODUCTION MODE
-----------------------------
DEMO_MODE=True (default): uses HeuristicSignClassifier below.
DEMO_MODE=False: expects ml/inference/inference.py's TrainedSignClassifier,
which loads a checkpoint exported by ml/training/export_model.py.

The demo model is intentionally limited. It should NEVER be presented to
end users as "sign language translation" -- the API always tags responses
with `"model": "demo"` so the frontend can show a "Demo Recognition Model"
badge (see §21 of the spec).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

# One frame of normalized landmarks. Coordinates are relative to the
# detected pose (wrist-centered, scale-normalized by shoulder width) so the
# classifier is invariant to the signer's distance from the camera.
# Each hand is 21 (x, y, z) points as produced by MediaPipe Hands.
FrameLandmarks = dict  # {"left_hand": [[x,y,z], ...] | None,
                        #  "right_hand": [[x,y,z], ...] | None,
                        #  "pose": [[x,y,z], ...] | None,   # 33 pts, optional
                        #  "timestamp_ms": int}


@dataclass
class SignPrediction:
    sign: str
    confidence: float
    model: str = "demo"  # "demo" | "production"

    def to_dict(self) -> dict:
        return {"sign": self.sign, "confidence": round(self.confidence, 3), "model": self.model}


DEMO_VOCAB = [
    "HELLO", "YES", "NO", "THANK_YOU", "PLEASE", "HELP", "WATER", "FOOD",
    "DOCTOR", "HOSPITAL", "WHERE", "WHAT", "WHO", "I", "YOU", "NEED",
    "GO", "STOP", "GOOD", "BAD", "SORRY",
]


# ---------------------------------------------------------------------------
# Helpers: finger curl / geometry, used by the heuristic demo classifier
# ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _finger_curls(hand) -> dict:
    """
    Returns approximate curl (0 = fully extended, 1 = fully curled) for
    each finger, using MediaPipe's 21-point hand landmark indexing:
      0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.
    Curl is estimated as normalized distance from fingertip to wrist versus
    the corresponding MCP-to-wrist distance (a cheap, rotation-tolerant proxy).
    """
    wrist = hand[0]
    fingers = {
        "thumb": (2, 4), "index": (5, 8), "middle": (9, 12),
        "ring": (13, 16), "pinky": (17, 20),
    }
    curls = {}
    for name, (mcp_idx, tip_idx) in fingers.items():
        mcp_dist = _dist(hand[mcp_idx], wrist) or 1e-6
        tip_dist = _dist(hand[tip_idx], wrist)
        ratio = tip_dist / mcp_dist
        # ratio ~1.6-2.2 => extended, ~0.8-1.2 => curled (empirical demo thresholds)
        curl = max(0.0, min(1.0, (2.0 - ratio) / 1.0))
        curls[name] = curl
    return curls


def _is_open_palm(curls: dict, threshold: float = 0.35) -> bool:
    return all(c < threshold for c in curls.values())


def _is_fist(curls: dict, threshold: float = 0.65) -> bool:
    return all(c > threshold for c in curls.values())


def _is_thumbs_up(curls: dict) -> bool:
    return curls["thumb"] < 0.3 and all(curls[f] > 0.6 for f in ("index", "middle", "ring", "pinky"))


def _is_pointing(curls: dict) -> bool:
    return curls["index"] < 0.3 and all(curls[f] > 0.55 for f in ("middle", "ring", "pinky"))


def _hand_height(hand, pose) -> str:
    """Rough vertical zone relative to pose (face/chest) if pose is available."""
    if not pose:
        return "unknown"
    wrist_y = hand[0][1]
    nose_y = pose[0][1] if len(pose) > 0 else None
    shoulder_y = (pose[11][1] + pose[12][1]) / 2 if len(pose) > 12 else None
    if nose_y is not None and wrist_y < nose_y:
        return "face"
    if shoulder_y is not None and wrist_y < shoulder_y:
        return "chest"
    return "low"


# ---------------------------------------------------------------------------
# Two-hand geometry helpers
# ---------------------------------------------------------------------------

def _both_hands_present(frame: FrameLandmarks) -> bool:
    """Check if both left and right hands are detected."""
    return frame.get("left_hand") is not None and frame.get("right_hand") is not None


def _wrist_pos(hand) -> tuple:
    """Get wrist (x, y) position."""
    return (hand[0][0], hand[0][1])


def _hands_horizontal_distance(left_hand, right_hand) -> float:
    """Horizontal distance between wrists."""
    return abs(left_hand[0][0] - right_hand[0][0])


def _hands_vertical_distance(left_hand, right_hand) -> float:
    """Vertical distance between wrists."""
    return abs(left_hand[0][1] - right_hand[0][1])


def _hands_euclidean_distance(left_hand, right_hand) -> float:
    """3D Euclidean distance between wrists."""
    return _dist(left_hand[0], right_hand[0])


def _hands_side_by_side(left_hand, right_hand, pose, max_horizontal_ratio: float = 0.5) -> bool:
    """
    Check if hands are side-by-side (similar Y, separated in X).
    Uses shoulder width as reference scale if pose available.
    """
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = pose[11], pose[12]
        shoulder_width = _dist(left_shoulder, right_shoulder)
        if shoulder_width > 1e-4:
            horiz_dist = _hands_horizontal_distance(left_hand, right_hand)
            vert_dist = _hands_vertical_distance(left_hand, right_hand)
            return horiz_dist > 0.3 * shoulder_width and vert_dist < 0.3 * shoulder_width
    # Fallback: similar Y, distinct X
    vert_dist = _hands_vertical_distance(left_hand, right_hand)
    horiz_dist = _hands_horizontal_distance(left_hand, right_hand)
    return horiz_dist > 3 * vert_dist and horiz_dist > 0.15


def _hands_stacked(left_hand, right_hand, pose, max_vertical_ratio: float = 0.5) -> bool:
    """
    Check if hands are stacked vertically (similar X, separated in Y).
    """
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = pose[11], pose[12]
        shoulder_width = _dist(left_shoulder, right_shoulder)
        if shoulder_width > 1e-4:
            horiz_dist = _hands_horizontal_distance(left_hand, right_hand)
            vert_dist = _hands_vertical_distance(left_hand, right_hand)
            return vert_dist > 0.3 * shoulder_width and horiz_dist < 0.3 * shoulder_width
    horiz_dist = _hands_horizontal_distance(left_hand, right_hand)
    vert_dist = _hands_vertical_distance(left_hand, right_hand)
    return vert_dist > 3 * horiz_dist and vert_dist > 0.15


def _hands_touching(left_hand, right_hand, pose, threshold_ratio: float = 0.08) -> bool:
    """
    Check if wrists are close enough to be touching/near each other.
    Threshold relative to shoulder width or hand size.
    """
    euclid_dist = _hands_euclidean_distance(left_hand, right_hand)
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = pose[11], pose[12]
        shoulder_width = _dist(left_shoulder, right_shoulder)
        if shoulder_width > 1e-4:
            return euclid_dist < threshold_ratio * shoulder_width
    # Fallback: use hand size (wrist to middle MCP)
    hand_size = _dist(left_hand[0], left_hand[9])
    return euclid_dist < 1.5 * hand_size


def _palms_facing_each_other(left_hand, right_hand) -> bool:
    """
    Heuristic: if both hands are open palms and wrists are horizontally
    separated with palms roughly facing inward.
    """
    left_curls = _finger_curls(left_hand)
    right_curls = _finger_curls(right_hand)
    both_open = _is_open_palm(left_curls) and _is_open_palm(right_curls)
    if not both_open:
        return False
    # Check horizontal separation with palms facing
    horiz_dist = _hands_horizontal_distance(left_hand, right_hand)
    return horiz_dist > 0.1  # Some separation, not touching


def _one_hand_fist_one_open(left_hand, right_hand) -> tuple[bool, str]:
    """
    Check if one hand is fist and other is open palm.
    Returns (is_mixed, which_hand_is_fist) where which_hand_is_fist is "left" or "right".
    """
    left_curls = _finger_curls(left_hand)
    right_curls = _finger_curls(right_hand)
    left_fist = _is_fist(left_curls)
    right_fist = _is_fist(right_curls)
    left_open = _is_open_palm(left_curls)
    right_open = _is_open_palm(right_curls)

    if left_fist and right_open:
        return True, "left"
    if right_fist and left_open:
        return True, "right"
    return False, ""


def _both_hands_same_shape(left_hand, right_hand) -> str | None:
    """
    Check if both hands have the same shape.
    Returns "open", "fist", "pointing", "thumbs_up" or None.
    """
    left_curls = _finger_curls(left_hand)
    right_curls = _finger_curls(right_hand)

    if _is_open_palm(left_curls) and _is_open_palm(right_curls):
        return "open"
    if _is_fist(left_curls) and _is_fist(right_curls):
        return "fist"
    if _is_pointing(left_curls) and _is_pointing(right_curls):
        return "pointing"
    if _is_thumbs_up(left_curls) and _is_thumbs_up(right_curls):
        return "thumbs_up"
    return None


# ---------------------------------------------------------------------------
# Demo heuristic classifier with two-hand support
# ---------------------------------------------------------------------------

class HeuristicSignClassifier:
    """
    Rule-based classifier over a short sequence of frames. Looks at the
    LAST frame's static hand shape(s) plus coarse motion across the sequence.
    Supports both one-hand and two-hand signs.
    """

    name = "demo-heuristic-v2"

    def recognize(self, sequence: list[FrameLandmarks]) -> Optional[SignPrediction]:
        if not sequence:
            return None
        last = sequence[-1]
        left_hand = last.get("left_hand")
        right_hand = last.get("right_hand")
        pose = last.get("pose")

        # No hands at all
        if left_hand is None and right_hand is None:
            return None

        # Determine if we have one or two hands
        both_hands = left_hand is not None and right_hand is not None
        single_hand = left_hand if right_hand is None else right_hand

        # Compute features for available hands
        motion = self._motion_score(sequence)

        if both_hands:
            return self._recognize_two_hand(left_hand, right_hand, pose, motion, sequence)
        else:
            return self._recognize_one_hand(single_hand, pose, motion)

    def _recognize_one_hand(self, hand, pose, motion: float) -> Optional[SignPrediction]:
        """Original one-hand recognition logic."""
        curls = _finger_curls(hand)
        zone = _hand_height(hand, pose)

        # Order matters: check more specific shapes first.
        if _is_thumbs_up(curls):
            return SignPrediction("YES", 0.88)
        if _is_fist(curls) and motion > 0.15:
            return SignPrediction("NO", 0.75)
        if _is_pointing(curls) and zone == "face":
            return SignPrediction("WHO", 0.6)
        if _is_pointing(curls):
            return SignPrediction("WHERE", 0.55)
        if _is_open_palm(curls) and zone == "face":
            return SignPrediction("HELLO", 0.82)
        if _is_open_palm(curls) and zone == "chest" and motion > 0.2:
            return SignPrediction("PLEASE", 0.5)
        if _is_open_palm(curls) and zone == "chest":
            return SignPrediction("I", 0.45)
        if _is_fist(curls) and zone == "chest":
            return SignPrediction("NEED", 0.4)

        return SignPrediction("UNKNOWN", 0.2)

    def _recognize_two_hand(self, left_hand, right_hand, pose, motion: float, sequence: list) -> Optional[SignPrediction]:
        """Two-hand sign recognition logic."""
        left_curls = _finger_curls(left_hand)
        right_curls = _finger_curls(right_hand)

        left_open = _is_open_palm(left_curls)
        right_open = _is_open_palm(right_curls)
        left_fist = _is_fist(left_curls)
        right_fist = _is_fist(right_curls)
        left_point = _is_pointing(left_curls)
        right_point = _is_pointing(right_curls)

        both_open = left_open and right_open
        both_fist = left_fist and right_fist
        both_point = left_point and right_point

        # Hand positions
        side_by_side = _hands_side_by_side(left_hand, right_hand, pose)
        stacked = _hands_stacked(left_hand, right_hand, pose)
        touching = _hands_touching(left_hand, right_hand, pose)
        palms_facing = _palms_facing_each_other(left_hand, right_hand)
        mixed_fist_open, fist_hand = _one_hand_fist_one_open(left_hand, right_hand)

        zone_left = _hand_height(left_hand, pose)
        zone_right = _hand_height(right_hand, pose)
        zone = zone_left if zone_left != "unknown" else zone_right

        # --- Two-hand signs ---

        # THANK_YOU: Both open palms, one hand moves from chin forward (simplified: both open near face/chest)
        if both_open and (zone == "face" or zone == "chest") and touching:
            return SignPrediction("THANK_YOU", 0.75)

        # PLEASE: Both open palms, circular motion on chest (simplified: both open on chest, some motion)
        if both_open and zone == "chest" and motion > 0.15:
            return SignPrediction("PLEASE", 0.7)

        # HELP: One fist on top of open palm (dominant fist on non-dominant palm)
        if mixed_fist_open and touching and stacked:
            # In ASL/ISL, dominant hand (usually right) is fist on non-dominant palm
            return SignPrediction("HELP", 0.75)

        # DOCTOR: Both fists, tapping wrists together (simplified: both fists touching)
        if both_fist and touching and side_by_side:
            return SignPrediction("DOCTOR", 0.7)

        # HOSPITAL: Cross wrists (both fists or both open, stacked vertically)
        if (both_fist or both_open) and stacked and touching:
            return SignPrediction("HOSPITAL", 0.65)

        # SORRY: Fist on chest, circular motion (simplified: one fist on chest, other hand nearby)
        if mixed_fist_open and zone == "chest" and touching:
            return SignPrediction("SORRY", 0.65)

        # GO: Both index fingers pointing forward, moving forward (simplified: both pointing, side by side, motion)
        if both_point and side_by_side and motion > 0.2:
            return SignPrediction("GO", 0.65)

        # STOP: One open palm facing out, other hand striking it (simplified: both open, touching, palms facing)
        if both_open and touching and palms_facing and side_by_side:
            return SignPrediction("STOP", 0.7)

        # GOOD: Thumbs up both hands (simplified: both thumbs up)
        if _is_thumbs_up(left_curls) and _is_thumbs_up(right_curls):
            return SignPrediction("GOOD", 0.75)

        # BAD: Both thumbs down or similar (simplified: both fists with downward motion)
        if both_fist and motion > 0.2 and zone == "low":
            return SignPrediction("BAD", 0.55)

        # WATER/FOOD: Complex handshapes (W-shape, fingertips together) - hard to detect with just curl
        # Skip for now - will need trained model

        # If no two-hand sign matched, fall back to one-hand logic on dominant hand
        # (Prefer right hand as dominant, fallback to left)
        dominant_hand = right_hand if right_hand else left_hand
        return self._recognize_one_hand(dominant_hand, pose, motion)

    @staticmethod
    def _motion_score(sequence: list[FrameLandmarks]) -> float:
        if len(sequence) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(1, len(sequence)):
            # Use both hands for motion if available
            h0_left = sequence[i - 1].get("left_hand")
            h0_right = sequence[i - 1].get("right_hand")
            h1_left = sequence[i].get("left_hand")
            h1_right = sequence[i].get("right_hand")

            # Average motion across available hands
            for h0, h1 in [(h0_left, h1_left), (h0_right, h1_right)]:
                if h0 and h1:
                    total += _dist(h0[0], h1[0])
                    count += 1
        return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Public entry point used by the rest of the app
# ---------------------------------------------------------------------------

_DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() != "false"
_classifier = HeuristicSignClassifier()
_trained_classifier = None

MIN_CONFIDENCE_HIGH = 0.75
MIN_CONFIDENCE_MEDIUM = 0.45


def _get_trained_classifier():
    global _trained_classifier
    if _trained_classifier is None:
        from ml.inference.inference import TrainedSignClassifier
        _trained_classifier = TrainedSignClassifier()
    return _trained_classifier


def recognize_sign(sequence: list[FrameLandmarks]) -> dict:
    """
    Stable public interface. Swap the implementation, not the signature.

    To plug in a trained model: set DEMO_MODE=false and implement
    ml/inference/inference.py::TrainedSignClassifier.recognize() with the
    same return contract, then import it here instead of HeuristicSignClassifier.
    """
    if _DEMO_MODE:
        prediction = _classifier.recognize(sequence)
    else:
        prediction = _get_trained_classifier().recognize(sequence)

    if prediction is None:
        return {"sign": None, "confidence": 0.0, "model": "demo" if _DEMO_MODE else "production"}
    return prediction.to_dict()

