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
shape + position. It recognizes a small, single-hand-friendly subset of
the requested vocabulary and is NOT a substitute for a trained model.

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
from dataclasses import dataclass, field
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
# Demo heuristic classifier
# ---------------------------------------------------------------------------

class HeuristicSignClassifier:
    """
    Rule-based classifier over a short sequence of frames. Looks at the
    LAST frame's static hand shape plus coarse motion across the sequence.
    This purposefully covers a subset of DEMO_VOCAB with reasonably
    distinct, well-documented hand shapes so the end-to-end pipeline
    (camera -> landmarks -> recognition -> sentence -> speech) is provably
    testable without a trained model.
    """

    name = "demo-heuristic-v1"

    def recognize(self, sequence: list[FrameLandmarks]) -> Optional[SignPrediction]:
        if not sequence:
            return None
        last = sequence[-1]
        hand = last.get("right_hand") or last.get("left_hand")
        pose = last.get("pose")
        if hand is None:
            return None

        curls = _finger_curls(hand)
        zone = _hand_height(hand, pose)
        motion = self._motion_score(sequence)

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

        # Below-threshold: caller should treat as "not sure".
        return SignPrediction("UNKNOWN", 0.2)

    @staticmethod
    def _motion_score(sequence: list[FrameLandmarks]) -> float:
        if len(sequence) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(1, len(sequence)):
            h0 = sequence[i - 1].get("right_hand") or sequence[i - 1].get("left_hand")
            h1 = sequence[i].get("right_hand") or sequence[i].get("left_hand")
            if h0 and h1:
                total += _dist(h0[0], h1[0])
                count += 1
        return total / count if count else 0.0


# ---------------------------------------------------------------------------
# Public entry point used by the rest of the app
# ---------------------------------------------------------------------------

_DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() != "false"
_classifier = HeuristicSignClassifier()

MIN_CONFIDENCE_HIGH = 0.75
MIN_CONFIDENCE_MEDIUM = 0.45


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
        from ml.inference.inference import TrainedSignClassifier  # lazy import
        prediction = TrainedSignClassifier().recognize(sequence)

    if prediction is None:
        return {"sign": None, "confidence": 0.0, "model": "demo" if _DEMO_MODE else "production"}
    return prediction.to_dict()
