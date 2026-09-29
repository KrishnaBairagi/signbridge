"""
ml/preprocessing/normalize.py

Normalizes raw MediaPipe landmark frames so the model is invariant to the
signer's position/distance from the camera:
  - translate so the wrist (or mid-shoulder point) is the origin
  - scale by shoulder width (or hand size, if pose isn't available)

This is applied identically at training time and inference time -- any
mismatch here is a classic, silent source of train/serve skew, so both
ml/training/prepare_dataset.py and the live backend's client-side JS
(frontend, see landmark_utils in index.html) implement the same formula.
"""

from __future__ import annotations

import numpy as np


def normalize_frame(frame: dict) -> dict:
    """Returns a new frame dict with normalized hand/pose coordinates."""
    scale = _reference_scale(frame)
    origin = _reference_origin(frame)

    def _norm(points):
        if points is None:
            return None
        arr = np.array(points, dtype=np.float32)
        arr = (arr - origin) / scale
        return arr.tolist()

    return {
        "left_hand": _norm(frame.get("left_hand")),
        "right_hand": _norm(frame.get("right_hand")),
        "pose": _norm(frame.get("pose")),
        "timestamp_ms": frame.get("timestamp_ms"),
    }


def _reference_origin(frame: dict) -> np.ndarray:
    pose = frame.get("pose")
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = np.array(pose[11]), np.array(pose[12])
        return (left_shoulder + right_shoulder) / 2
    hand = frame.get("right_hand") or frame.get("left_hand")
    if hand:
        return np.array(hand[0])
    return np.zeros(3, dtype=np.float32)


def _reference_scale(frame: dict) -> float:
    pose = frame.get("pose")
    if pose and len(pose) > 12:
        left_shoulder, right_shoulder = np.array(pose[11]), np.array(pose[12])
        width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if width > 1e-4:
            return width
    hand = frame.get("right_hand") or frame.get("left_hand")
    if hand:
        wrist, middle_mcp = np.array(hand[0]), np.array(hand[9])
        size = float(np.linalg.norm(wrist - middle_mcp))
        if size > 1e-4:
            return size
    return 1.0


def frames_to_feature_vector(frame: dict) -> np.ndarray:
    """
    Flattens a normalized frame into a fixed-length feature vector for the
    model: 21*3 (left hand) + 21*3 (right hand) + 33*3 (pose) = 225 dims.
    Missing components are zero-filled (with a presence flag concatenated
    so the model can distinguish "absent" from "at the origin").
    """
    def _flatten(points, n_points):
        if points is None:
            return np.zeros(n_points * 3, dtype=np.float32), 0.0
        arr = np.array(points, dtype=np.float32).flatten()
        return arr, 1.0

    left, left_present = _flatten(frame.get("left_hand"), 21)
    right, right_present = _flatten(frame.get("right_hand"), 21)
    pose, pose_present = _flatten(frame.get("pose"), 33)

    return np.concatenate([left, right, pose, [left_present, right_present, pose_present]])
