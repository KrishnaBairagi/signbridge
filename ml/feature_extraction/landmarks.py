"""
ml/feature_extraction/landmarks.py

Defines the landmark schema shared between:
  - the browser (MediaPipe Tasks Vision JS, extracting live landmarks)
  - the training pipeline (extracting landmarks from dataset videos with
    the Python `mediapipe` package, so train/inference feature spaces match)

Schema (one frame):
{
    "left_hand": [[x, y, z], ...21 points...] | None,
    "right_hand": [[x, y, z], ...21 points...] | None,
    "pose": [[x, y, z], ...33 points...] | None,
    "timestamp_ms": int
}

Point indexing follows MediaPipe's standard topology:
  Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
  Pose:  https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

`extract_landmarks_from_video` requires the `mediapipe` and `opencv-python`
packages (see requirements.txt) and is used only by the offline training
pipeline (ml/training/extract_landmarks.py) -- it is NOT imported by the
live backend, which receives landmarks already extracted client-side.
"""

from __future__ import annotations

from typing import Optional


def extract_landmarks_from_video(video_path: str, sample_fps: int = 30) -> list[dict]:
    """
    Runs MediaPipe Holistic (or separate Hands+Pose landmarkers) over a
    video file and returns a list of frame-landmark dicts matching the
    schema above. Used only during offline dataset preparation.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "extract_landmarks_from_video requires `opencv-python` and `mediapipe`. "
            "Install with: pip install opencv-python mediapipe"
        ) from e

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    frames: list[dict] = []
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    frame_interval = max(1, round(native_fps / sample_fps))

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
         mp_pose.Pose(static_image_mode=False) as pose:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = hands.process(rgb)
                pose_results = pose.process(rgb)
                frames.append(_build_frame(hand_results, pose_results, idx, native_fps))
            idx += 1
    cap.release()
    return frames


def _build_frame(hand_results, pose_results, frame_idx: int, fps: float) -> dict:
    left_hand: Optional[list] = None
    right_hand: Optional[list] = None

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            points = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            label = handedness.classification[0].label  # "Left" | "Right"
            if label == "Left":
                left_hand = points
            else:
                right_hand = points

    pose_points = None
    if pose_results.pose_landmarks:
        pose_points = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]

    return {
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pose": pose_points,
        "timestamp_ms": int((frame_idx / fps) * 1000),
    }
