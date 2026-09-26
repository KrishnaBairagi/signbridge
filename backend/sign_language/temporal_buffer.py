"""
sign_language/temporal_buffer.py

Implements §6/§8 of the spec: don't classify every frame independently.
Maintain a rolling window, detect gesture start/end via hand-presence and
motion, debounce repeated identical predictions, and apply a cooldown so
"HELLO HELLO HELLO" doesn't get emitted for one held sign.

This is deliberately independent of the recognition model (heuristic or
trained) -- it only needs recognize_sign() to return {"sign","confidence"}.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .recognizer import recognize_sign, MIN_CONFIDENCE_HIGH, MIN_CONFIDENCE_MEDIUM


@dataclass
class BufferConfig:
    window_size: int = 48          # frames kept for one recognition pass (~1.6s at 30fps)
    min_frames_for_inference: int = 48
    no_hand_frames_to_end_gesture: int = 6   # ~0.2s of no hands => gesture ended
    cooldown_seconds: float = 0.8            # min gap between two emitted signs
    stability_repeats_required: int = 2      # same top prediction N times running => confirm


@dataclass
class RecognitionEvent:
    sign: Optional[str]
    confidence: float
    status: str  # "confirmed" | "tentative" | "low_confidence" | "no_hands"
    model: str


class TemporalSignBuffer:
    """One instance per live session (per WebSocket connection)."""

    def __init__(self, config: BufferConfig | None = None):
        self.config = config or BufferConfig()
        self.frames: deque = deque(maxlen=self.config.window_size)
        self._no_hand_streak = 0
        self._last_emit_time = 0.0
        self._pending_sign: Optional[str] = None
        self._pending_repeats = 0
        self._last_confirmed_sign: Optional[str] = None

    def add_frame(self, landmarks: dict) -> Optional[RecognitionEvent]:
        """
        Feed one frame of normalized landmarks. Returns a RecognitionEvent
        only when there's something new to report (confirmed sign, or a
        state change like "no hands detected"); otherwise returns None so
        callers don't spam the UI/log every frame.
        """
        has_hand = bool(landmarks.get("left_hand") or landmarks.get("right_hand"))

        if not has_hand:
            self._no_hand_streak += 1
            if self._no_hand_streak >= self.config.no_hand_frames_to_end_gesture:
                self.frames.clear()
                self._pending_sign, self._pending_repeats = None, 0
                if self._no_hand_streak == self.config.no_hand_frames_to_end_gesture:
                    return RecognitionEvent(None, 0.0, "no_hands", "n/a")
            return None

        self._no_hand_streak = 0
        self.frames.append(landmarks)

        if len(self.frames) < self.config.min_frames_for_inference:
            return None

        result = recognize_sign(list(self.frames))
        sign, confidence, model = result["sign"], result["confidence"], result["model"]

        if sign is None or sign == "UNKNOWN" or confidence < MIN_CONFIDENCE_MEDIUM:
            self._pending_sign, self._pending_repeats = None, 0
            return RecognitionEvent(sign, confidence, "low_confidence", model)

        # Debounce: require the same sign to appear on consecutive windows
        # before treating it as a real, intentional gesture (not a blip).
        if sign == self._pending_sign:
            self._pending_repeats += 1
        else:
            self._pending_sign, self._pending_repeats = sign, 1

        now = time.monotonic()
        stable_enough = self._pending_repeats >= self.config.stability_repeats_required
        cooled_down = (now - self._last_emit_time) >= self.config.cooldown_seconds
        not_a_repeat_hold = sign != self._last_confirmed_sign or cooled_down

        if stable_enough and cooled_down and not_a_repeat_hold and confidence >= MIN_CONFIDENCE_HIGH:
            self._last_emit_time = now
            self._last_confirmed_sign = sign
            self._pending_repeats = 0
            return RecognitionEvent(sign, confidence, "confirmed", model)

        if stable_enough:
            return RecognitionEvent(sign, confidence, "tentative", model)

        return None
