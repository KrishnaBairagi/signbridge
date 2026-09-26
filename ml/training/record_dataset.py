#!/usr/bin/env python3
"""
ml/training/record_dataset.py

Capture webcam videos for a specific sign label and save them under
`data/raw/<LABEL>/sample_XXX.mp4`, ready for the training pipeline.

Example:
    python -m ml.training.record_dataset --label HELLO --clips 10 --duration 3 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def record_clip(cap: cv2.VideoCapture, output_path: Path, duration_sec: float, fps: int = 30):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    start = time.time()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame from camera for {output_path.name}.")

        writer.write(frame)
        frames += 1

        elapsed = time.time() - start
        if elapsed >= duration_sec:
            break

    writer.release()
    print(f"Saved {output_path} ({frames} frames)")


def main():
    parser = argparse.ArgumentParser(description="Record labelled sign videos from the webcam.")
    parser.add_argument("--label", required=True, help="Sign label, e.g. HELLO or THANK_YOU")
    parser.add_argument("--clips", type=int, default=10, help="How many clips to record")
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per clip")
    parser.add_argument("--output-dir", default="data/raw", help="Root output directory")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    label_dir = output_root / args.label.upper().replace(" ", "_")
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    print(f"Recording {args.clips} clips for label '{args.label}' into {label_dir}")
    print("Instructions:")
    print("  1. Position your hand clearly in the frame")
    print("  2. Perform the sign naturally")
    print("  3. Keep the sign stable for the full recording window")
    print("  4. Press Ctrl+C to stop early")

    try:
        for idx in range(1, args.clips + 1):
            target = label_dir / f"sample_{idx:03d}.mp4"
            print(f"\nRecording clip {idx}/{args.clips}: {target.name}")
            record_clip(cap, target, args.duration)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
