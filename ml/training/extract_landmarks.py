#!/usr/bin/env python3
"""
ml/training/extract_landmarks.py

Stage 1 of the training pipeline (§19): convert a directory of labeled sign
videos into landmark JSON files, ready for prepare_dataset.py.

Handles nested dataset layouts (e.g. Video_Dataset/Video_Dataset/<LABEL>/<video>.mp4),
incremental caching (skips existing JSONs), parallel multi-process extraction,
and subset filtering for fast testing.

Usage:
    python -m ml.training.extract_landmarks --dataset_root data/raw/isl_video_dataset --out_dir data/landmarks --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ml.feature_extraction.landmarks import extract_landmarks_from_video


def find_label_directories(dataset_root: Path) -> list[Path]:
    """Find all directories containing video files directly under dataset_root."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    label_dirs = []
    
    for path in dataset_root.rglob("*"):
        if path.is_dir():
            has_videos = any(f.suffix.lower() in video_extensions for f in path.iterdir() if f.is_file())
            if has_videos and path.name != "Sample Videos":
                label_dirs.append(path)
                
    return sorted(label_dirs, key=lambda p: p.name.upper())


def process_single_video(args_tuple: tuple[str, str, str, int, bool]) -> tuple[bool, str, str, int, int, int]:
    """Worker function for parallel video extraction."""
    video_path_str, out_path_str, label, sample_fps, overwrite = args_tuple
    video_path = Path(video_path_str)
    out_path = Path(out_path_str)

    if out_path.exists() and not overwrite:
        return True, video_path.name, "skipped", 0, 0, 0

    try:
        frames = extract_landmarks_from_video(str(video_path), sample_fps=sample_fps)
        
        h0_cnt, h1_cnt, h2_cnt = 0, 0, 0
        for f in frames:
            lh = f.get("left_hand") is not None
            rh = f.get("right_hand") is not None
            if lh and rh:
                h2_cnt += 1
            elif lh or rh:
                h1_cnt += 1
            else:
                h0_cnt += 1

        payload = {
            "label": label,
            "video_name": video_path.name,
            "total_frames": len(frames),
            "hand_stats": {"zero_hand": h0_cnt, "one_hand": h1_cnt, "two_hand": h2_cnt},
            "frames": frames,
        }
        out_path.write_text(json.dumps(payload))
        return True, video_path.name, f"{len(frames)} frames", h2_cnt, h1_cnt, h0_cnt
    except Exception as e:
        return False, video_path.name, str(e), 0, 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Folder containing sign videos")
    parser.add_argument("--out_dir", required=True, help="Where to write <LABEL>/<video>.json")
    parser.add_argument("--sample_fps", type=int, default=15)
    parser.add_argument("--max_classes", type=int, default=0, help="Limit number of classes (0 = all)")
    parser.add_argument("--max_videos_per_class", type=int, default=0, help="Limit videos per class (0 = all)")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel CPU workers (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted JSONs")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    label_dirs = find_label_directories(dataset_root)
    if args.max_classes > 0:
        label_dirs = label_dirs[:args.max_classes]

    print(f"Found {len(label_dirs)} label directories for extraction (workers={args.num_workers}, fps={args.sample_fps}).")

    tasks = []
    for label_dir in label_dirs:
        label = label_dir.name.upper().replace(" ", "_")
        out_label_dir = out_dir / label
        out_label_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in video_extensions])
        if args.max_videos_per_class > 0:
            videos = videos[:args.max_videos_per_class]

        for video_path in videos:
            out_path = out_label_dir / (video_path.stem + ".json")
            tasks.append((str(video_path), str(out_path), label, args.sample_fps, args.overwrite))

    print(f"Total video extraction tasks queued: {len(tasks)}")
    start_time = time.time()
    count, skipped, failures = 0, 0, []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_video, t): t for t in tasks}
            for future in as_completed(futures):
                success, v_name, status, h2, h1, h0 = future.result()
                if success:
                    if status == "skipped":
                        skipped += 1
                    else:
                        count += 1
                        print(f"[ok] {v_name} ({status}, 2-hand={h2}, 1-hand={h1})")
                else:
                    failures.append((v_name, status))
                    print(f"[FAIL] {v_name}: {status}")
    else:
        for t in tasks:
            success, v_name, status, h2, h1, h0 = process_single_video(t)
            if success:
                if status == "skipped":
                    skipped += 1
                else:
                    count += 1
                    print(f"[ok] {v_name} ({status}, 2-hand={h2}, 1-hand={h1})")
            else:
                failures.append((v_name, status))
                print(f"[FAIL] {v_name}: {status}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s. Extracted: {count}, Skipped: {skipped}, Failures: {len(failures)}.")
    if failures:
        print("Failures:", failures)


if __name__ == "__main__":
    main()


