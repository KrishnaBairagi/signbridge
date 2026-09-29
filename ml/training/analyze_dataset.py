#!/usr/bin/env python3
"""
ml/training/analyze_dataset.py

Generates a detailed statistical report and two-hand gesture analysis for the
SignBridge dataset. Saves JSON report to data/dataset_stats.json and Markdown report
to ml/reports/DATASET_REPORT.md (NEVER inside data/raw/).

Usage:
    python -m ml.training.analyze_dataset --landmarks_dir data/landmarks --prepared_npz data/prepared.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_landmarks(landmarks_dir: Path) -> dict:
    total_videos = 0
    class_stats = {}

    overall_h0 = 0
    overall_h1 = 0
    overall_h2 = 0
    overall_frames = 0

    one_hand_classes = 0
    two_hand_classes = 0
    mixed_classes = 0

    for label_dir in sorted(p for p in landmarks_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        json_files = list(label_dir.glob("*.json"))
        n_vids = len(json_files)
        total_videos += n_vids

        c_h0, c_h1, c_h2, c_frames = 0, 0, 0, 0
        for j_path in json_files:
            try:
                data = json.loads(j_path.read_text())
                stats = data.get("hand_stats", {})
                h0 = stats.get("zero_hand", 0)
                h1 = stats.get("one_hand", 0)
                h2 = stats.get("two_hand", 0)
                tot = data.get("total_frames", h0 + h1 + h2)

                c_h0 += h0
                c_h1 += h1
                c_h2 += h2
                c_frames += tot
            except Exception:
                pass

        overall_h0 += c_h0
        overall_h1 += c_h1
        overall_h2 += c_h2
        overall_frames += c_frames

        p0 = (c_h0 / c_frames * 100) if c_frames else 0.0
        p1 = (c_h1 / c_frames * 100) if c_frames else 0.0
        p2 = (c_h2 / c_frames * 100) if c_frames else 0.0

        if p2 >= 50:
            category = "Two-Hand"
            two_hand_classes += 1
        elif p1 >= 50:
            category = "One-Hand"
            one_hand_classes += 1
        else:
            category = "Mixed / Variable"
            mixed_classes += 1

        class_stats[label] = {
            "videos": n_vids,
            "total_frames": c_frames,
            "pct_0_hand": round(p0, 1),
            "pct_1_hand": round(p1, 1),
            "pct_2_hand": round(p2, 1),
            "category": category,
        }

    pct_overall_0 = round((overall_h0 / max(1, overall_frames)) * 100, 1)
    pct_overall_1 = round((overall_h1 / max(1, overall_frames)) * 100, 1)
    pct_overall_2 = round((overall_h2 / max(1, overall_frames)) * 100, 1)

    return {
        "total_classes": len(class_stats),
        "total_videos": total_videos,
        "total_frames": overall_frames,
        "overall_hand_stats": {
            "pct_0_hand": pct_overall_0,
            "pct_1_hand": pct_overall_1,
            "pct_2_hand": pct_overall_2,
        },
        "class_categories": {
            "one_hand_classes": one_hand_classes,
            "two_hand_classes": two_hand_classes,
            "mixed_classes": mixed_classes,
        },
        "class_breakdown": class_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_dir", required=True, help="Path to data/landmarks")
    parser.add_argument("--prepared_npz", default="data/prepared.npz", help="Path to prepared.npz")
    parser.add_argument("--out_json", default="data/dataset_stats.json")
    parser.add_argument("--out_md", default="ml/reports/DATASET_REPORT.md")
    args = parser.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    stats = analyze_landmarks(landmarks_dir)

    # Check prepared npz if available
    npz_path = Path(args.prepared_npz)
    npz_info = {}
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        npz_info = {
            "X_train_shape": list(data["X_train"].shape),
            "X_val_shape": list(data["X_val"].shape),
            "X_test_shape": list(data["X_test"].shape),
            "num_classes": len(data["labels"]),
        }
        stats["npz_info"] = npz_info

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"Saved dataset stats JSON to {out_json}")

    # Generate Markdown Report
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# SignBridge Dataset & Two-Hand Analysis Report\n")
    md.append(f"- **Total Classes**: {stats['total_classes']}")
    md.append(f"- **Total Extracted Videos**: {stats['total_videos']}")
    md.append(f"- **Total Processed Frames**: {stats['total_frames']}")
    md.append(f"- **Overall Hand Detection**: 0-hand: {stats['overall_hand_stats']['pct_0_hand']}%, 1-hand: {stats['overall_hand_stats']['pct_1_hand']}%, 2-hand: {stats['overall_hand_stats']['pct_2_hand']}%\n")

    md.append("## Class Categories")
    md.append(f"- **Two-Hand Predominant Classes**: {stats['class_categories']['two_hand_classes']}")
    md.append(f"- **One-Hand Predominant Classes**: {stats['class_categories']['one_hand_classes']}")
    md.append(f"- **Mixed / Variable Classes**: {stats['class_categories']['mixed_classes']}\n")

    if npz_info:
        md.append("## Model-Ready Dataset (.npz) Summary")
        md.append(f"- **Train Shape**: {npz_info['X_train_shape']} ({npz_info['X_train_shape'][0]} sequences)")
        md.append(f"- **Val Shape**: {npz_info['X_val_shape']} ({npz_info['X_val_shape'][0]} sequences)")
        md.append(f"- **Test Shape**: {npz_info['X_test_shape']} ({npz_info['X_test_shape'][0]} sequences)")
        md.append(f"- **Feature Dimension**: {npz_info['X_train_shape'][2] if len(npz_info['X_train_shape']) > 2 else 'N/A'}")
        md.append(f"- **Sequence Length**: {npz_info['X_train_shape'][1] if len(npz_info['X_train_shape']) > 1 else 'N/A'}\n")

    md.append("## Class Breakdown")
    md.append("| Class Name | Videos | Total Frames | 0-Hand % | 1-Hand % | 2-Hand % | Category |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for c_name, c_data in sorted(stats["class_breakdown"].items()):
        md.append(f"| {c_name} | {c_data['videos']} | {c_data['total_frames']} | {c_data['pct_0_hand']}% | {c_data['pct_1_hand']}% | {c_data['pct_2_hand']}% | {c_data['category']} |")

    out_md.write_text("\n".join(md))
    print(f"Saved Markdown report to {out_md}")


if __name__ == "__main__":
    main()
