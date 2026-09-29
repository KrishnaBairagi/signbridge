#!/usr/bin/env python3
"""
ml/training/prepare_dataset.py

Stage 2: normalize extracted landmark JSONs, pad/truncate to a fixed
sequence length (48), and produce a train/val/test split saved as .npz arrays
ready for ml/training/train.py.

PREVENTS DATA LEAKAGE:
Clips are grouped by recording base session (e.g. `WIN_20231102_12_38_17_Pro`),
ensuring original clips and augmented tilt variants (`_left_tilt`, `_right_tilt`)
stay in the exact same split (Train, Val, or Test).

Usage:
    python -m ml.training.prepare_dataset --landmarks_dir data/landmarks \
        --out data/prepared.npz --max_seq_len 48 --val_split 0.15 --test_split 0.15
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ml.preprocessing.normalize import normalize_frame, frames_to_feature_vector


def get_session_key(json_path: Path) -> str:
    """
    Extract base recording session key from filename stem.
    E.g. 'WIN_20231102_12_38_17_Pro_left_tilt' -> 'WIN_20231102_12_38_17_Pro'
    """
    stem = json_path.stem
    # Remove _left_tilt or _right_tilt suffix if present
    base_stem = re.sub(r'_(left|right)_tilt$', '', stem, flags=re.IGNORECASE)
    return base_stem


def load_and_vectorize(json_path: Path, max_seq_len: int) -> np.ndarray:
    data = json.loads(json_path.read_text())
    frames = data["frames"]
    vectors = [frames_to_feature_vector(normalize_frame(f)) for f in frames]
    arr = np.stack(vectors) if vectors else np.zeros((0, 228), dtype=np.float32)

    if len(arr) >= max_seq_len:
        return arr[:max_seq_len]
    pad = np.zeros((max_seq_len - len(arr), 228), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks_dir", required=True, help="Folder containing extracted landmark JSONs")
    parser.add_argument("--out", required=True, help="Output .npz file path")
    parser.add_argument("--max_seq_len", type=int, default=48)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_classes", type=int, default=0, help="Limit classes for testing")
    args = parser.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    labels = sorted(p.name for p in landmarks_dir.iterdir() if p.is_dir())
    if args.max_classes > 0:
        labels = labels[:args.max_classes]

    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Save explicit class mapping JSON
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_path = out_dir / "class_mapping.json"
    mapping_path.write_text(json.dumps(label_to_idx, indent=2))
    print(f"Saved class mapping to {mapping_path} ({len(labels)} classes).")

    # Group clips by session to prevent data leakage across splits
    # session_map: session_key -> list of (X_vec, label_idx)
    session_map: dict[str, list[tuple[np.ndarray, int]]] = {}

    total_clips = 0
    for label in labels:
        for json_path in sorted((landmarks_dir / label).glob("*.json")):
            s_key = f"{label}::{get_session_key(json_path)}"
            vec = load_and_vectorize(json_path, args.max_seq_len)
            
            if s_key not in session_map:
                session_map[s_key] = []
            session_map[s_key].append((vec, label_to_idx[label]))
            total_clips += 1

    print(f"Loaded {total_clips} clips across {len(session_map)} distinct recording sessions.")

    # Perform session-grouped train/val/test split
    session_keys = sorted(session_map.keys())
    rng = np.random.default_rng(args.seed)
    perm_sessions = rng.permutation(session_keys)

    n_val_sess = int(len(perm_sessions) * args.val_split)
    n_test_sess = int(len(perm_sessions) * args.test_split)

    val_sessions = set(perm_sessions[:n_val_sess])
    test_sessions = set(perm_sessions[n_val_sess:n_val_sess + n_test_sess])
    train_sessions = set(perm_sessions[n_val_sess + n_test_sess:])

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for s_key, items in session_map.items():
        if s_key in train_sessions:
            for vec, l_idx in items:
                X_train.append(vec)
                y_train.append(l_idx)
        elif s_key in val_sessions:
            for vec, l_idx in items:
                X_val.append(vec)
                y_val.append(l_idx)
        elif s_key in test_sessions:
            for vec, l_idx in items:
                X_test.append(vec)
                y_test.append(l_idx)

    X_train = np.stack(X_train) if X_train else np.zeros((0, args.max_seq_len, 228), dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.stack(X_val) if X_val else np.zeros((0, args.max_seq_len, 228), dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)
    X_test = np.stack(X_test) if X_test else np.zeros((0, args.max_seq_len, 228), dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    np.savez(
        args.out,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        labels=np.array(labels),
        feature_dim=np.array([228]),
        max_seq_len=np.array([args.max_seq_len]),
    )

    print(f"Saved prepared dataset to {args.out}:")
    print(f"  Train : {len(X_train)} samples ({len(train_sessions)} sessions)")
    print(f"  Val   : {len(X_val)} samples ({len(val_sessions)} sessions)")
    print(f"  Test  : {len(X_test)} samples ({len(test_sessions)} sessions)")
    print(f"  Feature shape: ({args.max_seq_len}, 228)")


if __name__ == "__main__":
    main()

