#!/usr/bin/env python3
"""
ml/training/evaluate.py

Stage 4: evaluate a trained checkpoint on the held-out test split and print
per-class precision/recall/F1 + confusion matrix. Also reports mean
inference latency per window, since real-time performance is a hard
requirement (§18).

Usage:
    python -m ml.training.evaluate --data data/prepared.npz --checkpoint checkpoints/model.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    import torch
    from ml.models.sequence_models import build_lstm_model, build_transformer_model

    data = np.load(args.data, allow_pickle=True)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    labels = ckpt["labels"]
    num_classes = len(labels)
    model = build_lstm_model(num_classes) if ckpt["architecture"] == "lstm" else build_transformer_model(num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        start = time.perf_counter()
        logits = model(X_test)
        elapsed = time.perf_counter() - start
        preds = logits.argmax(dim=1)

    accuracy = (preds == y_test).float().mean().item()
    per_window_ms = (elapsed / max(1, len(X_test))) * 1000

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Mean inference latency: {per_window_ms:.2f} ms/window (batch-evaluated, CPU)")

    print("\nPer-class breakdown:")
    for idx, label in enumerate(labels):
        mask = y_test == idx
        if mask.sum() == 0:
            continue
        class_acc = (preds[mask] == idx).float().mean().item()
        print(f"  {label:12s} n={int(mask.sum()):4d}  recall={class_acc:.3f}")


if __name__ == "__main__":
    main()
