#!/usr/bin/env python3
"""
ml/training/export_model.py

Stage 5: export a trained checkpoint to ONNX for fast, dependency-light
inference in the backend (avoids requiring the full PyTorch training stack
in production). See §4: "ONNX Runtime ... for optimized inference".

Usage:
    python -m ml.training.export_model --checkpoint checkpoints/model.pt \
        --out checkpoints/model.onnx --max_seq_len 48
"""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_seq_len", type=int, default=48)
    args = parser.parse_args()

    import torch
    from ml.models.sequence_models import build_lstm_model, build_transformer_model, FEATURE_DIM

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    labels = ckpt["labels"]
    model = build_lstm_model(len(labels)) if ckpt["architecture"] == "lstm" else build_transformer_model(len(labels))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.randn(1, args.max_seq_len, FEATURE_DIM)
    torch.onnx.export(
        model, dummy_input, args.out,
        input_names=["landmark_sequence"], output_names=["logits"],
        dynamic_axes={"landmark_sequence": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    labels_path = args.out.rsplit(".", 1)[0] + "_labels.json"
    with open(labels_path, "w") as f:
        json.dump(list(labels), f)

    print(f"Exported ONNX model to {args.out}")
    print(f"Exported label list to {labels_path}")


if __name__ == "__main__":
    main()
