#!/usr/bin/env python3
"""
ml/training/train.py

Stage 3: train the LSTM or Transformer sign classifier on the .npz produced
by prepare_dataset.py.

Usage:
    python -m ml.training.train --data data/prepared.npz --model lstm \
        --epochs 30 --batch_size 16 --out checkpoints/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.models.sequence_models import build_lstm_model, build_transformer_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    data = np.load(args.data, allow_pickle=True)
    labels = list(data["labels"])
    num_classes = len(labels)

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    model = build_lstm_model(num_classes) if args.model == "lstm" else build_transformer_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        print(f"epoch {epoch+1}/{args.epochs}  loss={total_loss/len(X_train):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "labels": labels,
                        "architecture": args.model}, str(out_path))

    print(f"Best val_acc={best_val_acc:.4f}. Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
